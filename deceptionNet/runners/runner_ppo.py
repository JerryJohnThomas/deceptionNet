"""PPO fine-tuning loop for the MindGames agent."""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import argparse
import csv
from datetime import datetime
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast


CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT.parent))

from deceptionNet.config import TrainingConfig, ModelDims, DEFAULT_INFERENCE_CONFIG
from deceptionNet.datatypes import StepMasks
from deceptionNet.utils import entropy_from_logits, masked_log_softmax, masked_softmax, save_jsonl, debug_log
from deceptionNet.online_agent import MindGamesAgent
from deceptionNet.human_play import SimpleSelfPlayEnv, TextArenaOfflineWrapper, run_human_play_session

from deceptionNet.agents.belief_net import BeliefState
from deceptionNet.agents.featurizer import ObservationFeatures
from deceptionNet.agents.policy import MultiHeadPolicy
from deceptionNet.agents.text_to_state_mapper import TextToStateMapper
from deceptionNet.runners.buffers import RolloutBuffer, RolloutTensors


from deceptionNet.rewards import RewardConfig, compute_reward_components, compute_team_outcome_bonus

@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    clip_frac: float


class PPORunner:
    """Handles PPO optimisation over stored rollouts."""

    def __init__(self, policy: MultiHeadPolicy, config: TrainingConfig) -> None:
        self.policy = policy
        self.config = config
        self.device = torch.device(config.device)
        self.policy.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.optim.learning_rate,
            betas=config.optim.betas,
            weight_decay=config.optim.weight_decay,
        )
        self.scaler = GradScaler(enabled=config.mixed_precision and self.device.type == "cuda")

    def update(self, buffer: RolloutBuffer, last_value: torch.Tensor) -> Dict[str, float]:
        cfg = self.config.ppo
        last_value = last_value.detach().cpu()
        advantages, returns = buffer.advantages_and_returns(last_value, cfg.gamma, cfg.gae_lambda)
        adv_flat = advantages.view(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        returns_flat = returns.view(-1)

        rollout = buffer.stacked_features()
        masks = buffer.stacked_masks()
        prev_belief = buffer.stacked_prev_belief()
        prev_memory = buffer.stacked_prev_memory()
        actions = buffer.stacked_actions()
        old_logprobs = buffer.stacked_logprobs()

        total_steps = adv_flat.shape[0]
        minibatch_size = max(1, total_steps // cfg.num_minibatches)
        stats_accumulator = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        updates = 0

        for _ in range(cfg.num_epochs):
            for idx in buffer.iterate_minibatches(minibatch_size):
                feat_batch = self._to_device_features(self._gather_features(rollout, idx))
                mask_batch = self._to_device_masks(self._gather_masks(masks, idx))
                belief_batch = self._to_device_belief(self._gather_belief(prev_belief, idx))
                memory_batch = prev_memory.view(-1, prev_memory.shape[-1])[idx].to(self.device)

                action_batch = {k: v.view(-1)[idx].to(self.device) for k, v in actions.items()}
                old_lp_batch = {k: v.view(-1)[idx].to(self.device) for k, v in old_logprobs.items()}
                adv_batch = adv_flat[idx].to(self.device)
                returns_batch = returns_flat[idx].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                with autocast(enabled=self.scaler.is_enabled()):
                    shared, outputs = self.policy(feat_batch, belief_batch, memory_batch)
                    new_logprob_dict = self._compute_logprobs(outputs, mask_batch, action_batch)
                    total_new_logprob = sum(new_logprob_dict.values())
                    total_old_logprob = sum(old_lp_batch.values())

                    ratio = torch.exp(total_new_logprob - total_old_logprob)
                    surr1 = ratio * adv_batch
                    surr2 = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * adv_batch
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = F.mse_loss(outputs.values, returns_batch)
                    entropy = self._compute_entropy(outputs, mask_batch).mean()

                    loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy

                self.scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()

                approx_kl = (total_old_logprob - total_new_logprob).mean().item()
                clip_frac = ((torch.abs(ratio - 1.0) > cfg.clip_ratio).float().mean().item())

                stats_accumulator["policy_loss"] += policy_loss.item()
                stats_accumulator["value_loss"] += value_loss.item()
                stats_accumulator["entropy"] += entropy.item()
                stats_accumulator["approx_kl"] += approx_kl
                stats_accumulator["clip_frac"] += clip_frac
                updates += 1

        if updates == 0:
            return {k: 0.0 for k in stats_accumulator}
        return {k: v / updates for k, v in stats_accumulator.items()}

    def _to_device_features(self, features: ObservationFeatures) -> ObservationFeatures:
        return ObservationFeatures(
            player_features=features.player_features.to(self.device),
            global_features=features.global_features.to(self.device),
            conversation_embedding=features.conversation_embedding.to(self.device),
            alive_mask=features.alive_mask.to(self.device),
            mafia_mask=features.mafia_mask.to(self.device),
            self_index=features.self_index.to(self.device),
            role_index=features.role_index.to(self.device),
            phase_encoding=features.phase_encoding.to(self.device),
            vote_matrix=features.vote_matrix.to(self.device),
            mention_scores=features.mention_scores.to(self.device),
            sentiment_scores=features.sentiment_scores.to(self.device),
            support_scores=features.support_scores.to(self.device),
            accusations=features.accusations.to(self.device),
            contradiction_scores=features.contradiction_scores.to(self.device),
            bandwagon_scores=features.bandwagon_scores.to(self.device),
            round_index=features.round_index.to(self.device),
        )

    def _to_device_masks(self, masks: StepMasks) -> StepMasks:
        return StepMasks(
            alive_mask=masks.alive_mask.to(self.device),
            mafia_mask=masks.mafia_mask.to(self.device),
            self_index=masks.self_index.to(self.device),
            night_valid=masks.night_valid.to(self.device),
            vote_valid=masks.vote_valid.to(self.device),
            talk_target_valid=masks.talk_target_valid.to(self.device),
        )

    def _to_device_belief(self, belief: BeliefState) -> BeliefState:
        return BeliefState(
            player_embeddings=belief.player_embeddings.to(self.device),
            global_hidden=belief.global_hidden.to(self.device),
            role_logits=belief.role_logits.to(self.device),
            suspicion=belief.suspicion.to(self.device),
            trust=belief.trust.to(self.device),
        )

    def _gather_features(self, tensors: RolloutTensors, idx: torch.Tensor) -> ObservationFeatures:
        T, B = tensors.player_features.shape[:2]
        flat = lambda x: x.view(T * B, *x.shape[2:])
        return ObservationFeatures(
            player_features=flat(tensors.player_features)[idx],
            global_features=flat(tensors.global_features)[idx],
            conversation_embedding=flat(tensors.conversation_embedding)[idx],
            alive_mask=flat(tensors.alive_mask)[idx],
            mafia_mask=flat(tensors.mafia_mask)[idx],
            self_index=flat(tensors.self_index)[idx],
            role_index=flat(tensors.role_index)[idx],
            phase_encoding=flat(tensors.phase_encoding)[idx],
            vote_matrix=flat(tensors.vote_matrix)[idx],
            mention_scores=flat(tensors.mention_scores)[idx],
            sentiment_scores=flat(tensors.sentiment_scores)[idx],
            support_scores=flat(tensors.support_scores)[idx],
            accusations=flat(tensors.accusations)[idx],
            contradiction_scores=flat(tensors.contradiction_scores)[idx],
            bandwagon_scores=flat(tensors.bandwagon_scores)[idx],
            round_index=flat(tensors.round_index)[idx],
        )

    def _gather_masks(self, masks: StepMasks, idx: torch.Tensor) -> StepMasks:
        T = masks.alive_mask.shape[0]
        B = masks.alive_mask.shape[1]
        reshape = lambda x: x.view(T * B, *x.shape[2:])[idx]
        return StepMasks(
            alive_mask=reshape(masks.alive_mask),
            mafia_mask=reshape(masks.mafia_mask),
            self_index=reshape(masks.self_index),
            night_valid=reshape(masks.night_valid),
            vote_valid=reshape(masks.vote_valid),
            talk_target_valid=reshape(masks.talk_target_valid),
        )

    def _gather_belief(self, belief: Dict[str, torch.Tensor], idx: torch.Tensor) -> BeliefState:
        T, B = belief["player_embeddings"].shape[:2]
        flat = lambda x: x.view(T * B, *x.shape[2:])[idx]
        return BeliefState(
            player_embeddings=flat(belief["player_embeddings"]),
            global_hidden=flat(belief["global_hidden"]),
            role_logits=flat(belief["role_logits"]),
            suspicion=flat(belief["suspicion"]),
            trust=flat(belief["trust"]),
        )

    def _safe_mask(self, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        valid = mask.sum(dim=-1, keepdim=True) > 0
        safe = torch.where(valid, mask, torch.ones_like(mask))
        return safe, valid.squeeze(-1)

    def _compute_logprobs(
        self,
        outputs,
        masks: StepMasks,
        actions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        logprobs: Dict[str, torch.Tensor] = {}

        vote_mask_safe, _ = self._safe_mask(masks.vote_valid)
        vote_logprob = masked_log_softmax(outputs.vote_logits, vote_mask_safe).gather(
            1, actions["vote"].unsqueeze(-1)
        ).squeeze(-1)
        logprobs["vote"] = vote_logprob

        intent_logprob = torch.log_softmax(outputs.talk_intent_logits, dim=-1).gather(
            1, actions["talk_intent"].unsqueeze(-1)
        ).squeeze(-1)
        logprobs["talk_intent"] = intent_logprob

        slot_actions = actions["talk_target"]
        slot_mask = slot_actions >= 0
        slot_mask_safe, _ = self._safe_mask(masks.talk_target_valid)
        slot_logits = masked_log_softmax(outputs.talk_slot_logits, slot_mask_safe)
        slot_lp = torch.zeros_like(intent_logprob)
        if slot_mask.any():
            slot_lp[slot_mask] = slot_logits[slot_mask].gather(
                1, slot_actions[slot_mask].unsqueeze(-1)
            ).squeeze(-1)
        logprobs["talk_target"] = slot_lp

        night_actions = actions["night"]
        night_mask_valid = night_actions >= 0
        night_mask_safe, _ = self._safe_mask(masks.night_valid)
        night_logits = masked_log_softmax(outputs.night_logits, night_mask_safe)
        night_lp = torch.zeros_like(intent_logprob)
        if night_mask_valid.any():
            night_lp[night_mask_valid] = night_logits[night_mask_valid].gather(
                1, night_actions[night_mask_valid].unsqueeze(-1)
            ).squeeze(-1)
        logprobs["night"] = night_lp

        return logprobs

    def _compute_entropy(self, outputs, masks: StepMasks) -> torch.Tensor:
        vote_mask_safe, _ = self._safe_mask(masks.vote_valid)
        slot_mask_safe, _ = self._safe_mask(masks.talk_target_valid)
        night_mask_safe, _ = self._safe_mask(masks.night_valid)
        vote_entropy = entropy_from_logits(outputs.vote_logits, vote_mask_safe)
        intent_entropy = entropy_from_logits(outputs.talk_intent_logits)
        slot_entropy = entropy_from_logits(outputs.talk_slot_logits, slot_mask_safe)
        night_entropy = entropy_from_logits(outputs.night_logits, night_mask_safe)
        return (vote_entropy + intent_entropy + slot_entropy + night_entropy) / 4.0


@dataclass
class CollectedTransition:
    episode_id: int
    step_idx: int
    player: int
    role: str
    phase: str
    state: Dict[str, Any]
    features: Dict[str, Any]
    memory_in: Dict[str, Any]
    memory_out: Dict[str, Any]
    logits: Dict[str, Any]
    value: float
    actions: Dict[str, Any]
    logprobs: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]
    reward_components: Dict[str, float] = field(default_factory=dict)


class CollectorBuffer:
    def __init__(self) -> None:
        self.data: List[CollectedTransition] = []

    def __len__(self) -> int:
        return len(self.data)

    def add(self, transition: CollectedTransition) -> None:
        self.data.append(transition)

    def save(self, base_path: Path, episode_index: Optional[List[Dict[str, Any]]] = None) -> None:
        records = [asdict(item) for item in self.data]
        base_path = Path(base_path)
        stem_path = base_path.with_suffix('') if base_path.suffix else base_path
        json_path = stem_path.with_suffix('.json')
        jsonl_path = stem_path.with_suffix('.jsonl')
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open('w', encoding='utf-8') as handle:
            json.dump(records, handle, indent=2)
        save_jsonl(records, jsonl_path)
        if episode_index is not None:
            index_path = stem_path.with_suffix('.index.json')
            with index_path.open('w', encoding='utf-8') as handle:
                json.dump(episode_index, handle, indent=2)


def _extract_vote_target(action_text: str) -> Optional[int]:
    digits = ''.join(ch for ch in action_text if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits[-1])
    except ValueError:
        return None


def run_selfplay(
    env: Any,
    agents: List[MindGamesAgent],
    buffer: CollectorBuffer,
    total_steps: int,
    use_shaping: bool = False,
    reward_config: Optional[RewardConfig] = None,
    debug: bool = False,
    debug_log_path: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Collect on-policy rollouts from an environment using shared weights.

    Args:
        env: Environment wrapper implementing reset/get_observation/step/close_episode.
        use_shaping: Whether to compute shaped reward components for each step.
        reward_config: Optional overrides for shaped reward hyper-parameters.
    """

    episode_id = 0
    step_idx = 0
    collected = 0
    episode_index: List[Dict[str, Any]] = []
    player_last_transition: Dict[int, int] = {}

    cfg = reward_config or RewardConfig()

    debug_path: Optional[Path] = None
    if debug:
        if debug_log_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            debug_log_path = Path('logs') / f'debug_run_{timestamp}.txt'
        debug_path = Path(debug_log_path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
    elif debug_log_path is not None:
        debug_path = Path(debug_log_path)

    def log_debug(message: str) -> None:
        if not debug:
            return
        line = f"[DEBUG] {message}"
        print(line)
        debug_log(line, path=debug_path)

    if debug:
        log_debug(f'Starting rollout collection: total_steps={total_steps}, players={len(agents)}')

    player_id, observation = env.reset()        # picks a random player id 
    for agent in agents:
        agent.reset_state()

    while collected < total_steps:
        if debug and isinstance(observation, dict):
            recent = observation.get('recent_text') or observation.get('current_message')
            if isinstance(recent, dict):
                recent = recent.get('text')
            recent_display = str(recent)[:120] if recent is not None else 'None'
            log_debug(f"Observation for player {player_id}: phase={observation.get('phase', '?')} recent={recent_display}")
        
        if debug and isinstance(observation, dict):
            print("[PHASE CHECK]", observation.get("phase"), flush=True)

        agent = agents[player_id]
        details = agent.act_with_details(observation, deterministic=False, log=False)       # This is get action method for all phases

        done, info = env.step(player_id, details["message"])        # steps one action for that player, episode does not move
        info = info or {}

        reward_components: Dict[str, float] = {}                    # reward calculation
        reward_value = 0.0
        if use_shaping:
            belief_snapshot = details.get("memory_out", {}).get("belief", {})
            reward_components = compute_reward_components(
                details["state"],
                details["features"],
                belief_snapshot,
                details["actions"],
                details.get("message"),
                info,
                cfg,
            )
            reward_value = float(reward_components.get("total", 0.0))

        # transition for buffer 
        transition = CollectedTransition(
            episode_id=episode_id,
            step_idx=step_idx,
            player=player_id,
            role=details["state"].get("role", "unknown"),
            phase=details["state"].get("phase", ""),
            state=details["state"],
            features=details["features"],
            memory_in=details["memory_in"],
            memory_out=details["memory_out"],
            logits=details["logits"],
            value=details["value"],
            actions=details["actions"],
            logprobs=details["logprobs"],
            reward=reward_value,
            done=done,
            info=info,
            reward_components=reward_components,
        )
        # Capture raw LLM output if available
        if "message" in details:        # details is the action done by the agent
            transition.info["talk_text"] = details["message"]

        buffer.add(transition)
        player_last_transition[player_id] = len(buffer.data) - 1

        if debug:               # debugger self play output
            log_debug(f"Player {player_id} presenter output: {details['message']}")
            if use_shaping:
                log_debug(f"Player {player_id} reward total={transition.reward} components={reward_components}")
                print(f"[LLM SENTENCE] Player {player_id}: {details['message']}")


        collected += 1
        step_idx += 1

        # episode close, total episode some reward update computation 
        if done:
            episode_rewards = None
            episode_meta = None
            close_fn = getattr(env, "close_episode", None)
            if callable(close_fn):
                try:
                    episode_rewards, episode_meta = close_fn()
                except Exception:
                    episode_rewards, episode_meta = None, None

            if use_shaping and episode_rewards is not None:
                for pid, idx in player_last_transition.items():
                    if idx is None or not (0 <= idx < len(buffer.data)):
                        continue
                    outcome_bonus = compute_team_outcome_bonus(episode_rewards, pid, cfg)
                    if outcome_bonus == 0.0:
                        continue
                    if debug:
                        log_debug(f"Outcome bonus for player {pid}: {outcome_bonus}")
                    transition_ref = buffer.data[idx]
                    prev_total = transition_ref.reward_components.get("total", transition_ref.reward)
                    transition_ref.reward += outcome_bonus
                    updated_components = dict(transition_ref.reward_components)
                    updated_components["outcome"] = updated_components.get("outcome", 0.0) + outcome_bonus
                    updated_components["total"] = prev_total + outcome_bonus
                    if episode_meta and pid < len(episode_meta):
                        summary = episode_meta[pid]
                        if isinstance(summary, dict):
                            transition_ref.info.setdefault("episode_summary", summary)
                    transition_ref.reward_components = updated_components

            episode_entry = {"episode_id": episode_id, "length": step_idx}
            if use_shaping and episode_rewards is not None:
                try:
                    episode_entry["outcome"] = [float(value) for value in episode_rewards]
                except Exception:
                    pass
            episode_index.append(episode_entry)
            if debug:
                log_debug(f"Episode {episode_entry['episode_id']} finished length={episode_entry['length']}")
            episode_id += 1
            step_idx = 0
            player_last_transition.clear()

            player_id, observation = env.reset()
            for agent in agents:
                agent.reset_state()
        else:
            # this rotates the playerid so that other players can play in the same episode
            player_id, observation = env.get_observation()      

    if debug:
        log_debug(f'Completed rollout collection: total_steps_recorded={collected}')
    return episode_index


def load_rollout_records(path: Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    if path.suffix == '.jsonl' or path.with_suffix('.jsonl').exists():
        jsonl_path = path if path.suffix == '.jsonl' else path.with_suffix('.jsonl')
        with jsonl_path.open('r', encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    json_path = path if path.suffix == '.json' else path.with_suffix('.json')
    with json_path.open('r', encoding='utf-8') as handle:
        data = json.load(handle)
        if isinstance(data, list):
            records.extend(data)
    return records


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_value = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1.0 - dones[t]) * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


class RolloutDataset:
    def __init__(self, records: List[Dict[str, Any]], gamma: float = 0.99, lam: float = 0.95) -> None:
        self.records = sorted(records, key=lambda item: (item.get('episode_id', 0), item.get('step_idx', 0)))
        self._compute_advantages(gamma, lam)
        advantages = np.array([rec['advantage'] for rec in self.records], dtype=np.float32)
        adv_mean = advantages.mean() if advantages.size else 0.0
        adv_std = advantages.std() if advantages.size else 1.0
        adv_std = adv_std if adv_std > 1e-8 else 1.0
        for rec in self.records:
            rec['advantage'] = (rec['advantage'] - adv_mean) / adv_std
        self.mean_return = float(np.mean([rec['return'] for rec in self.records])) if self.records else 0.0

    def _compute_advantages(self, gamma: float, lam: float) -> None:
        episodes: Dict[int, List[Dict[str, Any]]] = {}
        for rec in self.records:
            episodes.setdefault(rec.get('episode_id', 0), []).append(rec)
        for episode_id, steps in episodes.items():
            steps.sort(key=lambda item: item.get('step_idx', 0))
            rewards = np.array([step.get('reward', 0.0) for step in steps], dtype=np.float32)
            values = np.array([step.get('value', 0.0) for step in steps], dtype=np.float32)
            dones = np.array([1.0 if step.get('done', False) else 0.0 for step in steps], dtype=np.float32)
            advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
            for step, advantage, ret in zip(steps, advantages, returns):
                step['advantage'] = float(advantage)
                step['return'] = float(ret)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


class PPOTrainer:
    def __init__(self, policy: MultiHeadPolicy, optimizer: torch.optim.Optimizer, clip_range: float, vf_coef: float, ent_coef: float, consistency_weight: float, device: torch.device) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.clip_range = clip_range
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.consistency_weight = consistency_weight
        self.device = device
        self.max_grad_norm = 0.5

    def train_epoch(self, dataset: RolloutDataset, batch_size: int) -> Dict[str, float]:
        self.policy.train()
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        total_clip_frac = 0.0
        total_consistency = 0.0
        num_updates = 0

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            batch_records = [dataset[idx] for idx in batch_indices]
            batch_data = self._prepare_batch(batch_records)
            stats = self._update_batch(batch_data)
            total_policy_loss += stats['policy_loss']
            total_value_loss += stats['value_loss']
            total_entropy += stats['entropy']
            total_clip_frac += stats['clip_frac']
            total_consistency += stats['consistency']
            num_updates += 1

        if num_updates == 0:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'entropy': 0.0, 'clip_frac': 0.0, 'consistency': 0.0}

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'clip_frac': total_clip_frac / num_updates,
            'consistency': total_consistency / num_updates,
        }

    def _prepare_batch(self, batch_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        features = stack_features(batch_records, self.device)
        belief_state = stack_belief_states(batch_records, self.device)
        memory_state = stack_memory_states(batch_records, self.device)
        actions = stack_actions(batch_records, self.device)
        old_logprob_sum = torch.tensor([rec['logprobs']['sum'] for rec in batch_records], dtype=torch.float32, device=self.device)
        advantages = torch.tensor([rec['advantage'] for rec in batch_records], dtype=torch.float32, device=self.device)
        returns = torch.tensor([rec['return'] for rec in batch_records], dtype=torch.float32, device=self.device)
        return {
            'features': features,
            'belief_state': belief_state,
            'memory_state': memory_state,
            'actions': actions,
            'old_logprob_sum': old_logprob_sum,
            'advantages': advantages,
            'returns': returns,
        }

    def _update_batch(self, batch: Dict[str, Any]) -> Dict[str, float]:
        features = batch['features']
        belief_state = batch['belief_state']
        memory_state = batch['memory_state']
        actions = batch['actions']
        old_logprob_sum = batch['old_logprob_sum']
        advantages = batch['advantages']
        returns = batch['returns']

        shared, outputs = self.policy(features, belief_state, memory_state)

        new_logprob_dict = compute_logprobs(outputs, shared.masks, actions, self.device)
        new_logprob_sum = new_logprob_dict['sum']

        ratio = torch.exp(new_logprob_sum - old_logprob_sum)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        values = outputs.values.view(-1)
        value_loss = F.mse_loss(values, returns)

        entropy = compute_entropy(outputs, shared.masks).mean()
        consistency_loss = self._compute_consistency_loss(shared, outputs)

        loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy + self.consistency_weight * consistency_loss

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        clip_frac = (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'clip_frac': clip_frac,
            'consistency': consistency_loss.item(),
        }


    def _head_alignment_loss(
        self,
        logits: torch.Tensor,
        action_mask: Optional[torch.Tensor],
        alive_mask: torch.Tensor,
        target_distribution: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if logits is None or action_mask is None:
            return None
        action_mask = action_mask.float()
        alive_mask = alive_mask.float()
        valid_rows = (action_mask.sum(dim=-1, keepdim=True) > 0).float()
        if valid_rows.sum() == 0:
            return None
        safe_mask = torch.where(valid_rows > 0, action_mask, alive_mask)
        probs = masked_softmax(logits, safe_mask, dim=-1)
        probs = self._normalise_distribution(probs, alive_mask)
        diff = (probs - target_distribution) ** 2
        loss = (diff * alive_mask).sum(dim=-1) * valid_rows.squeeze(-1)
        return loss

    def _normalise_distribution(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.float()
        values = values.float()
        masked = values * mask
        denom = masked.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return masked / denom

    def _compute_consistency_loss(self, shared, outputs) -> torch.Tensor:
        alive_mask = shared.masks.alive_mask.float()
        target = self._normalise_distribution(shared.belief.suspicion, alive_mask)

        loss_terms = []

        vote_loss = self._head_alignment_loss(outputs.vote_logits, getattr(shared.masks, 'vote_valid', None), alive_mask, target)
        if vote_loss is not None:
            loss_terms.append(vote_loss)

        slot_loss = self._head_alignment_loss(outputs.talk_slot_logits, getattr(shared.masks, 'talk_target_valid', None), alive_mask, target)
        if slot_loss is not None:
            loss_terms.append(slot_loss)

        night_loss = self._head_alignment_loss(outputs.night_logits, getattr(shared.masks, 'night_valid', None), alive_mask, target)
        if night_loss is not None:
            loss_terms.append(night_loss)

        if not loss_terms:
            return torch.zeros((), device=self.device)

        stacked = torch.stack(loss_terms, dim=0)
        return stacked.mean(dim=0).mean()

def stack_features(records: List[Dict[str, Any]], device: torch.device) -> ObservationFeatures:
    def _tensor(data, dtype=torch.float32):
        return torch.tensor(data, dtype=dtype, device=device)

    feature_dicts = [rec['features'] for rec in records]
    player_features = _tensor([feat['player_features'] for feat in feature_dicts])
    global_features = _tensor([feat['global_features'] for feat in feature_dicts])
    conversation_embedding = _tensor([feat['conversation_embedding'] for feat in feature_dicts])
    alive_mask = _tensor([feat['alive_mask'] for feat in feature_dicts])
    mafia_mask = _tensor([feat['mafia_mask'] for feat in feature_dicts])
    self_index = torch.tensor([feat.get('self_index', 0) for feat in feature_dicts], dtype=torch.long, device=device)
    role_index = torch.tensor([feat.get('role_index', 0) for feat in feature_dicts], dtype=torch.long, device=device)
    phase_encoding = _tensor([feat['phase_encoding'] for feat in feature_dicts])
    vote_matrix = _tensor([feat['vote_matrix'] for feat in feature_dicts])
    mention_scores = _tensor([feat['mention_scores'] for feat in feature_dicts])
    sentiment_scores = _tensor([feat['sentiment_scores'] for feat in feature_dicts])
    support_scores = _tensor([feat['support_scores'] for feat in feature_dicts])
    accusations = _tensor([feat['accusations'] for feat in feature_dicts])
    contradiction_scores = _tensor([feat['contradiction_scores'] for feat in feature_dicts])
    bandwagon_scores = _tensor([feat['bandwagon_scores'] for feat in feature_dicts])
    round_index = _tensor([feat['round_index'] for feat in feature_dicts])

    return ObservationFeatures(
        player_features=player_features,
        global_features=global_features,
        conversation_embedding=conversation_embedding,
        alive_mask=alive_mask,
        mafia_mask=mafia_mask,
        self_index=self_index,
        role_index=role_index,
        phase_encoding=phase_encoding,
        vote_matrix=vote_matrix,
        mention_scores=mention_scores,
        sentiment_scores=sentiment_scores,
        support_scores=support_scores,
        accusations=accusations,
        contradiction_scores=contradiction_scores,
        bandwagon_scores=bandwagon_scores,
        round_index=round_index,
    )


def stack_belief_states(records: List[Dict[str, Any]], device: torch.device) -> BeliefState:
    belief_dicts = [rec["memory_in"]["belief"] for rec in records]
    player_embeddings = torch.tensor([b["player_embeddings"] for b in belief_dicts], dtype=torch.float32, device=device)
    if player_embeddings.dim() == 4:
        player_embeddings = player_embeddings.squeeze(1)
    global_hidden = torch.tensor([b["global_hidden"] for b in belief_dicts], dtype=torch.float32, device=device)
    if global_hidden.dim() == 3:
        global_hidden = global_hidden.squeeze(1)
    role_logits = torch.tensor([b["role_logits"] for b in belief_dicts], dtype=torch.float32, device=device)
    if role_logits.dim() == 4:
        role_logits = role_logits.squeeze(1)
    suspicion = torch.tensor([b["suspicion"] for b in belief_dicts], dtype=torch.float32, device=device)
    if suspicion.dim() == 3:
        suspicion = suspicion.squeeze(1)
    trust = torch.tensor([b["trust"] for b in belief_dicts], dtype=torch.float32, device=device)
    if trust.dim() == 3:
        trust = trust.squeeze(1)
    return BeliefState(
        player_embeddings=player_embeddings,
        global_hidden=global_hidden,
        role_logits=role_logits,
        suspicion=suspicion,
        trust=trust,
    )


def stack_memory_states(records: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
    memory_vectors = [rec["memory_in"].get("policy", []) for rec in records]
    memory_tensor = torch.tensor(memory_vectors, dtype=torch.float32, device=device)
    if memory_tensor.dim() == 3:
        memory_tensor = memory_tensor.squeeze(1)
    return memory_tensor


def stack_actions(records: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
    def _tensor_key(key: str, dtype=torch.long, default=-1):
        values = []
        for rec in records:
            value = rec['actions'].get(key, default)
            if isinstance(value, list):
                value = value[0] if value else default
            values.append(value)
        return torch.tensor(values, dtype=dtype, device=device)

    return {
        'night': _tensor_key('night'),
        'vote': _tensor_key('vote'),
        'talk_intent': _tensor_key('talk_intent'),
        'talk_target': _tensor_key('talk_target'),
    }


def compute_logprobs(outputs, masks, actions: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    result: Dict[str, torch.Tensor] = {}

    vote_logprob = masked_log_softmax(outputs.vote_logits, masks.vote_valid)
    result_vote = vote_logprob.gather(1, actions['vote'].unsqueeze(-1)).squeeze(-1)
    result['vote'] = result_vote

    intent_logprob = torch.log_softmax(outputs.talk_intent_logits, dim=-1)
    result_intent = intent_logprob.gather(1, actions['talk_intent'].unsqueeze(-1)).squeeze(-1)
    result['intent'] = result_intent

    slot_mask = actions['talk_target'] >= 0
    slot_logprob = masked_log_softmax(outputs.talk_slot_logits, masks.talk_target_valid)
    result_slot = torch.zeros_like(result_intent)
    if slot_mask.any():
        result_slot[slot_mask] = slot_logprob[slot_mask, actions['talk_target'][slot_mask]]
    result['slot'] = result_slot

    night_mask = actions['night'] >= 0
    night_logprob = masked_log_softmax(outputs.night_logits, masks.night_valid)
    result_night = torch.zeros_like(result_intent)
    if night_mask.any():
        result_night[night_mask] = night_logprob[night_mask, actions['night'][night_mask]]
    result['night'] = result_night

    result['sum'] = result_vote + result_intent + result_slot + result_night
    return result


def compute_entropy(outputs, masks) -> torch.Tensor:
    vote_entropy = entropy_from_logits(outputs.vote_logits, masks.vote_valid)
    intent_entropy = entropy_from_logits(outputs.talk_intent_logits)
    slot_entropy = entropy_from_logits(outputs.talk_slot_logits, masks.talk_target_valid)
    night_entropy = entropy_from_logits(outputs.night_logits, masks.night_valid)
    return (vote_entropy + night_entropy + slot_entropy + intent_entropy) / 4.0



def compute_belief_alignment_reward(*_args, **_kwargs) -> float:
    return 0.0


def compute_team_win_bonus(*_args, **_kwargs) -> float:
    return 0.0


def compute_contradiction_penalty(*_args, **_kwargs) -> float:
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='PPO rollout collection and training')
    parser.add_argument('--mode', choices=['collect', 'train'], default='collect', help='collect rollouts or train PPO')
    parser.add_argument('--steps', type=int, default=128, help='Number of environment steps to collect')
    parser.add_argument('--num-players', type=int, default=6, help='Number of self-play agents')
    parser.add_argument('--episode-length', type=int, default=60, help='Episode length for the mock environment')
    parser.add_argument('--env-source', choices=['synthetic', 'textarena'], default='synthetic', help='Select rollout environment source')
    parser.add_argument('--shaped', action='store_true', help='Enable shaped rewards during rollout collection')
    parser.add_argument('--weights', type=str, default=str(Path('deceptionNet/weights-il-v6.pt')), help='Path to IL weights for agents')
    parser.add_argument('--output', type=str, default=str(Path('logs/ppo_rollout_buffer')), help='Base path for collector outputs or PPO weights')
    parser.add_argument('--buffer', type=str, default=str(Path('logs/ppo_rollout_buffer.jsonl')), help='Path to rollout buffer for training')
    parser.add_argument('--from-il', type=str, default=str(Path('deceptionNet/weights-il-v6.pt')), help='Initial weights for PPO training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of PPO training epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Mini-batch size for PPO')
    parser.add_argument('--lr', type=float, default=3e-5, help='Learning rate for PPO')
    parser.add_argument('--clip', type=float, default=0.2, help='PPO clip range')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='Value loss coefficient')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='Entropy coefficient')
    parser.add_argument('--use-llm-listener', action='store_true', help='Enable LLM listener during rollouts')
    parser.add_argument('--llm-listener-model', type=str, default=None, help='Override model name for LLM listener')
    parser.add_argument('--use-llm-presenter', action='store_true', help='Enable LLM presenter for rollout agents')
    parser.add_argument('--presenter-model', type=str, default=None, help='HF model to use for the LLM presenter')
    parser.add_argument('--presenter-style', type=str, default='neutral', help='Tone requested from the LLM presenter')
    parser.add_argument('--presenter-max-lines', type=int, default=1, help='Maximum number of sentences emitted by the presenter')
    parser.add_argument('--device-preference', choices=['auto', 'gpu', 'cpu', 'template'], default='auto', help='Preferred device tier for LLM components')
    parser.add_argument('--human-play', action='store_true', help='Enable interactive human control for one player slot')
    parser.add_argument('--human-player', type=int, default=0, help='Player index controlled by the human')
    parser.add_argument('--human-max-steps', type=int, default=0, help='Optional max steps for human mode (0 = unlimited)')
    parser.add_argument('--consistency-weight', type=float, default=0.1, help='Belief-action consistency regulariser weight')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='GAE lambda')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run training on')
    parser.add_argument('--debug-talk', action='store_true', help='Enable verbose presenter/listener debug logs')
    return parser.parse_args()


def collect_main(args: argparse.Namespace) -> None:
    if args.env_source == 'textarena':
        env = TextArenaOfflineWrapper(num_players=args.num_players)
    else:
        env = SimpleSelfPlayEnv(num_players=args.num_players, episode_length=args.episode_length)

    debug_path = None
    if args.debug_talk:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        debug_path = Path('logs') / f'debug_run_{timestamp}.txt'

    human_player = max(0, min(args.human_player, args.num_players - 1))
    agent_kwargs = dict(
        device=args.device,
        use_llm_presenter=args.use_llm_presenter,
        presenter_model=args.presenter_model,
        presenter_style=args.presenter_style,
        presenter_max_lines=args.presenter_max_lines,
        debug_talk=args.debug_talk,
        debug_log_path=debug_path,
        use_llm_listener=args.use_llm_listener,
        llm_model=args.llm_listener_model,
        compare_listeners=False,
        device_preference=args.device_preference,
    )

    if args.human_play:
        agents = {
            idx: MindGamesAgent(
                model_path=args.weights if args.weights else None,
                self_player=idx,
                **agent_kwargs,
            )
            for idx in range(args.num_players) 
            if idx != human_player          # this is fancy one liner for agents variable , have all agents in this variabe
        }
        sample_agent = next(iter(agents.values()), None)        # lazyly selects one agent
        if sample_agent is not None:
            presenter_type = type(sample_agent.llm_presenter).__name__ if getattr(sample_agent, 'llm_presenter', None) else type(sample_agent.talk_presenter).__name__
            listener_type = (type(sample_agent.listener).__name__ if getattr(sample_agent, 'listener', None) else "None")
            message = f"Presenter class in use: {presenter_type} | Listener class in use: {listener_type}"
            print(message)
            debug_log(message, path=debug_path)

        result = run_human_play_session(
            env,
            agents,
            human_player=human_player,
            inference_config=DEFAULT_INFERENCE_CONFIG,
            debug=args.debug_talk,
            debug_log_path=debug_path,
            transcript_path=Path(args.output) if args.output else None,
            max_steps=args.steps if args.steps > 0 else None,
        )
        for agent in agents.values():
            agent.close()
        print(f"Human play transcript saved to {result.transcript_path}")
        if result.debug_log_path:
            print(f"Debug log recorded to {result.debug_log_path}")
        return

    # No Human Play Involved
    agents = [
        MindGamesAgent(
            model_path=args.weights if args.weights else None,
            self_player=idx,
            **agent_kwargs,
        )
        for idx in range(args.num_players)
    ]
    sample_agent = agents[0] if agents else None
    if sample_agent is not None:
        presenter_type = type(sample_agent.llm_presenter).__name__ if getattr(sample_agent, 'llm_presenter', None) else type(sample_agent.talk_presenter).__name__
        listener_type = (type(sample_agent.listener).__name__ if getattr(sample_agent, 'listener', None) else "None")
        message = f"Presenter class in use: {presenter_type} | Listener class in use: {listener_type}"
        print(message)
        debug_log(message, path=debug_path)
    buffer = CollectorBuffer()

    reward_cfg = RewardConfig() if args.shaped else None
    if args.shaped:
        print(f"Shaping enabled with config: {reward_cfg}")
    episode_index = run_selfplay(
        env,
        agents,
        buffer,
        total_steps=args.steps,
        use_shaping=args.shaped,
        reward_config=reward_cfg,
        debug=args.debug_talk,
        debug_log_path=debug_path,
    )

    output_base = Path(args.output)
    buffer.save(output_base, episode_index=episode_index)
    stem_path = output_base.with_suffix('') if output_base.suffix else output_base
    jsonl_output = stem_path.with_suffix('.jsonl')
    print(f"Collected {len(buffer)} steps -> {jsonl_output}")
    if args.debug_talk and debug_path is not None:
        print(f"Debug log recorded to {debug_path}")


def train_main(args: argparse.Namespace) -> None:
    buffer_path = Path(args.buffer)
    records = load_rollout_records(buffer_path)
    if not records:
        print(f"No rollout records found at {buffer_path}")
        return

    dataset = RolloutDataset(records, gamma=args.gamma, lam=args.gae_lambda)

    policy = MultiHeadPolicy(ModelDims())
    if args.from_il and Path(args.from_il).exists():
        state_dict = torch.load(args.from_il, map_location='cpu')
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(f'Loaded IL weights with missing keys={missing} unexpected={unexpected}')
        else:
            print('Loaded IL weights successfully - continuing PPO fine-tuning.')
    device = torch.device(args.device)
    policy.to(device)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    trainer = PPOTrainer(policy, optimizer, args.clip, args.vf_coef, args.ent_coef, args.consistency_weight, device)

    log_rows = []
    for epoch in range(args.epochs):
        stats = trainer.train_epoch(dataset, batch_size=args.batch_size)
        stats['epoch'] = epoch + 1
        stats['mean_return'] = dataset.mean_return
        log_rows.append(stats)
        print(f"epoch={epoch + 1} policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.4f} clip_frac={stats['clip_frac']:.4f} consistency={stats['consistency']:.4f}")

    output_path = Path(args.output)
    torch.save(policy.state_dict(), output_path)
    print(f"Saved PPO weights to {output_path}")

    log_path = output_path.with_suffix('.ppo_log.csv')
    with log_path.open('w', newline='') as handle:
        fieldnames = ['epoch', 'policy_loss', 'value_loss', 'entropy', 'clip_frac', 'consistency', 'mean_return']
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)
    print(f"Training log written to {log_path}")


def main() -> None:
    args = parse_args()
    if args.mode == 'collect':
        collect_main(args)
    else:
        train_main(args)


if __name__ == '__main__':
    main()


__all__ = ['PPORunner', 'PPOStats', 'CollectedTransition', 'CollectorBuffer', 'SimpleSelfPlayEnv', 'TextArenaOfflineWrapper', 'run_selfplay', 'collect_main', 'train_main', 'main']






