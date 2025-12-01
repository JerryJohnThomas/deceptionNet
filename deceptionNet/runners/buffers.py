"""Experience buffer utilities for PPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Tuple

import torch

from deceptionNet.datatypes import StepMasks

from ..agents.featurizer import ObservationFeatures
from ..agents.policy import MultiHeadPolicy


@dataclass
class RolloutTensors:
    player_features: torch.Tensor
    global_features: torch.Tensor
    conversation_embedding: torch.Tensor
    alive_mask: torch.Tensor
    mafia_mask: torch.Tensor
    self_index: torch.Tensor
    role_index: torch.Tensor
    phase_encoding: torch.Tensor
    vote_matrix: torch.Tensor
    mention_scores: torch.Tensor
    sentiment_scores: torch.Tensor
    support_scores: torch.Tensor
    accusations: torch.Tensor
    contradiction_scores: torch.Tensor
    bandwagon_scores: torch.Tensor
    round_index: torch.Tensor


class RolloutBuffer:
    """Stores on-policy rollouts for PPO updates."""

    def __init__(self, rollout_length: int, num_envs: int, device: torch.device) -> None:
        self.rollout_length = rollout_length
        self.num_envs = num_envs
        self.device = device
        self.reset()

    def reset(self) -> None:
        self.features: List[ObservationFeatures] = []
        self.masks: List[StepMasks] = []
        self.actions: List[Dict[str, torch.Tensor]] = []
        self.logprobs: List[Dict[str, torch.Tensor]] = []
        self.values: List[torch.Tensor] = []
        self.rewards: List[torch.Tensor] = []
        self.dones: List[torch.Tensor] = []
        self.prev_beliefs: List[Dict[str, torch.Tensor]] = []
        self.prev_memories: List[torch.Tensor] = []
        self.initial_belief = None
        self.initial_memory = None

    def set_initial_state(self, belief_state, memory_state) -> None:
        self.initial_belief = self._detach_belief(belief_state)
        self.initial_memory = memory_state.detach().cpu()

    def add(
        self,
        features: ObservationFeatures,
        masks: StepMasks,
        actions: Dict[str, torch.Tensor],
        logprobs: Dict[str, torch.Tensor],
        value: torch.Tensor,
        reward: torch.Tensor,
        done: torch.Tensor,
        prev_belief,
        prev_memory: torch.Tensor,
    ) -> None:
        self.features.append(self._detach_features(features))
        self.masks.append(self._detach_masks(masks))
        self.actions.append({k: v.detach().cpu() for k, v in actions.items()})
        self.logprobs.append({k: v.detach().cpu() for k, v in logprobs.items()})
        self.values.append(value.detach().cpu())
        self.rewards.append(reward.detach().cpu())
        self.dones.append(done.detach().cpu())
        self.prev_beliefs.append(self._detach_belief(prev_belief))
        self.prev_memories.append(prev_memory.detach().cpu())

    def __len__(self) -> int:
        return len(self.features)

    def stacked_features(self) -> RolloutTensors:
        return RolloutTensors(
            player_features=torch.stack([f.player_features for f in self.features]),
            global_features=torch.stack([f.global_features for f in self.features]),
            conversation_embedding=torch.stack([f.conversation_embedding for f in self.features]),
            alive_mask=torch.stack([f.alive_mask for f in self.features]),
            mafia_mask=torch.stack([f.mafia_mask for f in self.features]),
            self_index=torch.stack([f.self_index for f in self.features]),
            role_index=torch.stack([f.role_index for f in self.features]),
            phase_encoding=torch.stack([f.phase_encoding for f in self.features]),
            vote_matrix=torch.stack([f.vote_matrix for f in self.features]),
            mention_scores=torch.stack([f.mention_scores for f in self.features]),
            sentiment_scores=torch.stack([f.sentiment_scores for f in self.features]),
            support_scores=torch.stack([f.support_scores for f in self.features]),
            accusations=torch.stack([f.accusations for f in self.features]),
            contradiction_scores=torch.stack([f.contradiction_scores for f in self.features]),
            bandwagon_scores=torch.stack([f.bandwagon_scores for f in self.features]),
            round_index=torch.stack([f.round_index for f in self.features]),
        )

    def stacked_masks(self) -> StepMasks:
        alive = torch.stack([m.alive_mask for m in self.masks])
        mafia = torch.stack([m.mafia_mask for m in self.masks])
        self_idx = torch.stack([m.self_index for m in self.masks])
        night = torch.stack([m.night_valid for m in self.masks])
        vote = torch.stack([m.vote_valid for m in self.masks])
        talk = torch.stack([m.talk_target_valid for m in self.masks])
        return StepMasks(alive, mafia, self_idx, night, vote, talk)

    def stacked_actions(self) -> Dict[str, torch.Tensor]:
        keys = self.actions[0].keys()
        stacked = {k: torch.stack([step[k] for step in self.actions]) for k in keys}
        return stacked

    def stacked_logprobs(self) -> Dict[str, torch.Tensor]:
        keys = self.logprobs[0].keys()
        stacked = {k: torch.stack([step[k] for step in self.logprobs]) for k in keys}
        return stacked

    def stacked_values(self) -> torch.Tensor:
        return torch.stack(self.values)

    def stacked_rewards(self) -> torch.Tensor:
        return torch.stack(self.rewards)

    def stacked_dones(self) -> torch.Tensor:
        return torch.stack(self.dones)

    def stacked_prev_belief(self) -> Dict[str, torch.Tensor]:
        stacked = {
            "player_embeddings": torch.stack([b["player_embeddings"] for b in self.prev_beliefs]),
            "global_hidden": torch.stack([b["global_hidden"] for b in self.prev_beliefs]),
            "role_logits": torch.stack([b["role_logits"] for b in self.prev_beliefs]),
            "suspicion": torch.stack([b["suspicion"] for b in self.prev_beliefs]),
            "trust": torch.stack([b["trust"] for b in self.prev_beliefs]),
        }
        return stacked

    def stacked_prev_memory(self) -> torch.Tensor:
        return torch.stack(self.prev_memories)

    def advantages_and_returns(
        self,
        last_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = torch.cat([self.stacked_values(), last_value.unsqueeze(0)], dim=0)
        rewards = self.stacked_rewards()
        dones = self.stacked_dones()
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        last_gae = torch.zeros(self.num_envs, dtype=torch.float32)
        for t in reversed(range(T)):
            mask = 1.0 - dones[t]
            delta = rewards[t] + gamma * values[t + 1] * mask - values[t]
            last_gae = delta + gamma * gae_lambda * mask * last_gae
            adv[t] = last_gae
        returns = adv + values[:-1]
        return adv, returns

    def iterate_minibatches(self, batch_size: int) -> Iterator[torch.Tensor]:
        total = self.rollout_length * self.num_envs
        indices = torch.randperm(total)
        for start in range(0, total, batch_size):
            yield indices[start : start + batch_size]

    def _detach_features(self, features: ObservationFeatures) -> ObservationFeatures:
        return ObservationFeatures(
            player_features=features.player_features.detach().cpu(),
            global_features=features.global_features.detach().cpu(),
            conversation_embedding=features.conversation_embedding.detach().cpu(),
            alive_mask=features.alive_mask.detach().cpu(),
            mafia_mask=features.mafia_mask.detach().cpu(),
            self_index=features.self_index.detach().cpu(),
            role_index=features.role_index.detach().cpu(),
            phase_encoding=features.phase_encoding.detach().cpu(),
            vote_matrix=features.vote_matrix.detach().cpu(),
            mention_scores=features.mention_scores.detach().cpu(),
            sentiment_scores=features.sentiment_scores.detach().cpu(),
            support_scores=features.support_scores.detach().cpu(),
            accusations=features.accusations.detach().cpu(),
            contradiction_scores=features.contradiction_scores.detach().cpu(),
            bandwagon_scores=features.bandwagon_scores.detach().cpu(),
            round_index=features.round_index.detach().cpu(),
        )

    def _detach_masks(self, masks: StepMasks) -> StepMasks:
        return StepMasks(
            alive_mask=masks.alive_mask.detach().cpu(),
            mafia_mask=masks.mafia_mask.detach().cpu(),
            self_index=masks.self_index.detach().cpu(),
            night_valid=masks.night_valid.detach().cpu(),
            vote_valid=masks.vote_valid.detach().cpu(),
            talk_target_valid=masks.talk_target_valid.detach().cpu(),
        )


    def _detach_belief(self, belief_state) -> Dict[str, torch.Tensor]:
        return {
            "player_embeddings": belief_state.player_embeddings.detach().cpu(),
            "global_hidden": belief_state.global_hidden.detach().cpu(),
            "role_logits": belief_state.role_logits.detach().cpu(),
            "suspicion": belief_state.suspicion.detach().cpu(),
            "trust": belief_state.trust.detach().cpu(),
        }

__all__ = ["RolloutBuffer", "RolloutTensors"]
