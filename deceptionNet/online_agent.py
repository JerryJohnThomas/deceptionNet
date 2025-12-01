"""Online agent entrypoint wiring the full MindGames pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

if __package__ in (None, "", "__main__"):
    ROOT = os.path.dirname(os.path.dirname(__file__))
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

from typing import Any, Dict, List, Optional

import torch

try:
    import textarena as ta
except ImportError:  # pragma: no cover
    ta = None  # type: ignore

from deceptionNet.config import DEFAULT_INFERENCE_CONFIG, InferenceConfig, ModelDims
from deceptionNet.agents import (
    MultiHeadPolicy,
    ObservationFeaturizer,
    SimpleListener,
    TalkAux,
    TalkPresenter,
    VotePresenter,
    NightPresenter,
)
from deceptionNet.agents.featurizer import ObservationFeatures
from deceptionNet.utils import ComparisonLogger, debug_log
from deceptionNet.agents import TextToStateMapper


def _tensor_to_list(tensor: Optional[torch.Tensor]) -> Any:
    if tensor is None:
        return None
    return tensor.detach().cpu().tolist()


def _serialize_features(features: ObservationFeatures) -> Dict[str, Any]:
    return {
        "player_features": _tensor_to_list(features.player_features.squeeze(0)),
        "global_features": _tensor_to_list(features.global_features.squeeze(0)),
        "conversation_embedding": _tensor_to_list(features.conversation_embedding.squeeze(0)),
        "alive_mask": _tensor_to_list(features.alive_mask.squeeze(0)),
        "mafia_mask": _tensor_to_list(features.mafia_mask.squeeze(0)),
        "self_index": _tensor_to_list(features.self_index.squeeze(0)),
        "role_index": _tensor_to_list(features.role_index.squeeze(0)),
        "phase_encoding": _tensor_to_list(features.phase_encoding.squeeze(0)),
        "vote_matrix": _tensor_to_list(features.vote_matrix.squeeze(0)),
        "mention_scores": _tensor_to_list(features.mention_scores.squeeze(0)),
        "sentiment_scores": _tensor_to_list(features.sentiment_scores.squeeze(0)),
        "support_scores": _tensor_to_list(features.support_scores.squeeze(0)),
        "accusations": _tensor_to_list(features.accusations.squeeze(0)),
        "contradiction_scores": _tensor_to_list(features.contradiction_scores.squeeze(0)),
        "bandwagon_scores": _tensor_to_list(features.bandwagon_scores.squeeze(0)),
        "round_index": _tensor_to_list(features.round_index.squeeze(0)),
    }


def _serialize_belief_state(belief: BeliefState) -> Dict[str, Any]:
    return {
        "player_embeddings": _tensor_to_list(belief.player_embeddings.squeeze(0)),
        "global_hidden": _tensor_to_list(belief.global_hidden.squeeze(0)),
        "role_logits": _tensor_to_list(belief.role_logits.squeeze(0)),
        "suspicion": _tensor_to_list(belief.suspicion.squeeze(0)),
        "trust": _tensor_to_list(belief.trust.squeeze(0)),
    }


# Uncomment this for stage 2 our team submission
MODEL_NAME = "DeceptionNet_IL_v1"
MODEL_DESCRIPTION = "Stage 2 Submission"
TEAM_HASH = "MG25-38AF13423D"
DEFAULT_WEIGHTS_PATH = Path("deceptionNet/submission/weights-ppo-v3.pt")


# This is dummy thing to hit textarena without causing our stage 2 submission
# MODEL_NAME = "AmIaMafia"
# MODEL_DESCRIPTION = "ML baseline"
# TEAM_HASH = "4265621235121"
# DEFAULT_WEIGHTS_PATH = Path("deceptionNet/weights-il-v6.pt")
# DEFAULT_PRESENTER_MODEL = "microsoft/Phi-3-mini-128k-instruct"


class MindGamesAgent:
    """High-level agent compatible with the MindGames TextArena API."""


    def __init__(
        self,
        dims: Optional[ModelDims] = None,
        config: Optional[InferenceConfig] = None,
        model_path: Optional[str] = None,
        device: str = "cpu",
        use_llm_listener: bool = False,
        llm_model: Optional[str] = None,
        compare_listeners: bool = False,
        logger: Optional[ComparisonLogger] = None,
        self_player: int = 0,
        use_llm_presenter: bool = False,
        presenter_model: Optional[str] = None,
        presenter_style: str = "neutral",
        presenter_max_lines: int = 1,
        device_preference: str = "auto",
        debug_talk: bool = False,
        debug_log_path: Optional[Path] = None,
    ) -> None:
        self.dims = dims or ModelDims()
        self.config = config or DEFAULT_INFERENCE_CONFIG
        self.device = torch.device(device)

        self.compare_listeners = compare_listeners
        self.logger = logger
        self.presenter_style = (presenter_style or 'neutral').lower()
        self.presenter_max_lines = max(1, int(presenter_max_lines))
        self.presenter_model = presenter_model
        self.device_preference = (device_preference or 'auto').lower()
        self.use_llm_presenter = use_llm_presenter and self.device_preference != 'template'
        self.llm_presenter = None
        self.debug_talk = debug_talk
        if self.debug_talk:
            if debug_log_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                debug_log_path = Path('logs') / f'debug_run_{timestamp}.txt'
            self.debug_log_path = Path(debug_log_path)
            self.debug_log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            self.debug_log_path = Path(debug_log_path) if debug_log_path else None
        self.step_counter = 0
        self.self_player = self_player
        self.text_mapper = TextToStateMapper(self.dims.num_players, self_player=self_player)

        if self.debug_talk:
            debug_log(f"=== Debug session start (player {self.self_player}) ===", path=self.debug_log_path)

        simple_listener = SimpleListener(self.dims)
        self.listener = simple_listener
        self.llm_listener = None

        if use_llm_listener or compare_listeners:
            try:
                from deceptionNet.agents import LLMListener, LLMListenerConfig

                default_config = LLMListenerConfig()
                listener_config = LLMListenerConfig(
                    model_name=llm_model or default_config.model_name,
                    cache_dir=default_config.cache_dir,
                    load_in_8bit=default_config.load_in_8bit,
                    torch_dtype=default_config.torch_dtype,
                    fallback_model_name=default_config.fallback_model_name,
                )
                self.llm_listener = LLMListener(self.dims, listener_config)
            except Exception:
                self.llm_listener = None

        if use_llm_listener and not compare_listeners:
            if self.llm_listener is not None:
                self.listener = self.llm_listener
            else:
                print("Warning: LLM listener requested but unavailable; falling back to SimpleListener.")

        if compare_listeners:
            if self.llm_listener is None:
                print("Warning: compare-listeners requested but LLM listener unavailable. Logs will contain only simple listener data.")
            if self.logger is None:
                self.logger = ComparisonLogger()

        self.featurizer = ObservationFeaturizer(self.dims)
        self.policy = MultiHeadPolicy(self.dims).to(self.device)
        self.talk_presenter = TalkPresenter(self.config)
        self.vote_presenter = VotePresenter(self.config)
        self.night_presenter = NightPresenter(self.config)

        if self.use_llm_presenter:
            try:
                from deceptionNet.agents.presenter_llm import LLMPresenter  # lazy import

                model_name = self.presenter_model or DEFAULT_PRESENTER_MODEL
                self.llm_presenter = LLMPresenter(
                    model_name=model_name,
                    style=self.presenter_style,
                    max_lines=self.presenter_max_lines,
                )
            except Exception as exc:
                print(f"Warning: failed to initialise LLM presenter: {exc}. Falling back to templates.")
                self.llm_presenter = None
                self.use_llm_presenter = False

        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.policy.load_state_dict(state_dict)
            except RuntimeError as err:
                print(f"Warning: failed to load weights {model_path}: {err}.")
                print("Proceeding with randomly initialized policy weights.")

        self.reset_state()

    def reset_state(self) -> None:
        self.belief_state, self.memory_state = self.policy.initial_state(1, self.device)
        self.text_mapper.reset()
        self.step_counter = 0

    def _process_observation(self, observation: Any) -> Dict[str, Any]:
        if isinstance(observation, dict) and observation.get("reset", False):
            self.reset_state()
        return self.text_mapper.update(observation)

    def __call__(self, observation: Any) -> str:
        details = self.act_with_details(observation, deterministic=True, log=True)
        return details["message"]

    def act_with_details(self, observation: Any, deterministic: bool = False, log: bool = False):
        state = self._process_observation(observation)                                      # convert raw string observation i assume to texttoStateMapper processed obsevations
        public_details, internal_details = self._compute_action(state, deterministic=deterministic)     # compute the actions 
        # public_details: a clean dictionary for logging or training (message, actions, features, etc.).
        # internal_details: deeper info (listener outputs, belief states, logits, etc.).

        # compare listner snippet
        alt_message = None
        if log and self.compare_listeners and internal_details["alt_listener_output"] is not None and internal_details["action_info"].get("type") == "talk":
            target = internal_details["action_info"].get("target", -1)
            intent = internal_details["action_info"].get("intent", -1)
            alt_aux = self._build_talk_aux(internal_details["alt_listener_output"], internal_details["shared"].belief)
            alt_message = self.talk_presenter.render(intent, target, alt_aux)

        # logger snippet
        if log:
            self._log_step(
                internal_details["state"],
                internal_details["listener_output"],
                internal_details["alt_listener_output"],
                internal_details["shared"],
                internal_details["actions_raw"],
                internal_details["message"],
                alt_message,
                internal_details["action_info"],
            )
            if alt_message is not None:
                public_details["alt_message"] = alt_message

        return public_details

    def _compute_action(self, state: Dict[str, Any], deterministic: bool = True):
        listener_output = self.listener(state)      # Important, Todo Analysis , need to check implmentation of listnener
        alt_listener_output = None
        if deterministic and self.compare_listeners and self.llm_listener is not None:
            try:
                alt_listener_output = self.llm_listener(state)
            except Exception:
                alt_listener_output = None

        features_cpu = self.featurizer(state, listener_output)
        features_device = self._to_device_features(features_cpu)

        belief_snapshot = _serialize_belief_state(self.belief_state)
        memory_snapshot = _tensor_to_list(self.memory_state.squeeze(0))

        shared, outputs = self.policy(features_device, self.belief_state, self.memory_state)        # state, logits
        actions, logprobs = self.policy.act(outputs, shared.masks, deterministic=deterministic)     # actions, logprob of the action taken 

        value = float(outputs.values.detach().cpu().view(-1)[0].item()) if outputs.values.numel() > 0 else 0.0  # critic value is outputes.value

        # for logging
        logits = {
            "night": _tensor_to_list(outputs.night_logits.detach().cpu().squeeze(0)),
            "vote": _tensor_to_list(outputs.vote_logits.detach().cpu().squeeze(0)),
            "intent": _tensor_to_list(outputs.talk_intent_logits.detach().cpu().squeeze(0)),
            "slot": _tensor_to_list(outputs.talk_slot_logits.detach().cpu().squeeze(0)),
        }

        actions_serialized = {key: int(value.item()) if value.numel() == 1 else _tensor_to_list(value) for key, value in actions.items()}
        logprobs_serialized = {key: float(value.item()) if value.numel() == 1 else _tensor_to_list(value) for key, value in logprobs.items()}
        logprob_sum = 0.0
        for lp in logprobs.values():
            if lp.numel() == 1:
                logprob_sum += float(lp.item())
            elif lp.numel() > 0:
                logprob_sum += float(lp.view(-1)[0].item())
        logprobs_serialized["sum"] = logprob_sum

        memory_out_tensor = outputs.extra.get("memory_state")
        memory_out_serialized = _tensor_to_list(memory_out_tensor.detach().cpu().squeeze(0)) if memory_out_tensor is not None else None

        belief_out_serialized = _serialize_belief_state(shared.belief)

        message, action_info = self._format_action(state, actions, listener_output, shared)     # Important Presenter out

        public_details = {
            "state": json.loads(json.dumps(state)),
            "message": message,
            "action_info": action_info,
            "features": _serialize_features(features_cpu),
            "memory_in": {"belief": belief_snapshot, "policy": memory_snapshot},
            "memory_out": {"belief": belief_out_serialized, "policy": memory_out_serialized},
            "logits": logits,
            "value": value,
            "actions": actions_serialized,
            "logprobs": logprobs_serialized,
        }

        internal_details = {
            "state": state,
            "message": message,
            "action_info": action_info,
            "actions_raw": actions,
            "logprobs_raw": logprobs,
            "listener_output": listener_output,
            "alt_listener_output": alt_listener_output,
            "shared": shared,
        }

        self.belief_state = shared.belief.detach()
        self.memory_state = outputs.extra["memory_state"].detach()
        self.step_counter += 1

        return public_details, internal_details


    def _format_action(
        self,
        state: Dict[str, Any],
        actions: Dict[str, torch.Tensor],
        listener_output,
        shared_state,
    ) -> tuple[str, Dict[str, Any]]:
        phase = str(state.get("phase", "day_talk")).lower()     # tries to get the phase from the state, if not found default to day talk
        info: Dict[str, Any] = {"phase": phase}

        print("[Ronaldo Format Action being called]", phase)

        # both day vote and night is just [x] kind of outputs
        if phase == "day_vote":
            vote_idx = int(actions["vote"].item())
            message = self.vote_presenter.render(vote_idx)
            info.update({"type": "vote", "vote_index": vote_idx})
            self._debug_action_trace(state, message, info, listener_output, actions)
            return message, info
        if phase == "night":
            night_idx = int(actions["night"].item())
            message = self.night_presenter.render(night_idx)
            info.update({"type": "night", "night_index": night_idx})
            self._debug_action_trace(state, message, info, listener_output, actions)
            return message, info
        
        # conversation part
        intent_idx = int(actions["talk_intent"].item())
        target_idx = int(actions["talk_target"].item()) if "talk_target" in actions else -1
        aux = self._build_talk_aux(listener_output, shared_state.belief)
        info.update({"type": "talk", "intent": intent_idx, "target": target_idx})

        message: str
        if self.llm_presenter is not None and self.use_llm_presenter:
            # LLM Presenter
            intent_label = self._intent_label(intent_idx)
            target_label = self._format_player_label(target_idx)
            belief_summary = aux.belief_summary or None
            message = self.llm_presenter.render(
                intent=intent_label,
                target=target_label,
                phase=phase,
                belief_summary=belief_summary,
                player_id=self.self_player,
            )
            print("[MESSI LLM OUTPUT PHASE]", "LLM presenter", message, "[:: OVER]")
            
            # Template Presenter fall back if LLM Presenter is not working
            if not message:
                message = self.talk_presenter.render(intent_idx, target_idx, aux)
                print("[MESSI LLM OUTPUT PHASE]", "Tempate Presenter Fall back due to llm presenter failing", message), "[:: OVER]"
        else:
            # Template Presenter
            message = self.talk_presenter.render(intent_idx, target_idx, aux)
            print("[MESSI LLM OUTPUT PHASE]", "Tempate Presenter Mode", message, "[:: OVER]")
        self._debug_action_trace(state, message, info, listener_output, actions)
        return message, info



    def _intent_label(self, intent_idx: int) -> str:
        intents = {
            0: "accuse",
            1: "defend_self",
            2: "defend_other",
            3: "claim",
            4: "agree",
            5: "question",
            6: "filler",
        }
        return intents.get(intent_idx, "silent")

    def _format_player_label(self, index: int) -> str:
        if index < 0:
            return 'no target'
        return self.config.vote_token_fmt.format(index=int(index))

    def _debug_action_trace(self, state: Dict[str, Any], message: str, info: Dict[str, Any], listener_output, actions: Dict[str, torch.Tensor]) -> None:
        if not self.debug_talk:
            return
        phase = str(state.get('phase', '?'))
        prefix = f"Player {self.self_player} ({phase})"
        self._debug_print(f"{prefix} presenter says: {message}")
        print("[OMEGA] Presenter message only:", repr(message))

        if listener_output is not None:
            summary = getattr(listener_output, 'summary_text', '')
            if summary:
                self._debug_print(f"{prefix} listener summary: {summary}")
        self._debug_print(f"{prefix} action info: {info}")
        policy_actions = self._actions_to_dict(actions)
        self._debug_print(f"{prefix} policy actions: {policy_actions}")

    def _debug_print(self, message: str) -> None:
        if not self.debug_talk:
            return
        line = f"[DEBUG] {message}"
        print(line)
        debug_log(line, path=self.debug_log_path)

    def _build_talk_aux(self, listener_output, belief) -> TalkAux:
        suspicion_scores = self._tensor_to_list(belief.suspicion)
        trust_scores = self._tensor_to_list(belief.trust)
        summary = getattr(listener_output, "summary_text", "") if listener_output is not None else ""
        support_scores = self._tensor_to_list(getattr(listener_output, "support_scores", None)) if listener_output is not None else []
        accusation_scores = self._tensor_to_list(getattr(listener_output, "accusation_scores", None)) if listener_output is not None else []
        contradiction_scores = self._tensor_to_list(getattr(listener_output, "contradiction_scores", None)) if listener_output is not None else []
        bandwagon_scores = self._tensor_to_list(getattr(listener_output, "bandwagon_scores", None)) if listener_output is not None else []
        # prepares extra information about what the agent currently believes about the game and conversation, so that the presenter can speak intelligently
        return TalkAux(
            belief_summary=summary,
            suspicion_scores=suspicion_scores,
            trust_scores=trust_scores,
            support_scores=support_scores,
            accusation_scores=accusation_scores,
            contradiction_scores=contradiction_scores,
            bandwagon_scores=bandwagon_scores,
        )

    def _tensor_to_list(self, tensor: Optional[torch.Tensor]) -> List[float]:
        if tensor is None:
            return []
        array = tensor.detach().cpu()
        if array.numel() == 0:
            return []
        return [float(x) for x in array.reshape(-1).tolist()]

    def _listener_to_dict(self, output) -> Optional[Dict[str, Any]]:
        if output is None:
            return None
        return {
            "summary": getattr(output, "summary_text", ""),
            "mentions": self._tensor_to_list(output.player_mentions),
            "accusation": self._tensor_to_list(output.accusation_scores),
            "support": self._tensor_to_list(output.support_scores),
            "sentiment": self._tensor_to_list(output.sentiment_scores),
            "contradiction": self._tensor_to_list(output.contradiction_scores),
            "bandwagon": self._tensor_to_list(output.bandwagon_scores),
        }

    def _belief_to_dict(self, belief) -> Dict[str, Any]:
        role_logits = belief.role_logits.detach().cpu()
        role_list = role_logits.view(role_logits.shape[0], -1, role_logits.shape[-1]).tolist()
        return {
            "suspicion": self._tensor_to_list(belief.suspicion),
            "trust": self._tensor_to_list(belief.trust),
            "role_logits": role_list,
        }

    def _actions_to_dict(self, actions: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in actions.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    result[key] = int(value.item())
                else:
                    result[key] = [int(v) for v in value.detach().cpu().reshape(-1).tolist()]
            else:
                result[key] = value
        return result

    def _log_step(
        self,
        state: Dict[str, Any],
        simple_output,
        llm_output,
        shared_state,
        actions: Dict[str, torch.Tensor],
        message: str,
        alt_message: Optional[str],
        action_info: Dict[str, Any],
    ) -> None:
        if not self.logger:
            return
        record: Dict[str, Any] = {
            "step": self.step_counter,
            "phase": action_info.get("phase"),
            "role": state.get("role"),
            "action_type": action_info.get("type"),
            "actions": self._actions_to_dict(actions),
            "presenter_output": message,
            "round": state.get("round"),
            "turn": state.get("turn"),
            "compare_mode": self.compare_listeners,
            "llm_listener_available": self.llm_listener is not None,
        }
        action_type = action_info.get("type")
        if action_type == "talk":
            record["regex_talk"] = message
            record["llm_talk"] = alt_message
            record["talk_intent"] = action_info.get("intent")
            record["talk_target"] = action_info.get("target")
        elif action_type == "vote":
            record["vote_index"] = action_info.get("vote_index")
        elif action_type == "night":
            record["night_index"] = action_info.get("night_index")

        record["simple_listener"] = self._listener_to_dict(simple_output)
        record["llm_listener"] = self._listener_to_dict(llm_output) if llm_output is not None else None
        record["belief"] = self._belief_to_dict(shared_state.belief)

        self.logger.log(record)

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

    def close(self) -> Optional[str]:
        path_str: Optional[str] = None
        if self.logger:
            path_str = str(self.logger.filepath)
            self.logger.close()
            self.logger = None
        if self.debug_talk and self.debug_log_path is not None:
            print(f"Debug log written to {self.debug_log_path}")
        return path_str

def _test_observations() -> List[Dict[str, Any]]:
    """Return a small deterministic sequence for smoke testing."""

    return [
        {
            "reset": True,
            "phase": "day_talk",
            "round": 1,
            "turn": 1,
            "self_player": 0,
            "role": "villager",
            "alive": [True, True, True, True, True, True],
            "talk_history": [
                {"speaker": 1, "text": "[2] is acting odd."},
                {"speaker": 2, "text": "Maybe [4] is safer."},
            ],
            "vote_history": [],
        },
        {
            "phase": "day_vote",
            "round": 1,
            "turn": 2,
            "self_player": 0,
            "role": "villager",
            "alive": [True, True, True, True, True, True],
            "talk_history": [],
            "vote_history": [
                {"voter": 1, "target": 2},
                {"voter": 3, "target": 2},
            ],
        },
        {
            "phase": "night",
            "round": 1,
            "turn": 0,
            "self_player": 0,
            "role": "doctor",
            "alive": [True, True, True, True, True, True],
            "talk_history": [
                {"speaker": "system", "text": "Night actions resolving."}
            ],
            "vote_history": [],
        },
        {
            "phase": "day_talk",
            "round": 2,
            "turn": 1,
            "self_player": 0,
            "role": "villager",
            "alive": [True, True, True, True, True, True],
            "talk_history": [
                {"speaker": 4, "text": "Let's hear from [0]."},
            ],
            "vote_history": [
                {"voter": 0, "target": 2},
                {"voter": 3, "target": 2},
            ],
        },
    ]


def _run_test_mode(agent: MindGamesAgent, steps: int) -> None:
    sequence = _test_observations()
    if steps <= 0 or steps > len(sequence):
        steps = len(sequence)
    print(f"Running smoke test for {steps} steps...")
    for idx in range(steps):
        obs = sequence[idx]
        action = agent(obs)
        phase = obs.get("phase", "?")
        print(f"step={idx + 1} phase={phase} action={action}")
    print("Test rollout complete.")


def _run_arena_mode(agent: MindGamesAgent, args: argparse.Namespace) -> None:
    if ta is None:  # pragma: no cover - optional dependency
        raise ImportError("textarena package is not installed. Install textarena to use arena mode.")

    env = ta.make_mgc_online(
        track="Social Detection",
        model_name=args.model_name,
        model_description=args.model_description,
        team_hash=args.team_hash,
        agent=agent,
        small_category=args.small_category,
    )

    try:
        for game_idx in range(args.num_games):
            print(f"Starting arena game {game_idx + 1}/{args.num_games}")
            env.reset(num_players=args.num_players)
            agent.reset_state()
            done = False
            step_info = {}
            while not done:
                try:
                    _, observation = env.get_observation()
                except RuntimeError:
                    print("Warning: observation fetch failed; ending game early.")
                    break
                action = agent(observation)
                try:
                    done, step_info = env.step(action=action)
                except RuntimeError:
                    print("Warning: env step failed; ending game early.")
                    break
            print(f"Game {game_idx + 1} finished. Step summary: {step_info}")
    finally:
        env.close()



def _run_human_mode(args: argparse.Namespace, weights_path: Optional[Path]) -> None:
    num_players = max(1, args.num_players)
    env = SimpleSelfPlayEnv(num_players=num_players, episode_length=args.steps or 60)

    debug_path = None
    if args.debug_talk:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        debug_path = Path('logs') / f'debug_run_{timestamp}.txt'

    human_player = max(0, min(args.human_player, num_players - 1))

    agent_kwargs = dict(
        device=args.device,
        use_llm_presenter=args.use_llm_presenter,
        presenter_model=args.presenter_model,
        presenter_style=args.presenter_style,
        presenter_max_lines=args.presenter_max_lines,
        device_preference=args.device_preference,
        debug_talk=args.debug_talk,
        debug_log_path=debug_path,
    )

    agents = {
        idx: MindGamesAgent(
            model_path=str(weights_path) if weights_path else None,
            self_player=idx,
            **agent_kwargs,
        )
        for idx in range(num_players)
        if idx != human_player
    }

    result = run_human_play_session(
        env,
        agents,
        human_player=human_player,
        inference_config=DEFAULT_INFERENCE_CONFIG,
        debug=args.debug_talk,
        debug_log_path=debug_path,
        transcript_path=None,
        max_steps=args.human_max_steps if args.human_max_steps > 0 else (args.steps if args.steps > 0 else None),
    )

    for agent in agents.values():
        agent.close()

    print(f"Human play transcript saved to {result.transcript_path}")
    if result.debug_log_path:
        print(f"Debug log written to {result.debug_log_path}")


    env = ta.make_mgc_online(
        track="Social Detection",
        model_name=args.model_name,
        model_description=args.model_description,
        team_hash=args.team_hash,
        agent=agent,
        small_category=args.small_category,
    )

    try:
        for game_idx in range(args.num_games):
            print(f"Starting arena game {game_idx + 1}/{args.num_games}")
            env.reset(num_players=args.num_players)
            agent.reset_state()
            done = False
            step_info = {}
            while not done:
                try:
                    _, observation = env.get_observation()
                except RuntimeError:
                    print('Warning: observation fetch failed; ending game early.')
                    break
                action = agent(observation)
                try:
                    done, step_info = env.step(action=action)
                except RuntimeError:
                    print('Warning: env step failed; ending game early.')
                    break
            print(f"Game {game_idx + 1} finished. Step summary: {step_info}")
    finally:
        env.close()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MindGames agent CLI")
    parser.add_argument("--mode", choices=["test", "arena"], default="test", help="Execution mode")
    parser.add_argument("--weights", type=str, default=str(DEFAULT_WEIGHTS_PATH), help="Path to model weights")
    parser.add_argument("--device", type=str, default="cpu", help="Device for inference (cpu or cuda)")
    parser.add_argument("--steps", type=int, default=0, help="How many test steps to run (0 = full sequence)")
    parser.add_argument("--model-name", type=str, default=MODEL_NAME, help="Model name for TextArena submissions")
    parser.add_argument("--model-description", type=str, default=MODEL_DESCRIPTION, help="Model description for submissions")
    parser.add_argument("--team-hash", type=str, default=TEAM_HASH, help="Registered competition team hash")
    parser.add_argument("--num-games", type=int, default=1, help="Number of arena games to run")
    parser.add_argument("--num-players", type=int, default=6, help="Number of players when resetting the arena env")
    parser.add_argument("--small-category", action="store_true", help="Submit in the small resource category")
    parser.add_argument("--use-llm-listener", action="store_true", help="Enable LLM-backed conversation summarizer")
    parser.add_argument("--llm-listener-model", type=str, default=None, help="Override LLM model name for listener")
    parser.add_argument("--compare-listeners", action="store_true", help="Log outputs from both simple and LLM listeners")
    parser.add_argument("--use-llm-presenter", action="store_true", help="Enable LLM-driven talk presenter")
    parser.add_argument("--presenter-model", type=str, default=None, help="Override HF model for the LLM presenter")
    parser.add_argument("--presenter-style", type=str, default="neutral", help="Tone to request from the LLM presenter")
    parser.add_argument("--presenter-max-lines", type=int, default=1, help="Maximum number of sentences the presenter may emit")
    parser.add_argument("--device-preference", choices=["auto", "gpu", "cpu", "template"], default="auto", help="Preferred device tier for LLM components")
    parser.add_argument("--debug-talk", action="store_true", help="Enable verbose presenter/listener debug logs")
    parser.add_argument("--human-play", action="store_true", help="Enable human play mode (optional)")
    args = parser.parse_args(argv)

    weights_path = Path(args.weights) if args.weights else None
    if weights_path and not weights_path.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_path}")

    if args.human_play:
        _run_human_mode(args, weights_path)
        return

    if args.compare_listeners and args.use_llm_listener:
        print("Warning: --compare-listeners overrides --use-llm-listener; using comparison mode with the simple listener driving actions.")
        args.use_llm_listener = False

    logger = ComparisonLogger() if args.compare_listeners else None

    agent = MindGamesAgent(
        model_path=str(weights_path) if weights_path else None,
        device=args.device,
        use_llm_listener=args.use_llm_listener,
        llm_model=args.llm_listener_model,
        compare_listeners=args.compare_listeners,
        logger=logger,
        use_llm_presenter=args.use_llm_presenter,
        presenter_model=args.presenter_model,
        presenter_style=args.presenter_style,
        presenter_max_lines=args.presenter_max_lines,
        debug_talk=args.debug_talk,
        device_preference=args.device_preference, 
    )

    log_path: Optional[str] = None
    try:
        if args.mode == "test":
            _run_test_mode(agent, args.steps)
        elif args.mode == "arena":
            _run_arena_mode(agent, args)
        else:
            raise ValueError(f"Unsupported mode: {args.mode}")
    finally:
        log_path = agent.close()
        if log_path:
            print(f"Comparison log written to {log_path}")


if __name__ == "__main__":
    main()


__all__ = ["MindGamesAgent", "main"]




