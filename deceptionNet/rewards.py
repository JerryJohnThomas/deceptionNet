"""Reward shaping utilities for PPO rollouts."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Set


_ROLE_LEAK_PATTERN = re.compile(
    r"\b(mafia|werewolf|doctor|seer|role|kill\s+me|i\s+am\s+the|we\s+are)\b",
    re.IGNORECASE,
)


@dataclass
class RewardConfig:
    """Hyper-parameters controlling shaped reward components."""

    legality_bonus: float = 1.0
    suspicion_bonus: float = 0.3
    suspicion_top_k: int = 2
    leak_penalty: float = 0.3
    survival_bonus_scale: float = 0.0
    outcome_scale: float = 5.0


def compute_legality_reward(
    state: Dict[str, Any],
    features: Dict[str, Any],
    actions: Dict[str, Any],
    info: Optional[Dict[str, Any]] = None,
    bonus: float = 1.0,
) -> float:
    """Return a bonus when the chosen action targets a legal entity."""

    if info and info.get("invalid_move"):
        return 0.0

    phase = str(state.get("phase", "")).lower()
    alive = _alive_index_set(features)
    self_idx = _to_int(state.get("self_player"), default=-1)

    if phase == "day_vote":
        target = _to_int(actions.get("vote"), default=-1)
        if target in alive and target != self_idx:
            return bonus
        return 0.0

    if phase == "night":
        target = _to_int(actions.get("night"), default=-1)
        if target in alive:
            return bonus
        return 0.0

    if phase == "day_talk":
        target = _to_int(actions.get("talk_target"), default=-1)
        if target < 0:
            return bonus  # choosing no target is acceptable
        if target in alive:
            return bonus
        return 0.0

    return bonus


def compute_suspicion_alignment_reward(
    state: Dict[str, Any],
    features: Dict[str, Any],
    belief: Dict[str, Any],
    actions: Dict[str, Any],
    bonus: float,
    top_k: int,
) -> float:
    """Reward choosing targets that align with the agent's internal suspicion ordering."""

    suspicion = belief.get("suspicion")
    if not isinstance(suspicion, Sequence):
        return 0.0

    suspicion_values = [_to_float(value) for value in suspicion]
    alive = [idx for idx in _alive_index_set(features) if idx < len(suspicion_values)]
    if not alive:
        return 0.0

    phase = str(state.get("phase", "")).lower()
    if phase == "day_vote":
        target = _to_int(actions.get("vote"), default=-1)
    elif phase == "night":
        target = _to_int(actions.get("night"), default=-1)
    elif phase == "day_talk":
        target = _to_int(actions.get("talk_target"), default=-1)
    else:
        return 0.0

    if target < 0 or target not in alive:
        return 0.0

    self_idx = _to_int(state.get("self_player"), default=-1)
    candidates = [idx for idx in alive if idx != self_idx]
    if not candidates:
        return 0.0

    top_k = max(1, min(top_k, len(candidates)))
    ranked = sorted(candidates, key=lambda idx: suspicion_values[idx], reverse=True)[:top_k]
    if target in ranked:
        return bonus
    return 0.0


def compute_leak_penalty(
    phase: str,
    message: Optional[str],
    penalty: float,
) -> float:
    """Penalise talk messages that leak hidden role information."""

    if penalty <= 0.0 or not message:
        return 0.0
    if phase != "day_talk":
        return 0.0
    if _ROLE_LEAK_PATTERN.search(message):
        return penalty
    return 0.0


def compute_survival_bonus(state: Dict[str, Any], scale: float) -> float:
    """Small positive reward for surviving deeper into the game."""

    if scale <= 0.0:
        return 0.0
    round_idx = _to_float(state.get("round"), default=1.0)
    turn_idx = _to_float(state.get("turn"), default=0.0)
    return max(0.0, (round_idx + turn_idx / 10.0) * scale)


def compute_reward_components(
    state: Dict[str, Any],
    features: Dict[str, Any],
    belief: Dict[str, Any],
    actions: Dict[str, Any],
    message: Optional[str],
    info: Optional[Dict[str, Any]] = None,
    config: Optional[RewardConfig] = None,
) -> Dict[str, float]:
    """Aggregate per-step shaped reward components."""

    cfg = config or RewardConfig()
    components: Dict[str, float] = {}

    legality = compute_legality_reward(state, features, actions, info, cfg.legality_bonus)
    components["legality"] = legality

    suspicion = compute_suspicion_alignment_reward(
        state,
        features,
        belief,
        actions,
        cfg.suspicion_bonus,
        cfg.suspicion_top_k,
    )
    if suspicion:
        components["suspicion"] = suspicion

    phase = str(state.get("phase", "")).lower()
    leak = compute_leak_penalty(phase, message, cfg.leak_penalty)
    if leak:
        components["leak"] = leak

    survival = compute_survival_bonus(state, cfg.survival_bonus_scale)
    if survival:
        components["survival"] = survival

    total = legality + suspicion - leak + survival
    components["total"] = total
    return components


def compute_team_outcome_bonus(
    outcome_rewards: Optional[Sequence[Any]],
    player_id: int,
    config: Optional[RewardConfig] = None,
) -> float:
    """Scale terminal environment rewards into a shaping bonus."""

    if outcome_rewards is None:
        return 0.0

    cfg = config or RewardConfig()
    try:
        value = _to_float(outcome_rewards[player_id])
    except (TypeError, ValueError, IndexError):
        return 0.0

    if value > 0:
        return cfg.outcome_scale
    if value < 0:
        return -cfg.outcome_scale
    return 0.0


def _alive_index_set(features: Dict[str, Any]) -> Set[int]:
    alive = features.get("alive_mask")
    if not isinstance(alive, Sequence):
        return set()
    return {idx for idx, value in enumerate(alive) if _to_float(value) > 0.5}


def _to_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


__all__ = [
    "RewardConfig",
    "compute_reward_components",
    "compute_team_outcome_bonus",
]
