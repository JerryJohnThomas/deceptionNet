"""Shared datatypes for deceptionNet agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch

Tensor = torch.Tensor


@dataclass
class StepMasks:
    """Action masks for the different heads at a single timestep."""

    alive_mask: Tensor
    mafia_mask: Tensor
    self_index: Tensor
    night_valid: Tensor
    vote_valid: Tensor
    talk_target_valid: Tensor


@dataclass
class SharedTorsoOutput:
    """Output payload produced by the state builder."""

    player_embeddings: Tensor
    convo_embedding: Tensor
    memory_state: Tensor
    masks: StepMasks


@dataclass
class PolicyOutputs:
    """Policy logits at a timestep."""

    night_logits: Tensor
    vote_logits: Tensor
    talk_intent_logits: Tensor
    talk_slot_logits: Optional[Tensor]
    values: Optional[Tensor] = None
    extra: Optional[Dict[str, Tensor]] = None


@dataclass
class RolloutStep:
    """Storage for PPO rollouts."""

    obs: Dict[str, Tensor]
    actions: Dict[str, Tensor]
    masks: StepMasks
    logits: PolicyOutputs
    rewards: Tensor
    dones: Tensor
    values: Tensor
    logprobs: Dict[str, Tensor]


@dataclass
class RolloutBatch:
    """Mini-batched rollout data after flattening time and batch dims."""

    obs: Dict[str, Tensor]
    actions: Dict[str, Tensor]
    advantages: Dict[str, Tensor]
    returns: Tensor
    old_logprobs: Dict[str, Tensor]
    values: Tensor
    masks: StepMasks


__all__ = [
    "Tensor",
    "StepMasks",
    "SharedTorsoOutput",
    "PolicyOutputs",
    "RolloutStep",
    "RolloutBatch",
]
