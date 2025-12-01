"""Smoke tests to validate the end-to-end agent wiring."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from deceptionNet.config import ModelDims
from deceptionNet.agents import (
    MultiHeadPolicy,
    ObservationFeaturizer,
    SimpleListener,
)


def _dummy_observation(num_players: int = 6) -> dict:
    return {
        "phase": "day_talk",
        "round": 1,
        "turn": 1,
        "self_player": 0,
        "role": "villager",
        "alive": [True] * num_players,
        "vote_history": [],
        "talk_history": [
            {"speaker": 1, "text": "[0] feels off to me."},
            {"speaker": 2, "text": "I am voting [3] today."},
        ],
    }


def smoke_test_policy_forward() -> None:
    if torch is None:
        print("Skipping smoke test: torch not installed")
        return

    dims = ModelDims()
    listener = SimpleListener(dims)
    featurizer = ObservationFeaturizer(dims)
    policy = MultiHeadPolicy(dims)

    obs = _dummy_observation(dims.num_players)
    listener_output = listener(obs)
    features = featurizer(obs, listener_output)

    belief_state, memory_state = policy.initial_state(batch_size=1)
    shared, outputs = policy(features, belief_state, memory_state)
    assert shared.player_embeddings.shape[-1] == dims.hidden_size
    assert outputs.vote_logits.shape[-1] == dims.num_players
    assert outputs.talk_intent_logits.shape[-1] == dims.num_talk_intents

    actions, _ = policy.act(outputs, shared.masks, deterministic=True)
    assert "vote" in actions and actions["vote"].shape[0] == 1


if __name__ == "__main__":
    smoke_test_policy_forward()
    print("OK")
