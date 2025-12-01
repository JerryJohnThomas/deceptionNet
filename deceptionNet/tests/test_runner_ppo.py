"""Smoke tests for the rollout collector."""

from __future__ import annotations

import os
import sys
from pathlib import Path

def _setup_path() -> None:
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

_setup_path()

from deceptionNet.runners.runner_ppo import (
    CollectorBuffer,
    SimpleSelfPlayEnv,
    run_selfplay,
)
from deceptionNet.online_agent import MindGamesAgent
from deceptionNet.rewards import RewardConfig


def test_rollout_collects(tmp_path):
    env = SimpleSelfPlayEnv(num_players=2, episode_length=6)
    agents = [MindGamesAgent(self_player=i) for i in range(2)]
    buffer = CollectorBuffer()

    episode_index = run_selfplay(
        env,
        agents,
        buffer,
        total_steps=5,
        use_shaping=True,
        reward_config=RewardConfig(),
    )

    assert len(buffer) == 5
    output_base = tmp_path / "buffer"
    buffer.save(output_base, episode_index=episode_index)
    assert (output_base.with_suffix(".json")).exists()
    assert (output_base.with_suffix(".jsonl")).exists()

    first = buffer.data[0]
    assert "player_features" in first.features
    assert "belief" in first.memory_in
    assert "sum" in first.logprobs
    assert "total" in first.reward_components


if __name__ == "__main__":
    test_rollout_collects(Path(".") / "tmp")
    print("OK")
