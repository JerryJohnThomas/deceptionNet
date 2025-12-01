"""Smoke test for the debug logging layer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deceptionNet.runners.runner_ppo import run_selfplay, CollectorBuffer, SimpleSelfPlayEnv
from deceptionNet.online_agent import MindGamesAgent


def test_debug_logging_creates_file(tmp_path):
    debug_path = tmp_path / "debug_trace.txt"
    env = SimpleSelfPlayEnv(num_players=2, episode_length=4)
    agents = [
        MindGamesAgent(self_player=i, debug_talk=True, debug_log_path=debug_path)
        for i in range(2)
    ]
    buffer = CollectorBuffer()

    run_selfplay(
        env,
        agents,
        buffer,
        total_steps=3,
        use_shaping=False,
        reward_config=None,
        debug=True,
        debug_log_path=debug_path,
    )

    assert debug_path.exists()
    contents = debug_path.read_text(encoding="utf-8")
    assert "[DEBUG]" in contents
    assert len(buffer) == 3
