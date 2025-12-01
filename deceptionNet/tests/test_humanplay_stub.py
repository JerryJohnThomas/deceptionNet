"""Stub test ensuring human play session runs without crashing."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deceptionNet.human_play import run_human_play_session, SimpleSelfPlayEnv
from deceptionNet.config import DEFAULT_INFERENCE_CONFIG


def test_human_play_session_single_step(tmp_path):
    env = SimpleSelfPlayEnv(num_players=1, episode_length=2)
    agents = {}

    responses = iter(["hello", "quit"])

    def fake_input(prompt: str) -> str:  # pragma: no cover - simple helper
        try:
            return next(responses)
        except StopIteration:
            return "quit"

    result = run_human_play_session(
        env,
        agents,
        human_player=0,
        inference_config=DEFAULT_INFERENCE_CONFIG,
        debug=False,
        transcript_path=tmp_path / "transcript.txt",
        input_func=fake_input,
        max_steps=1,
    )

    assert result.transcript_path.exists()
    contents = result.transcript_path.read_text(encoding="utf-8")
    assert "[HUMAN]" in contents
