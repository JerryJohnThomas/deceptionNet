
"""Basic tests for TextToStateMapper heuristics."""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from deceptionNet.agents import TextToStateMapper


def test_role_and_phase_detection() -> None:
    mapper = TextToStateMapper(num_players=6)
    state = mapper.update("You are the Doctor.")
    assert state["role"] == "doctor"

    state = mapper.update("Night phase begins. Mafia, make your move.")
    assert state["phase"] == "night"

    state = mapper.update("Day 2 begins. Discussion phase.")
    assert state["phase"] == "day_talk"
    assert state["round"] == 2

    mapper.update({"self_player": 4, "phase": "day_vote", "alive": [1, 1, 1, 1, 1, 1]})
    state = mapper.get_state()
    assert state["self_player"] == 4


def test_elimination_and_talk_tracking() -> None:
    mapper = TextToStateMapper(num_players=6)
    mapper.update("Player 3: I think [2] is suspicious.")
    state = mapper.get_state()
    assert state["talk_history"][-1]["speaker"] == 3

    mapper.update("Player 2 was eliminated by vote.")
    state = mapper.get_state()
    assert state["alive"][2] == 0


def test_vote_tracking() -> None:
    mapper = TextToStateMapper(num_players=6)
    mapper.update("Player 1 voted [4]")
    state = mapper.get_state()
    assert state["vote_history"][-1]["voter"] == 1
    assert state["vote_history"][-1]["target"] == 4


if __name__ == "__main__":
    test_role_and_phase_detection()
    test_elimination_and_talk_tracking()
    test_vote_tracking()
    print("OK")
