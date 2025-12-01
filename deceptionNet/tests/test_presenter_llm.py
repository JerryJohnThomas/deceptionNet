"""Tests for the LLM presenter wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deceptionNet.agents.presenter_llm import LLMPresenter


def test_llm_presenter_returns_text_without_pipeline():
    presenter = LLMPresenter(model_name="mock/model", max_lines=2)
    text = presenter.render(
        intent="accuse",
        target="[2]",
        phase="day_talk",
        belief_summary="They keep dodging questions.",
        player_id=3,
    )
    assert isinstance(text, str)
    assert text
    sentence_endings = sum(text.count(mark) for mark in ".!?")
    assert sentence_endings <= 2


def test_llm_presenter_truncate():
    presenter = LLMPresenter(model_name="fake", max_lines=1)
    text = presenter.render(
        intent="question",
        target="[4]",
        phase="day_talk",
        belief_summary="Multiple players doubted them.",
        player_id=1,
    )
    assert "\n" not in text
    endings = sum(text.count(mark) for mark in ".!?")
    assert endings <= 1
