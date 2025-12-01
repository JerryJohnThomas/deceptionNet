"""Tests covering device fallback for presenter and listener."""

from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from deceptionNet.agents.presenter_llm import LLMPresenter
from deceptionNet.agents.listener_llm import LLMListener, LLMListenerConfig
from deceptionNet.config import ModelDims


class DummyGenerator:
    def __init__(self, response: str = "PROMPT result") -> None:
        self._response = response
        self.tokenizer = types.SimpleNamespace(eos_token_id=0)

    def __call__(self, prompt: str, **kwargs):  # pragma: no cover - simple stub
        return [{"generated_text": prompt + " " + self._response}]


def _restore_module(name: str, original):
    if original is None:
        sys.modules.pop(name, None)
    else:
        sys.modules[name] = original


def test_presenter_falls_back_to_cpu_on_cuda_error():
    original_transformers = sys.modules.get("transformers")
    call_state = {"count": 0}

    def pipeline_stub(task: str, **kwargs):
        call_state["count"] += 1
        if call_state["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        return DummyGenerator()

    sys.modules["transformers"] = types.SimpleNamespace(pipeline=pipeline_stub)
    original_cuda_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: True  # force initial CUDA attempt
    try:
        presenter = LLMPresenter(model_name="mock/presenter", device="auto")
        assert presenter.active_device == "cpu"
        assert presenter._generator is not None
    finally:
        torch.cuda.is_available = original_cuda_available
        _restore_module("transformers", original_transformers)


def test_presenter_uses_cpu_when_cuda_unavailable():
    original_transformers = sys.modules.get("transformers")

    def pipeline_stub(task: str, **kwargs):
        return DummyGenerator()

    sys.modules["transformers"] = types.SimpleNamespace(pipeline=pipeline_stub)
    original_cuda_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        presenter = LLMPresenter(model_name="mock/presenter", device="auto")
        assert presenter.active_device == "cpu"
    finally:
        torch.cuda.is_available = original_cuda_available
        _restore_module("transformers", original_transformers)


def test_listener_fallbacks_to_cpu_on_cuda_error():
    original_transformers = sys.modules.get("transformers")
    call_state = {"count": 0}

    class DummyModel:
        def eval(self):  # pragma: no cover - simple stub
            return None

    def model_from_pretrained(model_name: str, **kwargs):
        call_state["count"] += 1
        if call_state["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        return DummyModel()

    def tokenizer_from_pretrained(model_name: str, **kwargs):
        return types.SimpleNamespace()

    sys.modules["transformers"] = types.SimpleNamespace(
        AutoModelForCausalLM=types.SimpleNamespace(from_pretrained=model_from_pretrained),
        AutoTokenizer=types.SimpleNamespace(from_pretrained=tokenizer_from_pretrained),
    )
    original_cuda_available = torch.cuda.is_available
    torch.cuda.is_available = lambda: True
    try:
        config = LLMListenerConfig(model_name="mock/listener", device="auto")
        listener = LLMListener(ModelDims(), config=config)
        assert listener.device == "cpu"
    finally:
        torch.cuda.is_available = original_cuda_available
        _restore_module("transformers", original_transformers)
