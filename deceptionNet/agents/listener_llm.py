"""Optional listener that can summarize conversations with a local HF model."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore

AutoTokenizer = None  # type: ignore
AutoModelForCausalLM = None  # type: ignore

from deceptionNet.config import ModelDims
from deceptionNet.utils import debug_log
from .listener import ListenerOutput, SimpleListener


DEFAULT_MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"
DEFAULT_PROMPT = (
    "Summarize the following social deduction conversation as JSON. "
    "Keys: accuses (dict[str, list[int]]), defends (dict[str, list[int]]), "
    "claims (dict[str, str]), tone (dict[str, str in {{confident, unsure, neutral}}).\n\n"
    "Text:\n{conversation}\n"
)





@dataclass
class LLMListenerConfig:
    model_name: str = DEFAULT_MODEL_NAME
    device: Optional[str] = None
    max_new_tokens: int = 160
    temperature: float = 0.0
    prompt_template: str = DEFAULT_PROMPT
    max_history_chars: int = 2000
    cache_size: int = 16
    cache_dir: Optional[str] = None
    load_in_8bit: bool = False
    torch_dtype: Optional[str] = "float16"
    fallback_model_name: Optional[str] = "Qwen/Qwen2.5-1.8B-Instruct"


class LLMListener(SimpleListener):
    """Wrap the rule-based listener and optionally enrich with a local HF model."""

    _MODEL = None
    _TOKENIZER = None
    _MODEL_NAME = None
    _MODEL_CONFIG = None
    _ANNOUNCED = False

    def __init__(self, dims: ModelDims, config: Optional[LLMListenerConfig] = None) -> None:
        super().__init__(dims)
        self.config = config or LLMListenerConfig()
        if not self.config.cache_dir:
            self.config.cache_dir = os.getenv("HF_HOME")
        self.device = self.config.device or ("cuda" if torch and torch.cuda.is_available() else "cpu")
        self._ensure_model(self.config)
        self.device = getattr(self.__class__, '_MODEL_DEVICE', self.device)
        if self._MODEL_NAME and self.config.model_name != self._MODEL_NAME:
            self.config.model_name = self._MODEL_NAME
        if self._MODEL is not None and not self._ANNOUNCED:
            message = f"LLM Listener active: {self.config.model_name} (device={self.device})"
            print(message)
            debug_log(message)
            self.__class__._ANNOUNCED = True
        self._cache: OrderedDict[str, Dict] = OrderedDict()
        self._last_conversation: Optional[str] = None
        self._last_summary: Optional[Dict] = None

    @classmethod
    def _resolve_dtype(cls, dtype_value):
        if torch is None or dtype_value is None:
            return None
        if isinstance(dtype_value, str):
            mapping = {
                "float16": torch.float16,
                "fp16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            return mapping.get(dtype_value.lower())
        return dtype_value

    @classmethod
    def _ensure_model(cls, config: LLMListenerConfig) -> None:  # pragma: no cover - heavy load
        model_name = config.model_name
        if torch is None:
            cls._MODEL = None
            cls._TOKENIZER = None
            cls._MODEL_NAME = None
            cls._MODEL_CONFIG = None
            return
        resolved_dtype = cls._resolve_dtype(config.torch_dtype)
        cache_dir = config.cache_dir
        config_key = (model_name, cache_dir, bool(config.load_in_8bit), str(resolved_dtype))
        if (
            cls._MODEL is not None
            and cls._MODEL_NAME == model_name
            and cls._MODEL_CONFIG == config_key
        ):
            return
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

            model_kwargs = {
                "device_map": "auto",
                "cache_dir": cache_dir,
                "low_cpu_mem_usage": True,
            }
            if resolved_dtype is not None:
                model_kwargs["torch_dtype"] = resolved_dtype
            if config.load_in_8bit:
                model_kwargs["load_in_8bit"] = True

            if os.path.isdir(model_name):
                tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        except Exception as exc:
            if config.load_in_8bit:
                try:
                    print(f"Warning: load_in_8bit requested but unavailable ({exc}). Falling back to full precision.")
                    # retry without 8bit flag
                    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
                    model_kwargs.pop("load_in_8bit", None)
                    if os.path.isdir(model_name):
                        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
                        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                    else:
                        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
                        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                except Exception:
                    tokenizer = None
                    model = None
                    fallback_exc = exc
            else:
                tokenizer = None
                model = None
                fallback_exc = exc

            if model is None:
                fallback = config.fallback_model_name
                if fallback and fallback != model_name:
                    print(f"Warning: failed to load LLM listener model '{model_name}': {fallback_exc}. Trying fallback '{fallback}'.")
                    fallback_config = replace(config, model_name=fallback)
                    cls._ensure_model(fallback_config)
                    return
                print(f"Warning: failed to load LLM listener model '{model_name}': {fallback_exc}")
                cls._MODEL = None
                cls._TOKENIZER = None
                cls._MODEL_NAME = None
                cls._MODEL_CONFIG = None
                return
        model.eval()
        cls._MODEL = model
        cls._TOKENIZER = tokenizer
        cls._MODEL_NAME = model_name
        cls._MODEL_CONFIG = config_key
    def __call__(self, state: Dict) -> ListenerOutput:
        base_output = super().__call__(state)
        if self._MODEL is None or self._TOKENIZER is None or torch is None:
            return base_output
        conversation_raw = self._conversation_text(state)
        conversation = self._trim_history(conversation_raw)
        if not conversation.strip():
            return base_output
        if conversation == self._last_conversation and self._last_summary is not None:
            summary = self._last_summary
        else:
            summary = self._summarize(conversation)
            self._last_conversation = conversation
            self._last_summary = summary if summary is not None else {}
        if summary:
            self._apply_summary(summary, base_output)
        return base_output

    def _conversation_text(self, state: Dict) -> str:
        history = state.get("talk_history") or []
        lines = [f"{item.get('speaker')}: {item.get('text')}" for item in history if item.get("text")]
        if not lines and state.get("recent_text"):
            lines.append(f"system: {state['recent_text']}")
        return "\n".join(lines)

    def _trim_history(self, conversation: str) -> str:
        limit = max(0, int(self.config.max_history_chars))
        if limit <= 0 or len(conversation) <= limit:
            return conversation
        lines = conversation.splitlines()
        collected = []
        total = 0
        for line in reversed(lines):
            length = len(line)
            if collected:
                length += 1  # account for newline separation
            if total + length > limit and collected:
                break
            collected.append(line)
            total += length
            if total >= limit:
                break
        return "\n".join(reversed(collected))

    def _summarize(self, conversation: str) -> Optional[Dict]:  # pragma: no cover - heavy
        if conversation in self._cache:
            cached = self._cache[conversation]
            self._cache.move_to_end(conversation)
            return cached
        try:
            prompt = self.config.prompt_template.format(conversation=conversation)
            tokenizer = self._TOKENIZER
            model = self._MODEL
            assert tokenizer is not None and model is not None
            inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature,
                )
            text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            summary = self._safe_json_parse(text)
        except Exception:
            summary = None
        cached = summary or {}
        self._cache[conversation] = cached
        self._cache.move_to_end(conversation)
        while len(self._cache) > max(1, int(self.config.cache_size)):
            self._cache.popitem(last=False)
        return summary

    def _safe_json_parse(self, text: str) -> Optional[Dict]:
        text = text.strip()
        if not text:
            return None
        if text.startswith("{") and text.endswith("}"):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                pass
        # Attempt to extract JSON substring
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    def _apply_summary(self, summary: Dict, output: ListenerOutput) -> None:
        accuses = summary.get("accuses", {}) or {}
        defends = summary.get("defends", {}) or {}
        claims = summary.get("claims", {}) or {}
        tone = summary.get("tone", {}) or {}
        for src, targets in accuses.items():
            for tgt in targets or []:
                idx = int(tgt) if isinstance(tgt, (int, str)) else None
                if idx is not None and 0 <= idx < len(output.player_mentions):
                    output.accusation_scores[idx] += 0.5
                    output.player_mentions[idx] += 0.2
        for src, targets in defends.items():
            for tgt in targets or []:
                idx = int(tgt) if isinstance(tgt, (int, str)) else None
                if idx is not None and 0 <= idx < len(output.player_mentions):
                    output.support_scores[idx] += 0.5
        for player, role in claims.items():
            idx = int(player) if isinstance(player, (int, str)) else None
            if idx is not None and 0 <= idx < len(output.sentiment_scores):
                output.sentiment_scores[idx] += 0.1
        for player, tone_value in tone.items():
            idx = int(player) if isinstance(player, (int, str)) else None
            if idx is not None and 0 <= idx < len(output.sentiment_scores):
                if str(tone_value).lower() == "confident":
                    output.sentiment_scores[idx] += 0.2
                elif str(tone_value).lower() == "unsure":
                    output.sentiment_scores[idx] -= 0.1

        contradictions = summary.get("contradictions", []) or []
        for player in contradictions:
            idx = int(player) if isinstance(player, (int, str)) else None
            if idx is not None and 0 <= idx < len(output.contradiction_scores):
                output.contradiction_scores[idx] = max(float(output.contradiction_scores[idx]), 1.0)

        accuser_sets = {}
        for src, targets in accuses.items():
            for tgt in targets or []:
                idx = int(tgt) if isinstance(tgt, (int, str)) else None
                if idx is not None and 0 <= idx < len(output.bandwagon_scores):
                    accuser_sets.setdefault(idx, set()).add(str(src))

        for idx, accusers in accuser_sets.items():
            if len(accusers) >= 2:
                boost = min(1.0, 0.25 * len(accusers))
                output.bandwagon_scores[idx] += boost

        if hasattr(output.bandwagon_scores, "clamp_"):
            output.bandwagon_scores.clamp_(0.0, 1.0)
        if hasattr(output.contradiction_scores, "clamp_"):
            output.contradiction_scores.clamp_(0.0, 1.0)


__all__ = ["LLMListener", "LLMListenerConfig"]




