"""Utility helpers for deceptionNet package."""

from __future__ import annotations

from typing import Optional

import json
import os
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F


Tensor = torch.Tensor


def masked_log_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Compute log softmax with additive mask of invalid entries."""

    mask = mask.to(dtype=logits.dtype)
    very_negative = torch.finfo(logits.dtype).min
    masked_logits = torch.where(mask > 0, logits, torch.full_like(logits, very_negative))
    return F.log_softmax(masked_logits, dim=dim)


def masked_softmax(logits: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    """Softmax that respects a binary mask."""

    log_probs = masked_log_softmax(logits, mask, dim=dim)
    return log_probs.exp()


def apply_masks(logits: Tensor, mask: Tensor) -> Tensor:
    """Mask logits with a very negative number for invalid entries."""

    very_negative = torch.finfo(logits.dtype).min
    return logits.masked_fill(mask <= 0, very_negative)


def sequence_mask(lengths: Tensor, max_len: Optional[int] = None) -> Tensor:
    """Create a boolean mask of shape (batch, max_len) with valid positions."""

    if max_len is None:
        max_len = int(lengths.max().item())
    range_ = torch.arange(max_len, device=lengths.device)
    return range_.unsqueeze(0) < lengths.unsqueeze(1)


def chunk_time(batch: Tensor, num_chunks: int) -> Tensor:
    """Reshape (time, batch, ...) into (chunks, time/chunks, batch, ...)."""

    T = batch.shape[0]
    assert T % num_chunks == 0, "Unroll length must be divisible by num_chunks"
    new_shape = (num_chunks, T // num_chunks) + batch.shape[1:]
    return batch.reshape(new_shape)


def normalize(tensor: Tensor, eps: float = 1e-6) -> Tensor:
    """Layer-wise normalization helper."""

    mean = tensor.mean(dim=-1, keepdim=True)
    var = tensor.var(dim=-1, keepdim=True, unbiased=False)
    return (tensor - mean) / torch.sqrt(var + eps)


def entropy_from_logits(logits: Tensor, mask: Optional[Tensor] = None, dim: int = -1) -> Tensor:
    """Compute categorical entropy from logits, optionally applying a mask."""

    if mask is not None:
        probs = masked_softmax(logits, mask, dim=dim)
    else:
        probs = torch.softmax(logits, dim=dim)
    log_probs = torch.log(probs + 1e-12)
    entropy = -(probs * log_probs).sum(dim=dim)
    return entropy


class ComparisonLogger:
    """Utility to log listener/presenter comparisons as JSONL."""

    def __init__(self, directory: str = "logs", prefix: str = "listener_compare", filename: Optional[str] = None) -> None:
        base_dir = Path(directory)
        base_dir.mkdir(parents=True, exist_ok=True)
        if filename:
            self.path = base_dir / filename
        else:
            timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
            self.path = base_dir / f"{prefix}_{timestamp}.jsonl"
        self._file = self.path.open("w", encoding="utf-8")
        self._closed = False

    def log(self, record: dict) -> None:
        if self._closed:
            return
        json.dump(record, self._file, ensure_ascii=True)
        self._file.write("\n")
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except OSError:
            pass

    def flush(self) -> None:
        if self._closed:
            return
        self._file.flush()
        try:
            os.fsync(self._file.fileno())
        except OSError:
            pass

    def close(self) -> None:
        if self._closed:
            return
        self.flush()
        self._file.close()
        self._closed = True

    @property
    def filepath(self) -> Path:
        return self.path

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

def save_jsonl(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def truncate_to_lines(text: str, max_lines: int) -> str:
    """Collapse text into at most `max_lines` sentences/lines."""

    if max_lines <= 0:
        return ""
    cleaned = text.replace("\r", "\n").strip()
    if not cleaned:
        return ""
    segments: list[str] = []
    for line in cleaned.split("\n"):
        trimmed = line.strip()
        if not trimmed:
            continue
        cursor = 0
        for idx, char in enumerate(trimmed):
            if char in ".!?":
                segment = trimmed[cursor:idx + 1].strip()
                if segment:
                    segments.append(segment)
                cursor = idx + 1
        tail_segment = trimmed[cursor:].strip()
        if tail_segment:
            segments.append(tail_segment)
    if not segments:
        segments = [cleaned]
    limited = segments[: max_lines]
    return " ".join(limited).strip()



def truncate_to_lines(text: str, max_lines: int) -> str:
    """Collapse text into at most `max_lines` sentences/lines."""

    if max_lines <= 0:
        return ""
    cleaned = text.replace("\r", "\n").strip()
    if not cleaned:
        return ""
    segments: list[str] = []
    for line in cleaned.split("\n"):
        trimmed = line.strip()
        if not trimmed:
            continue
        cursor = 0
        for idx, char in enumerate(trimmed):
            if char in ".!?":
                segment = trimmed[cursor:idx + 1].strip()
                if segment:
                    segments.append(segment)
                cursor = idx + 1
        tail_segment = trimmed[cursor:].strip()
        if tail_segment:
            segments.append(tail_segment)
    if not segments:
        segments = [cleaned]
    limited = segments[: max_lines]
    return " ".join(limited).strip()


def debug_log(message: str, *, path: Path | None = None) -> None:
    """Append debug output to a rolling log file."""

    path = path or Path("logs/debug_latest.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message + "\n")


__all__ = [
    "Tensor",
    "masked_log_softmax",
    "masked_softmax",
    "apply_masks",
    "sequence_mask",
    "chunk_time",
    "normalize",
    "entropy_from_logits",
    "ComparisonLogger",
    "save_jsonl",
    "truncate_to_lines",
    "debug_log",
]
