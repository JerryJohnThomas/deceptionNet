"""Listener module that converts raw observation text into structured signals."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch

from deceptionNet.config import ModelDims

Tensor = torch.Tensor


@dataclass
class ListenerOutput:
    """Structured representation of conversational signals."""

    conversation_embedding: Tensor  # (hidden,)
    player_mentions: Tensor  # (num_players,)
    accusation_scores: Tensor  # (num_players,)
    support_scores: Tensor  # (num_players,)
    sentiment_scores: Tensor  # (num_players,)
    contradiction_scores: Tensor  # (num_players,)
    bandwagon_scores: Tensor  # (num_players,)
    summary_text: str


class SimpleListener:
    """Rule-based listener that extracts lightweight features from text logs.

    The listener is intentionally lightweight and deterministic so we can
    swap it out later for a stronger LLM summarizer without touching the
    rest of the pipeline.
    """

    POSITIVE_CUES = {
        "good",
        "town",
        "village",
        "trust",
        "clear",
        "helpful",
        "safe",
    }
    NEGATIVE_CUES = {
        "suspect",
        "mafia",
        "wolf",
        "scum",
        "kill",
        "eliminate",
        "lynch",
        "vote",
        "bad",
        "lying",
    }
    ACCUSE_CUES = {"suspect", "accuse", "vote", "eliminate", "push"}
    SUPPORT_CUES = {"defend", "clear", "support", "town", "ally"}

    def __init__(self, dims: ModelDims):
        self.num_players = dims.num_players
        self.hidden_size = dims.convo_hidden_size
        self._mention_pattern = re.compile(r"\[(\d+)\]")

    def __call__(self, state: dict) -> ListenerOutput:
        messages = self._extract_messages(state)
        embedding = self._encode_messages(messages)
        mentions = torch.zeros(self.num_players, dtype=torch.float32)
        accusation = torch.zeros_like(mentions)
        support = torch.zeros_like(mentions)
        sentiment = torch.zeros_like(mentions)
        contradictions = torch.zeros_like(mentions)
        bandwagon = torch.zeros_like(mentions)

        accusation_tally = [0.0] * self.num_players
        speaker_accuse_targets = {idx: set() for idx in range(self.num_players)}
        speaker_support_targets = {idx: set() for idx in range(self.num_players)}

        for speaker, text in messages:
            if speaker == "meta":
                continue
            indices = self._mention_pattern.findall(text)
            indices = [int(idx) for idx in indices if int(idx) < self.num_players]
            cue_vector = self._score_text(text.lower())
            accuse_flag = cue_vector[0] > 0
            support_flag = cue_vector[1] > 0

            if speaker.isdigit():
                speaker_idx = int(speaker)
                if 0 <= speaker_idx < self.num_players:
                    if accuse_flag and indices:
                        speaker_accuse_targets[speaker_idx].update(indices)
                    if support_flag and indices:
                        speaker_support_targets[speaker_idx].update(indices)
                    if speaker_accuse_targets[speaker_idx] and speaker_support_targets[speaker_idx]:
                        contradictions[speaker_idx] = 1.0

            if accuse_flag:
                for idx in indices:
                    if accusation_tally[idx] >= 1.0:
                        bandwagon[idx] += 1.0

            for idx in indices:
                mentions[idx] += 1.0
                accusation[idx] += cue_vector[0]
                support[idx] += cue_vector[1]
                sentiment[idx] += cue_vector[2]
                if accuse_flag:
                    accusation_tally[idx] += 1.0

        if mentions.sum() > 0:
            mentions = mentions / mentions.sum().clamp(min=1.0)
        if bandwagon.max() > 0:
            bandwagon = bandwagon / bandwagon.max().clamp(min=1.0)
        summary_text = " ".join(text for _, text in messages[-5:])

        return ListenerOutput(
            conversation_embedding=embedding,
            player_mentions=mentions,
            accusation_scores=accusation,
            support_scores=support,
            sentiment_scores=sentiment,
            contradiction_scores=contradictions,
            bandwagon_scores=bandwagon,
            summary_text=summary_text,
        )

    def _extract_messages(self, state: dict) -> List[tuple[str, str]]:
        history = state.get("talk_history") or []
        messages: List[tuple[str, str]] = []
        role_tag = state.get("role", "unknown")
        phase_tag = state.get("phase", "unknown")
        prefix = f"[ROLE={role_tag.upper()}] [PHASE={phase_tag.upper()}]"
        messages.append(("meta", prefix))
        for entry in history[-50:]:
            speaker = entry.get("speaker", "")
            text = entry.get("text", "")
            if text:
                messages.append((str(speaker), str(text)))
        recent = state.get("recent_text")
        if recent:
            recent = str(recent)
            if not messages or messages[-1][1] != recent:
                messages.append(("system", recent))
        return messages

    def _encode_messages(self, messages: Sequence[tuple[str, str]]) -> Tensor:
        vector = torch.zeros(self.hidden_size, dtype=torch.float32)
        if not messages:
            return vector
        joined = " \n ".join(f"{speaker}: {text}" for speaker, text in messages)
        encoded = joined.encode("utf-8", "ignore")
        limit = min(len(encoded), self.hidden_size)
        if limit > 0:
            vector[:limit] = torch.tensor(
                [byte / 255.0 for byte in encoded[:limit]], dtype=torch.float32
            )
        if len(encoded) > self.hidden_size:
            # simple hashed tail aggregation
            tail = torch.tensor(
                [byte / 255.0 for byte in encoded[self.hidden_size :]], dtype=torch.float32
            )
            vector += tail.mean().repeat(self.hidden_size)
        norm = vector.norm(p=2)
        if norm > 0:
            vector = vector / norm
        return vector

    def _score_text(self, text: str) -> tuple[float, float, float]:
        tokens = re.findall(r"[a-zA-Z]+", text)
        if not tokens:
            return (0.0, 0.0, 0.0)
        positives = sum(token in self.POSITIVE_CUES for token in tokens)
        negatives = sum(token in self.NEGATIVE_CUES for token in tokens)
        accuse = sum(token in self.ACCUSE_CUES for token in tokens)
        support = sum(token in self.SUPPORT_CUES for token in tokens)
        sentiment = positives - negatives
        total = max(len(tokens), 1)
        return (
            accuse / total,
            support / total,
            sentiment / total,
        )


__all__ = ["ListenerOutput", "SimpleListener"]
