"""Presenter modules that convert discrete actions into environment messages."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional

from deceptionNet.config import InferenceConfig


class VotePresenter:
    """Formats vote actions according to guardrails."""

    def __init__(self, config: InferenceConfig) -> None:
        self.token_fmt = config.vote_token_fmt

    def render(self, target_index: int) -> str:
        if target_index < 0:
            return self.token_fmt.format(index=0)
        return self.token_fmt.format(index=int(target_index))


class NightPresenter:
    """Formats night actions, ensuring mask compliance."""

    def __init__(self, config: InferenceConfig) -> None:
        self.token_fmt = config.vote_token_fmt

    def render(self, target_index: int) -> str:
        if target_index < 0:
            return self.token_fmt.format(index=0)
        return self.token_fmt.format(index=int(target_index))


@dataclass
class TalkAux:
    """Auxiliary context to help craft richer utterances."""

    belief_summary: Optional[str] = None
    claimed_role: Optional[str] = None
    investigative_result: Optional[str] = None
    suspicion_scores: Optional[List[float]] = None
    trust_scores: Optional[List[float]] = None
    support_scores: Optional[List[float]] = None
    accusation_scores: Optional[List[float]] = None
    contradiction_scores: Optional[List[float]] = None
    bandwagon_scores: Optional[List[float]] = None


class TalkPresenter:
    """Template-based presenter driving day talk outputs."""

    VALID_PATTERN = re.compile(r"^[A-Za-z0-9\[\]\s,\.\?\!\-']+$")

    def __init__(self, config: InferenceConfig) -> None:
        self.templates = config.presenter
        self.max_tokens = config.max_tokens_talk
        self.silence_token = config.silence_token
        self.vote_token_fmt = config.vote_token_fmt

    def render(self, intent: int, target: int, aux: Optional[TalkAux] = None) -> str:
        aux = aux or TalkAux()
        if intent == 0:
            sentence = self._sample(self.templates.accuse_templates).format(target=self._idx(target))
        elif intent == 1:
            sentence = self._sample(self.templates.defend_self_templates)
        elif intent == 2:
            sentence = self._sample(self.templates.defend_other_templates).format(target=self._idx(target))
        elif intent == 3:
            target_idx = self._idx(target)
            template = self._sample(self.templates.claim_templates)
            if aux.claimed_role:
                sentence = f"I'm the {aux.claimed_role}. {template.format(target=target_idx)}"
            else:
                sentence = template.format(target=target_idx)
        elif intent == 4:
            sentence = f"I agree with {self._idx(target)}; their reasoning tracks with my reads."
        elif intent == 5:
            sentence = self._sample(self.templates.ask_templates).format(target=self._idx(target))
        elif intent == 6:
            sentence = self._sample(self.templates.filler_templates)
        else:
            sentence = self.silence_token

        justification = self._build_justification(intent, target, aux)
        if justification:
            if sentence and sentence[-1] not in ".!?":
                sentence = f"{sentence}."
            sentence = f"{sentence} Because {justification}"

        sentence = self._guardrail(sentence)
        return sentence

    def _idx(self, index: int) -> str:
        if index < 0:
            return "unknown"
        return self.vote_token_fmt.format(index=int(index))

    def _sample(self, choices) -> str:
        return random.choice(choices) if choices else ""

    def _guardrail(self, text: str) -> str:
        tokens = text.split()
        if len(tokens) > self.max_tokens:
            tokens = tokens[: self.max_tokens]
        filtered = " ".join(tokens)
        filtered = filtered.replace('I will say:', '').strip()
        filtered = re.sub(r"\s+", " ", filtered).strip()
        filtered = filtered.encode("ascii", "ignore").decode("ascii")
        if filtered and not self.VALID_PATTERN.fullmatch(filtered):
            filtered = ""
        if not filtered:
            return self.silence_token
        if filtered[-1] not in ".!?":
            filtered = f"{filtered}."
        return filtered

    def _safe_score(self, values: Optional[List[float]], index: int) -> Optional[float]:
        if values is None or index < 0 or index >= len(values):
            return None
        try:
            return float(values[index])
        except (TypeError, ValueError):
            return None

    def _format_percent(self, value: float) -> str:
        return f"{int(round(max(0.0, min(1.0, value)) * 100))}%"

    def _build_justification(self, intent: int, target: int, aux: TalkAux) -> str:
        suspicion = self._safe_score(aux.suspicion_scores, target)
        trust = self._safe_score(aux.trust_scores, target)
        support = self._safe_score(aux.support_scores, target)
        contradiction = self._safe_score(aux.contradiction_scores, target)
        bandwagon = self._safe_score(aux.bandwagon_scores, target)

        if intent == 0 and target >= 0:
            if suspicion is not None and suspicion >= 0.55:
                return f"{self._idx(target)} sits at suspicion {self._format_percent(suspicion)}."
            if contradiction is not None and contradiction >= 0.5:
                return f"{self._idx(target)} contradicted earlier statements."
            if bandwagon is not None and bandwagon >= 0.5:
                return f"multiple players already pushed {self._idx(target)} and the pattern fits."
        elif intent == 2 and target >= 0:
            if trust is not None and trust >= 0.55:
                return f"their trust score remains {self._format_percent(trust)} in my tracker."
            if support is not None and support > 0:
                return f"chat sentiment keeps backing {self._idx(target)}."
        elif intent == 5 and target >= 0:
            if bandwagon is not None and bandwagon >= 0.5:
                return "the pressure on them looks like bandwagoning."
        elif intent == 1:
            if aux.belief_summary:
                return aux.belief_summary.split('.', 1)[0].strip()

        if aux.belief_summary:
            return aux.belief_summary.split('.', 1)[0].strip()
        return ""


__all__ = ["VotePresenter", "NightPresenter", "TalkPresenter", "TalkAux"]
