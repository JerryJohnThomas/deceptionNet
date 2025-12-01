"""Feature extraction from Listener output and environment observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from deceptionNet.config import ModelDims
from deceptionNet.utils import Tensor

from .listener import ListenerOutput


@dataclass
class ObservationFeatures:
    """Structured tensors that feed into the learning modules."""

    player_features: Tensor  # (B, N, F)
    global_features: Tensor  # (B, G)
    conversation_embedding: Tensor  # (B, D)
    alive_mask: Tensor  # (B, N)
    mafia_mask: Tensor  # (B, N)
    self_index: Tensor  # (B,)
    role_index: Tensor  # (B,)
    phase_encoding: Tensor  # (B, P)
    vote_matrix: Tensor  # (B, N, N)
    mention_scores: Tensor  # (B, N)
    sentiment_scores: Tensor  # (B, N)
    support_scores: Tensor  # (B, N)
    accusations: Tensor  # (B, N)
    contradiction_scores: Tensor  # (B, N)
    bandwagon_scores: Tensor  # (B, N)
    round_index: Tensor  # (B,)


class ObservationFeaturizer:
    """Converts raw game state dictionaries into dense tensors."""

    PHASE_ORDER = ["night", "day_talk", "day_vote"]

    def __init__(self, dims: ModelDims):
        self.num_players = dims.num_players
        self.hidden_size = dims.hidden_size
        self.phase_to_index = {name: idx for idx, name in enumerate(self.PHASE_ORDER)}

    def __call__(self, observation: dict, listener_output: ListenerOutput) -> ObservationFeatures:
        batch_player = self._build_player_features(observation, listener_output)
        global_feats = self._build_global_features(observation)
        conversation = listener_output.conversation_embedding.unsqueeze(0)
        alive_mask = self._alive_mask(observation)
        mafia_mask = self._mafia_mask(observation)
        self_index = torch.tensor([observation.get("self_player", 0)], dtype=torch.long)
        role_index = torch.tensor([self._role_to_index(observation.get("role")) or 0], dtype=torch.long)
        phase = self._phase_encoding(observation)
        vote_matrix = self._vote_matrix(observation)
        round_index = torch.tensor([observation.get("round", 0)], dtype=torch.float32)
        mention = listener_output.player_mentions.unsqueeze(0)
        sentiment = listener_output.sentiment_scores.unsqueeze(0)
        support = listener_output.support_scores.unsqueeze(0)
        accusations = listener_output.accusation_scores.unsqueeze(0)
        contradictions = listener_output.contradiction_scores.unsqueeze(0)
        bandwagon = listener_output.bandwagon_scores.unsqueeze(0)

        return ObservationFeatures(
            player_features=batch_player,
            global_features=global_feats,
            conversation_embedding=conversation,
            alive_mask=alive_mask,
            mafia_mask=mafia_mask,
            self_index=self_index,
            role_index=role_index,
            phase_encoding=phase,
            vote_matrix=vote_matrix,
            mention_scores=mention,
            sentiment_scores=sentiment,
            support_scores=support,
            accusations=accusations,
            contradiction_scores=contradictions,
            bandwagon_scores=bandwagon,
            round_index=round_index,
        )

    def _alive_mask(self, observation: dict) -> Tensor:
        alive = observation.get("alive")
        if alive is None:
            mask = torch.ones(self.num_players, dtype=torch.float32)
        else:
            padded = list(alive) + [False] * max(0, self.num_players - len(alive))
            mask = torch.tensor([1.0 if flag else 0.0 for flag in padded[: self.num_players]])
        return mask.unsqueeze(0)

    def _mafia_mask(self, observation: dict) -> Tensor:
        known_mafia = observation.get("known_mafia", [])
        mask = torch.zeros(self.num_players, dtype=torch.float32)
        for idx in known_mafia:
            if 0 <= idx < self.num_players:
                mask[idx] = 1.0
        return mask.unsqueeze(0)

    def _phase_encoding(self, observation: dict) -> Tensor:
        phase_name = observation.get("phase", "day_talk").lower()
        vec = torch.zeros(len(self.PHASE_ORDER), dtype=torch.float32)
        if phase_name not in self.phase_to_index:
            phase_name = "day_talk"
        vec[self.phase_to_index[phase_name]] = 1.0
        return vec.unsqueeze(0)

    def _vote_matrix(self, observation: dict) -> Tensor:
        history = observation.get("vote_history", [])
        matrix = torch.zeros((self.num_players, self.num_players), dtype=torch.float32)
        for vote in history:
            voter = int(vote.get("voter", -1))
            target = int(vote.get("target", -1))
            if 0 <= voter < self.num_players and 0 <= target < self.num_players:
                matrix[voter, target] += 1.0
        return matrix.unsqueeze(0)

    def _build_player_features(self, observation: dict, listener_output: ListenerOutput) -> Tensor:
        alive_mask = self._alive_mask(observation).squeeze(0)
        vote_matrix = self._vote_matrix(observation).squeeze(0)
        votes_for = vote_matrix.sum(dim=1)
        votes_against = vote_matrix.sum(dim=0)
        mention = listener_output.player_mentions
        sentiment = listener_output.sentiment_scores
        support = listener_output.support_scores
        accusations = listener_output.accusation_scores
        contradictions = listener_output.contradiction_scores
        bandwagon = listener_output.bandwagon_scores

        claims = observation.get("role_claims", {})
        claim_features = torch.zeros(self.num_players, dtype=torch.float32)
        for idx, claim in claims.items():
            idx_int = int(idx)
            if 0 <= idx_int < self.num_players:
                claim_features[idx_int] = 1.0

        role_hist = observation.get("role_history", {})
        role_embeddings = torch.zeros((self.num_players, 4), dtype=torch.float32)
        for idx, role_name in role_hist.items():
            pos = self._role_to_index(role_name)
            idx_int = int(idx)
            if pos is not None and 0 <= idx_int < self.num_players:
                role_embeddings[idx_int, pos] = 1.0

        base = torch.stack(
            [
                alive_mask,
                votes_for,
                votes_against,
                mention,
                sentiment,
                support,
                accusations,
                contradictions,
                bandwagon,
                claim_features,
            ],
            dim=1,
        )
        features = torch.cat([base, role_embeddings], dim=1)
        return features.unsqueeze(0)

    def _build_global_features(self, observation: dict) -> Tensor:
        round_idx = observation.get("round", 0)
        talk_turn = observation.get("turn", 0)
        remaining = observation.get("alive", [])
        num_alive = sum(bool(flag) for flag in remaining) if remaining else self.num_players
        alive_ratio = num_alive / max(self.num_players, 1)
        mafia_estimate = observation.get("mafia_count_estimate", 1) / max(self.num_players, 1)
        global_vec = torch.tensor(
            [
                float(round_idx) / 10.0,
                float(talk_turn) / 10.0,
                alive_ratio,
                mafia_estimate,
                float(num_alive),
            ],
            dtype=torch.float32,
        )
        return global_vec.unsqueeze(0)

    def _role_to_index(self, name: Optional[str]) -> Optional[int]:
        if not name:
            return None
        name = str(name).lower()
        mapping = {"villager": 0, "mafia": 1, "doctor": 2, "detective": 3}
        return mapping.get(name)


__all__ = ["ObservationFeatures", "ObservationFeaturizer"]
