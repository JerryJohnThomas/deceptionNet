"""Belief tracker for opponent role inference and suspicion modelling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from deceptionNet.config import ModelDims
from deceptionNet.utils import Tensor

from .featurizer import ObservationFeatures

PHASE_DIM = 3


@dataclass
class BeliefState:
    """Latent belief representation at a timestep."""

    player_embeddings: Tensor  # (B, N, D)
    global_hidden: Tensor  # (B, D)
    role_logits: Tensor  # (B, N, R)
    suspicion: Tensor  # (B, N)
    trust: Tensor  # (B, N)

    def detach(self) -> "BeliefState":
        return BeliefState(
            player_embeddings=self.player_embeddings.detach(),
            global_hidden=self.global_hidden.detach(),
            role_logits=self.role_logits.detach(),
            suspicion=self.suspicion.detach(),
            trust=self.trust.detach(),
        )


class BeliefNet(nn.Module):
    """Neural Bayes tracker implementing the specification from plan1."""

    def __init__(
        self,
        dims: ModelDims,
        player_feature_dim: int = 14,
        extra_player_dim: int = 6,
        global_feature_dim: int = 5,
    ) -> None:
        super().__init__()
        hidden = dims.belief_hidden_size
        self.num_players = dims.num_players
        self.num_roles = dims.num_roles
        self.hidden_size = hidden

        self.player_encoder = nn.Sequential(
            nn.Linear(player_feature_dim + extra_player_dim, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.global_encoder = nn.Sequential(
            nn.Linear(global_feature_dim + 3, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.convo_encoder = nn.Sequential(
            nn.Linear(dims.convo_hidden_size, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.global_gru = nn.GRUCell(hidden, hidden)

        self.role_head = nn.Linear(hidden, self.num_roles)
        self.suspicion_head = nn.Linear(hidden, 1)
        self.trust_head = nn.Linear(hidden, 1)

    def initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> BeliefState:
        device = device or torch.device("cpu")
        player_embeds = torch.zeros(batch_size, self.num_players, self.hidden_size, device=device)
        global_hidden = torch.zeros(batch_size, self.hidden_size, device=device)
        role_logits = torch.zeros(batch_size, self.num_players, self.num_roles, device=device)
        suspicion = torch.zeros(batch_size, self.num_players, device=device)
        trust = torch.zeros(batch_size, self.num_players, device=device)
        return BeliefState(player_embeds, global_hidden, role_logits, suspicion, trust)

    def forward(self, features: ObservationFeatures, state: Optional[BeliefState] = None) -> BeliefState:
        batch_size = features.player_features.shape[0]
        device = features.player_features.device
        if state is None:
            state = self.initial_state(batch_size, device)

        player_inputs = torch.cat(
            [
                features.player_features,
                features.mention_scores.unsqueeze(-1),
                features.sentiment_scores.unsqueeze(-1),
                features.support_scores.unsqueeze(-1),
                features.accusations.unsqueeze(-1),
                features.contradiction_scores.unsqueeze(-1),
                features.bandwagon_scores.unsqueeze(-1),
            ],
            dim=-1,
        )
        player_encoded = self.player_encoder(player_inputs)
        convo_encoded = self.convo_encoder(features.conversation_embedding)
        global_encoded = self.global_encoder(torch.cat([features.global_features, features.phase_encoding], dim=-1))
        fused_context = convo_encoded + global_encoded
        player_contextual = player_encoded + fused_context.unsqueeze(1)

        transformer_out = self.transformer(player_contextual)
        alive_mask = features.alive_mask
        mask_sum = alive_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_pool = (transformer_out * alive_mask.unsqueeze(-1)).sum(dim=1) / mask_sum
        global_hidden = self.global_gru(mean_pool, state.global_hidden)

        role_logits = self.role_head(transformer_out)
        suspicion = torch.sigmoid(self.suspicion_head(transformer_out)).squeeze(-1)
        trust = torch.sigmoid(self.trust_head(transformer_out)).squeeze(-1)

        return BeliefState(
            player_embeddings=transformer_out,
            global_hidden=global_hidden,
            role_logits=role_logits,
            suspicion=suspicion,
            trust=trust,
        )


__all__ = ["BeliefState", "BeliefNet"]
