"""Transforms belief outputs and features into the shared state S_t."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from deceptionNet.config import ModelDims
from deceptionNet.datatypes import StepMasks

from .belief_net import BeliefState
from .featurizer import ObservationFeatures


@dataclass
class SharedState:
    """Shared representation consumed by the control heads."""

    player_embeddings: torch.Tensor  # (B, N, D)
    convo_embedding: torch.Tensor  # (B, D)
    memory_state: torch.Tensor  # (B, D)
    masks: StepMasks
    belief: BeliefState


class GraphLayer(nn.Module):
    """Lightweight message-passing over the player graph."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, hidden)
        self.update = nn.GRUCell(hidden, hidden)

    def forward(self, nodes: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # nodes: (B, N, D), adjacency: (B, N, N), mask: (B, N)
        deg = adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)
        messages = adjacency @ nodes / deg
        transformed = self.linear(messages)
        B, N, D = nodes.shape
        nodes_flat = nodes.reshape(B * N, D)
        transformed_flat = transformed.reshape(B * N, D)
        updated = self.update(transformed_flat, nodes_flat)
        updated = updated.reshape(B, N, D)
        return nodes + updated * mask.unsqueeze(-1)


class StateBuilder(nn.Module):
    """Implements the specification of the shared state S_t."""

    def __init__(self, dims: ModelDims) -> None:
        super().__init__()
        hidden = dims.hidden_size
        self.num_players = dims.num_players
        self.hidden = hidden

        self.player_fuser = nn.Sequential(
            nn.Linear(dims.belief_hidden_size + dims.hidden_size // 2 + 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.convo_fuser = nn.Sequential(
            nn.Linear(dims.convo_hidden_size + dims.hidden_size // 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.cross_attn = nn.MultiheadAttention(hidden, num_heads=dims.gnn_num_heads, batch_first=True)
        self.graph_layers = nn.ModuleList(GraphLayer(hidden) for _ in range(dims.gnn_num_layers))
        self.memory_gru = nn.GRUCell(hidden * 3, hidden)

        self.player_side_encoder = nn.Sequential(
            nn.Linear(ObservationFeatureDim.player, dims.hidden_size // 2),
            nn.ReLU(),
        )
        self.convo_side_encoder = nn.Sequential(
            nn.Linear(ObservationFeatureDim.global_features + 3, dims.hidden_size // 2),
            nn.ReLU(),
        )

    def forward(
        self,
        features: ObservationFeatures,
        belief: BeliefState,
        prev_memory: Optional[torch.Tensor] = None,
    ) -> SharedState:
        batch = features.player_features.shape[0]
        device = features.player_features.device
        if prev_memory is None:
            prev_memory = torch.zeros(batch, self.hidden, device=device)

        player_side = self.player_side_encoder(features.player_features)
        player_stack = torch.cat(
            [
                belief.player_embeddings,
                player_side,
                belief.suspicion.unsqueeze(-1),
                belief.trust.unsqueeze(-1),
            ],
            dim=-1,
        )
        player_embed = self.player_fuser(player_stack)

        convo_side = self.convo_side_encoder(torch.cat([features.global_features, features.phase_encoding], dim=-1))
        convo_embed = self.convo_fuser(torch.cat([features.conversation_embedding, convo_side], dim=-1))

        # Cross-attention to inject conversation context into player embeddings
        convo_query = convo_embed.unsqueeze(1)
        attn_out, _ = self.cross_attn(convo_query, player_embed, player_embed)
        convo_embed = (convo_embed + attn_out.squeeze(1)) / 2.0
        player_embed = player_embed + attn_out

        adjacency = self._build_adjacency(features)
        mask = features.alive_mask
        for layer in self.graph_layers:
            player_embed = layer(player_embed, adjacency, mask)

        mean_pool = (player_embed * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        sum_pool = (player_embed * mask.unsqueeze(-1)).sum(dim=1)
        memory_input = torch.cat([mean_pool, convo_embed, sum_pool], dim=-1)
        memory_state = self.memory_gru(memory_input, prev_memory)

        masks = self._build_masks(features)
        return SharedState(
            player_embeddings=player_embed,
            convo_embedding=convo_embed,
            memory_state=memory_state,
            masks=masks,
            belief=belief,
        )

    def _build_adjacency(self, features: ObservationFeatures) -> torch.Tensor:
        vote = features.vote_matrix
        vote_t = vote.transpose(1, 2)
        mention = features.mention_scores.unsqueeze(2) * features.mention_scores.unsqueeze(1)
        adjacency = vote + vote_t + mention
        adjacency = adjacency / adjacency.max(dim=-1, keepdim=True).values.clamp(min=1.0)
        return adjacency

    def _build_masks(self, features: ObservationFeatures) -> StepMasks:
        alive = features.alive_mask
        mafia = features.mafia_mask
        batch = alive.shape[0]
        self_indices = features.self_index
        self_one_hot = F.one_hot(self_indices, num_classes=self.num_players).to(alive.dtype)
        not_self = 1.0 - self_one_hot
        vote_valid = alive * not_self

        night_valid = torch.zeros_like(alive)
        role_indices = features.role_index.tolist()
        for b, role_idx in enumerate(role_indices):
            if role_idx == 0:
                continue  # Villager has no night action
            if role_idx == 1:
                night_valid[b] = alive[b] * not_self[b] * (1.0 - mafia[b])
            elif role_idx == 2:
                night_valid[b] = alive[b]
            elif role_idx == 3:
                night_valid[b] = alive[b] * not_self[b]

        talk_valid = alive
        night_valid = night_valid.clamp(min=0.0, max=1.0)
        vote_valid = vote_valid.clamp(min=0.0, max=1.0)

        return StepMasks(
            alive_mask=alive,
            mafia_mask=mafia,
            self_index=self_indices,
            night_valid=night_valid,
            vote_valid=vote_valid,
            talk_target_valid=talk_valid,
        )


class ObservationFeatureDim:
    player = 14
    global_features = 5


__all__ = ["SharedState", "StateBuilder"]
