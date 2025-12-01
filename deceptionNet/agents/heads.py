"""Policy and value heads operating on the shared state."""

from __future__ import annotations

import torch
import torch.nn as nn

from deceptionNet.config import ModelDims

from .state_builder import SharedState

Tensor = torch.Tensor


class NightPolicyHead(nn.Module):
    """Maskable discrete head used for night actions."""

    def __init__(self, dims: ModelDims) -> None:
        super().__init__()
        hidden = dims.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, state: SharedState) -> Tensor:
        players = state.player_embeddings
        memory = state.memory_state.unsqueeze(1).expand_as(players)
        stacked = torch.cat([players, memory], dim=-1)
        scores = self.out(self.mlp(stacked)).squeeze(-1)
        return scores


class VotePolicyHead(nn.Module):
    """Maskable head for day vote selection."""

    def __init__(self, dims: ModelDims) -> None:
        super().__init__()
        hidden = dims.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, state: SharedState) -> Tensor:
        players = state.player_embeddings
        convo = state.convo_embedding.unsqueeze(1).expand_as(players)
        stacked = torch.cat([players, convo], dim=-1)
        logits = self.out(self.mlp(stacked)).squeeze(-1)
        return logits


class TalkIntentHead(nn.Module):
    """Discrete intent classifier for conversation actions."""

    def __init__(self, dims: ModelDims) -> None:
        super().__init__()
        hidden = dims.hidden_size
        self.num_intents = dims.num_talk_intents
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.out = nn.Linear(hidden, self.num_intents)

    def forward(self, state: SharedState) -> Tensor:
        stacked = torch.cat([state.convo_embedding, state.memory_state], dim=-1)
        logits = self.out(self.mlp(stacked))
        return logits


class TalkSlotHead(nn.Module):
    """Slot selector used when intents target another player."""

    def __init__(self, dims: ModelDims) -> None:
        super().__init__()
        hidden = dims.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
        )
        self.out = nn.Linear(hidden, 1)

    def forward(self, state: SharedState) -> Tensor:
        players = state.player_embeddings
        convo = state.convo_embedding.unsqueeze(1).expand_as(players)
        stacked = torch.cat([players, convo], dim=-1)
        return self.out(self.mlp(stacked)).squeeze(-1)


class ValueHead(nn.Module):
    """State-value estimator for PPO."""

    def __init__(self, dims: ModelDims) -> None:
        super().__init__()
        hidden = dims.hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.ReLU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.out = nn.Linear(hidden // 2, 1)

    def forward(self, state: SharedState) -> Tensor:
        mean_pool = state.player_embeddings.mean(dim=1)
        summed = state.player_embeddings.sum(dim=1)
        stacked = torch.cat([state.memory_state, state.convo_embedding, mean_pool + summed], dim=-1)
        return self.out(self.mlp(stacked)).squeeze(-1)


__all__ = [
    "NightPolicyHead",
    "VotePolicyHead",
    "TalkIntentHead",
    "TalkSlotHead",
    "ValueHead",
]
