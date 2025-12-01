"""Configuration dataclasses for deceptionNet MindGames agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional


@dataclass
class ModelDims:
    """Model dimensionalities and key sizes."""

    num_players: int = 6
    num_roles: int = 4
    hidden_size: int = 256
    belief_hidden_size: int = 256
    convo_hidden_size: int = 256
    num_talk_intents: int = 12
    gnn_num_layers: int = 2
    gnn_num_heads: int = 4


@dataclass
class OptimConfig:
    """Optimization hyperparameters shared across training stages."""

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.999)
    gradient_clip_norm: float = 1.0


@dataclass
class ILConfig:
    """Configuration for imitation-learning training."""

    batch_size: int = 16
    unroll_length: int = 32
    role_loss_weight: float = 1.0
    suspicion_loss_weight: float = 0.5
    trust_loss_weight: float = 0.5
    intent_loss_weight: float = 1.0
    slot_loss_weight: float = 1.0
    night_loss_weight: float = 1.0
    vote_loss_weight: float = 1.0
    aux_calibration_weight: float = 0.1
    checkpoint_interval: int = 1000


@dataclass
class PPOConfig:
    """Configuration for PPO fine-tuning."""

    rollout_length: int = 256
    num_minibatches: int = 8
    num_epochs: int = 4
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    il_aux_coef: float = 0.1
    gamma: float = 0.99
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5


@dataclass
class TrainingConfig:
    """Aggregated configuration for IL and RL stages."""

    dims: ModelDims = field(default_factory=ModelDims)
    optim: OptimConfig = field(default_factory=OptimConfig)
    il: ILConfig = field(default_factory=ILConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    device: str = "cuda"
    mixed_precision: bool = True
    checkpoint_dir: str = "checkpoints"
    log_every: int = 50


@dataclass
class PresenterTemplates:
    """Templates to drive deterministic presenter outputs."""

    accuse_templates: List[str] = field(
        default_factory=lambda: [
            "I think {target} is mafia; their votes and timing don't add up.",
            "{target} feels off to me. Their arguments aren't consistent with village logic.",
        ]
    )
    defend_self_templates: List[str] = field(
        default_factory=lambda: [
            "I'm not mafia. Check my vote history-it lines up with catching scum.",
            "You're barking up the wrong tree. My reasoning has backed village interests all game.",
        ]
    )
    defend_other_templates: List[str] = field(
        default_factory=lambda: [
            "I don't buy the case on {target}; the push feels opportunistic.",
            "{target} has been solving. Let's not mis-eliminate them without stronger evidence.",
        ]
    )
    ask_templates: List[str] = field(
        default_factory=lambda: [
            "{target}, can you walk us through why you switched your vote?",
            "I'd like {target} to explain that last-minute decision-what changed?",
        ]
    )
    claim_templates: List[str] = field(
        default_factory=lambda: [
            "I'm the Detective. I checked {target} last night.",
            "Claiming Detective: {target} is the result of my investigation.",
        ]
    )
    filler_templates: List[str] = field(
        default_factory=lambda: [
            "I'll wait for more info before locking a vote.",
            "Let's gather more claims before rushing a verdict.",
        ]
    )


@dataclass
class InferenceConfig:
    """Configuration holder for runtime agent behaviour."""

    dims: ModelDims = field(default_factory=ModelDims)
    presenter: PresenterTemplates = field(default_factory=PresenterTemplates)
    max_tokens_talk: int = 80
    vote_token_fmt: str = "[{index}]"
    silence_token: str = "..."
    talk_temperature: float = 1.0
    talk_top_p: float = 0.9


DEFAULT_TRAINING_CONFIG = TrainingConfig()
DEFAULT_INFERENCE_CONFIG = InferenceConfig()

__all__ = [
    "ModelDims",
    "OptimConfig",
    "ILConfig",
    "PPOConfig",
    "TrainingConfig",
    "PresenterTemplates",
    "InferenceConfig",
    "DEFAULT_TRAINING_CONFIG",
    "DEFAULT_INFERENCE_CONFIG",
]
