"""Imitation learning training loop for the MindGames agent."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import csv

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

from deceptionNet.config import TrainingConfig

from ..agents.featurizer import ObservationFeatures, ObservationFeaturizer
from ..agents.listener import SimpleListener
from ..agents.policy import MultiHeadPolicy


@dataclass
class ImitationBatch:
    """Container expected by the imitation learner."""

    features: ObservationFeatures
    role_labels: torch.Tensor  # (B, N)
    suspicion_targets: torch.Tensor  # (B, N)
    trust_targets: torch.Tensor  # (B, N)
    night_targets: torch.Tensor  # (B,)
    vote_targets: torch.Tensor  # (B,)
    talk_intents: torch.Tensor  # (B,)
    talk_targets: torch.Tensor  # (B,)


class ImitationLearner:
    """Supervised training harness for the IL stage."""

    def __init__(self, policy: MultiHeadPolicy, config: TrainingConfig) -> None:
        self.policy = policy
        self.config = config
        self.device = torch.device(config.device)
        self.policy.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.optim.learning_rate,
            betas=config.optim.betas,
            weight_decay=config.optim.weight_decay,
        )
        self.scaler = GradScaler(enabled=config.mixed_precision and self.device.type == "cuda") 
        # Mixed Precision Training
        # if cuda float16 multiplication are much faster. Supported by TPU and GPUs which has Volta architecture (2017) and newer. Not gtx1650ti actually
        # does not help on cpu

        # so they scale up the loss and compute gradients by this scaler
        # then before updating the weights they again scale down the gradients by this scalre
        # autocast does the conversions

        ## More formal definition of Mixed Precision Training
        # During forward pass with autocast():
        # Weights copied to float16 temporarily for computation
        # Gradients computed in float16 (scaled up)
        # Gradients converted back to float32 for optimizer
        # Weights updated in float32
        # This is called mixed precision - you're mixing float16 (computation) and float32 (storage

        # Historical Context  Before 2018: Everyone trained in float32, Slow, memory-hungry
        # 2018-2020: # NVIDIA introduced Tensor Cores. Mixed precision became feasible. # Papers showed it works: "Mixed Precision Training" (Micikevicius et al., 2018)
        # 2020-Present: Default practice in industry. PyTorch/TensorFlow have built-in support. Required for competitive training speeds

        self.loss_history: List[float] = []

    def train_epoch(self, dataloader: Iterable[ImitationBatch], epoch: int = 0) -> Dict[str, float]:
        self.policy.train()
        metrics_accumulator: Dict[str, float] = {}
        steps = 0
        for batch in dataloader:
            steps += 1
            losses = self._train_step(batch)
            for key, value in losses.items():
                metrics_accumulator[key] = metrics_accumulator.get(key, 0.0) + value
        if steps == 0:
            return {"loss": 0.0}
        return {k: v / steps for k, v in metrics_accumulator.items()}

    def _train_step(self, batch: ImitationBatch) -> Dict[str, float]:
        # batch is the data from the dataloader for one single batch
        features = self._to_device(batch.features)
        role_labels = batch.role_labels.to(self.device)
        suspicion_targets = batch.suspicion_targets.to(self.device)
        trust_targets = batch.trust_targets.to(self.device)
        night_targets = batch.night_targets.to(self.device)
        vote_targets = batch.vote_targets.to(self.device)
        talk_intents = batch.talk_intents.to(self.device)
        talk_targets = batch.talk_targets.to(self.device)

        self.optimizer.zero_grad(set_to_none=True)      # reseting gradients
        belief_state, memory_state = self.policy.initial_state(features.player_features.shape[0], self.device)
        
        with autocast(enabled=self.scaler.is_enabled()):            
            shared, outputs = self.policy(features, belief_state, memory_state) # Forward pass with Mixed Precision Training done via autocast
            loss_dict = self._compute_losses(
                shared,
                outputs,
                role_labels,
                suspicion_targets,
                trust_targets,
                night_targets,
                vote_targets,
                talk_intents,
                talk_targets,
            )
            total_loss = loss_dict["total"]
        self.loss_history.append(float(total_loss.detach().cpu().item()))
        self.scaler.scale(total_loss).backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.optim.gradient_clip_norm)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}

    def _compute_losses(
        self,
        shared,
        outputs,
        role_labels,
        suspicion_targets,
        trust_targets,
        night_targets,
        vote_targets,
        talk_intents,
        talk_targets,
    ) -> Dict[str, torch.Tensor]:
        config = self.config.il
        device = role_labels.device
        losses: Dict[str, torch.Tensor] = {}

        belief = shared.belief
        alive_mask = shared.masks.alive_mask

        # Role loss
        role_logits = belief.role_logits.view(-1, belief.role_logits.shape[-1])
        role_labels_flat = role_labels.view(-1)
        valid_roles = role_labels_flat >= 0
        if valid_roles.any():
            role_loss = F.cross_entropy(role_logits[valid_roles], role_labels_flat[valid_roles])
        else:
            role_loss = torch.tensor(0.0, device=device)
        losses["role_loss"] = role_loss * config.role_loss_weight

        # Suspicion and trust BCE
        suspicion_loss = self._weighted_bce(belief.suspicion, suspicion_targets, alive_mask)
        trust_loss = self._weighted_bce(belief.trust, trust_targets, alive_mask)
        losses["suspicion_loss"] = suspicion_loss * config.suspicion_loss_weight
        losses["trust_loss"] = trust_loss * config.trust_loss_weight

        # Calibration auxiliary (Brier score)
        calibration_loss = self._weighted_brier(belief.suspicion, suspicion_targets, alive_mask)
        losses["calibration_loss"] = calibration_loss * config.aux_calibration_weight

        # Policy heads
        night_mask = shared.masks.night_valid
        night_loss = self._masked_ce(outputs.night_logits, night_targets, night_mask.sum(dim=-1) > 0)
        vote_loss = self._masked_ce(outputs.vote_logits, vote_targets, shared.masks.vote_valid.sum(dim=-1) > 0)
        intent_loss = F.cross_entropy(outputs.talk_intent_logits, talk_intents)

        slot_needed = talk_targets >= 0
        if slot_needed.any():
            slot_loss = F.cross_entropy(outputs.talk_slot_logits[slot_needed], talk_targets[slot_needed])
        else:
            slot_loss = torch.tensor(0.0, device=device)

        losses["night_loss"] = night_loss * config.night_loss_weight
        losses["vote_loss"] = vote_loss * config.vote_loss_weight
        losses["intent_loss"] = intent_loss * config.intent_loss_weight
        losses["slot_loss"] = slot_loss * config.slot_loss_weight

        total = sum(losses.values())
        losses["total"] = total
        return losses

    def _masked_ce(self, logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if logits.dim() == 2:
            valid_mask = valid_mask.to(logits.device)
            target_mask = targets.to(logits.device) >= 0
            combined = valid_mask & target_mask
            if combined.any():
                loss = F.cross_entropy(logits[combined], targets[combined])
            else:
                loss = torch.tensor(0.0, device=logits.device)
            return loss
        raise ValueError("Unexpected logits shape for masked CE")

    def _weighted_bce(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask
        loss = F.binary_cross_entropy(preds, targets, reduction="none")
        loss = (loss * weight).sum() / weight.sum().clamp(min=1.0)
        return loss

    def _weighted_brier(self, preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        loss = (preds - targets) ** 2
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

    def _to_device(self, features: ObservationFeatures) -> ObservationFeatures:
        return ObservationFeatures(
            player_features=features.player_features.to(self.device),
            global_features=features.global_features.to(self.device),
            conversation_embedding=features.conversation_embedding.to(self.device),
            alive_mask=features.alive_mask.to(self.device),
            mafia_mask=features.mafia_mask.to(self.device),
            self_index=features.self_index.to(self.device),
            role_index=features.role_index.to(self.device),
            phase_encoding=features.phase_encoding.to(self.device),
            vote_matrix=features.vote_matrix.to(self.device),
            mention_scores=features.mention_scores.to(self.device),
            sentiment_scores=features.sentiment_scores.to(self.device),
            support_scores=features.support_scores.to(self.device),
            accusations=features.accusations.to(self.device),
            contradiction_scores=features.contradiction_scores.to(self.device),
            bandwagon_scores=features.bandwagon_scores.to(self.device),
            round_index=features.round_index.to(self.device),
        )


def _pad(values: List[float], target_len: int, fill: float = 0.0) -> List[float]:
    sequence = list(values)
    if len(sequence) < target_len:
        sequence.extend([fill] * (target_len - len(sequence)))
    return sequence[:target_len]


def load_jsonl_dataset(path: Path, dims) -> List[ImitationBatch]:
    listener = SimpleListener(dims)
    featurizer = ObservationFeaturizer(dims)
    dataset: List[ImitationBatch] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            observation = record.get("obs") or record.get("observation") or {}
            listener_output = listener(observation)
            features = featurizer(observation, listener_output)

            role_labels = torch.tensor(
                _pad(record.get("role_labels", [-1] * dims.num_players), dims.num_players, fill=-1),
                dtype=torch.long,
            ).unsqueeze(0)
            suspicion = torch.tensor(
                _pad(record.get("suspicion", [0.0] * dims.num_players), dims.num_players),
                dtype=torch.float32,
            ).unsqueeze(0)
            trust = torch.tensor(
                _pad(record.get("trust", [0.0] * dims.num_players), dims.num_players),
                dtype=torch.float32,
            ).unsqueeze(0)
            night_target = torch.tensor([record.get("night_target", -1)], dtype=torch.long)
            vote_target = torch.tensor([record.get("vote_target", 0)], dtype=torch.long)
            talk_intent = torch.tensor([record.get("talk_intent", 6)], dtype=torch.long)
            talk_target = torch.tensor([record.get("talk_target", -1)], dtype=torch.long)

            dataset.append(
                ImitationBatch(
                    features=features,
                    role_labels=role_labels,
                    suspicion_targets=suspicion,
                    trust_targets=trust,
                    night_targets=night_target,
                    vote_targets=vote_target,
                    talk_intents=talk_intent,
                    talk_targets=talk_target,
                )
            )
    return dataset


def _stack_features(batch: List[ObservationFeatures]) -> ObservationFeatures:
    return ObservationFeatures(
        player_features=torch.cat([item.player_features for item in batch], dim=0),
        global_features=torch.cat([item.global_features for item in batch], dim=0),
        conversation_embedding=torch.cat([item.conversation_embedding for item in batch], dim=0),
        alive_mask=torch.cat([item.alive_mask for item in batch], dim=0),
        mafia_mask=torch.cat([item.mafia_mask for item in batch], dim=0),
        self_index=torch.cat([item.self_index for item in batch], dim=0),
        role_index=torch.cat([item.role_index for item in batch], dim=0),
        phase_encoding=torch.cat([item.phase_encoding for item in batch], dim=0),
        vote_matrix=torch.cat([item.vote_matrix for item in batch], dim=0),
        mention_scores=torch.cat([item.mention_scores for item in batch], dim=0),
        sentiment_scores=torch.cat([item.sentiment_scores for item in batch], dim=0),
        support_scores=torch.cat([item.support_scores for item in batch], dim=0),
        accusations=torch.cat([item.accusations for item in batch], dim=0),
        contradiction_scores=torch.cat([item.contradiction_scores for item in batch], dim=0),
        bandwagon_scores=torch.cat([item.bandwagon_scores for item in batch], dim=0),
        round_index=torch.cat([item.round_index for item in batch], dim=0),
    )


def collate_imitation(samples: List[ImitationBatch]) -> ImitationBatch:
    features = _stack_features([sample.features for sample in samples])
    return ImitationBatch(
        features=features,
        role_labels=torch.cat([sample.role_labels for sample in samples], dim=0),
        suspicion_targets=torch.cat([sample.suspicion_targets for sample in samples], dim=0),
        trust_targets=torch.cat([sample.trust_targets for sample in samples], dim=0),
        night_targets=torch.cat([sample.night_targets for sample in samples], dim=0),
        vote_targets=torch.cat([sample.vote_targets for sample in samples], dim=0),
        talk_intents=torch.cat([sample.talk_intents for sample in samples], dim=0),
        talk_targets=torch.cat([sample.talk_targets for sample in samples], dim=0),
    )


def build_dataloader(dataset: List[ImitationBatch], batch_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_imitation)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run imitation-learning fine-tuning")
    parser.add_argument("--data", required=True, help="Path to JSONL dataset of IL examples")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train")
    parser.add_argument("--batch-size", type=int, default=4, help="Mini-batch size")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cpu or cuda)")
    parser.add_argument("--output", type=str, default="weights.pt", help="Where to save model weights")
    args = parser.parse_args(argv)

    config = TrainingConfig()
    if args.device:
        config.device = args.device
    config.il.batch_size = args.batch_size

    dataset_path = Path(args.data)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    dataset = load_jsonl_dataset(dataset_path, config.dims)
    if not dataset:
        raise RuntimeError("Dataset is empty; provide at least one example")

    dataloader = build_dataloader(dataset, batch_size=args.batch_size)
    policy = MultiHeadPolicy(config.dims)
    learner = ImitationLearner(policy, config)

    for epoch in range(args.epochs):
        metrics = learner.train_epoch(dataloader, epoch)
        metrics_str = " ".join(f"{key}={value:.4f}" for key, value in sorted(metrics.items()))
        print(f"epoch={epoch + 1} {metrics_str}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), output_path)
    print(f"Saved weights to {output_path}")

    loss_history = learner.loss_history
    if loss_history:
        plot_dir = Path("deceptionNet")
        plot_dir.mkdir(parents=True, exist_ok=True)

        curve_path = plot_dir / "il_loss_curve.png"
        epoch_path = None
        if np is not None and plt is not None:
            steps_per_epoch = max(1, len(dataloader))
            steps = np.arange(len(loss_history))
            plt.figure(figsize=(8, 4))
            plt.plot(steps, loss_history, label="total_loss", color="orange")
            plt.title("IL Training Loss Curve")
            plt.xlabel("Training Step")
            plt.ylabel("Loss")
            plt.legend()
            plt.tight_layout()
            plt.savefig(curve_path)
            plt.close()

            epoch_means = [float(np.mean(loss_history[i : i + steps_per_epoch]))
                           for i in range(0, len(loss_history), steps_per_epoch)
                           if loss_history[i : i + steps_per_epoch]]
            if epoch_means:
                epoch_path = plot_dir / "il_loss_epoch.png"
                plt.figure(figsize=(6, 3))
                plt.plot(range(1, len(epoch_means) + 1), epoch_means, marker="o")
                plt.title("Mean Loss per Epoch")
                plt.xlabel("Epoch")
                plt.ylabel("Mean Loss")
                plt.tight_layout()
                plt.savefig(epoch_path)
                plt.close()
        else:
            print("matplotlib or numpy not available; skipping loss plots.")

        csv_path = plot_dir / "il_loss_log.csv"
        with csv_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "loss"])
            for idx, loss in enumerate(loss_history):
                writer.writerow([idx, loss])
        artifact_msg = f"Saved loss artifacts to {csv_path}"
        if np is not None and plt is not None:
            artifact_msg = f"Saved loss artifacts to {curve_path}"
            if epoch_path is not None:
                artifact_msg += f", {epoch_path}"
            artifact_msg += f" and {csv_path}"
        print(artifact_msg)


if __name__ == "__main__":
    main()


__all__ = [
    "ImitationLearner",
    "ImitationBatch",
    "load_jsonl_dataset",
    "collate_imitation",
    "build_dataloader",
    "main",
]
