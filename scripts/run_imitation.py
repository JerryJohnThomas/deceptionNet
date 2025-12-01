#!/usr/bin/env python
"""Run Stage-1 imitation learning with the simple vote-supervised model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from deceptionnet.models.simple import SimpleImitationModel
from deceptionnet.training.imitation import ImitationConfig, run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run imitation training for DeceptionNet")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "games.jsonl",
        help="Path to normalized JSONL dataset",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Training device (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per step")
    parser.add_argument("--max-epochs", type=int, default=5, help="Number of epochs to run")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--embedding-dim", type=int, default=256, help="Model embedding width")
    parser.add_argument("--belief-hidden-dim", type=int, default=256, help="Belief GRU hidden size")
    parser.add_argument("--num-players", type=int, default=12, help="Maximum player count for action heads")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "imitation",
        help="Directory for TensorBoard logs (set to 'none' to disable)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_dir = None if str(args.log_dir).lower() == "none" else args.log_dir
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)

    cfg = ImitationConfig(
        batch_size=args.batch_size,
        num_workers=0,
        device=args.device,
        learning_rate=args.learning_rate,
        max_epochs=args.max_epochs,
        embedding_dim=args.embedding_dim,
        belief_hidden_dim=args.belief_hidden_dim,
        num_players=args.num_players,
        talk_action_dim=args.num_players,
        vote_action_dim=args.num_players,
        night_action_dim=args.num_players,
        log_dir=log_dir,
    )

    model = SimpleImitationModel(cfg.agent_dimensions())
    run_training(args.data, cfg, model)


if __name__ == "__main__":
    main()
