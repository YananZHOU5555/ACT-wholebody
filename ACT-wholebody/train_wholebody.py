#!/usr/bin/env python

"""
Training script for ACT-wholebody policy

Usage:
    # Full mode (torque + base)
    python train_wholebody.py --use_torque --mix

    # Torque only
    python train_wholebody.py --use_torque

    # Base only
    python train_wholebody.py --mix

    # Baseline (neither)
    python train_wholebody.py
"""

import sys
import os
from pathlib import Path
import argparse

# Add ACT-wholebody to path
sys.path.insert(0, str(Path(__file__).parent))

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent / "lerobot/src"))

from ACT_wholebody.configuration_act_wholebody import ACTWholeBodyConfig
from ACT_wholebody.modeling_act_wholebody import ACTWholeBodyPolicy

# Import LeRobot training infrastructure
from lerobot.scripts.train import train as lerobot_train
from lerobot.configs import TrainingConfig
import hydra
from omegaconf import DictConfig, OmegaConf


def create_wholebody_config(
    dataset_repo_id: str,
    dataset_root: Path,
    output_dir: Path,
    use_torque: bool = True,
    mix: bool = False,
    batch_size: int = 32,
    steps: int = 80000,
    log_freq: int = 100,
    eval_freq: int = 5000,
    save_freq: int = 5000,
    wandb_enable: bool = True,
):
    """Create configuration for ACT-wholebody training."""

    # Base config
    cfg = {
        "policy": {
            "_target_": "ACT_wholebody.modeling_act_wholebody.ACTWholeBodyPolicy",
            "config": {
                "_target_": "ACT_wholebody.configuration_act_wholebody.ACTWholeBodyConfig",
                # Wholebody-specific
                "use_torque": use_torque,
                "mix": mix,
                "state_dim": 17,
                # Architecture
                "vision_backbone": "resnet18",
                "pretrained_backbone_weights": "ResNet18_Weights.IMAGENET1K_V1",
                "dim_model": 512,
                "n_heads": 8,
                "dim_feedforward": 3200,
                "n_encoder_layers": 4,
                "n_decoder_layers": 1,
                "n_vae_encoder_layers": 4,
                "latent_dim": 64,
                "chunk_size": 100,
                "n_action_steps": 100,
                "use_vae": True,
                "dropout": 0.1,
                "kl_weight": 10.0,
                # Optimizer
                "optimizer_lr": 1e-5,
                "optimizer_weight_decay": 1e-4,
            }
        },
        "dataset": {
            "repo_id": dataset_repo_id,
            "root": str(dataset_root),
            "video_backend": "pyav",
        },
        "training": {
            "device": "cuda",
            "batch_size": batch_size,
            "num_workers": 4,
            "steps": steps,
            "log_freq": log_freq,
            "eval_freq": eval_freq,
            "save_freq": save_freq,
            "save_checkpoint": True,
            "num_gpus": 1,
        },
        "output_dir": str(output_dir),
        "job_name": f"wholebody_torque{use_torque}_mix{mix}",
        "wandb": {
            "enable": wandb_enable,
            "project": "act-wholebody",
            "name": f"torque{use_torque}_mix{mix}",
        }
    }

    return OmegaConf.create(cfg)


def main():
    parser = argparse.ArgumentParser(description="Train ACT-wholebody policy")

    # Dataset
    parser.add_argument("--dataset_repo_id", type=str, default="ACT-100-wholebody-v17",
                        help="Dataset repository ID")
    parser.add_argument("--dataset_root", type=str,
                        default="/home/zeno/piper_ros/data_collect/ACT-100-wholebody-v17",
                        help="Dataset root directory")

    # Wholebody-specific parameters
    parser.add_argument("--use_torque", action="store_true",
                        help="Use torque information (if False, torque is padded with zeros)")
    parser.add_argument("--mix", action="store_true",
                        help="Use mobile base velocity (if False, base dims are zeros)")

    # Training parameters
    parser.add_argument("--output_dir", type=str,
                        default="/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/ACT-wholebody",
                        help="Output directory for checkpoints")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--steps", type=int, default=80000,
                        help="Number of training steps")
    parser.add_argument("--log_freq", type=int, default=100,
                        help="Logging frequency")
    parser.add_argument("--eval_freq", type=int, default=5000,
                        help="Evaluation frequency")
    parser.add_argument("--save_freq", type=int, default=5000,
                        help="Checkpoint saving frequency")
    parser.add_argument("--wandb", action="store_true", default=True,
                        help="Enable Weights & Biases logging")
    parser.add_argument("--no_wandb", action="store_false", dest="wandb",
                        help="Disable Weights & Biases logging")

    args = parser.parse_args()

    # Print configuration
    print("\n" + "=" * 60)
    print("ACT-wholebody Training Configuration")
    print("=" * 60)
    print(f"Dataset: {args.dataset_repo_id}")
    print(f"  Path: {args.dataset_root}")
    print(f"\nWholebody Parameters:")
    print(f"  use_torque: {args.use_torque}")
    print(f"  mix (base velocity): {args.mix}")
    print(f"\nTraining:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Steps: {args.steps}")
    print(f"  Output: {args.output_dir}")
    print(f"  W&B logging: {args.wandb}")
    print("=" * 60 + "\n")

    # Create output directory
    output_dir = Path(args.output_dir) / f"torque{args.use_torque}_mix{args.mix}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create config
    cfg = create_wholebody_config(
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=Path(args.dataset_root),
        output_dir=output_dir,
        use_torque=args.use_torque,
        mix=args.mix,
        batch_size=args.batch_size,
        steps=args.steps,
        log_freq=args.log_freq,
        eval_freq=args.eval_freq,
        save_freq=args.save_freq,
        wandb_enable=args.wandb,
    )

    # Save config for reference
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        OmegaConf.save(cfg, f)
    print(f"Configuration saved to: {config_path}\n")

    # Train
    print("Starting training...\n")

    # TODO: Integrate with LeRobot's training loop
    # For now, we provide a simplified version

    # Initialize model
    policy_config = ACTWholeBodyConfig(
        use_torque=args.use_torque,
        mix=args.mix,
        state_dim=17,
    )

    policy = ACTWholeBodyPolicy(policy_config)

    print(f"\n{'=' * 60}")
    print("Model initialized successfully!")
    print(f"Parameters: {sum(p.numel() for p in policy.parameters() if p.requires_grad) / 1e6:.2f}M")
    print(f"{'=' * 60}\n")

    # TODO: Add actual training loop using LeRobot's trainer
    print("Training implementation TODO: Integrate with LeRobot trainer")
    print("For now, model is initialized and config is saved.")

    return policy


if __name__ == "__main__":
    main()
