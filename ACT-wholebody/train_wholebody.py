#!/usr/bin/env python
"""
ACT-wholebody Training Script
Complete training loop with LeRobotDataset support.
"""

import sys
import os
from pathlib import Path
import argparse
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np

# Add paths
sys.path.insert(0, str(Path(__file__).parent))

from lerobot.datasets.lerobot_dataset import LeRobotDataset


class ACTWholeBodyConfig:
    """Configuration for ACT-wholebody model."""
    def __init__(
        self,
        use_torque: bool = True,
        mix: bool = False,
        state_dim: int = 17,
        action_dim: int = 17,
        chunk_size: int = 100,
        n_obs_steps: int = 1,
        # Architecture
        vision_backbone: str = "resnet18",
        dim_model: int = 512,
        n_heads: int = 8,
        dim_feedforward: int = 3200,
        n_encoder_layers: int = 4,
        n_decoder_layers: int = 1,
        n_vae_encoder_layers: int = 4,
        dropout: float = 0.1,
        # VAE
        use_vae: bool = True,
        latent_dim: int = 32,
        kl_weight: float = 10.0,
        # Optimizer
        lr: float = 1e-5,
        weight_decay: float = 1e-4,
    ):
        self.use_torque = use_torque
        self.mix = mix
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.n_obs_steps = n_obs_steps
        self.vision_backbone = vision_backbone
        self.dim_model = dim_model
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.n_encoder_layers = n_encoder_layers
        self.n_decoder_layers = n_decoder_layers
        self.n_vae_encoder_layers = n_vae_encoder_layers
        self.dropout = dropout
        self.use_vae = use_vae
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.lr = lr
        self.weight_decay = weight_decay


def get_sinusoid_encoding_table(n_position: int, d_hid: int):
    """Generate sinusoidal positional encoding."""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ACTWholeBodyPolicy(nn.Module):
    """ACT-wholebody Policy with VAE."""

    def __init__(self, config: ACTWholeBodyConfig):
        super().__init__()
        self.config = config

        # Vision backbone
        from torchvision import models
        if config.vision_backbone == "resnet18":
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            backbone_out_dim = 512
        else:
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            backbone_out_dim = 2048
        self.backbone.fc = nn.Identity()

        # Freeze early layers
        for name, param in self.backbone.named_parameters():
            if 'layer4' not in name:
                param.requires_grad = False

        # Image projection
        self.img_proj = nn.Linear(backbone_out_dim, config.dim_model)

        # State/action projections
        self.state_proj = nn.Linear(config.state_dim, config.dim_model)
        self.torque_proj = nn.Linear(config.state_dim, config.dim_model)
        self.action_proj = nn.Linear(config.action_dim, config.dim_model)

        # VAE encoder
        if config.use_vae:
            self.vae_cls_token = nn.Parameter(torch.randn(1, 1, config.dim_model))
            vae_encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.dim_model,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
                batch_first=True,
            )
            self.vae_encoder = nn.TransformerEncoder(vae_encoder_layer, num_layers=config.n_vae_encoder_layers)
            self.latent_proj = nn.Linear(config.dim_model, config.latent_dim * 2)

        # Latent to embedding
        self.latent_out_proj = nn.Linear(config.latent_dim, config.dim_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_encoder_layers)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.dim_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_decoder_layers)

        # Query embeddings
        self.query_embed = nn.Parameter(torch.randn(1, config.chunk_size, config.dim_model))

        # Action head
        self.action_head = nn.Linear(config.dim_model, config.action_dim)

        # Positional encoding
        self.register_buffer('pos_table', get_sinusoid_encoding_table(config.chunk_size + 10, config.dim_model))

    def encode_images(self, images):
        """Encode images using backbone."""
        # images: list of (B, C, H, W) or (B, num_cams, C, H, W)
        batch_size = images[0].shape[0]
        all_features = []

        for img in images:
            if img.dim() == 5:  # (B, num_cams, C, H, W)
                B, N, C, H, W = img.shape
                img = img.view(B * N, C, H, W)
                feat = self.backbone(img)  # (B*N, feat_dim)
                feat = feat.view(B, N, -1)  # (B, N, feat_dim)
            else:  # (B, C, H, W)
                feat = self.backbone(img).unsqueeze(1)  # (B, 1, feat_dim)
            all_features.append(feat)

        features = torch.cat(all_features, dim=1)  # (B, total_cams, feat_dim)
        features = self.img_proj(features)  # (B, total_cams, dim_model)
        return features

    def forward(self, batch, actions=None):
        """
        Forward pass.

        Args:
            batch: dict with observation data
            actions: (B, chunk_size, action_dim) target actions for training

        Returns:
            pred_actions: (B, chunk_size, action_dim)
            mu, logvar: VAE parameters (None if not training)
        """
        device = next(self.parameters()).device

        # Get state (14D arm joints)
        state = batch['observation.state'].to(device)  # (B, 14)
        batch_size = state.shape[0]

        # Get base velocity (3D)
        base_vel = batch.get('observation.base_velocity', torch.zeros(batch_size, 3, device=device))
        if isinstance(base_vel, torch.Tensor):
            base_vel = base_vel.to(device)

        # Get torque/effort (14D)
        torque = batch.get('observation.effort', torch.zeros(batch_size, 14, device=device))
        if isinstance(torque, torch.Tensor):
            torque = torque.to(device)

        # Pad to 17D: [base_vel(3), arm_joints(14)]
        if self.config.mix:
            state_17 = torch.cat([base_vel, state], dim=-1)
        else:
            state_17 = torch.cat([torch.zeros(batch_size, 3, device=device), state], dim=-1)

        if self.config.use_torque:
            torque_17 = torch.cat([torch.zeros(batch_size, 3, device=device), torque], dim=-1)
        else:
            torque_17 = torch.zeros(batch_size, 17, device=device)

        # Encode images
        images = []
        for key in ['observation.images.main', 'observation.images.secondary_0',
                    'observation.images.secondary_1', 'observation.images.secondary_2']:
            if key in batch:
                img = batch[key].to(device)
                if img.dim() == 3:  # (C, H, W) -> (1, C, H, W)
                    img = img.unsqueeze(0)
                images.append(img)

        if images:
            img_features = self.encode_images(images)  # (B, num_cams, dim_model)
        else:
            img_features = torch.zeros(batch_size, 1, self.config.dim_model, device=device)

        # Project state and torque
        state_embed = self.state_proj(state_17).unsqueeze(1)  # (B, 1, dim_model)
        torque_embed = self.torque_proj(torque_17).unsqueeze(1)  # (B, 1, dim_model)

        # VAE encoding (training only)
        mu, logvar = None, None
        if self.config.use_vae and actions is not None:
            actions = actions.to(device)
            action_embed = self.action_proj(actions)  # (B, chunk_size, dim_model)

            # [CLS, state, torque, action_sequence]
            cls_token = self.vae_cls_token.expand(batch_size, -1, -1)
            vae_input = torch.cat([cls_token, state_embed, torque_embed, action_embed], dim=1)

            vae_output = self.vae_encoder(vae_input)
            cls_output = vae_output[:, 0]  # (B, dim_model)

            latent_params = self.latent_proj(cls_output)  # (B, latent_dim*2)
            mu = latent_params[:, :self.config.latent_dim]
            logvar = latent_params[:, self.config.latent_dim:]

            # Reparameterization
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            latent = mu + eps * std
        else:
            latent = torch.zeros(batch_size, self.config.latent_dim, device=device)

        latent_embed = self.latent_out_proj(latent).unsqueeze(1)  # (B, 1, dim_model)

        # Encoder input: [latent, state, torque, images]
        encoder_input = torch.cat([latent_embed, state_embed, torque_embed, img_features], dim=1)
        encoder_output = self.encoder(encoder_input)

        # Decoder
        queries = self.query_embed.expand(batch_size, -1, -1)
        decoder_output = self.decoder(queries, encoder_output)

        # Action prediction
        pred_actions = self.action_head(decoder_output)  # (B, chunk_size, action_dim)

        return pred_actions, (mu, logvar)


def compute_loss(pred_actions, target_actions, mu, logvar, kl_weight=10.0):
    """Compute L1 loss + KL divergence."""
    # L1 loss
    l1_loss = F.l1_loss(pred_actions, target_actions)

    # KL divergence
    if mu is not None and logvar is not None:
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.shape[0]
    else:
        kl_loss = torch.tensor(0.0, device=pred_actions.device)

    total_loss = l1_loss + kl_weight * kl_loss
    return total_loss, l1_loss, kl_loss


def collate_fn(batch):
    """Custom collate function for LeRobotDataset."""
    result = {}
    for key in batch[0].keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            result[key] = torch.stack(values)
        else:
            result[key] = values
    return result


def train(args):
    """Main training loop with H200 optimizations."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # H200 optimizations
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    # Create output directory
    exp_name = f"torque{args.use_torque}_mix{args.mix}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"\nLoading dataset from: {args.dataset_root}")
    dataset = LeRobotDataset(args.dataset_root)
    print(f"Dataset: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    # Create dataloader with optimizations
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
        persistent_workers=True if args.num_workers > 0 else False,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )

    # Create model
    config = ACTWholeBodyConfig(
        use_torque=args.use_torque,
        mix=args.mix,
        chunk_size=args.chunk_size,
        lr=args.lr,
        kl_weight=args.kl_weight,
    )

    model = ACTWholeBodyPolicy(config).to(device)

    # Compile model for speed (PyTorch 2.0+)
    if args.compile:
        print("Compiling model with torch.compile()...")
        model = torch.compile(model, mode="reduce-overhead")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params/1e6:.2f}M total, {trainable_params/1e6:.2f}M trainable")

    # Optimizer with fused AdamW (faster on CUDA)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        fused=True,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=1e-6)

    # Mixed precision with bfloat16 (better for H200)
    scaler = GradScaler('cuda', enabled=args.use_amp)

    # Save config
    config_dict = {
        'use_torque': args.use_torque,
        'mix': args.mix,
        'batch_size': args.batch_size,
        'steps': args.steps,
        'lr': args.lr,
        'chunk_size': args.chunk_size,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting training: {exp_name}")
    print(f"  use_torque: {args.use_torque}")
    print(f"  mix (base): {args.mix}")
    print(f"  batch_size: {args.batch_size}")
    print(f"  steps: {args.steps}")
    print(f"{'='*60}\n")

    model.train()
    global_step = 0
    start_time = time.time()
    data_iter = iter(dataloader)

    while global_step < args.steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Get actions (target)
        actions = batch['action']  # (B, action_dim)

        # Expand actions to chunk_size for training
        # In real scenario, you'd use action chunking from dataset
        target_actions = actions.unsqueeze(1).repeat(1, args.chunk_size, 1)  # (B, chunk_size, action_dim)

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.bfloat16, enabled=args.use_amp):
            pred_actions, (mu, logvar) = model(batch, target_actions)
            loss, l1_loss, kl_loss = compute_loss(
                pred_actions, target_actions.to(device),
                mu, logvar,
                kl_weight=args.kl_weight
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        global_step += 1

        # Logging
        if global_step % args.log_freq == 0:
            elapsed = time.time() - start_time
            steps_per_sec = global_step / elapsed
            eta_hours = (args.steps - global_step) / steps_per_sec / 3600

            print(f"[{global_step}/{args.steps}] loss: {loss.item():.4f} "
                  f"(l1: {l1_loss.item():.4f}, kl: {kl_loss.item():.4f}) "
                  f"lr: {scheduler.get_last_lr()[0]:.2e} "
                  f"ETA: {eta_hours:.1f}h")

        # Save checkpoint
        if global_step % args.save_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_{global_step}.pt"
            torch.save({
                'step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config_dict,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = output_dir / "model_final.pt"
    torch.save({
        'step': global_step,
        'model_state_dict': model.state_dict(),
        'config': config_dict,
    }, final_path)
    print(f"\nTraining complete! Final model saved to: {final_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train ACT-wholebody policy")

    # Dataset
    parser.add_argument("--dataset_repo_id", type=str, default="ACT-100-wholebody")
    parser.add_argument("--dataset_root", type=str,
                        default="/workspace/ACT-wholebody/ACT-wholebody/data/ACT-100-wholebody")

    # Model
    parser.add_argument("--use_torque", action="store_true", help="Use torque information")
    parser.add_argument("--mix", action="store_true", help="Use mobile base velocity")
    parser.add_argument("--chunk_size", type=int, default=100)

    # Training
    parser.add_argument("--output_dir", type=str,
                        default="/workspace/ACT-wholebody/ACT-wholebody/checkpoint")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=80000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--kl_weight", type=float, default=10.0)
    parser.add_argument("--num_workers", type=int, default=16)

    # Speed optimizations
    parser.add_argument("--compile", action="store_true", default=True, help="Use torch.compile()")
    parser.add_argument("--no_compile", action="store_false", dest="compile")
    parser.add_argument("--use_amp", action="store_true", default=True, help="Use mixed precision")
    parser.add_argument("--no_amp", action="store_false", dest="use_amp")

    # Logging
    parser.add_argument("--log_freq", type=int, default=100)
    parser.add_argument("--save_freq", type=int, default=5000)
    parser.add_argument("--eval_freq", type=int, default=5000)

    # W&B (placeholder)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--no_wandb", action="store_false", dest="wandb")

    args = parser.parse_args()

    train(args)


if __name__ == "__main__":
    main()
