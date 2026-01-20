#!/usr/bin/env python

"""
ACT-wholebody Policy Implementation
Based on act_bak/detr_vae.py with LeRobot interface adaptation

Key features:
- VAE Encoder with torque: [CLS, qpos_embed, qtor_embed, action_embed]
- Transformer Decoder 3-stack: [latent_token, proprio_token, torque_token]
- Dynamic control via use_torque and mix parameters
- 14D -> 17D padding support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Dict
import einops
import numpy as np
import sys
import os

# Add lerobot to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../lerobot/src'))

from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE
from .configuration_act_wholebody import ACTWholeBodyConfig


def reparametrize(mu: Tensor, logvar: Tensor) -> Tensor:
    """VAE reparameterization trick."""
    std = logvar.div(2).exp()
    eps = torch.randn_like(std)
    return mu + std * eps


def get_sinusoid_encoding_table(n_position: int, d_hid: int) -> Tensor:
    """Generate sinusoidal positional encoding."""
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class ACTWholeBodyPolicy(nn.Module):
    """
    ACT-wholebody: Action Chunking Transformer for whole-body mobile manipulation.

    Extends standard ACT with:
    1. Torque information as separate token stream
    2. Mobile base velocity support
    3. 14D->17D padding
    4. Dynamic control via use_torque and mix flags
    """

    def __init__(self, config: ACTWholeBodyConfig):
        super().__init__()
        self.config = config

        # Extract config parameters
        self.use_torque = config.use_torque
        self.mix = config.mix
        self.state_dim = config.state_dim
        self.chunk_size = config.chunk_size
        self.latent_dim = config.latent_dim
        self.dim_model = config.dim_model
        self.use_vae = config.use_vae

        # Vision backbone (ResNet)
        from lerobot.policies.act.modeling_act import ACTEncoder, ACTDecoder, ACTEncoderLayer, ACTDecoderLayer
        from torchvision import models as vision_models

        vision_backbone_class = getattr(vision_models, config.vision_backbone)
        weights_enum = getattr(
            vision_models,
            f"{config.vision_backbone.replace('resnet', 'ResNet')}_Weights",
        )
        weights = None
        if config.pretrained_backbone_weights:
            weights = weights_enum[config.pretrained_backbone_weights]

        self.backbone = vision_backbone_class(
            replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
            weights=weights,
            norm_layer=nn.BatchNorm2d,
        )

        # Remove classification head
        self.backbone.fc = nn.Identity()
        self.backbone_out_channels = 512 if "resnet18" in config.vision_backbone else 2048

        # Image feature projection
        self.encoder_img_feat_input_proj = nn.Conv2d(
            self.backbone_out_channels, config.dim_model, kernel_size=1
        )

        # Positional embedding for image features
        self.encoder_cam_feat_pos_embed = nn.Module()
        self.encoder_cam_feat_pos_embed.register_parameter(
            "weight",
            nn.Parameter(torch.randn(1, config.dim_model, 7, 7))  # Assuming 7x7 feature map
        )

        # ==================== VAE Encoder Components ====================
        if self.use_vae:
            # CLS token for VAE encoder
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)

            # Project qpos, qtor, action to embedding dim
            self.encoder_joint_proj = nn.Linear(self.state_dim, config.dim_model)  # qpos
            self.encoder_torque_proj = nn.Linear(self.state_dim, config.dim_model)  # qtor
            self.encoder_action_proj = nn.Linear(self.state_dim, config.dim_model)  # action

            # Latent projection: hidden_dim -> 2*latent_dim (mu and logvar)
            self.latent_proj = nn.Linear(config.dim_model, self.latent_dim * 2)

            # Positional encoding for VAE encoder: [CLS, qpos, qtor, action_sequence]
            # Length: 1 + 1 + 1 + chunk_size = chunk_size + 3
            self.register_buffer(
                'vae_pos_table',
                get_sinusoid_encoding_table(1 + 1 + 1 + config.chunk_size, config.dim_model)
            )

            # VAE Transformer Encoder
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)

        # ==================== Decoder Components ====================
        # Project latent sample to embedding
        self.latent_out_proj = nn.Linear(self.latent_dim, config.dim_model)

        # Project qpos and qtor for decoder
        self.input_proj_robot_state = nn.Linear(self.state_dim, config.dim_model)  # qpos for decoder
        self.input_proj_robot_torque = nn.Linear(self.state_dim, config.dim_model)  # qtor for decoder

        # Positional embeddings for additional tokens: [latent, proprio, torque]
        # Note: We use 3 embeddings (not 4, since we don't use lidar)
        self.additional_pos_embed = nn.Embedding(3, config.dim_model)

        # Query embeddings for action prediction
        self.query_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # Transformer Encoder-Decoder
        self.encoder = ACTEncoder(config, is_vae_encoder=False)
        self.decoder = ACTDecoder(config)

        # Action prediction head
        self.action_head = nn.Linear(config.dim_model, self.state_dim)

        # Initialize weights
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: Dict[str, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        Forward pass through ACT-wholebody.

        Args:
            batch: Dictionary containing:
                - 'observation.state': (B, 14) - dual-arm joint positions
                - 'observation.effort': (B, 14) - dual-arm joint torques
                - 'observation.base_velocity': (B, 3) - mobile base velocity [vx, vy, omega]
                - 'observation.images': List of (B, C, H, W) tensors
                - 'action': (B, chunk_size, 17) - target actions (only in training)
                - 'action_is_pad': (B, chunk_size) - padding mask (only in training)

        Returns:
            - actions: (B, chunk_size, 17) predicted action sequence
            - (mu, logvar): VAE latent distribution parameters (None if not training with VAE)
        """
        is_training = ACTION in batch and self.training

        # Extract batch data
        qpos_14 = batch[OBS_STATE]  # (B, 14)
        batch_size = qpos_14.shape[0]
        device = qpos_14.device

        # Get torque and base velocity
        qtor_14 = batch.get('observation.effort', torch.zeros_like(qpos_14))  # (B, 14)
        base_vel_3 = batch.get('observation.base_velocity', torch.zeros(batch_size, 3, device=device))  # (B, 3)

        # ==================== Step 1: Pad 14D -> 17D ====================
        # Apply use_torque and mix parameters
        qpos_17 = self._pad_to_17d(qpos_14, base_vel_3, use_base=self.mix)
        qtor_17 = self._pad_to_17d(qtor_14, torch.zeros_like(base_vel_3), use_base=self.use_torque)

        # ==================== Step 2: VAE Encoder (Training) ====================
        mu, logvar = None, None
        if self.use_vae and is_training:
            # Get actions and padding mask
            actions = batch[ACTION]  # (B, chunk_size, 17)
            is_pad = batch.get('action_is_pad', torch.zeros(batch_size, self.chunk_size, dtype=torch.bool, device=device))

            # Embed action sequence
            action_embed = self.encoder_action_proj(actions)  # (B, chunk_size, dim_model)

            # Embed qpos and qtor
            qpos_embed = self.encoder_joint_proj(qpos_17).unsqueeze(1)  # (B, 1, dim_model)
            qtor_embed = self.encoder_torque_proj(qtor_17).unsqueeze(1)  # (B, 1, dim_model)

            # CLS token
            cls_embed = self.vae_encoder_cls_embed.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 1, dim_model)

            # Concatenate: [CLS, qpos, qtor, action_sequence]
            encoder_input = torch.cat([cls_embed, qpos_embed, qtor_embed, action_embed], dim=1)  # (B, chunk_size+3, dim_model)
            encoder_input = encoder_input.transpose(0, 1)  # (chunk_size+3, B, dim_model)

            # Padding mask: don't mask CLS, qpos, qtor
            cls_qpos_qtor_is_pad = torch.full((batch_size, 3), False, device=device)
            full_is_pad = torch.cat([cls_qpos_qtor_is_pad, is_pad], dim=1)  # (B, chunk_size+3)

            # Positional encoding
            pos_embed = self.vae_pos_table.clone().detach().transpose(0, 1)  # (chunk_size+3, 1, dim_model)

            # Forward through VAE encoder
            encoder_output = self.vae_encoder(encoder_input, pos_embed=pos_embed, key_padding_mask=full_is_pad)
            encoder_output = encoder_output[0]  # Take CLS output: (B, dim_model)

            # Project to latent distribution
            latent_info = self.latent_proj(encoder_output)  # (B, latent_dim*2)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]

            # Sample latent
            latent_sample = reparametrize(mu, logvar)  # (B, latent_dim)
        else:
            # Inference: use zero latent
            latent_sample = torch.zeros(batch_size, self.latent_dim, device=device)

        # Project latent to embedding space
        latent_input = self.latent_out_proj(latent_sample)  # (B, dim_model)

        # ==================== Step 3: Image Features ====================
        all_cam_features = []
        all_cam_pos = []

        for img in batch[OBS_IMAGES]:
            # Extract features using ResNet backbone
            # img: (B, C, H, W)
            with torch.no_grad():
                x = img
                x = self.backbone.conv1(x)
                x = self.backbone.bn1(x)
                x = self.backbone.relu(x)
                x = self.backbone.maxpool(x)
                x = self.backbone.layer1(x)
                x = self.backbone.layer2(x)
                x = self.backbone.layer3(x)
                cam_features = self.backbone.layer4(x)  # (B, 2048 or 512, H', W')

            # Project to dim_model
            cam_features = self.encoder_img_feat_input_proj(cam_features)  # (B, dim_model, H', W')

            # Positional embedding
            cam_pos_embed = self.encoder_cam_feat_pos_embed.weight.expand(batch_size, -1, -1, -1)  # (B, dim_model, H', W')

            # Flatten spatial dimensions
            cam_features_flat = einops.rearrange(cam_features, "b c h w -> (h w) b c")  # (H'W', B, dim_model)
            cam_pos_flat = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")  # (H'W', B, dim_model)

            all_cam_features.append(cam_features_flat)
            all_cam_pos.append(cam_pos_flat)

        # Concatenate all camera features
        image_features = torch.cat(all_cam_features, dim=0)  # (num_cams*H'W', B, dim_model)
        image_pos_embed = torch.cat(all_cam_pos, dim=0)  # (num_cams*H'W', B, dim_model)

        # ==================== Step 4: Transformer Decoder ====================
        # Prepare decoder input: [latent, proprio, torque]
        proprio_input = self.input_proj_robot_state(qpos_17)  # (B, dim_model)
        torque_input = self.input_proj_robot_torque(qtor_17)  # (B, dim_model)

        # Stack additional inputs
        additional_tokens = torch.stack([latent_input, proprio_input, torque_input], dim=0)  # (3, B, dim_model)

        # Get positional embeddings for additional tokens
        additional_pos = self.additional_pos_embed.weight.unsqueeze(1)  # (3, 1, dim_model)

        # Concatenate with image features
        encoder_input = torch.cat([additional_tokens, image_features], dim=0)  # (3+num_cams*H'W', B, dim_model)
        encoder_pos_embed = torch.cat([additional_pos, image_pos_embed], dim=0)  # (3+num_cams*H'W', B, dim_model)

        # Transformer encoder
        encoder_out = self.encoder(encoder_input, pos_embed=encoder_pos_embed)  # (3+num_cams*H'W', B, dim_model)

        # Decoder input: learnable queries
        decoder_in = torch.zeros(self.chunk_size, batch_size, self.dim_model, device=device)

        # Decoder positional embedding
        decoder_pos_embed = self.query_embed.weight.unsqueeze(1)  # (chunk_size, 1, dim_model)

        # Transformer decoder
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_pos_embed,
            decoder_pos_embed=decoder_pos_embed
        )  # (chunk_size, B, dim_model)

        # Transpose to (B, chunk_size, dim_model)
        decoder_out = decoder_out.transpose(0, 1)

        # ==================== Step 5: Action Prediction ====================
        actions = self.action_head(decoder_out)  # (B, chunk_size, 17)

        return actions, (mu, logvar)

    def _pad_to_17d(self, data_14d: Tensor, base_3d: Tensor, use_base: bool) -> Tensor:
        """
        Pad 14D data to 17D.

        Args:
            data_14d: (B, 14) tensor
            base_3d: (B, 3) base velocity or zeros
            use_base: If True, use base_3d. If False, use zeros.

        Returns:
            (B, 17) padded tensor
        """
        if use_base:
            return torch.cat([base_3d, data_14d], dim=-1)
        else:
            zeros = torch.zeros_like(base_3d)
            return torch.cat([zeros, data_14d], dim=-1)


# Import ACTDecoder from lerobot
# This is a simplified version - you may need to copy the full implementation
class ACTDecoder(nn.Module):
    """Simplified ACT Decoder."""
    def __init__(self, config):
        super().__init__()
        from lerobot.policies.act.modeling_act import ACTDecoderLayer
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(self, tgt, memory, encoder_pos_embed=None, decoder_pos_embed=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, encoder_pos_embed=encoder_pos_embed, decoder_pos_embed=decoder_pos_embed)
        output = self.norm(output)
        return output
