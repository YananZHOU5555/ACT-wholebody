#!/usr/bin/env python

"""
Configuration for ACT-wholebody Policy
Extends LeRobot's ACTConfig with torque and mobile base support
"""

from dataclasses import dataclass, field
import sys
import os

# Add lerobot to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../lerobot/src'))

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig


@PreTrainedConfig.register_subclass("act_wholebody")
@dataclass
class ACTWholeBodyConfig(PreTrainedConfig):
    """Configuration for ACT-wholebody policy with torque and mobile base support.

    Key additions compared to standard ACT:
    - use_torque: Whether to use torque information as additional input
    - mix: Whether to use mobile base velocity (if False, base dimensions are zeros)
    - state_dim: State dimension (17 for mobile base + dual-arm)
    - latent_dim: Increased to 64 (from 32) to accommodate additional information

    Args:
        use_torque: If True, uses torque data. If False, torque input is padded with zeros.
        mix: If True, uses base velocity [vx, vy, omega]. If False, first 3 dims are zeros.
        state_dim: Dimension of state/action space (17 for wholebody, 14 for arm-only).
        latent_dim: VAE latent dimension (64 recommended for wholebody).
    """

    # Input / output structure
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Architecture
    # Vision backbone
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False

    # Transformer layers
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1

    # VAE
    use_vae: bool = True
    latent_dim: int = 64  # Increased from 32 to accommodate torque information
    n_vae_encoder_layers: int = 4

    # Inference
    temporal_ensemble_coeff: float | None = None

    # Training and loss
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    # ==================== NEW: Wholebody-specific parameters ====================
    use_torque: bool = True
    """Whether to use torque information. If False, torque is padded with zeros."""

    mix: bool = False
    """Whether to use mobile base velocity. If False, first 3 state dims are zeros."""

    state_dim: int = 17
    """State dimension: 17 for wholebody (3 base + 7 left arm + 7 right arm), 14 for arm-only."""

    # ===========================================================================

    def __post_init__(self):
        super().__post_init__()

        # Input validation
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"Chunk size must be >= n_action_steps. Got {self.chunk_size} and {self.n_action_steps}."
            )
        if self.n_obs_steps != 1:
            raise ValueError(f"Multiple observation steps not handled yet. Got {self.n_obs_steps}")

        # Wholebody-specific validation
        if self.state_dim not in [14, 17]:
            raise ValueError(f"state_dim must be 14 or 17, got {self.state_dim}")

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> None:
        return None

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or environment state among inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None
