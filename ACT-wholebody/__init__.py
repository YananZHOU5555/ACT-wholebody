"""
ACT-wholebody: Action Chunking Transformer for Whole-body Mobile Manipulation

This package implements an extended ACT policy with:
- Support for torque information (use_torque parameter)
- Support for mobile base velocity (mix parameter)
- 3-stack architecture: [latent, proprioception, torque]
- VAE encoder including torque: [CLS, qpos, qtor, action]
- 14D -> 17D padding (base + dual-arm)
"""

from .configuration_act_wholebody import ACTWholeBodyConfig
from .modeling_act_wholebody import ACTWholeBodyPolicy

__all__ = ["ACTWholeBodyConfig", "ACTWholeBodyPolicy"]
