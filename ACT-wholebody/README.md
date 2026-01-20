# ACT-wholebody: Whole-body Mobile Manipulation Policy

ACT-wholebody extends the standard ACT (Action Chunking Transformer) policy with:
- **Torque information** as separate token stream (use_torque parameter)
- **Mobile base velocity** support (mix parameter)
- **17D state/action space** (3D base + 7D left arm + 7D right arm)
- **Dynamic control** via command-line flags for ablation studies

## 📁 Project Structure

```
ACT-wholebody/
├── __init__.py                    # Package initialization
├── configuration_act_wholebody.py # Policy configuration class
├── modeling_act_wholebody.py      # Core policy model
├── convert_bag_wholebody.py       # ROS bag → LeRobot dataset converter
├── train_wholebody.py             # Training script
├── piper_act_wholebody.sh         # Easy-to-use training launcher
└── README.md                      # This file
```

## 🚀 Quick Start

### Step 1: Convert ROS Bags to Dataset

Convert your ROS bag files to LeRobot dataset format with 17D support:

```bash
cd /home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/ACT-wholebody

# Edit paths in convert_bag_wholebody.py if needed
python convert_bag_wholebody.py
```

**What it does:**
- Reads ROS bags from `/home/zeno/piper_ros/data_collect/ACT-100`
- Extracts:
  - Images from 3 cameras
  - Arm states (position, velocity, effort) from both arms
  - Base velocity from `/ranger_base_node/odom`
  - Actions from teleop (dual-arm + base cmd_vel)
- Outputs LeRobot dataset to `/home/zeno/piper_ros/data_collect/ACT-100-wholebody-v17`

**Dataset structure:**
- `observation.state`: 14D (dual-arm positions)
- `observation.velocity`: 14D (dual-arm velocities)
- `observation.effort`: 14D (dual-arm torques)
- `observation.base_velocity`: 3D (base vx, vy, omega)
- `action`: 17D (base 3D + dual-arm 14D)

### Step 2: Train

Train the policy with one simple command:

```bash
# Full mode (torque + base velocity)
bash piper_act_wholebody.sh --use_torque --mix

# Torque only (no base, for comparison)
bash piper_act_wholebody.sh --use_torque

# Base only (no torque)
bash piper_act_wholebody.sh --mix

# Baseline (neither)
bash piper_act_wholebody.sh
```

## 🎛️ Parameters Explained

### `--use_torque`
- **True**: Uses real torque data from `observation.effort`
- **False**: Pads torque input with zeros

**Purpose:** Compare performance with/without torque information

### `--mix`
- **True**: Uses base velocity `[vx, vy, omega]` from `observation.base_velocity`
- **False**: Pads first 3 dims with zeros (base not used)

**Purpose:** Compare performance with/without mobile base information

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset_repo_id` | `ACT-100-wholebody-v17` | Dataset name |
| `--dataset_root` | `/home/zeno/piper_ros/data_collect/ACT-100-wholebody-v17` | Dataset path |
| `--batch_size` | 32 | Batch size |
| `--steps` | 80000 | Training steps |
| `--log_freq` | 100 | Logging frequency |
| `--eval_freq` | 5000 | Evaluation frequency |
| `--save_freq` | 5000 | Checkpoint save frequency |
| `--wandb` / `--no_wandb` | Enabled | W&B logging |

## 📊 Ablation Study (4 Configurations)

Train 4 policies to understand the contribution of each component:

```bash
# 1. Full (torque + base)
bash piper_act_wholebody.sh --use_torque --mix

# 2. Torque only
bash piper_act_wholebody.sh --use_torque

# 3. Base only
bash piper_act_wholebody.sh --mix

# 4. Baseline (arm-only)
bash piper_act_wholebody.sh
```

**Results will be saved to:**
```
/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/ACT-wholebody/
├── torqueTrue_mixTrue/     # Full mode
├── torqueTrue_mixFalse/    # Torque only
├── torqueFalse_mixTrue/    # Base only
└── torqueFalse_mixFalse/   # Baseline
```

## 🏗️ Architecture Details

### VAE Encoder (Training)
```
Input: [CLS, qpos_embed, qtor_embed, action_sequence]
  ↓
Transformer Encoder (4 layers)
  ↓
CLS output → Linear → [μ, logσ²]
  ↓
Reparameterization → latent z
```

### Transformer Decoder (Training + Inference)
```
3-stack input:
  [latent_token, proprio_token, torque_token]
  + image_features (from ResNet18)
  ↓
Transformer Encoder (4 layers)
  ↓
Transformer Decoder (1 layer)
  ↓
Action head → 17D actions
```

### Dimension Padding Logic

**14D → 17D padding:**
```python
# State (qpos)
if mix:
    qpos_17 = [vx, vy, omega, left_j1...j7, right_j1...j7]
else:
    qpos_17 = [0, 0, 0, left_j1...j7, right_j1...j7]

# Torque (qtor)
if use_torque:
    qtor_17 = [0, 0, 0, left_eff1...eff7, right_eff1...eff7]
else:
    qtor_17 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

## 🔧 Model Configuration

**Default hyperparameters (in `configuration_act_wholebody.py`):**
- `vision_backbone`: ResNet18
- `dim_model`: 512
- `n_heads`: 8
- `dim_feedforward`: 3200
- `n_encoder_layers`: 4
- `n_decoder_layers`: 1
- `n_vae_encoder_layers`: 4
- `latent_dim`: 64 (increased from 32 for torque info)
- `chunk_size`: 100
- `dropout`: 0.1
- `kl_weight`: 10.0

## 📝 Data Format

### ROS Topics Used
- **Images**: `/realsense_{left,right,top}/color/image_raw/compressed`
- **Arm states**: `/robot/arm_{left,right}/joint_states_single`
- **Arm actions**: `/teleop/arm_{left,right}/joint_states_single`
- **Base state**: `/ranger_base_node/odom`
- **Base action**: `/teleop/cmd_vel`

### JointState Message Fields
```python
sensor_msgs/JointState:
  - position[7]  → used for qpos (joint positions)
  - velocity[7]  → used for qvel (joint velocities)
  - effort[7]    → used for qtor (joint torques)
```

### Odometry Message Fields
```python
nav_msgs/Odometry:
  - twist.twist.linear.{x, y}  → base vx, vy
  - twist.twist.angular.z      → base omega
```

## ⚠️ Important Notes

1. **Data Requirements:**
   - Your ROS bags MUST contain all required topics
   - `effort` field in JointState must be populated (for torque)
   - `/ranger_base_node/odom` must exist (for base velocity)

2. **17D Action Space:**
   - Action includes both base velocity AND arm commands
   - Base action comes from `/teleop/cmd_vel`
   - If your task doesn't need base control, use `--mix=False` to ignore it

3. **Backward Compatibility:**
   - If you don't have base or torque data, the model can still work by padding with zeros
   - Use appropriate flags (`--use_torque`, `--mix`) to control what's used

4. **Training Time:**
   - Full mode (torque + base) takes ~10-12 hours on a single GPU (A100)
   - Model size: ~23M parameters

## 🐛 Troubleshooting

**Problem:** `KeyError: 'observation.effort'`
- **Solution:** Your dataset doesn't have torque data. Use `--use_torque=False` or re-convert bags.

**Problem:** `KeyError: 'observation.base_velocity'`
- **Solution:** Your dataset doesn't have odom data. Use `--mix=False` or re-convert bags.

**Problem:** Shape mismatch errors
- **Solution:** Check that your dataset has been converted with the wholebody converter script.

## 📚 References

Based on:
- ACT (Action Chunking Transformers): https://github.com/tonyzhaozh/act
- LeRobot: https://github.com/huggingface/lerobot
- Your advisor's modifications in `/home/zeno/piper_ros/act_bak/`

## 📧 Contact

For questions about this implementation, check:
- Original implementation plan: `/home/zeno/详细实施计划.md`
- Advisor's reference code: `/home/zeno/piper_ros/act_bak/`
