# Cloud Server Setup Guide

## Quick Setup for 4x H200 GPU Server

### 1. Clone Repository
```bash
git clone https://github.com/YananZHOU5555/ACT-wholebody.git
cd ACT-wholebody
```

### 2. Setup Python Environment
```bash
# Create conda environment (recommended)
conda create -n act-wholebody python=3.10 -y
conda activate act-wholebody

# Or use venv
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

**Option A: Install LeRobot + ACT-wholebody (Recommended)**
```bash
# Install PyTorch first (CUDA 11.8 or 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install LeRobot in editable mode
pip install -e lerobot/

# Install additional dependencies
pip install -r requirements.txt
```

**Option B: Use LeRobot's requirements directly**
```bash
# For Ubuntu
pip install -r lerobot/requirements-ubuntu.txt

# For macOS
pip install -r lerobot/requirements-macos.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "from lerobot.common.policies.act.modeling_act import ACTPolicy; print('LeRobot OK')"
```

### 5. Prepare Dataset
Upload your dataset to the server or use Hugging Face:
```bash
# If using local dataset
mkdir -p data/
# Copy your dataset here

# If using Hugging Face
# Set environment variable
export HF_DATASETS_CACHE=/path/to/cache
```

### 6. Run Training

**Single GPU Training:**
```bash
cd ACT-wholebody
bash piper_act_wholebody.sh --use_torque --mix
```

**4-GPU Parallel Training (Ablation Study):**
```bash
cd ACT-wholebody
bash train_parallel_4gpu.sh
```

**Monitor Training:**
```bash
bash monitor_training.sh
```

### 7. Expected GPU Usage
- Each training process: ~10-15GB VRAM
- 4x H200 (141GB each): Can easily run 4 parallel trainings
- Training speed: ~80k steps in 10-15 hours per GPU

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in training scripts:
```bash
--batch_size 16  # Default is 32
```

### Missing ROS Dependencies
If you need to convert ROS bags on the server:
```bash
# Install ROS dependencies (Ubuntu 20.04/22.04)
sudo apt-get update
sudo apt-get install python3-rosbag python3-rospy python3-sensor-msgs
```

### Hugging Face Token
If using private datasets:
```bash
huggingface-cli login
# Or set token
export HF_TOKEN=your_token_here
```

## Training Outputs
- Checkpoints: `ACT-wholebody/outputs/<config_name>/checkpoints/`
- Logs: `ACT-wholebody/outputs/<config_name>/train.log`
- WandB: Check your wandb dashboard for real-time metrics

## Next Steps
After training completes:
1. Download checkpoints from `outputs/` directory
2. Use for deployment or evaluation
3. Compare 4 configurations for ablation analysis
