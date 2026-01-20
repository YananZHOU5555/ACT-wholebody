#!/usr/bin/env bash
set -e

# 完全本地训练配置
# dataset.root 应该直接指向数据集目录本身，而不是父目录

python -m lerobot.scripts.lerobot_train \
  --policy.type=act \
  --dataset.repo_id=ACT-100-v30 \
  --dataset.root=/home/zeno/piper_ros/data_collect/ACT-100-v30 \
  --output_dir=/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/ACT-100-v30 \
  --job_name=ACT-100-v30 \
  --policy.device=cuda \
  --batch_size=32 \
  --steps=80000 \
  --log_freq=100 \
  --eval_freq=5000 \
  --save_freq=5000 \
  --wandb.enable=true \
  --policy.repo_id=false \
  --resume=false
