#!/usr/bin/env bash
set -e

# 使用本地数据集训练
# dataset.root 直接指向数据集目录

python -m lerobot.scripts.lerobot_train \
  --policy.type=act \
  --dataset.repo_id=ACT-100-v30 \
  --dataset.root=/home/zeno/piper_ros/data_collect/ACT-100-v30 \
  --dataset.video_backend=pyav \
  --output_dir "/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/ACT-100-v30" \
  --job_name="ACT-100-v30" \
  --policy.device=cuda \
  --batch_size=32 \
  --steps=80000 \
  --log_freq=100 \
  --eval_freq=5000 \
  --save_freq=5000 \
  --wandb.enable=true \
  --policy.repo_id=false
