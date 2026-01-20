#!/usr/bin/env bash
set -e

# ============================================================
# ACT-wholebody Training Script
# ============================================================
# Train ACT policy with whole-body control (17D: base + dual-arm)
#
# Usage:
#   bash piper_act_wholebody.sh [OPTIONS]
#
# Options:
#   --use_torque    Enable torque information
#   --mix           Enable mobile base velocity
#   --no_wandb      Disable Weights & Biases logging
#
# Examples:
#   # Full mode (torque + base)
#   bash piper_act_wholebody.sh --use_torque --mix
#
#   # Torque only (no base)
#   bash piper_act_wholebody.sh --use_torque
#
#   # Base only (no torque)
#   bash piper_act_wholebody.sh --mix
#
#   # Baseline (neither)
#   bash piper_act_wholebody.sh
# ============================================================

# Default values
DATASET_REPO_ID="ACT-100-wholebody-v17"
DATASET_ROOT="/home/zeno/piper_ros/data_collect/ACT-100-wholebody-v17"
OUTPUT_DIR="/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/ACT-wholebody"
BATCH_SIZE=32
STEPS=80000
LOG_FREQ=100
EVAL_FREQ=5000
SAVE_FREQ=5000

# Parse command line arguments
USE_TORQUE=""
MIX=""
WANDB="--wandb"

for arg in "$@"; do
    case $arg in
        --use_torque)
            USE_TORQUE="--use_torque"
            shift
            ;;
        --mix)
            MIX="--mix"
            shift
            ;;
        --no_wandb)
            WANDB="--no_wandb"
            shift
            ;;
        --dataset_repo_id=*)
            DATASET_REPO_ID="${arg#*=}"
            shift
            ;;
        --dataset_root=*)
            DATASET_ROOT="${arg#*=}"
            shift
            ;;
        --batch_size=*)
            BATCH_SIZE="${arg#*=}"
            shift
            ;;
        --steps=*)
            STEPS="${arg#*=}"
            shift
            ;;
        *)
            # Unknown option
            ;;
    esac
done

# Print configuration
echo "========================================================"
echo "ACT-wholebody Training"
echo "========================================================"
echo "Dataset:     $DATASET_REPO_ID"
echo "Data path:   $DATASET_ROOT"
echo ""
echo "Parameters:"
echo "  use_torque: $([ -n "$USE_TORQUE" ] && echo "✓ Enabled" || echo "✗ Disabled")"
echo "  mix:        $([ -n "$MIX" ] && echo "✓ Enabled (using base velocity)" || echo "✗ Disabled (base dims=0)")"
echo ""
echo "Training:"
echo "  Batch size: $BATCH_SIZE"
echo "  Steps:      $STEPS"
echo "  Output:     $OUTPUT_DIR"
echo "  W&B:        $([ "$WANDB" = "--wandb" ] && echo "✓ Enabled" || echo "✗ Disabled")"
echo "========================================================"
echo ""

# Change to the script directory
cd "$(dirname "$0")"

# Run training
python train_wholebody.py \
    --dataset_repo_id "$DATASET_REPO_ID" \
    --dataset_root "$DATASET_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    $USE_TORQUE \
    $MIX \
    --batch_size "$BATCH_SIZE" \
    --steps "$STEPS" \
    --log_freq "$LOG_FREQ" \
    --eval_freq "$EVAL_FREQ" \
    --save_freq "$SAVE_FREQ" \
    $WANDB

echo ""
echo "========================================================"
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "========================================================"
