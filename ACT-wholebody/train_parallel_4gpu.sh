#!/usr/bin/env bash
set -e

# ============================================================
# ACT-wholebody 4-GPU Parallel Training
# ============================================================
# 在4张GPU上并行训练4种配置，最快完成消融实验
#
# 配置分配：
#   GPU 0: torqueTrue_mixTrue   (全开)
#   GPU 1: torqueTrue_mixFalse  (仅力矩)
#   GPU 2: torqueFalse_mixTrue  (仅底座)
#   GPU 3: torqueFalse_mixFalse (基线)
#
# 使用方法：
#   bash train_parallel_4gpu.sh
# ============================================================

# ==================== 配置参数 ====================
DATASET_REPO_ID="ACT-100-wholebody-v17"
DATASET_ROOT="/home/zeno/piper_ros/data_collect/ACT-100-wholebody-v17"
OUTPUT_DIR="/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/ACT-wholebody"
BATCH_SIZE=32
STEPS=80000
LOG_FREQ=100
EVAL_FREQ=5000
SAVE_FREQ=5000

# ==================== 颜色输出 ====================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ==================== 打印配置 ====================
echo -e "${GREEN}========================================================"
echo "ACT-wholebody 4-GPU Parallel Training"
echo "========================================================${NC}"
echo ""
echo "Dataset:     $DATASET_REPO_ID"
echo "Data path:   $DATASET_ROOT"
echo "Output dir:  $OUTPUT_DIR"
echo "Batch size:  $BATCH_SIZE"
echo "Steps:       $STEPS"
echo ""
echo -e "${BLUE}GPU Allocation:${NC}"
echo "  GPU 0 → torqueTrue_mixTrue   (全开: 力矩+底座)"
echo "  GPU 1 → torqueTrue_mixFalse  (仅力矩)"
echo "  GPU 2 → torqueFalse_mixTrue  (仅底座)"
echo "  GPU 3 → torqueFalse_mixFalse (基线)"
echo ""
echo -e "${YELLOW}========================================================"
echo "Starting parallel training in 5 seconds..."
echo "Press Ctrl+C to cancel"
echo "========================================================${NC}"
sleep 5

# ==================== 创建日志目录 ====================
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

# ==================== 定义训练函数 ====================
train_config() {
    local GPU_ID=$1
    local USE_TORQUE=$2
    local MIX=$3
    local CONFIG_NAME=$4

    # 构建命令行参数
    local TORQUE_FLAG=""
    local MIX_FLAG=""
    if [ "$USE_TORQUE" = "true" ]; then
        TORQUE_FLAG="--use_torque"
    fi
    if [ "$MIX" = "true" ]; then
        MIX_FLAG="--mix"
    fi

    # 日志文件
    local LOG_FILE="$LOG_DIR/${CONFIG_NAME}.log"

    echo -e "${GREEN}[GPU $GPU_ID] Starting: $CONFIG_NAME${NC}"
    echo "  Log: $LOG_FILE"

    # 运行训练
    CUDA_VISIBLE_DEVICES=$GPU_ID python train_wholebody.py \
        --dataset_repo_id "$DATASET_REPO_ID" \
        --dataset_root "$DATASET_ROOT" \
        --output_dir "$OUTPUT_DIR" \
        $TORQUE_FLAG \
        $MIX_FLAG \
        --batch_size "$BATCH_SIZE" \
        --steps "$STEPS" \
        --log_freq "$LOG_FREQ" \
        --eval_freq "$EVAL_FREQ" \
        --save_freq "$SAVE_FREQ" \
        --wandb \
        > "$LOG_FILE" 2>&1

    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo -e "${GREEN}[GPU $GPU_ID] ✓ $CONFIG_NAME completed successfully${NC}"
    else
        echo -e "${RED}[GPU $GPU_ID] ✗ $CONFIG_NAME failed with exit code $EXIT_CODE${NC}"
    fi

    return $EXIT_CODE
}

# ==================== 启动4个并行训练 ====================
START_TIME=$(date +%s)

echo -e "\n${BLUE}========================================================${NC}"
echo -e "${BLUE}Launching 4 parallel training processes...${NC}"
echo -e "${BLUE}========================================================${NC}\n"

# GPU 0: 全开 (torqueTrue_mixTrue)
train_config 0 true true "torqueTrue_mixTrue" &
PID_0=$!

# GPU 1: 仅力矩 (torqueTrue_mixFalse)
train_config 1 true false "torqueTrue_mixFalse" &
PID_1=$!

# GPU 2: 仅底座 (torqueFalse_mixTrue)
train_config 2 false true "torqueFalse_mixTrue" &
PID_2=$!

# GPU 3: 基线 (torqueFalse_mixFalse)
train_config 3 false false "torqueFalse_mixFalse" &
PID_3=$!

echo -e "${YELLOW}All training processes launched in background${NC}"
echo ""
echo "Process IDs:"
echo "  GPU 0 (torqueTrue_mixTrue):   PID $PID_0"
echo "  GPU 1 (torqueTrue_mixFalse):  PID $PID_1"
echo "  GPU 2 (torqueFalse_mixTrue):  PID $PID_2"
echo "  GPU 3 (torqueFalse_mixFalse): PID $PID_3"
echo ""
echo -e "${BLUE}========================================================${NC}"
echo "Monitoring logs in real-time:"
echo "  tail -f $LOG_DIR/torqueTrue_mixTrue.log"
echo "  tail -f $LOG_DIR/torqueTrue_mixFalse.log"
echo "  tail -f $LOG_DIR/torqueFalse_mixTrue.log"
echo "  tail -f $LOG_DIR/torqueFalse_mixFalse.log"
echo -e "${BLUE}========================================================${NC}\n"

# ==================== 等待所有训练完成 ====================
echo -e "${YELLOW}Waiting for all training processes to complete...${NC}\n"

# 等待所有后台进程
wait $PID_0
EXIT_0=$?

wait $PID_1
EXIT_1=$?

wait $PID_2
EXIT_2=$?

wait $PID_3
EXIT_3=$?

# ==================== 汇总结果 ====================
END_TIME=$(date +%s)
TOTAL_TIME=$((END_TIME - START_TIME))
HOURS=$((TOTAL_TIME / 3600))
MINUTES=$(((TOTAL_TIME % 3600) / 60))
SECONDS=$((TOTAL_TIME % 60))

echo -e "\n${GREEN}========================================================"
echo "All Training Completed!"
echo "========================================================${NC}"
echo ""
echo "Results:"
echo "  GPU 0 (torqueTrue_mixTrue):   $([ $EXIT_0 -eq 0 ] && echo -e "${GREEN}✓ Success${NC}" || echo -e "${RED}✗ Failed${NC}")"
echo "  GPU 1 (torqueTrue_mixFalse):  $([ $EXIT_1 -eq 0 ] && echo -e "${GREEN}✓ Success${NC}" || echo -e "${RED}✗ Failed${NC}")"
echo "  GPU 2 (torqueFalse_mixTrue):  $([ $EXIT_2 -eq 0 ] && echo -e "${GREEN}✓ Success${NC}" || echo -e "${RED}✗ Failed${NC}")"
echo "  GPU 3 (torqueFalse_mixFalse): $([ $EXIT_3 -eq 0 ] && echo -e "${GREEN}✓ Success${NC}" || echo -e "${RED}✗ Failed${NC}")"
echo ""
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Checkpoints saved to:"
echo "  $OUTPUT_DIR/torqueTrue_mixTrue/"
echo "  $OUTPUT_DIR/torqueTrue_mixFalse/"
echo "  $OUTPUT_DIR/torqueFalse_mixTrue/"
echo "  $OUTPUT_DIR/torqueFalse_mixFalse/"
echo ""
echo "Logs saved to:"
echo "  $LOG_DIR/"
echo -e "${GREEN}========================================================${NC}\n"

# ==================== 退出码 ====================
# 如果所有训练都成功，返回0；否则返回1
if [ $EXIT_0 -eq 0 ] && [ $EXIT_1 -eq 0 ] && [ $EXIT_2 -eq 0 ] && [ $EXIT_3 -eq 0 ]; then
    echo -e "${GREEN}✓ All experiments completed successfully!${NC}\n"
    exit 0
else
    echo -e "${RED}✗ Some experiments failed. Check logs for details.${NC}\n"
    exit 1
fi
