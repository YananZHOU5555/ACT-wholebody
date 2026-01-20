#!/usr/bin/env bash

# ============================================================
# 实时监控 4-GPU 训练进度
# ============================================================
# 使用方法：
#   bash monitor_training.sh
#
# 功能：
#   - 实时显示4个GPU的训练状态
#   - 显示最新的loss、进度等
#   - 每10秒刷新一次
# ============================================================

LOG_DIR="/home/zeno/NPM-VLA-Project/NPM-VLA/IL_policies/checkpoints/ACT-wholebody/logs"

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

# 清屏
clear

while true; do
    # 移动光标到顶部
    tput cup 0 0

    echo -e "${GREEN}========================================================"
    echo "ACT-wholebody 4-GPU Training Monitor"
    echo "========================================================${NC}"
    echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # 检查日志文件是否存在
    if [ ! -d "$LOG_DIR" ]; then
        echo -e "${RED}Log directory not found: $LOG_DIR${NC}"
        echo "Training may not have started yet."
        sleep 10
        continue
    fi

    # 监控每个GPU的训练状态
    for config in "torqueTrue_mixTrue" "torqueTrue_mixFalse" "torqueFalse_mixTrue" "torqueFalse_mixFalse"; do
        LOG_FILE="$LOG_DIR/${config}.log"

        echo -e "${BLUE}----------------------------------------${NC}"
        echo -e "${YELLOW}Configuration: $config${NC}"

        if [ ! -f "$LOG_FILE" ]; then
            echo -e "${RED}  Status: Log file not found${NC}"
        else
            # 检查进程是否在运行
            if pgrep -f "train_wholebody.py.*$config" > /dev/null; then
                echo -e "${GREEN}  Status: Running ✓${NC}"
            else
                echo -e "${RED}  Status: Not running or completed${NC}"
            fi

            # 显示最后10行日志（过滤掉空行）
            echo "  Latest logs:"
            tail -n 15 "$LOG_FILE" | grep -v "^$" | tail -n 10 | sed 's/^/    /'
        fi
        echo ""
    done

    echo -e "${GREEN}========================================================${NC}"
    echo "Press Ctrl+C to exit monitor"
    echo "Refreshing in 10 seconds..."
    echo ""

    # 等待10秒，可以被Ctrl+C中断
    sleep 10
done
