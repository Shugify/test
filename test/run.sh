#!/bin/bash

# --- 脚本设置 ---
CONFIG_FILE="config.yaml"

export CUDA_VISIBLE_DEVICES=7

if [ ! -f "$CONFIG_FILE" ]; then
    echo " 错误: 配置文件 '$CONFIG_FILE' 未找到！"
    exit 1
fi

# --- 健壮的路径解析 ---
OUTPUT_DIR=$(grep 'output_dir:' "$CONFIG_FILE" | sed -e 's/#.*//' -e 's/.*: *//' -e 's/["'\'']//g')
EXP_NAME=$(grep 'name:' "$CONFIG_FILE" -A 1 | grep 'name:' | tail -n1 | sed -e 's/#.*//' -e 's/.*: *//' -e 's/["'\'']//g')

# 拼接日志文件路径，但不再在这里创建目录
FULL_OUTPUT_DIR="$OUTPUT_DIR/$EXP_NAME"
LOG_FILE="$FULL_OUTPUT_DIR/log.txt"


# --- 执行部分 ---
# 注意：因为日志文件所在的目录现在由Python创建，
# 我们需要先创建目录，再重定向日志。
# 最简单的做法是让Python脚本自己处理目录创建，然后我们再记录日志。
# 为了简化，我们暂时只让python创建目录，日志先不重定向，或者用一个临时日志

echo "============================================="
echo "开始执行训练任务在 GPU ${CUDA_VISIBLE_DEVICES}上..."
echo "配置文件: $CONFIG_FILE"
echo "启动脚本: $0"
# 日志记录暂时简化，因为目录由python创建
echo "实验目录将是: $FULL_OUTPUT_DIR"
echo "============================================="
echo ""

# Python脚本会自己处理目录创建
# 我们把日志重定向到脚本执行之后，但这会丢失实时性
# 一个折中的方案是先创建目录

python train.py --config "$CONFIG_FILE" --shell_path "$0" 2>&1 | tee "$LOG_FILE"


echo ""
echo "任务执行完毕。"