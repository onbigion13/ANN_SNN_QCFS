#!/bin/bash
set -e          # 命令失败即退出
set -u          # 使用未定义的变量即退出（防止变量名写错导致 rm -rf $UNSET_VAR/ ）
set -o pipefail # 管道命令中只要有一个环节失败即退出

# --- 强力清理代理，防止 wandb 报错 ---
unset http_proxy
unset https_proxy
unset ALL_PROXY

# --- 基础环境配置 ---
GPU_ID=7 # 1 or "0 1 3"
SEED=42  # 13, 42, 2026
# endregion

# region --- 数据集相关参数 --- 批量大小，已在 train.py 中，为每个数据集对应指定。
NAME_DATASET="CIFAR10-DVS"
NUM_FRAMES=10
# endregion

# region --- 模型相关参数 ---
NAME_NET="vgg11"
# endregion

# region --- 训练、损失相关参数 ---
NUM_WORKERS=8
EPOCHS=300
# endregion

# region --- 优化器相关参数 ---
INITIAL_LR=0.001
WEIGHT_DECAY=2e-5
# endregion

SAVE_DIR="./logs_of_shell/QCFS_dataset-${NAME_DATASET}_start-at-$(date +%Y%m%d_%H%M%S)"
mkdir -p $SAVE_DIR

# --- region 运行命令 ---
# 使用 nohup 配合 & 可以在退出终端后继续运行，并将日志重定向到文件
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Starting Experiment on ${NAME_DATASET} with QCFS..."

python main_train.py \
  --suffix "Training of VGG11 on CIFAR10-DVS" \
  --seed $SEED \
  --epochs $EPOCHS \
  --batch_size 32 \
  --workers $NUM_WORKERS \
  --dataset $NAME_DATASET \
  --dataset $NAME_DATASET \
  --time $NUM_FRAMES \
  --L 8 \
  --model $NAME_NET \
  --lr $INITIAL_LR \
  --weight_decay $WEIGHT_DECAY \
  --imChannels 2 \
  >"${SAVE_DIR}/training.log" 2>&1 &

echo "Experiment is running in background. Log saved at: ${SAVE_DIR}/training.log"