#!/usr/bin/env bash
# ============================================================
# 批量训练 + 多数据集评估脚本（硬编码参数版）
#
# 流程：依次训练 5 个模型（各自的 base_lr / epoch 不同），
#       每个模型训练完成后，自动找到刚生成的 checkpoint，
#       分别在 3 个数据集上运行评估。
# 说明：全程使用第 0 号 GPU，串行执行，不做断电/断点恢复；
#       所有终端输出同步保存到 batch_logs/ 下。
# 用法：运行前修改下方数组参数，然后 bash run_batch.sh
# ============================================================
set -eo pipefail
cd "$(dirname "$0")"

# ===== 硬编码参数区（按需修改） =====
GPU=0
TRAIN_DATA_DIR="./data/busi"        # 训练所用数据集

MODELS=("U_Net" "AttU_Net" "UNetplus_L3" "Mobile_U_ViT" "CMUNeXt")
BASE_LRS=(0.01   0.01       0.005          0.001           0.0005)
EPOCHS=(  20     20         40             60              80)

EVAL_DATASETS=(                     # 评估所用的 3 个数据集
    "./data/busi"
    "./data/dataset_A"
    "./data/dataset_B"
)

LOG_DIR="batch_logs"
# ===== 参数区结束 =====

mkdir -p "$LOG_DIR"

for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    lr="${BASE_LRS[$i]}"
    ep="${EPOCHS[$i]}"
    run_log="$LOG_DIR/${model}_$(date +%Y%m%d_%H%M%S).log"

    {
        echo "============================================================"
        echo "[$((i + 1))/${#MODELS[@]}] model=$model  base_lr=$lr  epoch=$ep"
        echo "============================================================"

        # ---- 训练（在 TRAIN_DATA_DIR 上） ----
        python train.py \
            --model "$model" \
            --epoch "$ep" \
            --base_lr "$lr" \
            --batch_size 8 \
            --img_size 256 \
            --data_dir "$TRAIN_DATA_DIR" \
            --seed 41 \
            --gpu "$GPU" \
            --custom_message "batch_${model}_lr${lr}_ep${ep}"

        # ---- 找到刚生成的 checkpoint（取最新目录） ----
        CKPT_STEM="$(basename "$(ls -td checkpoint/*/ | head -1)")"
        echo "=> Trained checkpoint: $CKPT_STEM"

        # ---- 在 3 个数据集上分别评估 ----
        for ds in "${EVAL_DATASETS[@]}"; do
            echo "------------------------------------------------------------"
            echo "Evaluating $CKPT_STEM on dataset: $ds"
            echo "------------------------------------------------------------"
            python evaluate.py \
                --model "$CKPT_STEM" \
                --data_dir "$ds" \
                --batch_size 8 \
                --img_size 256 \
                --gpu "$GPU"
        done
    } 2>&1 | tee "$run_log"

    echo "=> Log saved to $run_log"
done

echo "============================================================"
echo "ALL DONE. Batch logs are in $LOG_DIR/"
echo "============================================================"
