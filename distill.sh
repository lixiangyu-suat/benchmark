#!/usr/bin/env bash
# ============================================================
# 知识蒸馏启动脚本（硬编码参数版）
# 用法：运行前修改下方 python 命令中的参数，然后 bash distill.sh
# 参数说明：python distill.py --help
# 说明：--teacher / --student 传检查点 stem 或模型架构名；
#       --resume 表示 student 从检查点续训（不需要时删除该行）
# ============================================================
set -e
cd "$(dirname "$0")"

python distill.py \
    --teacher 20260709_1430_U_Net \
    --student Mobile_U_ViT \
    --epoch 20 \
    --base_lr 0.01 \
    --batch_size 8 \
    --img_size 256 \
    --data_dir ./data/busi \
    --seed 41 \
    --train_file busi_train1.txt \
    --val_file busi_val1.txt \
    --num_classes 1 \
    --gpu 0 \
    --custom_message "distill_U_Net_to_Mobile_U_ViT"
