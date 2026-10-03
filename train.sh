#!/usr/bin/env bash
# ============================================================
# 训练启动脚本（硬编码参数版）
# 用法：运行前修改下方 python 命令中的参数，然后 bash train.sh
# 参数说明：python train.py --help
# ============================================================
# 断点续训：删除 --model 行，取消最后一行 --ckpt 的注释
# ============================================================
set -e
cd "$(dirname "$0")"
   
python train.py \
    --model Mobile_U_ViT \
    --epoch 5 \
    --base_lr 0.01 \
    --batch_size 8 \
    --img_size 256 \
    --data_dir ./data/busi \
    --seed 41 \
    --train_file busi_train1.txt \
    --val_file busi_valid1.txt \
    --mask_target binary \
    --num_classes 1 \
    --gpu 0 \
    --custom_message "测试busi数据集能否正常使用并测试效果" \
#    --ckpt 20261003_1538_55786_Mobile_U_ViT
