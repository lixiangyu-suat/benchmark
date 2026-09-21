#!/usr/bin/env bash
# ============================================================
# 评估启动脚本（硬编码参数版）
# 用法：运行前修改下方 python 命令中的参数，然后 bash evaluate.sh
# 参数说明：python evaluate.py --help
# ============================================================
set -e
cd "$(dirname "$0")"

python evaluate.py \
    --model 20260921_1715_47410_UNetplus_L3 \
    --batch_size 8 \
    --img_size 256 \
    --data_dir ./data/busi \
    --seed 41 \
    --val_file busi_val1.txt \
    --num_classes 1 \
    --gpu 0 \
    --save_viz
