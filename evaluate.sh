#!/usr/bin/env bash
# ============================================================
# 评估启动脚本（硬编码参数版）
# 用法：运行前修改下方 python 命令中的参数，然后 bash evaluate.sh
# 参数说明：python evaluate.py --help
# 用法：bash evaluate.sh --test_file test1.txt [其他参数，例如 --model <stem>]
# 当前示例传入验证集清单；独立测试时改为实际测试清单。
# ============================================================
set -e
cd "$(dirname "$0")"

python evaluate.py \
    --model 20261003_1538_55786_Mobile_U_ViT \
    --batch_size 8 \
    --img_size 256 \
    --data_dir ./data/busi \
    --test_file busi_valid1.txt \
    --mask_target binary \
    --seed 41 \
    --num_classes 1 \
    --gpu 0 \
    --save_viz \
    "$@"
