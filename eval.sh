#!/bin/bash

# $1 表示你在终端执行脚本时输入的第一个参数
MODEL_NAME=$1

# 检查用户是否传递了参数，如果没有则报错退出
if [ -z "$MODEL_NAME" ]; then
    echo "错误: 请提供模型名称！使用示例: ./eval.sh <model_name>"
    exit 1
fi

echo
echo "----------------正在使用模型: ${MODEL_NAME} 进行测试...----------------"
echo
# 运行第一个脚本
python infer.py --model "${MODEL_NAME}" --base_dir ./data/busi --val_file_dir busi_val.txt --img_size 256 --num_classes 1

echo
echo "----------------正在输出模型 ${MODEL_NAME} 的底层框架...----------------"
echo
# 运行第二个脚本
python eval_info.py --model "${MODEL_NAME}"