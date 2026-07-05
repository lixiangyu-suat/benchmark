#!/bin/bash

# $1 表示你在终端执行脚本时输入的第一个参数
MODEL_NAME=$1

# 检查用户是否传递了参数，如果没有则报错退出
if [ -z "$MODEL_NAME" ]; then
    echo "错误: 请提供模型名称！使用示例: ./eval.sh <model_name>"
    echo "不予运行测试代码."
    # 判断脚本是否是被 source 运行的
    if [ "$ZSH_EVAL_CONTEXT" = "toplevel" ] || [ "$BASH_SOURCE" != "$0" ]; then
        return 1  # 被 source 运行，安全返回，不杀终端
    else
        exit 1    # 正常独立运行，直接退出
    fi
fi

echo
echo "----------------正在输出模型 ${MODEL_NAME} 的底层框架...----------------"
echo
# 运行第一个脚本
python eval_info.py --model "${MODEL_NAME}"

echo
echo "----------------正在使用模型: ${MODEL_NAME} 进行测试...----------------"
echo
# 运行第二个脚本
python infer.py --cfg ./configs/config.yaml --ckpt "${MODEL_NAME}"