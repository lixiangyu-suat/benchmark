#!/bin/bash

# $1 表示你在终端执行脚本时输入的第一个参数
TEACHER_NAME=$1
STUDENT_NAME=$2

# 检查用户是否传递了参数，如果没有则报错退出
if [ -z "$TEACHER_NAME" ]; then
    echo "错误: 请提供teacher名称！使用示例: ./eval.sh <teacher_name> <stu_name>"
    echo "不予运行测试代码."
    # 判断脚本是否是被 source 运行的
    if [ "$ZSH_EVAL_CONTEXT" = "toplevel" ] || [ "$BASH_SOURCE" != "$0" ]; then
        return 1  # 被 source 运行，安全返回，不杀终端
    else
        exit 1    # 正常独立运行，直接退出
    fi
fi

if [ -z "$STUDENT_NAME" ]; then
    echo "错误: 请提供student名称！使用示例: ./eval.sh <teacher_name> <stu_name>"
    echo "不予运行测试代码."
    # 判断脚本是否是被 source 运行的
    if [ "$ZSH_EVAL_CONTEXT" = "toplevel" ] || [ "$BASH_SOURCE" != "$0" ]; then
        return 1  # 被 source 运行，安全返回，不杀终端
    else
        exit 1    # 正常独立运行，直接退出
    fi
fi


echo
echo "----------------蒸馏：$TEACHER_NAME >>> $STUDENT_NAME ----------------"
echo
# 运行第二个脚本
python distill.py --cfg ./configs/config.yaml --teacher_model_ckpt "$TEACHER_NAME" --student_model_ckpt "$STUDENT_NAME"