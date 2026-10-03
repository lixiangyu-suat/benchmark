"""直接将指定的 PyTorch 检查点导出为 ONNX，不启动训练或加载数据集。"""

import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, type=Path, help="输入 .pth 权重路径")
    parser.add_argument("--output", required=True, type=Path, help="输出 .onnx 路径")
    parser.add_argument("--model", default="Mobile_U_ViT", help="模型架构名，须与权重一致")
    parser.add_argument("--img_size", type=int, default=256, help="输入宽高，须与推理设置一致")
    parser.add_argument("--num_classes", type=int, default=1, help="模型输出通道数，须与训练一致")
    parser.add_argument("--opset", type=int, default=16, help="ONNX opset；当前 TensorRT 8.5 使用 16")
    parser.add_argument("--device", default="cpu", help="导出设备，例如 cpu 或 cuda:0")
    parser.add_argument("--dynamic_batch", action="store_true", help="启用动态 batch；默认固定 batch=1")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"输入权重不存在: {args.input}")
    if args.input.suffix.lower() != ".pth" or args.output.suffix.lower() != ".onnx":
        parser.error("输入须为 .pth，输出须为 .onnx")
    if args.img_size <= 0 or args.num_classes <= 0 or args.opset < 9:
        parser.error("img_size 和 num_classes 须为正数，opset 须至少为 9")
    return args


def main():
    args = parse_args()
    # 延迟导入，让 --help 不依赖 PyTorch/ONNX 环境。
    import torch
    from src.utils.config import build_config
    from src.utils.convert_to_onnx import convert_pth_to_onnx
    from src.utils.model_loader import build_model, validate_model

    model_name = validate_model(args.model)
    if model_name == "SwinUnet" and args.img_size != 224:
        raise ValueError("SwinUnet 请指定 --img_size 224")
    model = build_model(build_config(args), model_name, torch.device(args.device))
    print(f"输入权重: {args.input.resolve()}")
    print(f"输出模型: {args.output.resolve()}")
    print(f"输入尺寸: 1 x 3 x {args.img_size} x {args.img_size}")
    print(f"导出 opset: {args.opset}，设备: {args.device}")
    dynamic_axes = ({"input": {0: "batch_size"}, "output": {0: "batch_size"}}
                    if args.dynamic_batch else None)
    convert_pth_to_onnx(
        model=model,
        pth_path=args.input,
        onnx_path=args.output,
        dummy_input=(1, 3, args.img_size, args.img_size),
        dynamic_axes=dynamic_axes,
        device=args.device,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()
