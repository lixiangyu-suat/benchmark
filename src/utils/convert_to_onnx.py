from pathlib import Path

import torch
import torch.nn as nn
import onnx

from src.utils.model_loader import load_checkpoint_meta

OPSETVERSION = 16

def convert_pth_to_onnx(
    model: nn.Module,
    pth_path,
    onnx_path,
    dummy_input,
    input_names: list = None,
    output_names: list = None,
    dynamic_axes: dict = None,
    device: str = "cpu",
    opset_version: int = OPSETVERSION,
) -> None:
    """将训练保存的 .pth 检查点导出为 .onnx 并校验。

    Parameters
    ----------
    model : nn.Module
        已构建好的模型（权重会被 pth_path 覆盖加载）。
    pth_path : path-like
        save_checkpoint 保存的检查点（dict 格式或裸 state_dict 均可）。
    onnx_path : path-like
        输出 .onnx 文件路径。
    dummy_input : torch.Tensor 或 tuple
        示例输入张量，或输入形状元组如 ``(1, 3, 256, 256)``。
    """
    input_names = input_names or ["input"]
    output_names = output_names or ["output"]

    # 1. 载入权重（兼容 checkpoint dict 格式与 DataParallel 前缀）并切换为评估模式
    state_dict, _, _ = load_checkpoint_meta(pth_path, device)
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    target.load_state_dict(state_dict)
    target.to(device)
    target.eval()

    # dummy_input 支持直接传形状元组
    if not isinstance(dummy_input, torch.Tensor):
        dummy_input = torch.randn(*dummy_input)
    dummy_input = dummy_input.to(device)

    # 2. 导出为 ONNX
    Path(onnx_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        target,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # 3. 校验导出的 ONNX 文件格式是否完整有效
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"✓ ONNX 导出成功: {onnx_path}")
    print(f"  实际 opset: {[(item.domain or 'ai.onnx', item.version) for item in onnx_model.opset_import]}")
    print(f"  LayerNormalization 节点数量: "
          f"{sum(node.op_type == 'LayerNormalization' for node in onnx_model.graph.node)}")
