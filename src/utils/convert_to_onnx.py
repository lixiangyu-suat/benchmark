from pathlib import Path
import torch
import torch.nn as nn
import onnx


def convert_pth_to_onnx(
    model: nn.Module,
    pth_path,
    onnx_path,
    dummy_input: torch.Tensor,
    input_names: list[str] = None,
    output_names: list[str] = None,
    dynamic_axes: dict = None,
    device: str = "cpu",
) -> None:
    """将 PyTorch 的 .pth 模型权重转换为 .onnx 格式并校验。"""
    input_names = input_names or ["input"]
    output_names = output_names or ["output"]

    # 1. 载入权重并切换为评估模式
    state_dict = torch.load(pth_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    dummy_input = dummy_input.to(device)

    # 2. 导出为 ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    # 3. 校验导出的 ONNX 文件格式是否完整有效
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"✓ 转换成功: {onnx_path}")


if __name__ == "__main__":
    # 使用示例：
    # from my_network import MyModel
    #
    # model = MyModel()
    # dummy_input = torch.randn(1, 3, 224, 224)  # 依据你的网络输入形状构造
    #
    # convert_pth_to_onnx(
    #     model=model,
    #     pth_path="weights/model.pth",
    #     onnx_path="weights/model.onnx",
    #     dummy_input=dummy_input,
    #     dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}} # 若需要动态 batch
    # )
    pass