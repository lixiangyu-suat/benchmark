import argparse
import os
import sys
import time

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
from torchinfo import summary

from src.utils.config import add_common_args, build_config
from src.utils.dataset import get_test_loader
from src.utils.eval_report import format_tree
from src.utils.helpers import format_duration, timestamp
from src.utils.losses import BCEDiceLoss
from src.utils.metrics import iou_score
from src.utils.model_loader import build_model, load_checkpoint_meta, resolve_ckpt_path
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="读取指定测试清单评估分割模型（架构摘要 + 指标）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="检查点 stem（如 20260708_1624_U_Net）；仅加载对应 _best.pth")
    parser.add_argument("--img_size", type=int, default=256,
                        help="评估输入分辨率 (SwinUnet 固定 224)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="评估批大小")
    parser.add_argument("--save_viz", action="store_true",
                        help="保存预测掩码可视化到 test_results/")
    parser.add_argument("--test_file", type=str, required=True,
                        help="测试集清单文件名，例如 test1.txt；相对 data_dir 解析")
    add_common_args(parser, include_split_args=False)
    return parser.parse_args()


def main():
    task_start = time.perf_counter()  # 任务总计时起点
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = build_config(args)
    # Check manifest and test paths before loading a potentially large model.
    test_loader = get_test_loader(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = resolve_ckpt_path(args.model, best_only=True)
    print("\n" + format_tree("Evaluation setup", [
        ("Model", [("Name", args.model), ("Checkpoint", ckpt_path)]),
        ("Data", [("Directory", config["data"]["base_dir"]),
                  ("Manifest", config["data"]["test_file"]),
                  ("Samples", len(test_loader.dataset))]),
    ]) + "\n")
    model = build_model(config, args.model, device)
    state_dict, ckpt_epoch, best_iou = load_checkpoint_meta(ckpt_path, device)
    model.load_state_dict(state_dict)
    model.eval()

    img_size = config["eval"]["img_size"]
    summary(model, input_size=(1, 3, img_size, img_size))
    print()

    criterion = BCEDiceLoss().to(device)

    test_loss = 0.0
    test_iou, test_dice = 0.0, 0.0
    test_SE, test_PC, test_F1, test_ACC = 0.0, 0.0, 0.0, 0.0

    if args.save_viz:
        os.makedirs("test_results", exist_ok=True)

    eval_start = time.perf_counter()  # 评估计时起点
    with torch.no_grad():
        for i_batch, batch in enumerate(test_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            iou, dice, SE, PC, F1, _, ACC = iou_score(outputs, labels)

            test_loss += loss.item()
            test_iou += iou
            test_dice += dice
            test_SE += SE
            test_PC += PC
            test_F1 += F1
            test_ACC += ACC

            if args.save_viz:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                for idx in range(preds.size(0)):
                    save_path = os.path.join(
                        "test_results", f"batch_{i_batch}_img_{idx}.png"
                    )
                    save_image(preds[idx].cpu(), save_path)

    n = len(test_loader)
    test_loss /= n
    test_iou /= n
    test_dice /= n
    test_SE /= n
    test_PC /= n
    test_F1 /= n
    test_ACC /= n

    eval_elapsed = time.perf_counter() - eval_start

    # Append one readable report per evaluation; the console uses the same text.
    ckpt_dir = os.path.join("checkpoint", args.model)
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = "checkpoint"
        os.makedirs(ckpt_dir, exist_ok=True)
    eval_log = os.path.join(ckpt_dir, f"{args.model}_eval.log")
    task_elapsed = time.perf_counter() - task_start
    report = format_tree(f"Evaluation [{timestamp()}]", [
        ("Model", [
            ("Name", args.model),
            ("Checkpoint", ckpt_path),
            ("Checkpoint epoch", ckpt_epoch),
            ("Best validation IoU", f"{best_iou:.4f}"),
        ]),
        ("Data", [
            ("Directory", config["data"]["base_dir"]),
            ("Manifest", config["data"]["test_file"]),
            ("Samples", len(test_loader.dataset)),
            ("Mask target", config["data"]["mask_target"]),
            ("Input shape", f"N x 3 x {img_size} x {img_size}"),
            ("Batch size", config["eval"]["batch_size"]),
        ]),
        ("Runtime", [("Device", device), ("GPU selection", args.gpu)]),
        ("Results", [
            ("test_loss", f"{test_loss:.4f}"),
            ("test_iou", f"{test_iou:.4f}"),
            ("test_dice", f"{test_dice:.4f}"),
            ("test_SE", f"{test_SE:.4f}"),
            ("test_PC", f"{test_PC:.4f}"),
            ("test_F1", f"{test_F1:.4f}"),
            ("test_ACC", f"{test_ACC:.4f}"),
        ]),
        ("Timing", [
            ("Evaluation loop", format_duration(eval_elapsed)),
            ("Total task", format_duration(task_elapsed)),
        ]),
        ("Outputs", [
            ("Log file", eval_log),
            ("Visualizations", "test_results/" if args.save_viz else "Disabled"),
        ]),
    ])
    with open(eval_log, "a", encoding="utf-8") as f:
        f.write(report + "\n\n\n")  # Two empty lines between evaluations.
    print("\n" + report + "\n\n")


if __name__ == "__main__":
    main()
