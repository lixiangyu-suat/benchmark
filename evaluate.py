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
from src.utils.dataset import get_val_loader
from src.utils.helpers import format_duration, timestamp
from src.utils.losses import BCEDiceLoss
from src.utils.metrics import iou_score
from src.utils.model_loader import build_model, load_checkpoint_meta, resolve_ckpt_path
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="评估分割模型（架构摘要 + 指标）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        help="检查点 stem（如 20260708_1624_U_Net）")
    parser.add_argument("--img_size", type=int, default=256,
                        help="评估输入分辨率 (SwinUnet 固定 224)")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="评估批大小")
    parser.add_argument("--save_viz", action="store_true",
                        help="保存预测掩码可视化到 validation_results/")
    add_common_args(parser)
    return parser.parse_args()


def main():
    task_start = time.perf_counter()  # 任务总计时起点
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = build_config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Model: {args.model}")
    print(f"{'='*60}\n")

    model = build_model(config, args.model, device)
    state_dict, _, _ = load_checkpoint_meta(
        resolve_ckpt_path(args.model), device)
    model.load_state_dict(state_dict)
    model.eval()

    img_size = config["eval"]["img_size"]
    summary(model, input_size=(1, 3, img_size, img_size))
    print()

    val_loader = get_val_loader(config)
    criterion = BCEDiceLoss().to(device)

    val_loss = 0.0
    val_iou, val_dice = 0.0, 0.0
    val_SE, val_PC, val_F1, val_ACC = 0.0, 0.0, 0.0, 0.0

    if args.save_viz:
        os.makedirs("validation_results", exist_ok=True)

    eval_start = time.perf_counter()  # 评估计时起点
    with torch.no_grad():
        for i_batch, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            iou, dice, SE, PC, F1, _, ACC = iou_score(outputs, labels)

            val_loss += loss.item()
            val_iou += iou
            val_dice += dice
            val_SE += SE
            val_PC += PC
            val_F1 += F1
            val_ACC += ACC

            if args.save_viz:
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                for idx in range(preds.size(0)):
                    save_path = os.path.join(
                        "validation_results", f"batch_{i_batch}_img_{idx}.png"
                    )
                    save_image(preds[idx].cpu(), save_path)

    n = len(val_loader)
    val_loss /= n
    val_iou /= n
    val_dice /= n
    val_SE /= n
    val_PC /= n
    val_F1 /= n
    val_ACC /= n

    print(f"{'='*60}")
    print(f"Evaluation Results")
    print(f"{'='*60}")
    print(f"  val_loss:  {val_loss:.4f}")
    print(f"  val_iou:   {val_iou:.4f}")
    print(f"  val_dice:  {val_dice:.4f}")
    print(f"  val_SE:    {val_SE:.4f}")
    print(f"  val_PC:    {val_PC:.4f}")
    print(f"  val_F1:    {val_F1:.4f}")
    print(f"  val_ACC:   {val_ACC:.4f}")
    print(f"{'='*60}\n")

    eval_elapsed = time.perf_counter() - eval_start
    task_elapsed = time.perf_counter() - task_start
    print(f"=> Evaluation time (val loop): {format_duration(eval_elapsed)}")
    print(f"=> Total task time (whole script): {format_duration(task_elapsed)}")

    # ===== 评估结果与耗时追加写入检查点目录下的 <stem>_eval.log =====
    ckpt_dir = os.path.join("checkpoint", args.model)
    if not os.path.isdir(ckpt_dir):
        ckpt_dir = "checkpoint"
        os.makedirs(ckpt_dir, exist_ok=True)
    eval_log = os.path.join(ckpt_dir, f"{args.model}_eval.log")
    with open(eval_log, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp()}] model={args.model} "
                f"data_dir={config['data']['base_dir']} "
                f"img_size={img_size} batch_size={config['eval']['batch_size']}\n")
        f.write(f"  val_loss: {val_loss:.4f}  val_iou: {val_iou:.4f}  "
                f"val_dice: {val_dice:.4f}  val_SE: {val_SE:.4f}  "
                f"val_PC: {val_PC:.4f}  val_F1: {val_F1:.4f}  val_ACC: {val_ACC:.4f}\n")
        f.write(f"  Evaluation time: {format_duration(eval_elapsed)}  "
                f"Total task time: {format_duration(task_elapsed)}\n\n")
    print(f"=> Eval log appended to {eval_log}")

    if args.save_viz:
        print(f"Visualisations saved to validation_results/")


if __name__ == "__main__":
    main()
