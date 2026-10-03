import argparse
import os
import signal
import sys
import time

# ===== 确保项目根目录在 sys.path 上（支持从任意目录直接运行本脚本） =====
_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import torch.optim as optim
from tqdm import tqdm

from src.utils.config import add_common_args, add_train_args, build_config
from src.utils.dataset import get_dataloaders
from src.utils.helpers import AverageMeter, format_duration, seed_everything, timestamp_short
from src.utils.losses import BCEDiceLoss
from src.utils.logger import CheckpointLogger
from src.utils.metrics import iou_score
from src.utils.model_loader import (
    build_model,
    load_checkpoint_meta,
    save_checkpoint,
    validate_model,
    resolve_ckpt_path,
)
from src.utils.convert_to_onnx import convert_pth_to_onnx


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练医学图像分割模型",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, default=None,
                        help="模型架构名（如 U_Net），全新训练时必填；"
                             "--ckpt 续训时自动从检查点名推断，无需传入")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="断点续训的检查点 stem（如 20260708_1624_U_Net）")
    add_common_args(parser)
    add_train_args(parser)
    args = parser.parse_args()
    if args.model is None and args.ckpt is None:
        parser.error("必须指定 --model（全新训练）或 --ckpt（断点续训）之一")
    return args


def main():
    task_start = time.perf_counter()  # 任务总计时起点
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = build_config(args)

    seed_everything(config["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== 断点续训 or 全新训练 =====
    if args.ckpt is not None:
        model_name = validate_model(args.ckpt)
        model = build_model(config, args.ckpt, device)
        ckpt_path = resolve_ckpt_path(args.ckpt)
        sd, ckpt_epoch, best_iou = load_checkpoint_meta(ckpt_path, device)
        model.load_state_dict(sd)
        start_epoch = ckpt_epoch + 1

        print(f"=> Resumed from epoch {ckpt_epoch}  (best_iou: {best_iou:.4f})")
        ckpt_stem = args.ckpt.replace("_interrupted", "")
        ckpt_new_stem = f"{timestamp_short()}_{model_name}"
        os.makedirs(os.path.join("checkpoint", ckpt_stem), exist_ok=True)
    else:
        model_name = validate_model(args.model)
        model = build_model(config, args.model, device)
        start_epoch = 1
        best_iou = float("-inf")  # Save the first validated epoch even if IoU is zero.
        # 全新训练：创建全新的时间戳目录
        ckpt_stem = f"{timestamp_short()}_{model_name}"
        ckpt_new_stem = ckpt_stem
        os.makedirs(os.path.join("checkpoint", ckpt_stem), exist_ok=True)

    ckpt_best_path = os.path.join("checkpoint", ckpt_stem, f"{ckpt_stem}_best.pth")
    train_loader, val_loader = get_dataloaders(config)

    base_lr = config["train"]["base_lr"]
    add_epochs = config["train"]["epoch"]
    end_epoch = start_epoch + add_epochs - 1
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    criterion = BCEDiceLoss().to(device)

    # ===== 日志器 =====
    log_path = os.path.join("checkpoint", ckpt_stem, f"{ckpt_new_stem}.log")
    logger = CheckpointLogger(log_path)
    logger.log_pretrain(config)
    logger.log_custom_message(config.get("log", {}).get("custom_message", ""))
    logger.log_architecture(model, (1, 3, config["train"]["img_size"],
                                    config["train"]["img_size"]))

    print(f"{len(train_loader)} iterations per epoch")
    print(f"Epochs: {start_epoch} -> {end_epoch}  ({add_epochs} runs)\n")

    iter_num = 0
    max_iters = len(train_loader) * add_epochs

    interrupted = False

    def _on_interrupt(sig, frame):
        nonlocal interrupted
        if interrupted:
            sys.exit(1)
        interrupted = True

    signal.signal(signal.SIGINT, _on_interrupt)

    # ===== 训练主循环 =====
    last_completed_epoch = start_epoch - 1
    train_start = time.perf_counter()  # 训练计时起点

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            if interrupted:
                print("\n⚠  Interrupt received, finishing current epoch then saving...")
                break

            model.train()
            meters = {k: AverageMeter() for k in [
                "loss", "iou",
                "val_loss", "val_iou", "val_SE", "val_PC", "val_F1", "val_ACC",
            ]}

            with tqdm(train_loader, desc=f"Epoch {epoch}/{end_epoch} train",
                      unit="batch", dynamic_ncols=True, leave=False) as train_progress:
                for batch in train_progress:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    iou, _, _, _, _, _, _ = iou_score(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    iter_num += 1
                    lr = base_lr * (1.0 - iter_num / max_iters) ** 0.9
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                    meters["loss"].update(loss.item(), images.size(0))
                    meters["iou"].update(iou, images.size(0))
                    train_progress.set_postfix(
                        loss=f"{meters['loss'].avg:.4f}",
                        iou=f"{meters['iou'].avg:.4f}", lr=f"{lr:.2e}",
                        refresh=False,
                    )

            model.eval()
            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch {epoch}/{end_epoch} val",
                          unit="batch", dynamic_ncols=True, leave=False) as val_progress:
                    for batch in val_progress:
                        images = batch["image"].to(device)
                        labels = batch["label"].to(device)
                        outputs = model(images)

                        loss = criterion(outputs, labels)
                        iou, _, SE, PC, F1, _, ACC = iou_score(outputs, labels)

                        meters["val_loss"].update(loss.item(), images.size(0))
                        meters["val_iou"].update(iou, images.size(0))
                        meters["val_SE"].update(SE, images.size(0))
                        meters["val_PC"].update(PC, images.size(0))
                        meters["val_F1"].update(F1, images.size(0))
                        meters["val_ACC"].update(ACC, images.size(0))
                        val_progress.set_postfix(
                            loss=f"{meters['val_loss'].avg:.4f}",
                            iou=f"{meters['val_iou'].avg:.4f}", refresh=False,
                        )

            line = (
                f"Epoch Result[{epoch}/{end_epoch}]  "
                f"train_loss: {meters['loss'].avg:.4f}  "
                f"train_iou: {meters['iou'].avg:.4f}  |  "
                f"val_loss: {meters['val_loss'].avg:.4f}  "
                f"val_iou: {meters['val_iou'].avg:.4f}  "
                f"val_SE: {meters['val_SE'].avg:.4f}  "
                f"val_PC: {meters['val_PC'].avg:.4f}  "
                f"val_F1: {meters['val_F1'].avg:.4f}  "
                f"val_ACC: {meters['val_ACC'].avg:.4f}"
            )
            print(line)
            logger.log_training(line)

            last_completed_epoch = epoch

            if meters["val_iou"].avg > best_iou:
                best_iou = meters["val_iou"].avg
                save_checkpoint(
                    ckpt_best_path, model, epoch, best_iou,
                )
                print(f"  => saved best model (val_iou: {best_iou:.4f})")

    except KeyboardInterrupt:
        interrupted = True
        print("\n⚠  Caught KeyboardInterrupt.")

    # ===== 收尾：保存最终权重、重命名（续训场景）、导出 ONNX =====
    train_elapsed = time.perf_counter() - train_start

    if interrupted:
        logger.log_training("--- TRAINING INTERRUPTED ---")
        logger.log_training(f"Last completed epoch: {last_completed_epoch}")

    # 保存最终检查点 (.pth)
    final_ckpt = os.path.join("checkpoint", ckpt_stem, f"{ckpt_stem}_final.pth")
    save_checkpoint(final_ckpt, model, last_completed_epoch, best_iou)

    if args.ckpt is not None:
        # 续训：T0.pth -> T1.pth，目录 T0 -> T1 原子重命名
        t1_pth = os.path.join("checkpoint", ckpt_stem, f"{ckpt_new_stem}_final.pth")
        os.rename(final_ckpt, t1_pth)
        # Keep the best checkpoint resolvable under the new run stem, even when
        # resumed epochs never improve on the previous best.
        best_new_path = os.path.join("checkpoint", ckpt_stem, f"{ckpt_new_stem}_best.pth")
        if os.path.isfile(ckpt_best_path):
            os.rename(ckpt_best_path, best_new_path)
        old_dir = os.path.join("checkpoint", ckpt_stem)
        new_dir = os.path.join("checkpoint", ckpt_new_stem)
        os.rename(old_dir, new_dir)
        final_ckpt = os.path.join(new_dir, os.path.basename(t1_pth))
        ckpt_best_path = os.path.join(new_dir, os.path.basename(best_new_path))
        final_dir = new_dir
        log_path = os.path.join(new_dir, f"{ckpt_new_stem}.log")
        logger.log_path = log_path
        print(f"=> Renamed checkpoint folder: {ckpt_stem} -> {ckpt_new_stem}")
    else:
        final_dir = os.path.join("checkpoint", ckpt_stem)
        if interrupted:
            print(f"=> Saved checkpoint: {ckpt_stem}/{ckpt_stem}_final.pth "
                  f"(best: {ckpt_stem}/{ckpt_stem}_best.pth)")
        else:
            print(f"=> Training finished (best val_iou: {best_iou:.4f})")

    # 仅导出验证 IoU 最佳权重；final.pth 保留用于训练记录。
    best_onnx = os.path.join(final_dir, f"{ckpt_new_stem}_best.onnx")
    onnx_img_size = 224 if model_name == "SwinUnet" else config["train"]["img_size"]
    try:
        if not os.path.isfile(ckpt_best_path):
            raise FileNotFoundError(f"No best checkpoint available: {ckpt_best_path}; "
                                    "best ONNX export skipped")
        convert_pth_to_onnx(
            model=model,
            pth_path=ckpt_best_path,
            onnx_path=best_onnx,
            dummy_input=torch.randn(1, 3, onnx_img_size, onnx_img_size),
            dynamic_axes={"input": {0: "batch_size"},
                          "output": {0: "batch_size"}},  # 动态 batch
        )
    except Exception as e:  # 导出失败不影响已保存的 .pth 与日志
        print(f"⚠  ONNX 导出失败（.pth 权重不受影响）: {e}")

    final_metrics = {k: meters[k].avg for k in
                     ["val_loss", "val_iou", "val_SE", "val_PC", "val_F1", "val_ACC"]}
    task_elapsed = time.perf_counter() - task_start
    timing_lines = [
        f"Training time (train loop): {format_duration(train_elapsed)}",
        f"Total task time (whole script): {format_duration(task_elapsed)}",
    ]
    for tl in timing_lines:
        print(f"=> {tl}")
        logger.log_training(tl)
    logger.log_posttrain(best_iou, final_metrics)
    logger.flush()
    print(f"=> Log saved to {log_path}")


if __name__ == "__main__":
    main()
