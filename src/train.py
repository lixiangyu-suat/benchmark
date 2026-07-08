import argparse
import os
import signal
import sys

# ── Ensure project root is on sys.path ───────────────────
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import torch.optim as optim

from src.utils.config import load_config
from src.utils.dataset import get_dataloaders
from src.utils.helpers import AverageMeter, seed_everything, timestamp
from src.utils.losses import BCEDiceLoss
from src.utils.logger import CheckpointLogger
from src.utils.metrics import iou_score
from src.utils.model_loader import (
    build_model,
    load_checkpoint_meta,
    save_checkpoint,
    validate_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument("--model", type=str, required=True,
                        help="Model architecture name (see configs/modellists.yaml)")
    parser.add_argument("--cfg", type=str, default="configs/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint stem to resume from "
                             "(e.g. UNet_model_2026-07-04_22_09_42)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.cfg)

    seed_everything(config["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Resume or scratch ───────────────────────────────────────
    if args.ckpt is not None:
        model_name = validate_model(args.ckpt)
        model = build_model(config, args.ckpt, device)
        ckpt_path = f"./checkpoint/{args.ckpt}.pth"
        sd, ckpt_epoch, best_iou = load_checkpoint_meta(ckpt_path, device)
        model.load_state_dict(sd)
        start_epoch = ckpt_epoch + 1

        print(f"=> Resumed from epoch {ckpt_epoch}  (best_iou: {best_iou:.4f})")
    else:
        model_name = validate_model(args.model)
        model = build_model(config, args.model, device)
        start_epoch = 1
        best_iou = 0.0


    train_loader, val_loader = get_dataloaders(config)

    base_lr = config["train"]["base_lr"]
    add_epochs = config["train"]["epoch"]
    end_epoch = start_epoch + add_epochs - 1
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    criterion = BCEDiceLoss().to(device)

    # ── Logger ──────────────────────────────────────────────────
    ckpt_stem = f"{model_name}_model_{timestamp()}"
    log_path = os.path.join("checkpoint", f"{ckpt_stem}.log")
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
        print("\n\n\u26a0  Caught Ctrl+C \u2014 will save and exit after this epoch...")

    signal.signal(signal.SIGINT, _on_interrupt)

    # ── Training loop ───────────────────────────────────────────
    try:
        for epoch in range(start_epoch, end_epoch + 1):
            if interrupted:
                break

            model.train()
            meters = {k: AverageMeter() for k in [
                "loss", "iou",
                "val_loss", "val_iou", "val_SE", "val_PC", "val_F1", "val_ACC",
            ]}

            for batch in train_loader:
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

            model.eval()
            with torch.no_grad():
                for batch in val_loader:
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

            line = (
                f"Epoch [{epoch}/{end_epoch}]  "
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

            if meters["val_iou"].avg > best_iou:
                best_iou = meters["val_iou"].avg
                os.makedirs("checkpoint", exist_ok=True)
                save_checkpoint(
                    os.path.join("checkpoint", f"{ckpt_stem}.pth"),
                    model, epoch, best_iou,
                )
                print(f"  => saved best model (val_iou: {best_iou:.4f})")

    except KeyboardInterrupt:
        interrupted = True
        print("\n\u26a0  Caught KeyboardInterrupt.")

    # ── Finalise ────────────────────────────────────────────────
    if interrupted:
        os.makedirs("checkpoint", exist_ok=True)
        save_checkpoint(
            os.path.join("checkpoint", f"{ckpt_stem}_interrupted.pth"),
            model, epoch, best_iou,
        )
        logger.log_training("--- TRAINING INTERRUPTED ---")
        logger.log_training(f"Last completed epoch: {epoch}")
        print(f"=> Saved interrupted checkpoint: {ckpt_stem}_interrupted.pth")
    else:
        print(f"=> Training finished (best val_iou: {best_iou:.4f})")

    final_metrics = {k: meters[k].avg for k in
                     ["val_loss", "val_iou", "val_SE", "val_PC", "val_F1", "val_ACC"]}
    logger.log_posttrain(best_iou, final_metrics)
    logger.flush()
    print(f"=> Log saved to {log_path}")


if __name__ == "__main__":
    main()
