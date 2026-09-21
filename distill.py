import argparse
import os
import signal
import sys
import shutil
import time

_PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.config import add_common_args, add_train_args, build_config
from src.utils.dataset import (DistillationDataset, MedicalDataset,
                                load_split_ids,
                                train_transform, val_transform)
from src.utils.helpers import seed_everything, timestamp_short, format_duration
from src.utils.losses import BCEDiceLoss
from src.utils.logger import CheckpointLogger
from src.utils.model_loader import (
    build_model,
    load_checkpoint_meta,
    save_checkpoint,
    validate_model,
    resolve_ckpt_path,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="知识蒸馏（Teacher 软标签 + Student 多任务优化）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--teacher", type=str, required=True,
                        help="Teacher 检查点 stem")
    parser.add_argument("--student", type=str, required=True,
                        help="Student 架构名（全新训练）或检查点 stem（配合 --resume）")
    parser.add_argument("--resume", action="store_true",
                        help="将 --student 视为检查点 stem，加载权重续训")
    add_common_args(parser)
    add_train_args(parser)
    return parser.parse_args()


@torch.no_grad()
def generate_teacher_probs(config, teacher_model, save_dir):
    teacher_model.eval()
    device = next(teacher_model.parameters()).device

    img_size = config["train"]["img_size"]
    base_dir = config["data"]["base_dir"]
    train_ids, _ = load_split_ids(
        base_dir, config["data"]["seed"], config["data"].get("val_split", 0.3)
    )

    loader = DataLoader(
        MedicalDataset(base_dir, "train", val_transform(img_size), train_ids),
        batch_size=config["train"]["batch_size"],
        shuffle=False, num_workers=4, pin_memory=True,
    )

    os.makedirs(save_dir, exist_ok=True)
    for batch in loader:
        images = batch["image"].to(device)
        names = batch["name"]
        outputs = teacher_model(images)
        probs = torch.sigmoid(outputs).cpu()
        for i, name in enumerate(names):
            torch.save(probs[i], os.path.join(save_dir, f"{name}.pt"))

    return save_dir


def train_student(config, student_model, teacher_prob_dir, logger,
                  start_epoch=1):
    device = next(student_model.parameters()).device
    student_model.train()

    img_size = config["train"]["img_size"]
    base_dir = config["data"]["base_dir"]
    train_ids, _ = load_split_ids(
        base_dir, config["data"]["seed"], config["data"].get("val_split", 0.3)
    )
    batch_size = config["train"]["batch_size"]
    base_lr = config["train"]["base_lr"]
    add_epochs = config["train"]["epoch"]

    end_epoch = start_epoch + add_epochs - 1

    base_dataset = MedicalDataset(base_dir, "train",
                                  train_transform(img_size), train_ids)
    distill_dataset = DistillationDataset(base_dataset, teacher_prob_dir)
    train_loader = DataLoader(
        distill_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )

    hard_loss_fn = BCEDiceLoss().to(device)
    print(f"{len(train_loader)} iterations per epoch")
    print(f"Epochs: {start_epoch} -> {end_epoch}  ({add_epochs} runs)\n")

    distill_loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(student_model.parameters(), lr=base_lr)

    best_loss = float("inf")
    interrupted = False
    last_completed_epoch = start_epoch - 1

    def _on_interrupt(sig, frame):
        nonlocal interrupted
        if interrupted:
            sys.exit(1)
        interrupted = True
        print("\n\n\u26a0  Caught Ctrl+C \u2014 will save and exit after this epoch...")

    signal.signal(signal.SIGINT, _on_interrupt)

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            if interrupted:
                break

            epoch_loss = 0.0
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                teacher_probs = batch["teacher_prob"].to(device)

                optimizer.zero_grad()
                outputs = student_model(images)
                student_probs = torch.sigmoid(outputs)

                hard_loss = hard_loss_fn(outputs, labels)
                distill_loss = distill_loss_fn(student_probs, teacher_probs)
                loss = hard_loss + 0.5 * distill_loss

                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            epoch_loss /= len(train_loader)
            line = f"Epoch [{epoch}/{end_epoch}]  distill_loss: {epoch_loss:.4f}"
            print(line)
            logger.log_training(line)

            if epoch_loss < best_loss:
                best_loss = epoch_loss

            last_completed_epoch = epoch

    except KeyboardInterrupt:
        interrupted = True
        print("\n\u26a0  Caught KeyboardInterrupt.")

    return best_loss, interrupted, last_completed_epoch


def main():
    task_start = time.perf_counter()  # 任务总计时起点
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    config = build_config(args)

    seed_everything(config["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading teacher from: {args.teacher}")
    teacher = build_model(config, args.teacher, device)
    teacher_sd, _, _ = load_checkpoint_meta(
        resolve_ckpt_path(args.teacher), device)
    teacher.load_state_dict(teacher_sd)

    if args.resume:
        print(f"\nResuming student from: {args.student}")
        student_model_name = validate_model(args.student)
        student = build_model(config, args.student, device)
        student_sd, ckpt_epoch, _ = load_checkpoint_meta(
            resolve_ckpt_path(args.student), device)
        student.load_state_dict(student_sd)
        start_epoch = ckpt_epoch + 1

        print(f"=> Student resumed from epoch {ckpt_epoch}")
        ckpt_stem = args.student.replace("_interrupted", "")
        ckpt_new_stem = f"{timestamp_short()}_{student_model_name}"
        os.makedirs(os.path.join("checkpoint", ckpt_stem), exist_ok=True)
    else:
        print(f"\nBuilding student from architecture: {args.student}")
        student_model_name = validate_model(args.student)
        student = build_model(config, args.student, device)
        start_epoch = 1
        ckpt_stem = f"{timestamp_short()}_{student_model_name}"
        ckpt_new_stem = ckpt_stem
        os.makedirs(os.path.join("checkpoint", ckpt_stem), exist_ok=True)


    teacher_prob_dir = generate_teacher_probs(config, teacher, "teacher_probs")
    print(f"Teacher probabilities saved to {teacher_prob_dir}")

    log_path = os.path.join("checkpoint", ckpt_stem, f"{ckpt_new_stem}.log")
    logger = CheckpointLogger(log_path)
    logger.log_pretrain(config)
    custom_msg = config.get("log", {}).get("custom_message", "")
    logger.log_custom_message(f"{custom_msg}\n(distillation: teacher={args.teacher})")

    distill_start = time.perf_counter()  # 蒸馏训练计时起点
    best_loss, interrupted, last_epoch = train_student(
        config, student, teacher_prob_dir, logger,
        start_epoch=start_epoch,
    )
    train_elapsed = time.perf_counter() - distill_start

    if interrupted:
        logger.log_training("--- DISTILLATION INTERRUPTED ---")
        logger.log_training(f"Last completed epoch: {last_epoch}")

    # Final checkpoint save
    final_ckpt = os.path.join("checkpoint", ckpt_stem, f"{ckpt_stem}.pth")
    save_checkpoint(final_ckpt, student, last_epoch, best_loss)

    if args.resume:
        # Resume: rename T0.pth -> T1.pth, rename folder T0 -> T1
        t1_pth = os.path.join("checkpoint", ckpt_stem, f"{ckpt_new_stem}.pth")
        os.rename(final_ckpt, t1_pth)
        old_dir = os.path.join("checkpoint", ckpt_stem)
        new_dir = os.path.join("checkpoint", ckpt_new_stem)
        os.rename(old_dir, new_dir)
        log_path = os.path.join("checkpoint", ckpt_new_stem, f"{ckpt_new_stem}.log")
        logger.log_path = log_path
        print(f"=> Renamed checkpoint folder: {ckpt_stem} -> {ckpt_new_stem}")
    elif interrupted:
        print(f"=> Saved checkpoint: {ckpt_stem}/{ckpt_stem}.pth")
    else:
        print(f"=> Distillation finished (best loss: {best_loss:.4f})")
    shutil.rmtree("teacher_probs", ignore_errors=True)
    logger.log_architecture(student,
                            (1, 3, config["train"]["img_size"],
                             config["train"]["img_size"]))
    task_elapsed = time.perf_counter() - task_start
    timing_lines = [
        f"Distillation time (train loop): {format_duration(train_elapsed)}",
        f"Total task time (whole script): {format_duration(task_elapsed)}",
    ]
    for tl in timing_lines:
        print(f"=> {tl}")
        logger.log_training(tl)
    logger.log_posttrain(best_loss, {"distill_loss": best_loss})
    logger.flush()
    print(f"=> Log saved to {log_path}")


if __name__ == "__main__":
    main()
