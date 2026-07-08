import argparse
import os
import signal
import sys

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.dataset import (DistillationDataset, MedicalDataset,
                                train_transform, val_transform)
from src.utils.helpers import seed_everything, timestamp
from src.utils.losses import BCEDiceLoss
from src.utils.logger import CheckpointLogger
from src.utils.model_loader import (
    build_model,
    load_checkpoint_meta,
    save_checkpoint,
    validate_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge distillation")
    parser.add_argument("--teacher", type=str, required=True,
                        help="Teacher checkpoint stem")
    parser.add_argument("--student", type=str, required=True,
                        help="Student architecture name or checkpoint stem")
    parser.add_argument("--cfg", type=str, default="configs/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--resume", action="store_true",
                        help="Treat --student as a checkpoint stem (load weights)")
    return parser.parse_args()


@torch.no_grad()
def generate_teacher_probs(config, teacher_model, save_dir):
    teacher_model.eval()
    device = next(teacher_model.parameters()).device

    img_size = config["train"]["img_size"]
    base_dir = config["data"]["base_dir"]
    train_file = config["data"]["train_file_dir"]

    loader = DataLoader(
        MedicalDataset(base_dir, "train", val_transform(img_size), train_file),
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
    train_file = config["data"]["train_file_dir"]
    batch_size = config["train"]["batch_size"]
    base_lr = config["train"]["base_lr"]
    add_epochs = config["train"]["epoch"]

    end_epoch = start_epoch + add_epochs - 1

    base_dataset = MedicalDataset(base_dir, "train",
                                  train_transform(img_size), train_file)
    distill_dataset = DistillationDataset(base_dataset, teacher_prob_dir)
    train_loader = DataLoader(
        distill_dataset, batch_size=batch_size,
        shuffle=True, num_workers=4, pin_memory=True,
    )

    hard_loss_fn = BCEDiceLoss().to(device)
    distill_loss_fn = torch.nn.MSELoss()
    optimizer = optim.Adam(student_model.parameters(), lr=base_lr)

    best_loss = float("inf")
    interrupted = False

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

    except KeyboardInterrupt:
        interrupted = True
        print("\n\u26a0  Caught KeyboardInterrupt.")

    return best_loss, interrupted, epoch


def main():
    args = parse_args()
    config = load_config(args.cfg)

    seed_everything(config["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading teacher from: {args.teacher}")
    teacher = build_model(config, args.teacher, device)
    teacher_sd, _, _ = load_checkpoint_meta(
        f"./checkpoint/{args.teacher}.pth", device)
    teacher.load_state_dict(teacher_sd)

    if args.resume:
        print(f"Resuming student from: {args.student}")
        student_model_name = validate_model(args.student)
        student = build_model(config, args.student, device)
        student_sd, ckpt_epoch, _ = load_checkpoint_meta(
            f"./checkpoint/{args.student}.pth", device)
        student.load_state_dict(student_sd)
        start_epoch = ckpt_epoch + 1

        print(f"=> Student resumed from epoch {ckpt_epoch}")
    else:
        print(f"Building student from architecture: {args.student}")
        student_model_name = validate_model(args.student)
        student = build_model(config, args.student, device)
        start_epoch = 1


    teacher_prob_dir = generate_teacher_probs(config, teacher, "teacher_probs")
    print(f"Teacher probabilities saved to {teacher_prob_dir}")

    ckpt_stem = f"{student_model_name}_model_{timestamp()}"
    log_path = os.path.join("checkpoint", f"{ckpt_stem}.log")
    logger = CheckpointLogger(log_path)
    logger.log_pretrain(config)
    custom_msg = config.get("log", {}).get("custom_message", "")
    logger.log_custom_message(f"{custom_msg}\n(distillation: teacher={args.teacher})")

    best_loss, interrupted, last_epoch = train_student(
        config, student, teacher_prob_dir, logger,
        start_epoch=start_epoch,
    )

    os.makedirs("checkpoint", exist_ok=True)
    if interrupted:
        ckpt_path = os.path.join("checkpoint", f"{ckpt_stem}_interrupted.pth")
        logger.log_training("--- DISTILLATION INTERRUPTED ---")
        logger.log_training(f"Last completed epoch: {last_epoch}")
        print(f"=> Interrupted, saving to {ckpt_stem}_interrupted.pth")
    else:
        ckpt_path = os.path.join("checkpoint", f"{ckpt_stem}.pth")
        print(f"=> Distillation finished (best loss: {best_loss:.4f})")

    save_checkpoint(ckpt_path, student, last_epoch, best_loss)
    logger.log_architecture(student,
                            (1, 3, config["train"]["img_size"],
                             config["train"]["img_size"]))
    logger.log_posttrain(best_loss, {"distill_loss": best_loss})
    logger.flush()
    print(f"=> Log saved to {log_path}")


if __name__ == "__main__":
    main()
