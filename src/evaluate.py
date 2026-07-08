import argparse
import os
import sys

_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
from torchinfo import summary

from src.utils.config import load_config
from src.utils.dataset import get_val_loader
from src.utils.losses import BCEDiceLoss
from src.utils.metrics import iou_score
from src.utils.model_loader import build_model, load_checkpoint_meta, resolve_ckpt_path
from torchvision.utils import save_image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a segmentation model (architecture + metrics)")
    parser.add_argument("--model", type=str, required=True,
                        help="Checkpoint stem (e.g. UNet_model_2026-07-04_22_09_42)")
    parser.add_argument("--cfg", type=str, default="configs/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--save_viz", action="store_true",
                        help="Save prediction visualisations to validation_results/")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.cfg)
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

    if args.save_viz:
        print(f"Visualisations saved to validation_results/")


if __name__ == "__main__":
    main()
