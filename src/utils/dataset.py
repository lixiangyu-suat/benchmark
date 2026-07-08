import os

import cv2
import torch
from albumentations import RandomRotate90, Resize
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from torch.utils.data import DataLoader, Dataset


# ── Transforms ──────────────────────────────────────────────────

def val_transform(img_size):
    return Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])


def train_transform(img_size):
    return Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])


# ── Dataset ─────────────────────────────────────────────────────

class MedicalDataset(Dataset):
    """Load medical images and masks from a file list.

    Expects ``base_dir/images/*.png`` and ``base_dir/masks/0/*.png``.
    The file list (train or val) is a plain text file of image stems,
    one per line.
    """

    def __init__(self, base_dir, split, transform, file_list):
        """
        split : "train" | "val"
        file_list : path to the .txt file with image stems
        """
        self._base_dir = base_dir
        self.transform = transform

        with open(os.path.join(base_dir, file_list), "r") as f:
            self.sample_list = [line.strip() for line in f if line.strip()]

        print(f"MedicalDataset [{split}]: {len(self.sample_list)} samples")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        case_name = os.path.splitext(os.path.basename(case))[0]

        img_path = os.path.join(self._base_dir, "images", case + ".png")
        msk_path = os.path.join(self._base_dir, "masks", "0", case + ".png")

        image = cv2.imread(img_path)
        label = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)[..., None]

        augmented = self.transform(image=image, mask=label)
        image = augmented["image"].astype("float32") / 255
        label = augmented["mask"].astype("float32") / 255

        # Convert from HWC to CHW.
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)

        return {"image": image, "label": label, "idx": idx, "name": case_name}


# ── Distillation wrapper ────────────────────────────────────────

class DistillationDataset(Dataset):
    """Wrap a MedicalDataset and attach pre-computed teacher probabilities."""

    def __init__(self, base_dataset, teacher_prob_dir):
        self.base_dataset = base_dataset
        self.teacher_prob_dir = teacher_prob_dir

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        prob_path = os.path.join(self.teacher_prob_dir, f"{sample['name']}.pt")
        sample["teacher_prob"] = torch.load(prob_path, weights_only=True)
        return sample


# ── Loader factory ──────────────────────────────────────────────

def get_dataloaders(config):
    """Return (train_loader, val_loader) for the training pipeline.

    config must contain:
      data.base_dir, data.train_file_dir, data.val_file_dir
      train.img_size, train.batch_size
    """
    base_dir = config["data"]["base_dir"]
    train_file = config["data"]["train_file_dir"]
    val_file = config["data"]["val_file_dir"]
    img_size = config["train"]["img_size"]
    batch_size = config["train"]["batch_size"]

    # SwinUnet internally forces 224.
    if _extract_model_name_from_cfg(config) == "SwinUnet":
        img_size = 224

    train_set = MedicalDataset(base_dir, "train", train_transform(img_size), train_file)
    val_set = MedicalDataset(base_dir, "val", val_transform(img_size), val_file)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_loader, val_loader


def get_val_loader(config):
    """Return a single DataLoader for evaluation."""
    base_dir = config["data"]["base_dir"]
    val_file = config["data"]["val_file_dir"]
    img_size = config["eval"]["img_size"]
    batch_size = config["eval"]["batch_size"]

    val_set = MedicalDataset(base_dir, "val", val_transform(img_size), val_file)
    return DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)


def _extract_model_name_from_cfg(config):
    """Helper: get model name from anywhere it might be in config."""
    # Not very clean, but keeps backward compat with the old inferred field.
    return config.get("model", {}).get("name", "")

