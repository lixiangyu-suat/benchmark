import os
import json
from glob import glob

import cv2
import torch
from albumentations import RandomRotate90, Resize, Flip, Normalize
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


# ======== Transforms ========

def train_transform(img_size):
    """Training transforms: random 90° rotation, random H/V flip, resize, ImageNet normalization."""
    return Compose([
        RandomRotate90(),
        Flip(),
        Resize(img_size, img_size),
        Normalize(),
    ])


def val_transform(img_size):
    """Validation transforms: resize + ImageNet normalization (no augmentation)."""
    return Compose([
        Resize(img_size, img_size),
        Normalize(),
    ])


# ======== Dataset ========

class MedicalDataset(Dataset):
    """Load medical images and masks from a file list.

    ``base_dir/images/*.png`` and ``base_dir/masks/0/*.png``.
    ``sample_ids`` is a list of image stems (without extension).
    """
    def __init__(self, base_dir, split, transform, sample_ids):
        """
        split : "train" | "val"
        sample_ids : list of image stems (without extension)
        """
        self._base_dir = base_dir
        self.transform = transform
        self.sample_list = sample_ids
        print(f"MedicalDataset [{split}]: {len(self.sample_list)} samples")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        img_path = os.path.join(self._base_dir, "images", f"{case}.png")
        msk_path = os.path.join(self._base_dir, "masks", "0", f"{case}.png")
        image = cv2.imread(img_path)
        label = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)[..., None]
        augmented = self.transform(image=image, mask=label)
        image = augmented["image"].astype("float32") / 255.0
        label = augmented["mask"].astype("float32") / 255.0
        image = image.transpose(2, 0, 1)
        label = label.transpose(2, 0, 1)
        return {"image": image, "label": label, "idx": idx, "name": case}


# ======== Distillation wrapper ========

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


# ======== Split helpers ========

def load_txt_ids(base_dir, file_name):
    """Load sample IDs from a text file (one stem per line, no extension)."""
    path = os.path.join(base_dir, file_name)
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_split_ids(base_dir, seed, val_split=0.3):
    """Return (train_ids, val_ids), auto-persisted to a JSON file.

    On first call, scans ``base_dir/images/*.png``, creates the split,
    and saves ``{dataset_name}_split.json`` next to the data.
    Subsequent calls reuse the saved split as long as the seed matches.

    This is a fallback when ``train_file`` / ``val_file`` are not provided.
    """
    dataset_name = os.path.basename(base_dir.rstrip("/\\"))
    split_file = os.path.join(base_dir, f"{dataset_name}_split.json")
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            saved = json.load(f)
        if saved["seed"] == seed:
            return saved["train"], saved["val"]
    img_paths = glob(os.path.join(base_dir, "images", "*.png"))
    if not img_paths:
        raise FileNotFoundError(f"No .png images found in {os.path.join(base_dir, 'images')}")
    all_ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in img_paths)
    train_ids, val_ids = train_test_split(all_ids, test_size=val_split, random_state=seed)
    with open(split_file, "w") as f:
        json.dump({"seed": seed, "train": train_ids, "val": val_ids}, f, indent=2)
    return train_ids, val_ids


def _resolve_split_ids(base_dir, config):
    """Return (train_ids, val_ids) from txt files (preferred) or auto-split (fallback)."""
    train_file = config["data"].get("train_file", "")
    val_file = config["data"].get("val_file", "")
    if train_file and val_file:
        return load_txt_ids(base_dir, train_file), load_txt_ids(base_dir, val_file)
    seed = config["data"]["seed"]
    val_split = config["data"].get("val_split", 0.3)
    return load_split_ids(base_dir, seed, val_split)


# ======== Loader factories ========

def get_dataloaders(config):
    """Return (train_loader, val_loader) for the training pipeline.

    Expects config keys:
      data.base_dir, data.seed
      train.img_size, train.batch_size
    """
    base_dir = config["data"]["base_dir"]
    img_size = config["train"]["img_size"]
    batch_size = config["train"]["batch_size"]
    train_ids, val_ids = _resolve_split_ids(base_dir, config)
    if _extract_model_name_from_cfg(config) == "SwinUnet":
        img_size = 224
    train_set = MedicalDataset(base_dir, "train", train_transform(img_size), train_ids)
    val_set = MedicalDataset(base_dir, "val", val_transform(img_size), val_ids)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


def get_val_loader(config):
    """Return a single DataLoader for evaluation."""
    base_dir = config["data"]["base_dir"]
    img_size = config["eval"]["img_size"]
    batch_size = config["eval"]["batch_size"]
    _, val_ids = _resolve_split_ids(base_dir, config)
    val_set = MedicalDataset(base_dir, "val", val_transform(img_size), val_ids)
    return DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)


def _extract_model_name_from_cfg(config):
    """Helper: get model name from anywhere it might be in config."""
    return config.get("model", {}).get("name", "")
