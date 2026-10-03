import os
import json
from pathlib import Path

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

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _image_files(directory):
    return sorted(p for p in Path(directory).iterdir()
                  if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def _file_index(directory):
    index = {}
    for path in _image_files(directory):
        index.setdefault(path.stem, []).append(path)
    return index


def _resolve_file(directory, sample, index=None):
    """Resolve a filename or legacy stem; reject missing/ambiguous matches."""
    directory = Path(directory)
    if Path(sample).name != sample or sample in {".", ".."}:
        raise ValueError(f"Expected a filename, got {sample!r}")
    exact = directory / sample
    if exact.is_file() and exact.suffix.lower() in IMAGE_EXTENSIONS:
        return exact
    if Path(sample).suffix.lower() in IMAGE_EXTENSIONS:
        raise FileNotFoundError(f"Image listed in manifest does not exist: {exact}")
    stem = Path(sample).stem if Path(sample).suffix.lower() in IMAGE_EXTENSIONS else sample
    if index is None:
        index = _file_index(directory)
    matches = index.get(stem, [])
    if len(matches) != 1:
        raise ValueError(f"Expected one file for {sample!r} in {directory}; found {matches}")
    return matches[0]


class MedicalDataset(Dataset):
    """Load medical images and masks from a file list.

    Lists contain image filenames (including extension), or legacy stems.
    Masks match image stems and may use a different extension.
    All images live in images/ and all masks in masks/; manifests define splits.
    """
    def __init__(self, base_dir, split, transform, sample_ids, mask_target="binary"):
        """
        split : "train" | "val" | "test"
        sample_ids : list of image filenames or legacy stems
        """
        self._base_dir = base_dir
        self.transform = transform
        if mask_target not in {"binary", "cup"}:
            raise ValueError(f"Unsupported mask_target: {mask_target}")
        self.mask_target = mask_target
        self.sample_list = sample_ids
        self.sample_paths = []
        image_dir = Path(base_dir) / "images"
        mask_dir = Path(base_dir) / "masks"
        image_index = _file_index(image_dir)
        mask_index = _file_index(mask_dir)
        for sample in sample_ids:
            image_path = _resolve_file(image_dir, sample, image_index)
            mask_path = _resolve_file(mask_dir, image_path.stem, mask_index)
            self.sample_paths.append((image_path, mask_path))
        print(f"MedicalDataset [{split}]: {len(self.sample_list)} samples")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        img_path, msk_path = self.sample_paths[idx]
        case = img_path.stem  # Preserve teacher cache identifiers.
        image = cv2.imread(str(img_path))
        label = cv2.imread(str(msk_path), cv2.IMREAD_GRAYSCALE)
        if image is None or label is None:
            raise ValueError(f"Cannot decode image/mask: {img_path}, {msk_path}")
        if image.shape[:2] != label.shape:
            raise ValueError(f"Image/mask dimensions differ: {img_path}, {msk_path}")
        if self.mask_target == "cup":
            if ((label != 0) & (label != 128) & (label != 255)).any():
                raise ValueError(f"Expected REFUGE mask values 0/128/255: {msk_path}")
            label = (label == 0).astype("uint8") * 255
        elif ((label != 0) & (label != 255)).any():
            raise ValueError(f"Expected binary mask values 0/255: {msk_path}; "
                             "use mask_target='cup' for REFUGE optic cup masks")
        label = label[..., None]
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
    """Load image filenames or legacy stems, one per line (spaces allowed)."""
    path = os.path.join(base_dir, file_name)
    with open(path, "r", encoding="utf-8-sig") as f:
        return [line.strip() for line in f if line.strip()]


def load_split_ids(base_dir, seed, val_split=0.3):
    """Return (train_ids, val_ids), auto-persisted to a JSON file.

    On first call, scans supported image files, creates the split,
    and saves ``{dataset_name}_split.json`` next to the data.
    Subsequent calls reuse the saved split when seed, ratio and files match.

    This is a fallback when ``train_file`` / ``val_file`` are not provided.
    """
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")
    all_ids = [p.name for p in _image_files(Path(base_dir) / "images")]
    if not all_ids:
        raise FileNotFoundError(f"No supported images in {os.path.join(base_dir, 'images')}")
    dataset_name = os.path.basename(base_dir.rstrip("/\\"))
    split_file = os.path.join(base_dir, f"{dataset_name}_split.json")
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            saved = json.load(f)
        if (saved["seed"] == seed and saved.get("val_split") == val_split
                and sorted(saved["train"] + saved["val"]) == all_ids):
            return saved["train"], saved["val"]
    train_ids, val_ids = train_test_split(all_ids, test_size=val_split, random_state=seed)
    with open(split_file, "w") as f:
        json.dump({"seed": seed, "val_split": val_split,
                   "train": train_ids, "val": val_ids}, f, indent=2)
    return train_ids, val_ids


def _resolve_split_ids(base_dir, config):
    """Return (train_ids, val_ids) from txt files (preferred) or auto-split (fallback)."""
    train_file = config["data"].get("train_file", "")
    val_file = config["data"].get("val_file", "")
    if bool(train_file) != bool(val_file):
        raise ValueError("Provide both train_file and val_file, or leave both empty")
    if train_file and val_file:
        train_ids = load_txt_ids(base_dir, train_file)
        val_ids = load_txt_ids(base_dir, val_file)
        image_dir = Path(base_dir) / "images"
        image_index = _file_index(image_dir)
        train_names = [_resolve_file(image_dir, s, image_index).name for s in train_ids]
        val_names = [_resolve_file(image_dir, s, image_index).name for s in val_ids]
        if not train_names or not val_names:
            raise ValueError("Training and validation lists must be nonempty")
        if len(set(train_names)) != len(train_names) or len(set(val_names)) != len(val_names):
            raise ValueError("Duplicate samples in training/validation list")
        overlap = set(train_names) & set(val_names)
        if overlap:
            raise ValueError(f"Training/validation overlap: {sorted(overlap)[:10]}")
        return train_names, val_names
    seed = config["data"]["seed"]
    val_split = config["data"].get("val_split", 0.3)
    if config["data"].get("test_file") or any(Path(base_dir).glob("*test*.txt")):
        raise ValueError("A test manifest is present in the shared image pool; "
                         "provide explicit train_file and val_file to keep test samples held out")
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
    mask_target = config["data"].get("mask_target", "binary")
    train_set = MedicalDataset(base_dir, "train", train_transform(img_size), train_ids, mask_target)
    val_set = MedicalDataset(base_dir, "val", val_transform(img_size), val_ids, mask_target)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


def get_test_loader(config):
    """Read an explicit test manifest from the shared images/ directory."""
    base_dir = config["data"]["base_dir"]
    img_size = config["eval"]["img_size"]
    batch_size = config["eval"]["batch_size"]
    test_file = config["data"].get("test_file", "")
    if not test_file:
        raise ValueError("Evaluation requires --test_file, for example test1.txt")
    test_ids = load_txt_ids(base_dir, test_file)
    if not test_ids:
        raise ValueError("Test list must be nonempty")
    image_dir = Path(base_dir) / "images"
    image_index = _file_index(image_dir)
    test_names = [_resolve_file(image_dir, s, image_index).name for s in test_ids]
    if len(test_names) != len(set(test_names)):
        raise ValueError("Duplicate samples in test list")
    test_set = MedicalDataset(base_dir, "test", val_transform(img_size), test_names,
                              config["data"].get("mask_target", "binary"))
    return DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4)


def _extract_model_name_from_cfg(config):
    """Helper: get model name from anywhere it might be in config."""
    return config.get("model", {}).get("name", "")
