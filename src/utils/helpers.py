import os
import random
from datetime import datetime

import numpy as np
import torch


def seed_everything(seed):
    """Set seeds for reproducibility across all random number generators."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def timestamp():
    """Return current time formatted as YYYY-MM-DD_HH_MM_SS (full precision)."""
    return datetime.now().strftime("%Y-%m-%d_%H_%M_%S")


def timestamp_short():
    """Return current time formatted as YYYYMMDD_HHMM (8+4 digits, sortable).

    Designed for checkpoint naming so that alphabetical order equals
    chronological order.
    """
    return datetime.now().strftime("%Y%m%d_%H%M")


def count_params(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter:
    """Accumulates values and computes running average."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
