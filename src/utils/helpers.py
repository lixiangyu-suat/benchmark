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
    """紧凑时间戳，精确到毫秒，用于 checkpoint 命名。

    格式: YYYYMMDD_HHMM_SSmmm，例如 ``20260921_1639_10213``
    （秒=10，毫秒=213）。字母序即时间序；毫秒位用于避免多 GPU
    独立并发运行时同秒启动产生的 checkpoint 目录名冲突。
    """
    now = datetime.now()
    return now.strftime("%Y%m%d_%H%M_") + f"{now.second:02d}{now.microsecond // 1000:03d}"


def format_duration(seconds: float) -> str:
    """把秒数格式化为 ``HHH:MM:SS.mmm``（小时可超过 24，不归零）。

    例如 20 天 -> ``480:00:01.143``。用于在日志中记录训练/评估/任务耗时。
    """
    ms = int(round(seconds * 1000))
    hours, rem = divmod(ms, 3600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours}:{minutes:02d}:{secs:02d}.{millis:03d}"


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
