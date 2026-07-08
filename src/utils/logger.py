import os
import io
import sys
from contextlib import redirect_stdout

import torch
import yaml


class CheckpointLogger:
    """Creates a .log file that accompanies each .pth checkpoint.

    Sections in order:
      1. PRETRAIN PARAMS    — full YAML config
      2. CUSTOM MESSAGE     — user-defined notes from config
      3. POSTTRAIN RESULTS  — final metrics (structured display)
      4. MODEL ARCHITECTURE — torchinfo summary (captured at save time)
      5. TRAINING LOG       — per-epoch text output
    """

    def __init__(self, log_path):
        self.log_path = log_path
        self._sections = {}

    # ── section setters ──────────────────────────────────────────

    def log_pretrain(self, config_dict):
        self._sections["pretrain"] = yaml.dump(
            config_dict, default_flow_style=False, allow_unicode=True
        )

    def log_custom_message(self, message):
        self._sections["custom"] = message if message else "(none)"

    def log_architecture(self, model, input_size):
        """Capture torchinfo summary into a string."""
        from torchinfo import summary
        buf = io.StringIO()
        with redirect_stdout(buf):
            summary(model, input_size=input_size, col_names=("input_size", "output_size", "num_params"))
        self._sections["arch"] = buf.getvalue()

    def log_training(self, text):
        """Append a line to the training-log section."""
        self._sections.setdefault("training", []).append(text)

    def log_posttrain(self, best_iou, metrics):
        """
        metrics: dict like {"val_loss": 0.123, "val_iou": 0.851, ...}
        """
        self._sections["posttrain"] = (best_iou, metrics)

    # ── write ────────────────────────────────────────────────────

    def flush(self):
        """Write everything to disk as a single .log file."""
        lines = []

        # 1. Pretrain
        lines.append("=== PRETRAIN PARAMS ===")
        lines.append(self._sections.get("pretrain", "").rstrip())
        lines.append("")

        # 2. Custom message
        lines.append("=== CUSTOM MESSAGE ===")
        lines.append(str(self._sections.get("custom", "")))
        lines.append("")

        # 3. Posttrain results (formatted with alignment)
        lines.append("=== POSTTRAIN RESULTS ===")
        best_iou, metrics = self._sections.get("posttrain", (None, {}))
        if best_iou is not None:
            lines.append(f"Best IoU: {best_iou:.4f}")
            lines.append("")
            # Split into groups of 4 for readability
            keys = list(metrics.keys())
            for chunk_start in range(0, len(keys), 4):
                chunk = keys[chunk_start : chunk_start + 4]
                # Header line
                header = "   ".join(f"{k:<18}" for k in chunk)
                lines.append("   " + header)
                # Value line
                vals = "   ".join(f"{metrics[k]:<18.4f}" if isinstance(metrics[k], float)
                                  else f"{metrics[k]:<18}" for k in chunk)
                lines.append("   " + vals)
                lines.append("")
        lines.append("")

        # 4. Architecture
        lines.append("=== MODEL ARCHITECTURE ===")
        lines.append(self._sections.get("arch", "(not captured)").rstrip())
        lines.append("")

        # 5. Training log
        lines.append("=== TRAINING LOG ===")
        training = self._sections.get("training", [])
        if training:
            lines.extend(training)
        else:
            lines.append("(not available)")
        lines.append("")

        text = "\n".join(lines)
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write(text)
        return text

