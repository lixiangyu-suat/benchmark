import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["BCEDiceLoss"]


class BCEDiceLoss(nn.Module):
    """Binary cross-entropy + Dice loss (equal weighting)."""

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input_sig = torch.sigmoid(input)
        num = target.size(0)
        input_sig = input_sig.view(num, -1)
        target = target.view(num, -1)
        intersection = (input_sig * target)
        dice = (2. * intersection.sum(1) + smooth) / (input_sig.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


def compute_kl_loss(p, q):
    """Symmetric KL divergence (useful for semi-supervised learning)."""
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction="none")
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction="none")
    return (p_loss.mean() + q_loss.mean()) / 2

