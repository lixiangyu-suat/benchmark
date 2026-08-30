from argparse import Namespace
import os

import yaml
import torch


def _find_modellists():
    """Locate modellists.yaml relative to this file."""
    return os.path.join(os.path.dirname(__file__), "../../configs/modellists.yaml")


def load_model_list():
    """Read configs/modellists.yaml and return the model metadata dict."""
    path = _find_modellists()
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data["models"]


def validate_model(raw_name):
    """Check that *raw_name* corresponds to a known architecture.
    Raises ValueError with a clear message if not found.
    Returns the canonical architecture name on success.
    """
    name = _extract_model_name(raw_name)
    known = load_model_list()
    if name not in known:
        raise ValueError(
            f"Unknown model architecture: {name!r} (from {raw_name!r}).\n"
            f"Available: {', '.join(known)}"
        )
    return name


def _extract_model_name(raw_name):
    """Extract architecture name from a checkpoint or plain model name.

    Supports three formats:
      - New:   "20260708_1624_U_Net"            (split by ``_``, first 2 segments are 8+4 digits)
      - Old:   "U_Net_model_2026-07-08_16_24_02"  (contains ``_model_``)
      - Plain: "U_Net"
    """
    # Strip _interrupted suffix before parsing.
    clean = raw_name.removesuffix("_interrupted")

    # New format: first two segments are 8-digit date + 4-digit time.
    parts = clean.split("_", 2)
    if len(parts) == 3 and parts[0].isdigit() and len(parts[0]) == 8 \
                        and parts[1].isdigit() and len(parts[1]) == 4:
        return parts[2]
    # Old format: has a literal "_model_" segment.
    if "_model_" in clean:
        return clean.split("_model_")[0]
    # Plain architecture name.
    return clean


def build_model(config, raw_model_name, device):
    """Construct a model by registry lookup (no weight loading).

    Parameters
    ----------
    config : dict
    raw_model_name : str — architecture name or checkpoint stem
    device : torch.device
    """
    model_name = _extract_model_name(raw_model_name)
    num_classes = config["model"]["num_classes"]

    def _unet():
        from src.network.U_Net import U_Net
        return U_Net(output_ch=num_classes)

    def _unet_re():
        from src.network.U_Net_re import U_Net
        return U_Net(ch_out=num_classes)

    def _unetplus():
        from src.network.UNetplus import ResNet34UnetPlus
        return ResNet34UnetPlus(num_class=num_classes)

    def _unetplus_l3():
        from src.network.UNetplus_L3 import ResNet34UnetPlus
        return ResNet34UnetPlus(num_class=num_classes)
    
    def _unetplus_l5():
        from src.network.UNetplus_L5 import ResNet34UnetPlus
        return ResNet34UnetPlus(num_class=num_classes)
    
    def _unet3plus():
        from src.network.UNet3plus.UNet3plus import UNet3plus
        return UNet3plus(n_classes=num_classes)

    def _unext():
        from src.network.UNeXt import UNext
        return UNext(output_ch=num_classes)

    def _attu_net():
        from src.network.AttU_Net import AttU_Net
        return AttU_Net(output_ch=num_classes)

    def _cmunet():
        from src.network.CMUNet import CMUNet
        return CMUNet(output_ch=num_classes)

    def _cmunext():
        from src.network.CMUNeXt import cmunext
        return cmunext(num_classes=num_classes)

    def _mobile_uvit():
        from src.network.Mobile_U_ViT import mobileuvit
        return mobileuvit(out_channel=num_classes)

    def _med_t():
        from src.network.medicalT.axialnet import MedT
        return MedT(img_size=256, imgchan=3, num_classes=num_classes)

    def _transunet():
        from src.network.transUnet.transunet import TransUnet
        return TransUnet(img_ch=3, output_ch=num_classes)

    def _swinunet():
        from src.network.swinUnet.vision_transformer import SwinUnet
        from src.network.swinUnet.config import get_config
        swin_cfg = get_config(Namespace(**config["model"]["SwinUnet"]))
        return SwinUnet(swin_cfg, img_size=224, num_classes=num_classes)

    
    REGISTRY = {
        "U_Net": _unet,
        "U_Net_re": _unet_re,
        "UNetplus": _unetplus,
        "UNetplus_L3": _unetplus_l3,
        "UNetplus_L5": _unetplus_l5,
        "UNet3plus": _unet3plus,
        "UNext": _unext,
        "AttU_Net": _attu_net,
        "CMUNet": _cmunet,
        "CMUNeXt": _cmunext,
        "Mobile_U_ViT": _mobile_uvit,
        "MedT": _med_t,
        "TransUnet": _transunet,
        "SwinUnet": _swinunet,
    }

    if model_name not in REGISTRY:
        raise ValueError(
            f"Unknown model {model_name!r} (from {raw_model_name!r}). "
            f"Available: {list(REGISTRY.keys())}"
        )

    print(f"=> Building model: {model_name}")
    model = REGISTRY[model_name]()
    model.to(device)

    if torch.cuda.device_count() > 1:
        print(f"=> Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    return model


def load_checkpoint_meta(path, device):
    """Load a checkpoint and return (state_dict, epoch, best_iou).

    Supports both:
    - new format: dict {'epoch': N, 'model_state_dict': ..., 'best_iou': ...}
    - old format: bare state_dict  (epoch = 0, best_iou = 0.0)
    """
    data = torch.load(path, map_location=device, weights_only=False)
    if isinstance(data, dict) and "model_state_dict" in data:
        sd = data["model_state_dict"]
        epoch = data.get("epoch", 0)
        best_iou = data.get("best_iou", 0.0)
    else:
        sd = data
        epoch = 0
        best_iou = 0.0

    # Strip DataParallel wrapper prefix if present
    if any(k.startswith("module.") for k in sd):
        sd = {k.removeprefix("module."): v for k, v in sd.items()}

    return sd, epoch, best_iou


def save_checkpoint(path, model, epoch, best_iou):
    """Save model weights with training metadata."""
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "best_iou": best_iou,
    }, path)

def resolve_ckpt_path(stem, suffix=".pth"):
    """Resolve checkpoint path, trying variants: _best, _final, bare."""
    variants = [f"{stem}_best", f"{stem}_final", stem]
    for v in variants:
        flat = os.path.join("checkpoint", f"{v}{suffix}")
        if os.path.exists(flat):
            return flat
        sub = os.path.join("checkpoint", stem, f"{v}{suffix}")
        if os.path.exists(sub):
            return sub
    if stem.endswith("_interrupted"):
        base = stem[:-len("_interrupted")]
        for v in [f"{stem}_best", f"{stem}_final", stem]:
            sub_int = os.path.join("checkpoint", base, f"{v}{suffix}")
            if os.path.exists(sub_int):
                return sub_int
    raise FileNotFoundError(f"Checkpoint not found for stem: {stem}")
