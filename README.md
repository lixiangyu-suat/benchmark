# Medical Image Segmentation Benchmark

A refactored benchmark for medical image segmentation with **12 model architectures**, three independent pipelines (train / evaluate / distill), and a flat checkpoint system designed for easy comparison.

> Forked from [FengheTan9/Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks).
> Refactored from a monolithic script into decoupled pipelines with a unified model loader, checkpoint logging, and configuration system.

## At a glance

| Layer | Contents |
|---|---|
| **Pipelines** | train.py -- training with resume | evaluate.py -- metrics on val set | distill.py -- knowledge distillation |
| **Models** | 12 architectures: U-Net, U-Net++, U-Net 3+, AttU_Net, U-NeXt, CMU-Net, CMU-NeXt, Mobile U-ViT, Medical Transformer, TransUNet, SwinUNet |
| **Checkpoints** | Flat checkpoint/ directory, *.pth + *.log paired files, sortable by name |
| **Config** | Single configs/config.yaml shared by all three pipelines |
| **Data** | Auto-scan images/ dir, no manual file lists needed |

## Directory structure

```
benchmark/
├── configs/
│   ├── config.yaml             # shared config (train / eval / distill)
│   ├── modellists.yaml         # model registry
│   ├── config_mirror.txt       # batch-scan template
│   └── config_yaml.py          # batch hyper-parameter scanner
├── scripts/
│   ├── train.sh                # training entry point
│   ├── eval.sh                 # evaluation entry point
│   └── distill.sh              # distillation entry point
├── src/
│   ├── train.py                # training pipeline
│   ├── evaluate.py             # evaluation pipeline
│   ├── distill.py              # knowledge distillation pipeline
│   ├── network/                # model implementations (12 architectures)
│   └── utils/
│       ├── model_loader.py     # model registry + build + checkpoint I/O
│       ├── dataset.py          # data loading + split management
│       ├── metrics.py          # IoU, Dice, SE, PC, F1, SP, ACC
│       ├── losses.py           # BCEDiceLoss
│       ├── logger.py           # checkpoint-paired log system
│       ├── config.py           # YAML loader
│       └── helpers.py          # seed, timestamp, AverageMeter
├── checkpoint/                 # all .pth and .log files (flat)
├── data/
│   └── busi/
│       ├── images/             # PNG images
│       └── masks/0/            # corresponding masks
└── validation_results/         # eval visualizations (with --save_viz)
```

## Quick start

```bash
conda activate benchmark
cd benchmark
```

### Train

```bash
# Fresh training
source scripts/train.sh U_Net

# Resume from checkpoint (use the exact stem)
source scripts/train.sh 20260709_1430_U_Net
```

Checkpoints are stored flat in `checkpoint/`:

```
checkpoint/
├── 20260709_1430_U_Net.pth      # model weights
├── 20260709_1430_U_Net.log      # paired log
├── 20260709_1620_MedT.pth
├── 20260709_1620_MedT.log
└── ...
```

Names use `YYYYMMDD_HHMM_ModelName` format -- alphabetical order equals chronological order. `ls` shows them naturally sorted.

**Ctrl+C** handling: press once, current epoch finishes, checkpoint saved. Press again to force exit.

### Evaluate

```bash
source scripts/eval.sh 20260709_1430_U_Net
source scripts/eval.sh 20260709_1430_U_Net --save_viz   # save predictions as PNGs
```

Outputs: val_loss, val_iou, val_dice, val_SE, val_PC, val_F1, val_ACC.

### Distill (knowledge distillation)

```bash
# Teacher from checkpoint, student from scratch
source scripts/distill.sh 20260709_1430_U_Net Mobile_U_ViT

# Both teacher and student from checkpoints
source scripts/distill.sh 20260709_1430_U_Net 20260709_1620_Mobile_U_ViT
```

Flow: teacher generates soft labels, student trains on BCE-Dice (hard) + MSE (soft), temp files auto-cleaned.

## Configuration

`configs/config.yaml` key fields:

| Field | Description |
|---|---|
| `data.base_dir` | dataset root |
| `data.seed` | random seed (controls split reproducibility) |
| `data.val_split` | proportion of images held out for validation (default 0.3) |
| `train.epoch` | number of epochs per training session |
| `train.base_lr` | initial learning rate (polynomial decay) |
| `train.img_size` | training image size |
| `train.batch_size` | training batch size |
| `eval.img_size` / `eval.batch_size` | evaluation dimensions |
| `model.num_classes` | segmentation classes (1 for binary) |
| `log.custom_message` | annotation embedded in checkpoint log |

The dataset split is **auto-generated**: on first run, `dataset.py` scans `base_dir/images/`, splits by `data.seed` with `data.val_split` ratio, and persists the result as `{dataset}_split.json`. Manual file lists are not needed.

Checkpoint logs contain five sections:
1. **PRETRAIN PARAMS** -- full YAML config snapshot
2. **CUSTOM MESSAGE** -- from `log.custom_message` in config
3. **POSTTRAIN RESULTS** -- final metrics (formatted table)
4. **MODEL ARCHITECTURE** -- torchinfo summary
5. **TRAINING LOG** -- per-epoch output

## Available models

| Model | Category | Description |
|---|---|---|
| U_Net | CNN | original U-Net |
| U_Net_re | CNN | revised U-Net |
| AttU_Net | CNN | Attention U-Net |
| UNetplus | CNN | U-Net++ (ResNet34 backbone) |
| UNet3plus | CNN | U-Net 3+ |
| UNext | CNN | U-NeXt |
| CMUNet | CNN | CMU-Net |
| CMUNeXt | CNN | CMU-NeXt |
| Mobile_U_ViT | hybrid | Mobile U-ViT |
| MedT | transformer | Medical Transformer (axial attention) |
| TransUnet | transformer | TransUNet (ViT-CNN hybrid) |
| SwinUnet | transformer | SwinUNet (shifted-window) |

## Adding a new model

Three files to update:

1. Place the model code in `src/network/`
2. Register in `src/utils/model_loader.py` -- import + add to `REGISTRY` dict
3. Register in `configs/modellists.yaml` -- add an entry with description and category

## Data structure

```
data/{dataset}/
├── images/
│   ├── benign (1).png
│   ├── malignant (10).png
│   └── ...
└── masks/
    └── 0/
        ├── benign (1).png
        └── ...
```

Images and masks must be **PNG** with matching filenames. Masks are single-channel grayscale (0 for background, 255 for foreground). On first training run, the dataset is automatically split into train/val sets based on the configured seed and `val_split` ratio.

## Batch parameter scanning

```bash
python configs/config_yaml.py --mirror configs/config_mirror.txt --script scripts/train.sh
```

Fields marked `{value1, value2}` or `{start-end}` in the mirror file are expanded into all combinations, the YAML is updated for each combination, and the script is run. The original YAML is restored after all runs.

## Migration from old checkpoints

If you have existing checkpoints from before the naming change, run the migration script once:

```bash
python scripts/migrate_checkpoints.py
```

This renames old-format folders (`ModelName_model_YYYY-MM-DD_HH_MM_SS`) to the new flat format (`YYYYMMDD_HHMM_ModelName`).

## License

Based on [FengheTan9/Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks). See the original repository and `LICENSE` in the project root for details.

---

*Maintained by SnowWolf*
