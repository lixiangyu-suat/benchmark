# Medical Image Segmentation Benchmark

> Forked from [FengheTan9/Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks).
>
> 将原项目单体结构拆分为训练、评估、蒸馏三条独立管线；新增 checkpoint 配对日志系统（.pth + .log 按实验子目录存放）；统一模型加载接口，支持 checkpoint 恢复与中断续训；精简配置体系，删除大量遗留死代码。

---

## 目录结构

```
benchmark/
├── configs/
│   ├── config.yaml             # 训练/评估/蒸馏共用配置
│   ├── modellists.yaml         # 模型注册表（新增模型需同步更新）
│   ├── config_mirror.txt       # 超参批量扫描镜像文件
│   └── config_yaml.py          # 批量扫描工具（基于 mirror 展开组合）
├── scripts/
│   ├── train.sh                # 训练快捷入口（source-safe）
│   ├── eval.sh                 # 评估快捷入口
│   └── distill.sh              # 蒸馏快捷入口
├── src/
│   ├── train.py                # 训练管线
│   ├── evaluate.py             # 评估管线
│   ├── distill.py              # 蒸馏管线
│   ├── network/                # 模型架构实现
│   │   ├── U_Net.py            # 原始 U-Net
│   │   ├── U_Net_re.py         # 修订版 U-Net
│   │   ├── AttU_Net.py         # Attention U-Net
│   │   ├── UNetplus.py         # U-Net++（ResNet34 backbone）
│   │   ├── UNet3plus/          # U-Net 3+
│   │   ├── UNeXt.py            # U-NeXt
│   │   ├── CMUNet.py           # CMU-Net
│   │   ├── CMUNeXt.py          # CMU-NeXt
│   │   ├── Mobile_U_ViT.py     # Mobile U-ViT
│   │   ├── medicalT/           # Medical Transformer
│   │   ├── transUnet/          # TransUNet
│   │   └── swinUnet/           # SwinUNet
│   └── utils/
│       ├── model_loader.py     # 模型注册表 + 构建 + checkpoint 路径管理
│       ├── config.py           # YAML 配置加载
│       ├── dataset.py          # 数据加载 + 蒸馏数据集包装
│       ├── metrics.py          # 评估指标（IoU, Dice, SE, PC, F1, ACC）
│       ├── losses.py           # 损失函数（BCEDiceLoss）
│       ├── logger.py           # Checkpoint 配对日志系统
│       └── helpers.py          # 工具函数（种子、计时、参数统计）
├── checkpoint/                 # 实验子目录（.pth + .log 配对）
├── data/
│   ├── busi/                   # 乳腺超声数据集
│   │   ├── images/             # benign/malignant PNG
│   │   └── masks/0/            # 对应 mask
│   ├── busi_train*.txt         # 训练集文件列表
│   ├── busi_val*.txt           # 验证集文件列表
│   └── transunet_ACDC.zip      # ACDC 数据集
├── teacher_probs/              # 蒸馏时临时生成的 soft labels（运行后自动清理）
├── validation_results/         # 评估可视化输出
├── split.py                    # 数据集 7:3 随机划分工具
└── img/ushape.png              # 示意图
```

## 可用模型

见 `configs/modellists.yaml`，当前内置 **12 个架构**：

| 模型 | 类别 | 说明 |
|---|---|---|
| U_Net | cnn | 原始 U-Net |
| U_Net_re | cnn | 修订版 U-Net |
| AttU_Net | cnn | Attention U-Net |
| UNetplus | cnn | U-Net++（ResNet34 backbone） |
| UNet3plus | cnn | U-Net 3+ |
| UNext | cnn | U-NeXt |
| CMUNet | cnn | CMU-Net |
| CMUNeXt | cnn | CMU-NeXt |
| Mobile_U_ViT | hybrid | Mobile U-ViT |
| MedT | transformer | Medical Transformer |
| TransUnet | transformer | TransUNet（ViT + CNN） |
| SwinUnet | transformer | SwinUNet |

**新增模型**需同步修改三处：
1. 把模型代码放到 `src/network/` 下
2. 在 `src/utils/model_loader.py` 中 import 并注册工厂函数
3. 在 `configs/modellists.yaml` 中添加一条记录

---

## 使用方法

项目在 **WSL2 + conda** 环境下运行。所有 `.sh` 脚本设计为 `source` 执行，自动切到项目根目录。

```bash
conda activate benchmark
cd ./benchmark
```

### 训练

```bash
# 新训练（传入架构名）
source scripts/train.sh U_Net

# 从 checkpoint 恢复（传入 checkpoint stem，自动检测 _model_ 前缀）
source scripts/train.sh UNetplus_model_2026-07-04_23_17_55
```

每轮训练在 `checkpoint/` 下创建以 checkpoint stem 命名的子目录：

```
checkpoint/U_Net_model_2026-07-08_16_24_02/
├── U_Net_model_2026-07-08_16_24_02.pth    # 模型权重
└── U_Net_model_2026-07-08_16_24_02.log    # 配对日志
```

日志按以下顺序组织：
- `PRETRAIN PARAMS` — 完整 YAML 配置快照
- `CUSTOM MESSAGE` — config.yaml 中 `log.custom_message` 字段
- `POSTTRAIN RESULTS` — 最终指标对齐输出
- `MODEL ARCHITECTURE` — torchinfo 架构摘要
- `TRAINING LOG` — 逐 epoch 训练输出

训练中断时按 **Ctrl+C** 自动保存 `_interrupted.pth`，下次 resume 可继续。

### 评估

```bash
source scripts/eval.sh UNetplus_model_2026-07-04_23_17_55
```

输出 torchinfo 架构概览 + 验证集指标（val_loss, val_iou, val_dice, val_SE, val_PC, val_F1, val_ACC）。

可选 `--save_viz` 将预测 mask 保存到 `validation_results/`。

### 蒸馏

```bash
# 教师从 checkpoint 加载，学生从头训练
source scripts/distill.sh UNetplus_model_2026-07-04_23_17_55 Mobile_U_ViT

# 教师 + 学生都从 checkpoint 恢复
source scripts/distill.sh UNetplus_model_2026-07-04_23_17_55 Mobile_U_ViT_model_2026-07-06_15_41_15
```

流程：
1. 教师从 checkpoint 加载 → 对训练集生成 soft labels（存 `teacher_probs/`）
2. 学生按架构名构建（新训练）或从 checkpoint 恢复（含 `_model_` 自动检测）
3. BCE-Dice（hard target）+ MSE（soft target）联合训练
4. 保存学生 checkpoint + log，自动清理 `teacher_probs/`

---

## 配置说明

`configs/config.yaml` 主要字段：

| 字段 | 说明 |
|---|---|
| `train.epoch` | 训练/蒸馏轮数 |
| `train.base_lr` | 初始学习率（多项式衰减）|
| `train.img_size` | 训练图像尺寸 |
| `train.batch_size` | 批大小 |
| `eval.img_size` | 评估图像尺寸 |
| `eval.batch_size` | 评估批大小 |
| `data.base_dir` | 数据集根目录 |
| `data.train_file_dir` / `data.val_file_dir` | 训练/验证文件列表 |
| `model.num_classes` | 分割类别数 |
| `log.custom_message` | 嵌入 checkpoint log 的自定义标注 |

模型名不在 yaml 中配置，改为通过命令行 `--model` / `--student` 显式传入。

---

## 数据准备

`data/` 下存放数据集。以 busi（乳腺超声）为例：

```
data/busi/
├── images/
│   ├── benign (1).png
│   ├── benign (2).png
│   ├── malignant (1).png
│   └── ...
└── masks/
    └── 0/
        ├── benign (1).png
        └── ...
```

使用 `split.py` 将图片列表划分为训练集和验证集：

```bash
python split.py --dataset_name busi --dataset_root ./data
```

脚本读取 `data/busi/images/*.png`，按 7:3 随机分为 `busi_train.txt` 和 `busi_val.txt`（写入 `data/busi/`），然后配置 yaml 中引用对应的文件列表即可。

---

## 超参批量扫描

`configs/config_yaml.py` 配合 `configs/config_mirror.txt` 可批量修改 yaml 字段并依次运行脚本：

```bash
python configs/config_yaml.py --mirror configs/config_mirror.txt --script scripts/train.sh
```

`config_mirror.txt` 中标记 `{值1, 值2}` 或 `{起始-结束}` 的字段会被展开为所有组合，逐个修改 yaml 后执行脚本，全部跑完恢复原始 yaml。

---

## License

本项目基于上游 [FengheTan9/Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks) 进行重构。具体许可条款请参考上游仓库及项目根目录的 `LICENSE` 文件。

---

*Maintained by **SnowWolf** — example@123.com*
