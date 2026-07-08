# Medical Image Segmentation Benchmark

> 在 xxx 项目的基础上，做了如下修改：
> - 解耦了训练、评估、蒸馏三个管线，各自独立入口
> - 新增了 checkpoint 配对日志系统（每个 .pth 同级生成 .log）
> - 统一了模型加载接口，支持从 checkpoint 恢复训练
> - 清理了配置体系，删除了大量混乱的遗留代码
> - 将工具函数按职责拆分到 `src/utils/`，并补全了缺失的 `__init__.py`
> - 新增 `configs/modellists.yaml`，用于集中管理可用的模型列表和校验
> - 后续 TODO：补充新作者信息

---

## 目录结构

```
benchmark/
├── configs/
│   ├── config.yaml            # 主配置文件
│   └── modellists.yaml        # 模型注册表（新增模型时需同步更新）
├── src/
│   ├── train.py                # 训练入口
│   ├── evaluate.py             # 评估入口
│   ├── distill.py              # 蒸馏入口
│   ├── utils/
│   │   ├── config.py           # YAML 配置加载
│   │   ├── model_loader.py     # 模型注册表 + 构建
│   │   ├── dataset.py          # 数据加载 + 蒸馏数据集包装
│   │   ├── metrics.py          # 评估指标（IoU, Dice, SE, PC, F1, ACC）
│   │   ├── losses.py           # 损失函数（BCEDiceLoss, KL）
│   │   ├── logger.py           # Checkpoint 配对日志系统
│   │   └── helpers.py          # 工具函数（种子、计时、参数统计）
│   └── network/                # 所有模型架构（保留不动）
├── scripts/
│   ├── train.sh                # 训练快捷脚本（仅备忘用）
│   ├── eval.sh                 # 评估快捷脚本（仅备忘用）
│   └── distill.sh              # 蒸馏快捷脚本（仅备忘用）
├── checkpoint/                 # .pth + .log 配对存放
├── data/                       # 数据集
├── teacher_probs/              # 蒸馏时生成的教师 soft labels
└── validation_results/         # 评估时保存的预测可视化
```

---

## 可用模型

见 `configs/modellists.yaml`：

```bash
grep -E "^  [A-Z]" configs/modellists.yaml
```

当前内置 12 个架构：

| 模型 | 类别 | 说明 |
|---|---|---|
| U_Net | cnn | 原始 U-Net |
| U_Net_re | cnn | 修订版 U-Net |
| AttU_Net | cnn | Attention U-Net |
| UNetplus | cnn | U-Net++ (ResNet34 backbone) |
| UNet3plus | cnn | U-Net 3+ |
| UNext | cnn | U-NeXt |
| CMUNet | cnn | CMU-Net |
| CMUNeXt | cnn | CMU-NeXt |
| Mobile_U_ViT | hybrid | Mobile U-ViT |
| MedT | transformer | Medical Transformer |
| TransUnet | transformer | TransUNet |
| SwinUnet | transformer | SwinUNet |

新增模型时，需要：
1. 把模型代码放到 `src/network/` 下
2. 在 `src/utils/model_loader.py` 里 import 并注册工厂函数
3. 在 `configs/modellists.yaml` 里加一条记录

---

## 使用方法

项目在 WSL2 的 conda 环境下运行。推荐直接调用 Python 入口（避免 `bash` 子 shell 丢失 conda 环境的问题），`scripts/` 下的 `.sh` 文件保留仅做命令格式备忘。

```bash
conda activate your_env
cd /mnt/f/Workspace/valid_paper_AI/benchmark
```

### 训练

```bash
# 训练：新训练 U_Net
source scripts/train.sh U_Net
# 训练：从 checkpoint 恢复 UNetplus
source scripts/train.sh UNetplus_model_2026-07-04_23_17_55
```

模型名在校验时与 `modellists.yaml` 比对，输入有误会提前提示，不会走到 Python import 报错。

训练结束会在 `checkpoint/` 下生成一对文件：
- `{model_name}_model_{timestamp}.pth` — 模型权重
- `{model_name}_model_{timestamp}.log` — 日志

日志按以下顺序组织：
```
=== PRETRAIN PARAMS ===              ← 完整的 YAML 配置
=== CUSTOM MESSAGE ===               ← 来自 config.yaml log.custom_message
=== POSTTRAIN RESULTS ===            ← 最终指标（对齐排版）
=== MODEL ARCHITECTURE ===           ← torchinfo summary
=== TRAINING LOG ===                 ← 每 epoch 的输出
```

### 评估

```bash
# 评估：传 checkpoint stem
source scripts/eval.sh UNetplus_model_2026-07-04_23_17_55
```

输出内容：
1. `torchinfo summary` 打印模型架构
2. 验证集指标：val_loss, val_iou, val_dice, val_SE, val_PC, val_F1, val_ACC

### 蒸馏

```bash
# 蒸馏：教师从 checkpoint 加载，学生从头训练
source scripts/distill.sh UNetplus_model_2026-07-04_23_17_55 Mobile_U_ViT

# 蒸馏：教师 + 学生都从 checkpoint 恢复
source scripts/distill.sh UNetplus_model_2026-07-04_23_17_55 Mobile_U_ViT_model_2026-07-06_15_41_15
```

流程：
1. 教师从 checkpoint 加载 → 对训练集生成 soft labels
2. 学生构建（新模型或从 checkpoint 恢复）
3. 用 BCE-Dice + MSE distillation loss 训练
4. 保存学生 checkpoint + 配对 log

---

## 配置说明

详见 `configs/config.yaml`：

| 字段 | 说明 |
|---|---|
| `train.epoch` | 训练轮数 |
| `train.base_lr` | 初始学习率 |
| `log.custom_message` | 嵌入 checkpoint log 的自定义文本 |
| `data.*` | 数据集路径和划分文件 |

模型名不再放在 yaml 里，改为通过命令行 `--model` 传入，确保每次运行都能肉眼确认模型名字输对了没有。
