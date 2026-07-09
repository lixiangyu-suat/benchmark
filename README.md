# Medical Image Segmentation Benchmark

一个重构的医学图像分割基准，包含 **12 种模型架构**、三条独立流水线（train / evaluate / distill），以及便于对比的 checkpoint 子文件夹系统。

> Forked from [FengheTan9/Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks).
> 从单体脚本重构为解耦的流水线，统一了模型加载、checkpoint 日志和配置系统。

## 一瞥

| 层级 | 内容 |
|---|---|
| **流水线** | train.py — 训练（支持续传） | evaluate.py — 验证集指标 | distill.py — 知识蒸馏 |
| **模型** | 12 种：U-Net, U-Net++, U-Net 3+, AttU_Net, U-NeXt, CMU-Net, CMU-NeXt, Mobile U-ViT, Medical Transformer, TransUNet, SwinUNet |
| **检查点** | checkpoint/{时间戳}_{模型名}/ 子文件夹，内含同名 .pth 和 .log |
| **配置** | 单文件 configs/config.yaml，三条流水线共享 |
| **数据** | 自动扫描 images/ 目录，无需手动维护文件列表 |

## 目录结构

`
benchmark/
├── configs/
│   ├── config.yaml             # 共享配置（train / eval / distill）
│   ├── modellists.yaml         # 模型注册表
│   ├── config_mirror.txt       # 批量扫描模板
│   └── config_yaml.py          # 批量超参扫描
├── scripts/
│   ├── train.sh                # 训练入口
│   ├── eval.sh                 # 评估入口
│   └── distill.sh              # 蒸馏入口
├── src/
│   ├── train.py                # 训练流水线
│   ├── evaluate.py             # 评估流水线
│   ├── distill.py              # 知识蒸馏流水线
│   ├── network/                # 模型实现（12 种架构）
│   └── utils/
│       ├── model_loader.py     # 模型注册 + 构建 + 检查点读写
│       ├── dataset.py          # 数据加载 + 切分管理
│       ├── metrics.py          # IoU, Dice, SE, PC, F1, SP, ACC
│       ├── losses.py           # BCEDiceLoss
│       ├── logger.py           # 与 .pth 配对的日志系统
│       ├── config.py           # YAML 加载器
│       └── helpers.py          # seed、时间戳、AverageMeter
├── checkpoint/                 # 所有 .pth 和 .log 文件（按模型分文件夹存放）
├── data/
│   └── busi/
│       ├── images/             # PNG 图像
│       └── masks/0/            # 对应的掩码
└── validation_results/         # 评估可视化输出（配合 --save_viz）
`

## 快速开始

`ash
conda activate benchmark
cd benchmark
`

### 训练

`ash
# 新训练
source scripts/train.sh U_Net

# 从 checkpoint 续传（直接用文件夹名）
source scripts/train.sh 20260709_1430_U_Net
`

**训练机制**：

| 场景 | 流程 |
|---|---|
| **新训练** | T0 = 20260709_1430_U_Net → 建文件夹 checkpoint/T0/ → 训练过程中 best model 覆盖写入 T0.pth，日志写入 T0.log |
| **续传** | 从 checkpoint/T0/T0.pth 加载 → 训练中覆盖 T0.pth，日志**新写入** T1.log（T0.log 保留历史） → 结束时：保存 T0.pth → **重命名** T0.pth 为 T1.pth → **重命名** 文件夹 T0/ 为 T1/ → 日志路径自动更新 |

续传后的文件夹结构：

`
checkpoint/
└── 20260709_1600_Mobile_U_ViT/    # T1（续传后的新名称）
    ├── 20260709_1600_Mobile_U_ViT.pth   # 最终权重
    ├── 20260709_1536_Mobile_U_ViT.log   # 第一次训练的日志（保留）
    └── 20260709_1600_Mobile_U_ViT.log   # 续传训练的日志
`

Checkpoint 名称使用 YYYYMMDD_HHMM_ModelName 格式 — 按字母顺序即按时间顺序。

**Ctrl+C 处理**：按一次，当前 epoch 完成后保存 checkpoint；再按一次强制退出。

### 评估

`ash
source scripts/eval.sh 20260709_1430_U_Net
source scripts/eval.sh 20260709_1430_U_Net --save_viz   # 保存预测结果为 PNG
`

输出指标：val_loss, val_iou, val_dice, val_SE, val_PC, val_F1, val_ACC

### 蒸馏（Knowledge Distillation）

`ash
# 老师来自 checkpoint，学生从头训练
source scripts/distill.sh 20260709_1430_U_Net Mobile_U_ViT

# 老师和学生都来自 checkpoint (带 --resume)
source scripts/distill.sh 20260709_1430_U_Net 20260709_1620_Mobile_U_ViT --resume
`

流程：老师生成 soft labels → 学生用 BCE-Dice（硬标签）+ MSE（软标签）训练 → 临时文件自动清理。

蒸馏的学生模型保存逻辑与训练**完全一致**：新训练建子文件夹、续传重命名文件夹；老师模型只加载生成概率，不参与保存。

## 配置

configs/config.yaml 关键字段：

| 字段 | 说明 |
|---|---|
| data.base_dir | 数据集根目录 |
| data.seed | 随机种子（控制切分可复现性） |
| data.val_split | 验证集比例（默认 0.3） |
| train.epoch | 每轮训练的 epoch 数量（续传时追加同样数量的 epoch） |
| train.base_lr | 初始学习率（多项式衰减） |
| train.img_size | 训练图像尺寸 |
| train.batch_size | 训练 batch 大小 |
| eval.img_size / eval.batch_size | 评估使用的尺寸和 batch |
| model.num_classes | 分割类别数（二分类为 1） |
| log.custom_message | 嵌入 checkpoint 日志的备注 |

数据集切分是**自动生成**的：首次运行时，dataset.py 扫描 base_dir/images/，根据 data.seed 和 data.val_split 比例划分，结果持久化为 {dataset}_split.json，不需要手动维护文件列表。

Checkpoint 日志包含五个部分：
1. **PRETRAIN PARAMS** - 完整的 YAML 配置快照
2. **CUSTOM MESSAGE** - 来自 log.custom_message
3. **POSTTRAIN RESULTS** - 最终指标（格式化表格）
4. **MODEL ARCHITECTURE** - torchinfo 模型摘要
5. **TRAINING LOG** - 逐 epoch 输出

## 可用模型

| 模型 | 类别 | 说明 |
|---|---|---|
| U_Net | CNN | 原始 U-Net |
| U_Net_re | CNN | 修订版 U-Net |
| AttU_Net | CNN | Attention U-Net |
| UNetplus | CNN | U-Net++（ResNet34 骨干网络） |
| UNet3plus | CNN | U-Net 3+ |
| UNext | CNN | U-NeXt |
| CMUNet | CNN | CMU-Net |
| CMUNeXt | CNN | CMU-NeXt |
| Mobile_U_ViT | hybrid | Mobile U-ViT |
| MedT | transformer | Medical Transformer（轴向注意力） |
| TransUnet | transformer | TransUNet（ViT-CNN 混合） |
| SwinUnet | transformer | SwinUNet（滑动窗口） |

## 添加新模型

需要更新三个文件：

1. 将模型代码放入 src/network/
2. 在 src/utils/model_loader.py 中注册 - import + 添加到 REGISTRY 字典
3. 在 configs/modellists.yaml 中注册 - 添加描述和分类

## 数据格式

`
data/{dataset}/
├── images/
│   ├── benign (1).png
│   ├── malignant (10).png
│   └── ...
└── masks/
    └── 0/
        ├── benign (1).png
        └── ...
`

图片和掩码必须为 **PNG** 格式，文件名一一对应。掩码为单通道灰度图（0 为背景，255 为前景）。首次训练时自动按 seed 和 val_split 比例划分训练/验证集。

## 批量超参扫描

`ash
python configs/config_yaml.py --mirror configs/config_mirror.txt --script scripts/train.sh
`

在 mirror 文件中标记为 {value1, value2} 或 {start-end} 的字段会被展开为所有组合，逐次更新 YAML 后执行脚本。所有运行结束后恢复原始 YAML。

## 旧 checkpoint 迁移

如果已有旧格式（命名变更前）的 checkpoint，运行一次迁移脚本：

`ash
python scripts/migrate_checkpoints.py
`

会将旧格式文件夹（ModelName_model_YYYY-MM-DD_HH_MM_SS）重命名为新的格式（YYYYMMDD_HHMM_ModelName）。

## 许可证

基于 [FengheTan9/Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks)。详情见原始仓库和项目根目录的 LICENSE 文件。

---

*Maintained by SnowWolf*
