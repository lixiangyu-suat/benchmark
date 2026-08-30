# Medical Image Segmentation Benchmark

一个高度解耦、可扩展的医学图像分割基准测试平台。集成 **12 种主流模型架构**，提供训练、评估、知识蒸馏三条独立流水线，并配备严格按时间序追踪的检查点与日志管理系统。

> **Reference**: Forked from [FengheTan9/Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks)
> **Refactor Focus**: 单体脚本解耦为模块化 Pipeline、统一模型动态注册加载、统一 Checkpoint 目录生命周期与全局 YAML 配置。

---

## 核心特性

| 模块 | 说明 | 核心组件 / 文件 |
| --- | --- | --- |
| **流水线** | 支持断点续训、标准指标评测、软硬标签知识蒸馏 | `train.py` / `evaluate.py` / `distill.py` |
| **模型库** | 涵盖 CNN、混合架构及 Transformer 类 12 种分割模型 | `src/network/` |
| **检查点** | 按 `checkpoint/{时间戳}_{模型名}/` 统一隔离权重与日志 | `src/utils/logger.py` |
| **配置中心** | 单一入口统一管理所有流水线参数与超参数 | `configs/config.yaml` |
| **数据管理** | 目录扫描 + 自动化划分持久化，无需手动维护列表 | `src/utils/dataset.py` |

---

## 目录结构

```text
benchmark/
├── configs/
│   ├── config.yaml             # 共享配置（训练 / 评估 / 蒸馏）
│   ├── modellists.yaml         # 模型注册元数据
│   ├── config_mirror.txt       # 批量扫描模板
│   └── config_yaml.py          # 批量超参扫描生成器
├── scripts/
│   ├── train.sh                # 训练执行脚本
│   ├── eval.sh                 # 评估执行脚本
│   ├── distill.sh              # 蒸馏执行脚本
│   └── migrate_checkpoints.py  # 旧版检查点迁移工具
├── src/
│   ├── train.py                # 训练流水线入口
│   ├── evaluate.py             # 评估流水线入口
│   ├── distill.py              # 蒸馏流水线入口
│   ├── network/                # 12 种模型架构定义
│   └── utils/
│       ├── model_loader.py     # 模型注册、动态构建与权重加载
│       ├── dataset.py          # 数据集读取与划分持久化
│       ├── metrics.py          # 评估指标库（IoU, Dice, SE, PC, F1, ACC 等）
│       ├── losses.py           # 损失函数库（BCEDiceLoss 等）
│       ├── logger.py           # Checkpoint 绑定的日志记录器
│       ├── config.py           # YAML 配置文件解析器
│       └── helpers.py          # 随机种子、时间戳与 AverageMeter 等工具
├── checkpoint/                 # 模型权重与日志持久化目录（按时间戳隔离）
├── data/
│   └── busi/
│       ├── images/             # 原始图像（PNG）
│       └── masks/0/            # 分割掩码（单通道灰度图）
└── validation_results/         # 评估预测可视化结果保存目录

```

---

## 快速开始

### 环境激活

```bash
conda activate benchmark
cd benchmark

```

### 1. 模型训练 (Train)

```bash
# 启动全新训练
source scripts/train.sh U_Net

# 从指定检查点断点续传（直接传入时间戳文件夹名）
source scripts/train.sh 20260709_1430_U_Net

```

**训练与检查点流转机制**：

| 运行场景 | 目录与文件流转逻辑 |
| --- | --- |
| **全新训练** | 1. 生成初始标识 $T_0$（如 `20260709_1430_U_Net`）并建立子目录 `checkpoint/T0/`<br>

<br>2. 训练中最佳权重覆盖写入 `T0.pth`，过程输出记录至 `T0.log` |
| **断点续训** | 1. 读取 `checkpoint/T0/T0.pth` 作为基础权重<br>

<br>2. 训练过程中覆盖 `T0.pth`，历史日志保留，新日志写入 `$T_1$.log`<br>

<br>3. 训练完成：将 `T0.pth` 重命名为 `$T_1$.pth`，并将目录 `T0/` 原子重命名为 `$T_1$/` |

续传后的目录结构变化示例：

```text
checkpoint/
└── 20260709_1600_Mobile_U_ViT/          # 最终目录以续传完成时间戳 (T1) 命名
    ├── 20260709_1600_Mobile_U_ViT.pth   # 续传后的最终权重
    ├── 20260709_1536_Mobile_U_ViT.log   # 初始训练阶段历史日志 (T0)
    └── 20260709_1600_Mobile_U_ViT.log   # 续传阶段日志 (T1)

```

> **中断保护**：
> * 单次按下 `Ctrl + C`：等待当前 Epoch 完成后优雅保存 Checkpoint 并退出。
> * 连续按下 `Ctrl + C`：直接强制中断退出。
> 
> 

---

### 2. 模型评估 (Evaluate)

```bash
# 标准评估
source scripts/eval.sh 20260709_1430_U_Net

# 评估并保存预测掩码可视化图像
source scripts/eval.sh 20260709_1430_U_Net --save_viz

```

* **评估指标输出**：`val_loss`, `val_iou`, `val_dice`, `val_SE`, `val_PC`, `val_F1`, `val_ACC`。

---

### 3. 知识蒸馏 (Distill)

```bash
# Teacher 来自检查点，Student 从头初始化训练
source scripts/distill.sh 20260709_1430_U_Net Mobile_U_ViT

# Teacher 与 Student 均来自检查点（启用 --resume 续传）
source scripts/distill.sh 20260709_1430_U_Net 20260709_1620_Mobile_U_ViT --resume

```

* **蒸馏逻辑**：Teacher 模型前向推理生成 Soft Labels；Student 模型结合真实标签（BCE-Dice Loss）与软标签（MSE Loss）进行多任务优化，临时生成的中间概率文件在训练完成后自动清理。

---

## 配置文件说明

全局配置集中在 `configs/config.yaml`：

| 配置键 | 默认/示例 | 说明 |
| --- | --- | --- |
| `data.base_dir` | `data/busi` | 数据集根目录路径 |
| `data.seed` | `42` | 数据集切分随机种子（保证切分可复现） |
| `data.val_split` | `0.3` | 验证集所占比例 |
| `train.epoch` | `100` | 训练 Epoch 轮数（续传时追加同等轮数） |
| `train.base_lr` | `1e-3` | 初始学习率（搭配多项式衰减策略） |
| `train.img_size` | `256` | 训练阶段输入分辨率 |
| `train.batch_size` | `8` | 训练批大小 |
| `eval.img_size` | `256` | 评估阶段输入分辨率 |
| `eval.batch_size` | `1` | 评估批大小 |
| `model.num_classes` | `1` | 分割通道数（二分类设为 1） |
| `log.custom_message` | `""` | 嵌入 Checkpoint 头部元信息的自定义备注 |

### 数据切分与持久化

数据集无需维护繁琐的 `train.txt` / `val.txt`。首次启动时，`dataset.py` 自动扫描 `images/`，按 `seed` 与 `val_split` 划分并生成 `{dataset}_split.json`。后续所有流水线均直接复用该 JSON，确保基准对比的一致性。

### 日志结构

生成的 `.log` 文件统一包含五个结构化区块：

1. **PRETRAIN PARAMS**：当前训练所使用的完整 YAML 快照
2. **CUSTOM MESSAGE**：用户注入的实验备注
3. **MODEL ARCHITECTURE**：通过 `torchinfo` 导出的模型参数量与层结构摘要
4. **TRAINING LOG**：逐 Epoch 训练损失与指标变化
5. **POSTTRAIN RESULTS**：训练结束后的最佳指标概览表

---

## 支持的模型架构

| 模型标识 (Model Name) | 架构分类 | 特点与说明 |
| --- | --- | --- |
| `U_Net` | CNN | 经典对称医学分割网络 |
| `U_Net_re` | CNN | 改进版经典 U-Net |
| `AttU_Net` | CNN | 融合注意力门控机制的 U-Net |
| `UNetplus` | CNN | U-Net++（ResNet-34 骨干网络与密集跳跃连接） |
| `UNet3plus` | CNN | U-Net 3+（全尺度跳跃连接与深度监督） |
| `UNext` | CNN | 基于 MLP 与轻量卷积的高效分割架构 |
| `CMUNet` | CNN | 结合上下文多尺度提取的医学分割模型 |
| `CMUNeXt` | CNN | 融合现代 ConvNeXt 块的高效结构 |
| `Mobile_U_ViT` | Hybrid | 结合轻量 CNN 与 ViT 的低延迟混合网络 |
| `MedT` | Transformer | Medical Transformer（基于轴向注意力机制） |
| `TransUnet` | Transformer | 结合 CNN 局部特征与 ViT 全局自注意力的混合架构 |
| `SwinUnet` | Transformer | 基于滑动窗口分层自注意力（Swin Transformer）的纯 Transformer 架构 |

---

## 扩展与高级功能

### 扩展新模型

新增自定义模型需完成以下三步：

1. **放置代码**：将模型实现源码添加至 `src/network/your_model.py`。
2. **注册构造器**：在 `src/utils/model_loader.py` 中引入该类并加入 `REGISTRY` 字典。
3. **注册元数据**：在 `configs/modellists.yaml` 中配置模型类别、描述及默认参数。

### 数据集目录规范

存入 `data/{dataset}/` 的数据需遵循以下结构：

```text
data/{dataset}/
├── images/
│   ├── case_001.png
│   ├── case_002.png
│   └── ...
└── masks/
    └── 0/
        ├── case_001.png
        ├── case_002.png
        └── ...

```

* 图像与掩码必须为 **PNG** 格式且文件名严格一致。
* 掩码需为单通道灰度图（$0$ 代表背景，$255$ 代表分割前景）。

### 批量超参数网格搜索

使用内置扫描工具批量生成配置并依次运行：

```bash
python configs/config_yaml.py \
    --mirror configs/config_mirror.txt \
    --script scripts/train.sh

```

* 在 `config_mirror.txt` 中使用 `{val1, val2}` 或 `{start-end}` 标记待扫描参数，脚本将全排列组合后自动修改 YAML、执行训练并在结束后还原配置文件。

### 旧版 Checkpoint 格式迁移

将旧版 `ModelName_model_YYYY-MM-DD_HH_MM_SS` 格式目录平滑迁移为标准格式：

```bash
python scripts/migrate_checkpoints.py

```

---

## 开源协议

本项目基于 [FengheTan9/Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks) 构建，遵循项目根目录下的 `LICENSE` 声明与相关开源许可。