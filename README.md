# Medical Image Segmentation Benchmark

医学图像分割基准测试平台：集成 **14 种主流分割架构**，提供**训练 / 评估 / 知识蒸馏**三条命令行流水线，配套时间戳隔离的检查点目录、结构化日志与自动 ONNX 导出。

> **Reference**: Forked from [FengheTan9/Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks)
> **本仓库的改造点**: 单体脚本解耦为模块化 Pipeline；模型统一注册加载；**所有超参数通过纯命令行传递**（默认值定格在 `src/utils/config.py`，无 YAML、无配置脚本）。

---

## 目录

- [一、项目结构与运行原理](#一项目结构与运行原理)
- [二、环境准备](#二环境准备)
- [三、场景 1：新训练一个已知模型](#三场景-1新训练一个已知模型)
- [四、场景 2：接着训练（断点续训）](#四场景-2接着训练断点续训)
- [五、场景 3：加入、训练并测试一个新模型架构](#五场景-3加入训练并测试一个新模型架构)
- [六、场景 4：批量训练 5 个模型并在 3 个数据集上评估](#六场景-4批量训练-5-个模型并在-3-个数据集上评估)
- [七、知识蒸馏（可选进阶）](#七知识蒸馏可选进阶)
- [八、命令行参数速查](#八命令行参数速查)
- [九、Checkpoint 与日志规则](#九checkpoint-与日志规则)
- [十、支持的模型架构](#十支持的模型架构)
- [十一、数据集规范](#十一数据集规范)
- [十二、常见问题](#十二常见问题)

---

## 一、项目结构与运行原理

### 目录结构

```text
benchmark/
├── train.py                  # 【入口】训练流水线
├── evaluate.py               # 【入口】评估流水线
├── distill.py                # 【入口】知识蒸馏流水线
├── export_onnx.py            # 【入口】从指定 .pth 独立导出 ONNX，无需训练
├── train.sh                  # 硬编码参数启动脚本：训练
├── evaluate.sh               # 硬编码参数启动脚本：评估
├── distill.sh                # 硬编码参数启动脚本：蒸馏
├── run_batch.sh              # 硬编码参数启动脚本：批量训练 + 多数据集评估
│
├── src/
│   ├── network/              # 14 种模型架构定义（每个模型一个文件/目录）
│   └── utils/
│       ├── config.py         # 参数中心：全部超参数默认值 + build_config(args)
│       ├── model_loader.py   # 模型注册表（MODEL_LIST/REGISTRY）、权重存取、stem 解析
│       ├── dataset.py        # 数据集读取、自动划分与清单持久化
│       ├── metrics.py        # 指标库（IoU, Dice, SE, PC, F1, ACC 等）
│       ├── losses.py         # 损失函数（BCEDiceLoss 等）
│       ├── logger.py         # 结构化日志记录器（5 段式 .log）
│       ├── convert_to_onnx.py# .pth -> .onnx 导出与格式校验
│       └── helpers.py        # 随机种子、毫秒时间戳、耗时格式化、AverageMeter
│
├── checkpoint/               # 每次训练一个独立子目录：权重 + 日志 + onnx
├── test_results/             # evaluate --save_viz 的预测掩码输出
├── batch_logs/               # run_batch.sh 的终端输出留存
└── data/
    └── busi/
        ├── images/           # 各划分的图片统一存放，支持多种后缀
        ├── masks/            # 分割掩码（单通道灰度图）
        ├── busi_train1.txt    # 训练清单，每行完整图片文件名
        └── busi_valid1.txt    # 验证清单
```

### 运行原理（10 句话看懂）

1. **参数只有一个来源**：`src/utils/config.py` 里的 `DEFAULTS` 定格了所有默认值；每个入口脚本用 `argparse` 接收命令行覆盖，再经 `build_config(args)` 组装成配置字典传给下游。**改参数 = 改命令行，或改 .sh 里的硬编码值**。
2. **模型通过注册表构建**：`model_loader.py` 的 `REGISTRY` 把架构名映射到构造函数；输入 `--model U_Net` 或一个 checkpoint stem（如 `20260921_1639_10213_U_Net`）都能得到对应模型——stem 中的架构名由 `_extract_model_name` 自动解析。
3. **每次训练得到一个独立目录**：`checkpoint/{毫秒时间戳}_{模型名}/`，内含 `.pth` 权重、`.log` 日志、`.onnx` 导出模型，按名字母序排列即时间序。
4. **训练循环**：SGD(momentum=0.9, weight_decay=1e-4) + 多项式学习率衰减 `lr = base_lr * (1 - iter/max_iter)^0.9`；损失为 BCE+Dice；每个 epoch 结束后在验证集上算全套指标。
5. **最佳权重即时保存**：每当 `val_iou` 刷新纪录，写入 `{stem}_best.pth`；训练结束保留 `{stem}_final.pth`，并仅将最佳权重导出为 `{stem}_best.onnx`。

   ONNX 导出明确加载 `best.pth`，以验证集 IoU 选择最佳轮；不再生成 `final.onnx`。评估入口仅加载 `best.pth`，缺失时直接报错，并在终端和日志中记录实际检查点路径。
6. **检查点内容是字典**：`{"epoch", "model_state_dict", "best_iou"}`，因此续训时知道从第几个 epoch、什么成绩继续。
7. **续训即换时间戳**：续训加载已有检查点的模型权重与 epoch/best IoU，优化器重新创建；新日志/新权重使用**新的时间戳 stem**，结束时把旧目录重命名为新 stem，保留旧日志。
8. **数据划分可复现**：显式指定 `--train_file / --val_file` 时沿用清单；仅当两个清单参数同时置空时，才按 `seed` 自动划分并落盘 `{dataset}_split.json`。指定清单缺失时直接报错。
9. **日志是 5 段式纯文本**：参数快照 → 自定义备注 → 模型结构（torchinfo）→ 逐 epoch 指标 → 结束指标总表 + **训练耗时 / 任务总耗时**（格式 `HHH:MM:SS.mmm`，如 `480:00:01.143`）。
10. **评估也会留痕**：`evaluate.py` 将检查点、数据清单、标签目标、指标与耗时以树形格式打印并追加到 `checkpoint/{stem}/{stem}_eval.log`，每次评估记录之间空两行。

---

## 二、环境准备

```bash
# WSL2 Ubuntu-22.04，conda 环境名：benchmark
conda activate benchmark
cd /mnt/f/Workspace/valid_paper_AI/benchmark

# 依赖（首次配置时）
bash download.sh
```

目录约定：以下所有命令都假设**当前目录是 `benchmark/`**（四个 .sh 脚本内置了自动 `cd`，在别处调用也没问题）。

---

## 三、场景 1：新训练一个已知模型

**方式 A —— 直接命令行（适合临时跑一次）：**

```bash
python train.py --model U_Net --train_file busi_train1.txt --val_file busi_valid1.txt --mask_target binary
```

常用覆盖项：

```bash
python train.py \
    --model UNetplus_L5 \
    --epoch 100 \
    --base_lr 0.001 \
    --batch_size 8 \
    --img_size 256 \
    --train_file busi_train1.txt \
    --val_file busi_valid1.txt \
    --mask_target binary \
    --gpu 0 \
    --custom_message "unetpp_l5_lr1e-3"
```

**方式 B —— 改脚本再运行（适合固定实验配置）：**

打开 `train.sh`，把硬编码参数改成你想要的（每行一个参数，一目了然），然后：

```bash
bash train.sh
```

**你会得到什么**：

```text
checkpoint/20260921_1639_10213_U_Net/
├── 20260921_1639_10213_U_Net.log         # 结构化日志（含耗时）
├── 20260921_1639_10213_U_Net_best.pth    # 训练中 val_iou 最好的权重
├── 20260921_1639_10213_U_Net_final.pth   # 最后一个 epoch 的权重
└── 20260921_1639_10213_U_Net_best.onnx   # 最佳权重的部署格式，自动导出并校验
```

### 单独导出已有权重

无需重新训练，也不要通过 `--epoch 0` 触发训练收尾；直接指定输入权重和输出路径：

```bash
python export_onnx.py \
    --input checkpoint/<stem>/<stem>_best.pth \
    --output model_best_opset16.onnx \
    --model Mobile_U_ViT \
    --img_size 256 \
    --num_classes 1 \
    --opset 16
```

`--model` 是权重对应的架构名，输入尺寸与输出通道数须与模型设置一致。默认使用 CPU，固定输入为 `1×3×256×256`；加 `--dynamic_batch` 可启用动态 batch。训练结束的自动导出使用动态 batch，当前默认 opset 同样为 16。导出后进行 ONNX 格式校验，并打印实际 opset 与 `LayerNormalization` 节点数量；格式校验通过不代表目标 TensorRT 版本一定支持所有算子。

---

## 四、场景 2：接着训练（断点续训）

一个命令，架构名、已训 epoch、历史最好成绩全部自动从 checkpoint 读取：

```bash
python train.py --ckpt 20260921_1639_10213_U_Net --train_file busi_train1.txt --val_file busi_valid1.txt
```

- 会在旧 epoch 基础上**追加** `--epoch` 指定的轮数（默认再训 20 轮）；
- 训练结束后，目录会被**原子重命名**为新时间戳（如 `20260921_1800_45678_U_Net/`），旧日志保留在内，新旧一段一段可溯源；
- 想改学习率等参数，照常加 `--base_lr` 等参数即可；
- 也可以编辑 `train.sh`：删掉 `--model` 行、取消末尾 `--ckpt` 行的注释并填上 stem。

> **训练中途想停**：按一次 `Ctrl+C` —— 跑完当前 epoch 后优雅保存退出；连按两次强制退出。

---

## 五、场景 3：加入、训练并测试一个新模型架构

以新增 `MyNet` 为例，共三步：

**第 1 步：放代码** —— 新建 `src/network/MyNet.py`，模型类需满足：

- 构造函数接受输出通道数参数（如 `num_classes` / `output_ch` / `n_classes`）；
- 前向输入 `(B, 3, H, W)`，输出 `(B, num_classes, H, W)` 的 logits（未过 sigmoid）。

**第 2 步：注册** —— 打开 `src/utils/model_loader.py`，做两处补充：

```python
# (a) MODEL_LIST 元数据
MODEL_LIST = {
    ...
    "MyNet": {"description": "My custom network", "category": "cnn"},
}

# (b) build_model 里的 REGISTRY
def _mynet():
    from src.network.MyNet import MyNet
    return MyNet(num_classes=num_classes)

REGISTRY = {
    ...
    "MyNet": _mynet,
}
```

**第 3 步：训练 + 评估** —— 与已知模型完全相同的用法：

```bash
python train.py --model MyNet --train_file busi_train1.txt --val_file busi_valid1.txt --custom_message "mynet_v1"
python evaluate.py --model <训练产生的 stem> --test_file test1.txt --save_viz
```

> 评估输出全套指标：`test_loss / test_iou / test_dice / test_SE / test_PC / test_F1 / test_ACC`，`--save_viz` 会把预测掩码存到 `test_results/`。指标与耗时同时追加到 `checkpoint/{stem}/{stem}_eval.log`。

训练时每个 epoch 的训练和验证各显示一个 tqdm 进度条，从 0 重新增长，显示 batch 数、预计剩余时间及 loss/IoU；训练条还显示学习率。每轮结束保留原有指标摘要和日志。蒸馏训练也显示每轮进度，生成教师软标签时另有进度提示。缺少依赖时运行 `pip install tqdm`。

---

## 六、场景 4：批量训练 5 个模型并在 3 个数据集上评估

打开 `run_batch.sh`，编辑顶部的硬编码数组：

```bash
GPU=0
TRAIN_DATA_DIR="./data/busi"                # 训练数据集

MODELS=("U_Net" "AttU_Net" "UNetplus_L3" "Mobile_U_ViT" "CMUNeXt")
BASE_LRS=(0.01   0.01       0.005          0.001           0.0005)
EPOCHS=(  20     20         40             60              80)

# 当前没有测试集，以下数组默认均为空；纳入测试数据后按顺序配套填写：
EVAL_DATASETS=( "./data/busi" "./data/iChallenge_GON" )
EVAL_TEST_FILES=( "test1.txt" "test1.txt" )
EVAL_MASK_TARGETS=( "binary" "cup" )
```

然后一条命令跑完全部：

```bash
bash run_batch.sh
```

**它会自动做这些事**（全程 GPU 0、串行、无人值守；当前测试数组为空时只执行训练）：

1. 依次训练 5 个模型，各自使用数组里对应的 `base_lr` 和 `epoch`；
2. 每训完一个，自动定位刚生成的 checkpoint（取 `checkpoint/` 下最新目录）；
3. 拿这个 checkpoint 分别在配置的测试数据集上各跑一次 `evaluate.py`；
4. 每个模型的全部终端输出用 `tee` 留存到 `batch_logs/{模型}_{时间}.log`；训练日志在 `checkpoint/{stem}/`，评估记录追加在 `{stem}_eval.log`。

> 该脚本不考虑断电恢复——定位 checkpoint 依赖"最新目录"，请勿在跑批期间并行启动其他训练。`set -eo pipefail` 保证任何一步失败立即停下并在日志中留痕。

---

## 七、知识蒸馏（可选进阶）

用大模型的软标签指导小模型训练：

```bash
# Teacher 来自检查点，Student 从头训练
python distill.py --teacher 20260921_1639_10213_U_Net --student Mobile_U_ViT

# Student 也从检查点续训
python distill.py --teacher <teacher_stem> --student <student_stem> --resume
```

- 损失 = 硬标签 BCE-Dice + 0.5 × 软标签 MSE；
- Teacher 概率缓存目录 `teacher_probs/` 训练结束自动清理；
- 同样记录蒸馏耗时与任务总耗时至日志；`distill.sh` 为对应硬编码启动脚本。

---

## 八、命令行参数速查

训练、评估和蒸馏的共享默认值位于 `src/utils/config.py` → `DEFAULTS`；独立导出参数见 `python export_onnx.py --help`。.sh 中显式填写的参数会覆盖 Python 默认值。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | — | 架构名（train 新训必填）或 checkpoint stem（evaluate 必填） |
| `--ckpt` | — | 断点续训的 stem，与 `--model` 二选一 |
| `--data_dir` | `./data/busi` | 数据集根目录 |
| `--seed` | `41` | 数据划分与训练随机种子 |
| `--train_file` / `--val_file` | `busi_train1.txt` / `busi_val1.txt` | 划分清单（相对 data_dir），同时置空则自动划分 |
| `--val_split` | `0.3` | 自动划分时的验证集比例 |
| `--test_file` | 必填 | evaluate 的测试清单（相对 data_dir），例如 test1.txt；不使用 val_file |
| `--mask_target` | `binary` | BUSI 用 binary；iChallenge_GON 视杯用 cup |
| `--epoch` | `20` | 训练轮数（续训为追加轮数） |
| `--base_lr` | `0.01` | 初始学习率（多项式衰减） |
| `--batch_size` | `8` | 批大小 |
| `--img_size` | `256` | 输入分辨率（SwinUnet 固定 224） |
| `--num_classes` | `1` | 分割通道数（二分类为 1） |
| `--gpu` | `0` | GPU 编号（写入 CUDA_VISIBLE_DEVICES） |
| `--custom_message` | `""` | 写入日志头部的实验备注 |
| `--save_viz` | 关 | （evaluate）保存预测掩码到 test_results/ |
| `--resume` | 关 | （distill）student 从 checkpoint 续训 |

完整列表：`python train.py --help` / `python evaluate.py --help` / `python distill.py --help`。

当前 `config.py` 的验证清单默认值仍为 `busi_val1.txt`，而实际文件为 `busi_valid1.txt`。直接调用 Python 入口时请显式传入实际清单；`train.sh` 已指定正确文件。更换数据集时，也需同步修改清单和 `--mask_target`，批量脚本的训练命令同样如此。

---

## 九、Checkpoint 与日志规则

### 命名规则

```text
{YYYYMMDD}_{HHMM}_{SSmmm}_{模型名}
例：20260921_1639_10213_U_Net
       │        │      │
       │        │      └─ 秒(2位)+毫秒(3位)：10秒213毫秒
       │        └──────── 时分
       └───────────────── 日期
```

毫秒位用于多块 GPU 各自独立训练时，同秒启动也不会目录撞名；字母序 = 时间序。

### 日志结构（train / distill 生成的 .log）

| 段落 | 内容 |
| --- | --- |
| PRETRAIN PARAMS | 本次运行完整参数快照 |
| CUSTOM MESSAGE | `--custom_message` 备注 |
| MODEL ARCHITECTURE | torchinfo 导出的层结构与参数量 |
| TRAINING LOG | 逐 epoch 指标 + **Training time / Total task time** |
| POSTTRAIN RESULTS | 历史最佳 IoU，以及最后完成轮的验证指标总表 |

### 耗时记录格式

统一为 `HHH:MM:SS.mmm`（小时可超过 24，不归零）。例如跑 20 天记录为 `480:00:01.143`：

- `train.py` 记录：**Training time**（训练+验证循环）与 **Total task time**（整个脚本，含建模型、加载数据、ONNX 导出）；
- `distill.py` 记录：**Distillation time** 与 **Total task time**；
- `evaluate.py` 记录：**Evaluation time**（测试循环）与 **Total task time**，并随指标追加到 `checkpoint/{stem}/{stem}_eval.log`。

---

## 十、支持的模型架构

| 模型标识 | 分类 | 说明 |
| --- | --- | --- |
| `U_Net` | CNN | 经典对称医学分割网络 |
| `U_Net_re` | CNN | 改进版经典 U-Net |
| `AttU_Net` | CNN | 注意力门控 U-Net |
| `UNetplus` | CNN | U-Net++（ResNet-34 骨干） |
| `UNetplus_L3` | CNN | 3 层 U-Net++（ResNet-34 骨干） |
| `UNetplus_L5` | CNN | 5 层 U-Net++（ResNet-34 骨干） |
| `UNet3plus` | CNN | U-Net 3+（全尺度跳跃连接 + 深度监督） |
| `UNext` | CNN | MLP + 轻量卷积高效架构 |
| `CMUNet` | CNN | 上下文多尺度提取 |
| `CMUNeXt` | CNN | 现代 ConvNeXt 块高效结构 |
| `Mobile_U_ViT` | Hybrid | 轻量 CNN + ViT 低延迟混合 |
| `MedT` | Transformer | 轴向注意力 Medical Transformer（内置 256） |
| `TransUnet` | Transformer | CNN 局部特征 + ViT 全局注意力 |
| `SwinUnet` | Transformer | Swin 滑窗纯 Transformer（固定 224） |

---

## 十一、数据集规范

```text
data/{dataset}/
├── images/                   # train/valid/test 图片统一存放
│   └── case_001.jpg
├── masks/                    # 所有图片的真值掩码，直接放在该目录
│   └── case_001.bmp
├── {dataset}_train_id.txt    # 每行完整图片文件名，例如 case_001.jpg
├── {dataset}_valid_id.txt
└── {dataset}_test_id.txt     # 以后纳入测试数据时自行创建
```

- 图片支持 `.png/.jpg/.jpeg/.bmp/.tif/.tiff`，图片与掩码按同名主干配对，后缀可以不同；清单保留文件名中的空格。
- 默认 `--mask_target binary`：单通道二值掩码，0=背景、255=前景；iChallenge_GON 使用 `--mask_target cup`，将原始 0 映射为视杯前景，128/255 映射为背景。训练与评估必须保持相同的标签目标；BUSI 使用 `cup` 会将背景误当成前景。
- 训练、验证、测试统一从 `images/` 和 `masks/` 读取；清单决定样本用途。训练和蒸馏使用 `--train_file/--val_file`，评估必须指定 `--test_file`，不会自动划分或改用验证集。清单相对 `--data_dir` 解析；上述 `_id.txt` 是推荐命名，原有清单名和 `test1.txt` 也可使用。
- 当前 iChallenge_GON 保留原始 400 张训练集和 400 张验证集，不再划分 8:2；BUSI 的样本归属和顺序保持不变，仅给清单补齐后缀。
- 当前未纳入独立测试数据。以后将测试图片和对应掩码加入统一目录，再创建测试清单；train/valid/test 清单必须互不重叠。优先保留官方划分；没有官方划分时，可离线按固定种子和比例生成清单，同一患者的数据应在同一集合。已有测试清单时，不要对整个 `images/` 再随机划分。
- `--test_file` 是评估入口的清单参数，不会改变样本的实际用途。当前 `evaluate.sh` 使用 `busi_valid1.txt`，结果属于验证集评估，即使输出字段名为 `test_iou`；独立测试时换成未参与训练和模型选择的测试清单。

```bash
# 从测试图片生成带后缀的清单（准备好测试数据后）
python predata/convert.py --folder <原始测试图片目录> --output ./data/iChallenge_GON/test1.txt

# 测试集评估
python evaluate.py --model <checkpoint_stem> --data_dir ./data/iChallenge_GON --test_file test1.txt --mask_target cup
bash evaluate.sh --test_file test1.txt --model <checkpoint_stem> --data_dir ./data/iChallenge_GON --mask_target cup
```

---

## 十二、常见问题

- **ONNX 导出失败怎么办？** 训练成果（.pth、.log）不受影响，终端会打印警告。使用 `export_onnx.py` 加载已有最佳权重重新导出；修改导出参数后必须重新生成文件，并确认部署命令使用的是新文件。
- **多 GPU 机器上只想用某张卡？** `--gpu 1` 即可（等效 `CUDA_VISIBLE_DEVICES=1`）；若可见多卡，代码自动启用 `DataParallel`（读取旧权重会自动剥离 `module.` 前缀）。
- **想精确复现实验？** 保持 `--seed`、`--data_dir`、清单文件与超参数一致即可；日志头部 PRETRAIN PARAMS 就是完整快照。
- **可以同时跑多个独立训练吗？** 可以（多卡各起一个），毫秒级时间戳保证目录不撞名；但**不要**与 `run_batch.sh` 混跑（见场景 4 说明）。
- **上古格式的 checkpoint（`*_model_2026-07-08_16_24_02`）还能用吗？** 可以，stem 解析器兼容 `_model_` 旧格式与无毫秒的旧时间戳格式。

---

## 开源协议

本项目基于 [FengheTan9/Medical-Image-Segmentation-Benchmarks](https://github.com/FengheTan9/Medical-Image-Segmentation-Benchmarks) 构建，遵循项目根目录下的 `LICENSE` 声明与相关开源许可。
