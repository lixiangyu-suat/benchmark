"""集中式超参数中心（纯 Python，不再依赖 YAML 配置文件）。

所有默认值为原 ``configs/config.yaml`` 的定格值；三条流水线
（train / evaluate / distill）通过命令行参数覆盖默认值。
"""

# ======== 定格默认值（原 configs/config.yaml） ========

DEFAULTS = {
    # data
    "seed": 41,
    "data_dir": "./data/busi",
    "train_file": "busi_train1.txt",   # 相对 data_dir；留空则自动按比例划分
    "val_file": "busi_val1.txt",
    "val_split": 0.3,
    # model
    "num_classes": 1,
    # train
    "epoch": 20,
    "base_lr": 0.01,
    "batch_size": 8,
    "img_size": 256,
    # log
    "custom_message": "",
}

# SwinUnet 专属构造参数（其他模型忽略）
SWINUNET_OPTS = {
    "opts": None,
    "cfg": "../../Swin-Unet/configs/swin_tiny_patch4_window7_224_lite.yaml",
    "zip": "../data/transunet_ACDC.zip",
    "cache_mode": "part",
    "resume": None,
    "accumulation_steps": None,
    "use_checkpoint": False,
    "amp_opt_level": "O1",
    "tag": None,
    "eval": False,
    "throughput": False,
    "batch_size": 8,
}


# ======== argparse 参数注册 ========

def add_common_args(parser):
    """注册三条流水线共享的命令行参数。"""
    d = DEFAULTS
    parser.add_argument("--data_dir", type=str, default=d["data_dir"],
                        help=f"数据集根目录 (default: {d['data_dir']})")
    parser.add_argument("--seed", type=int, default=d["seed"],
                        help=f"数据划分与训练随机种子 (default: {d['seed']})")
    parser.add_argument("--train_file", type=str, default=d["train_file"],
                        help="训练集清单文件名（相对 data_dir）；"
                             "与 --val_file 同时置空则按 val_split 自动划分")
    parser.add_argument("--val_file", type=str, default=d["val_file"],
                        help="验证集清单文件名（相对 data_dir）")
    parser.add_argument("--val_split", type=float, default=d["val_split"],
                        help=f"自动划分时验证集比例 (default: {d['val_split']})")
    parser.add_argument("--num_classes", type=int, default=d["num_classes"],
                        help=f"分割通道数，二分类为 1 (default: {d['num_classes']})")
    parser.add_argument("--gpu", type=str, default="0",
                        help="使用的 GPU 编号，写入 CUDA_VISIBLE_DEVICES (default: 0)")
    parser.add_argument("--custom_message", type=str, default=d["custom_message"],
                        help="写入日志头部的自定义实验备注")


def add_train_args(parser):
    """注册训练相关超参数（train / distill 共用）。"""
    d = DEFAULTS
    parser.add_argument("--epoch", type=int, default=d["epoch"],
                        help=f"训练轮数，续训时追加同等轮数 (default: {d['epoch']})")
    parser.add_argument("--base_lr", type=float, default=d["base_lr"],
                        help=f"初始学习率，多项式衰减 (default: {d['base_lr']})")
    parser.add_argument("--batch_size", type=int, default=d["batch_size"],
                        help=f"训练批大小 (default: {d['batch_size']})")
    parser.add_argument("--img_size", type=int, default=d["img_size"],
                        help=f"输入分辨率 (default: {d['img_size']}; SwinUnet 固定 224)")


# ======== 配置字典构建 ========

def build_config(args):
    """把 argparse Namespace 组装成与原 YAML 同构的配置字典。

    下游模块（dataset / logger / model_loader）读取方式保持不变。
    """
    g = lambda key: getattr(args, key, DEFAULTS.get(key))
    img_size = g("img_size")
    batch_size = g("batch_size")
    return {
        "data": {
            "seed": g("seed"),
            "base_dir": g("data_dir"),
            "train_file": g("train_file"),
            "val_file": g("val_file"),
            "val_split": g("val_split"),
        },
        "model": {
            "num_classes": g("num_classes"),
            "SwinUnet": dict(SWINUNET_OPTS),
        },
        "train": {
            "epoch": g("epoch"),
            "base_lr": g("base_lr"),
            "batch_size": batch_size,
            "img_size": img_size,
        },
        "eval": {
            "batch_size": batch_size,
            "img_size": img_size,
        },
        "log": {
            "custom_message": g("custom_message"),
        },
    }
