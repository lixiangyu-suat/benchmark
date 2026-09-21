import argparse
import os
import signal
import sys

# 闂傚倸鍊搁崐椋庣矆娓氣偓瀹曨垶宕稿Δ鈧崒銊︾節婵犲倻澧曠痪鎯ь煼閺岀喖宕滆鐢盯鏌ｉ幘鍐叉殻闁哄本绋栫粻娑㈠箼閸愨敩锔界箾?Ensure project root is on sys.path 闂傚倸鍊搁崐椋庣矆娓氣偓瀹曨垶宕稿Δ鈧崒銊︾節婵犲倻澧曠痪鎯ь煼閺岀喖宕滆鐢盯鏌ｉ幘鍐叉殻闁哄本绋栫粻娑㈠箼閸愨敩锔界箾鐎涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺?
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ_ROOT not in sys.path:
    sys.path.insert(0, _PROJ_ROOT)

import torch
import torch.optim as optim

from src.utils.config import load_config
from src.utils.dataset import get_dataloaders
from src.utils.helpers import AverageMeter, seed_everything, timestamp, timestamp_short
from src.utils.losses import BCEDiceLoss
from src.utils.logger import CheckpointLogger
from src.utils.metrics import iou_score
from src.utils.model_loader import (
    build_model,
    load_checkpoint_meta,
    save_checkpoint,
    validate_model,
    resolve_ckpt_path,
)
from src.utils.convert_to_onnx import convert_pth_to_onnx


def parse_args():
    parser = argparse.ArgumentParser(description="Train a segmentation model")
    parser.add_argument("--model", type=str, required=True,
                        help="Model architecture name (see configs/modellists.yaml)")
    parser.add_argument("--cfg", type=str, default="configs/config.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Checkpoint stem to resume from "
                             "(e.g. UNet_model_2026-07-04_22_09_42)")
    return parser.parse_args()

def config_dataset(train_loader):
    # 假设已经获取了 loader
    # train_loader, val_loader = get_dataloaders(config)

    print("========== 输出一下 Transform 阶段归一化的结果 ==========")

    # 1. 从 dataloader 中抽取第一个 batch 的数据（这里返回的是字典）
    batch_data = next(iter(train_loader))

    # 2. 从字典中通过 key 提取出图像 Tensor
    images = batch_data["image"]
    labels = batch_data["label"]

    # 3. 打印基础信息，确认接口正确
    print(f"提取到的 image 形状: {images.shape}")
    print(f"提取到的 label 形状: {labels.shape}")
    print(f"当前 batch 的 case_name: {batch_data['name']}")

    print("-" * 40)

    # 4. 打印统计信息，查看归一化结果
    print(f"全局最大值 (Max): {images.max().item():.4f}")
    print(f"全局最小值 (Min): {images.min().item():.4f}")
    print(f"全局均值 (Mean): {images.mean().item():.4f}")
    print(f"全局标准差 (Std): {images.std().item():.4f}")

    # 5. 打印局部具体数值验证
    print("\n[示例] 第一张图，第二通道，左上角 10x10 像素的实际数值:")
    print(images[0, 1, :10, :10])

    print("=====================================================")

def main():
    args = parse_args()
    config = load_config(args.cfg)

    seed_everything(config["data"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 闂傚倸鍊搁崐椋庣矆娓氣偓瀹曨垶宕稿Δ鈧崒銊︾節婵犲倻澧曠痪鎯ь煼閺岀喖宕滆鐢盯鏌ｉ幘鍐叉殻闁哄本绋栫粻娑㈠箼閸愨敩锔界箾?Resume or scratch 闂傚倸鍊搁崐椋庣矆娓氣偓瀹曨垶宕稿Δ鈧崒銊︾節婵犲倻澧曠痪鎯ь煼閺岀喖宕滆鐢盯鏌ｉ幘鍐叉殻闁哄本绋栫粻娑㈠箼閸愨敩锔界箾鐎涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮?
    if args.ckpt is not None:
        model_name = validate_model(args.ckpt)
        model = build_model(config, args.ckpt, device)
        ckpt_path = resolve_ckpt_path(args.ckpt)
        sd, ckpt_epoch, best_iou = load_checkpoint_meta(ckpt_path, device)
        model.load_state_dict(sd)
        start_epoch = ckpt_epoch + 1

        print(f"=> Resumed from epoch {ckpt_epoch}  (best_iou: {best_iou:.4f})")
        ckpt_stem = args.ckpt.replace("_interrupted", "")
        ckpt_new_stem = f"{timestamp_short()}_{model_name}"
        os.makedirs(os.path.join("checkpoint", ckpt_stem), exist_ok=True)
    else:
        model_name = validate_model(args.model)
        model = build_model(config, args.model, device)
        start_epoch = 1
        best_iou = 0.0
        # New training: create a fresh timestamped directory.
        ckpt_stem = f"{timestamp_short()}_{model_name}"
        ckpt_new_stem = ckpt_stem
        os.makedirs(os.path.join("checkpoint", ckpt_stem), exist_ok=True)


    train_loader, val_loader = get_dataloaders(config)
    config_dataset(train_loader)

    base_lr = config["train"]["base_lr"]
    add_epochs = config["train"]["epoch"]
    end_epoch = start_epoch + add_epochs - 1
    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)
    criterion = BCEDiceLoss().to(device)

    # 闂傚倸鍊搁崐椋庣矆娓氣偓瀹曨垶宕稿Δ鈧崒銊︾節婵犲倻澧曠痪鎯ь煼閺岀喖宕滆鐢盯鏌ｉ幘鍐叉殻闁哄本绋栫粻娑㈠箼閸愨敩锔界箾?Logger 闂傚倸鍊搁崐椋庣矆娓氣偓瀹曨垶宕稿Δ鈧崒銊︾節婵犲倻澧曠痪鎯ь煼閺岀喖宕滆鐢盯鏌ｉ幘鍐叉殻闁哄本绋栫粻娑㈠箼閸愨敩锔界箾鐎涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻?
    log_path = os.path.join("checkpoint", ckpt_stem, f"{ckpt_new_stem}.log")
    logger = CheckpointLogger(log_path)
    logger.log_pretrain(config)
    logger.log_custom_message(config.get("log", {}).get("custom_message", ""))
    logger.log_architecture(model, (1, 3, config["train"]["img_size"],
                                    config["train"]["img_size"]))

    print(f"{len(train_loader)} iterations per epoch")
    print(f"Epochs: {start_epoch} -> {end_epoch}  ({add_epochs} runs)\n")

    iter_num = 0
    max_iters = len(train_loader) * add_epochs

    interrupted = False

    def _on_interrupt(sig, frame):
        nonlocal interrupted
        if interrupted:
            sys.exit(1)
        interrupted = True


    signal.signal(signal.SIGINT, _on_interrupt)

    # 闂傚倸鍊搁崐椋庣矆娓氣偓瀹曨垶宕稿Δ鈧崒銊︾節婵犲倻澧曠痪鎯ь煼閺岀喖宕滆鐢盯鏌ｉ幘鍐叉殻闁哄本绋栫粻娑㈠箼閸愨敩锔界箾?Training loop 闂傚倸鍊搁崐椋庣矆娓氣偓瀹曨垶宕稿Δ鈧崒銊︾節婵犲倻澧曠痪鎯ь煼閺岀喖宕滆鐢盯鏌ｉ幘鍐叉殻闁哄本绋栫粻娑㈠箼閸愨敩锔界箾鐎涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺?
    last_completed_epoch = start_epoch - 1

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            if interrupted:
                print("\n\u26a0  Interrupt received, finishing current epoch then saving...")
                break

            model.train()
            meters = {k: AverageMeter() for k in [
                "loss", "iou",
                "val_loss", "val_iou", "val_SE", "val_PC", "val_F1", "val_ACC",
            ]}

            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                iou, _, _, _, _, _, _ = iou_score(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                iter_num += 1
                lr = base_lr * (1.0 - iter_num / max_iters) ** 0.9
                for pg in optimizer.param_groups:
                    pg["lr"] = lr

                meters["loss"].update(loss.item(), images.size(0))
                meters["iou"].update(iou, images.size(0))

            model.eval()
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    outputs = model(images)

                    loss = criterion(outputs, labels)
                    iou, _, SE, PC, F1, _, ACC = iou_score(outputs, labels)

                    meters["val_loss"].update(loss.item(), images.size(0))
                    meters["val_iou"].update(iou, images.size(0))
                    meters["val_SE"].update(SE, images.size(0))
                    meters["val_PC"].update(PC, images.size(0))
                    meters["val_F1"].update(F1, images.size(0))
                    meters["val_ACC"].update(ACC, images.size(0))

            line = (
                f"Epoch Result[{epoch}/{end_epoch}]  "
                f"train_loss: {meters['loss'].avg:.4f}  "
                f"train_iou: {meters['iou'].avg:.4f}  |  "
                f"val_loss: {meters['val_loss'].avg:.4f}  "
                f"val_iou: {meters['val_iou'].avg:.4f}  "
                f"val_SE: {meters['val_SE'].avg:.4f}  "
                f"val_PC: {meters['val_PC'].avg:.4f}  "
                f"val_F1: {meters['val_F1'].avg:.4f}  "
                f"val_ACC: {meters['val_ACC'].avg:.4f}"
            )
            print(line)
            logger.log_training(line)

            last_completed_epoch = epoch

            if meters["val_iou"].avg > best_iou:
                best_iou = meters["val_iou"].avg
                ckpt_best_path = os.path.join("checkpoint", ckpt_stem, f"{ckpt_stem}_best.pth")
                save_checkpoint(
                    ckpt_best_path, model, epoch, best_iou,
                )
                print(f"  => saved best model (val_iou: {best_iou:.4f})")

    except KeyboardInterrupt:
        interrupted = True
        print("\n\u26a0  Caught KeyboardInterrupt.")

    # 闂傚倸鍊搁崐椋庣矆娓氣偓瀹曨垶宕稿Δ鈧崒銊︾節婵犲倻澧曠痪鎯ь煼閺岀喖宕滆鐢盯鏌ｉ幘鍐叉殻闁哄本绋栫粻娑㈠箼閸愨敩锔界箾?Finalise 闂傚倸鍊搁崐椋庣矆娓氣偓瀹曨垶宕稿Δ鈧崒銊︾節婵犲倻澧曠痪鎯ь煼閺岀喖宕滆鐢盯鏌ｉ幘鍐叉殻闁哄本绋栫粻娑㈠箼閸愨敩锔界箾鐎涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ婊堟⒒閸屾艾鈧绮堟笟鈧畷顖炲锤濡も偓閸屻劍绻濇繝鍌滃缁炬儳顭烽弻鐔煎礈瑜忕敮娑㈡煟閹惧啿鏆ｉ柡灞剧缁犳盯骞欓崘鈹附绻涚€涙鐭掔紒鐘崇墪椤繐煤椤忓嫮顦ㄩ梺鍦帛鐢帗娼忛崨瀛樷拺缂佸顑欓崕鎰版煙閻熺増鍠樼€殿喛顕ч鍏煎緞婵犲嫷妲┑鐘灱濞夋盯鏁冮敐鍡欑彾闁哄洢鍨洪埛鎺懨归敐鍥剁劸闁哄棝浜堕弻娑樜熼懡銈囩厜閻庤娲橀崹鍧楃嵁濡偐纾兼俊顖滃帶鐢姊绘担渚劸缂佺粯鍔欒棟妞ゆ牗绋撻々鎻捨旈敐鍛殲闁抽攱鍨块弻娑㈠箛椤撶偟绁烽柣銏╁灛閸旀垿寮诲☉姘ｅ亾閿濆骸浜濈€规洖鐭傞弻锛勪沪閻ｅ睗褏鈧娲橀〃鍡楊嚗閸曨剛绡€濞达絽澹婂Λ?
    if interrupted:
        logger.log_training("--- TRAINING INTERRUPTED ---")
        logger.log_training(f"Last completed epoch: {last_completed_epoch}")

    # Final checkpoint save (.pth)
    final_ckpt = os.path.join("checkpoint", ckpt_stem, f"{ckpt_stem}_final.pth")
    save_checkpoint(final_ckpt, model, last_completed_epoch, best_iou)

    # Final checkpoint save (.onnx )
    final_onnx = os.path.join("checkpoint", ckpt_stem, f"{ckpt_stem}_final.onnx")
    convert_pth_to_onnx(
        model=model,
        pth_path=final_ckpt,
        onnx_path=final_onnx,
        dummy_input=(1, 3, 224, 224),
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}} # 若需要动态 batch
    )

    if args.ckpt is not None:
        # Resume: rename T0.pth -> T1.pth, rename folder T0 -> T1
        t1_pth = os.path.join("checkpoint", ckpt_stem, f"{ckpt_new_stem}_final.pth")
        os.rename(final_ckpt, t1_pth)
        old_dir = os.path.join("checkpoint", ckpt_stem)
        new_dir = os.path.join("checkpoint", ckpt_new_stem)
        os.rename(old_dir, new_dir)
        log_path = os.path.join("checkpoint", ckpt_new_stem, f"{ckpt_new_stem}.log")
        logger.log_path = log_path
        print(f"=> Renamed checkpoint folder: {ckpt_stem} -> {ckpt_new_stem}")
    elif interrupted:
        print(f"=> Saved checkpoint: {ckpt_stem}/{ckpt_stem}_final.pth (best: {ckpt_stem}/{ckpt_stem}_best.pth)")
    else:
        print(f"=> Training finished (best val_iou: {best_iou:.4f})")

    final_metrics = {k: meters[k].avg for k in
                     ["val_loss", "val_iou", "val_SE", "val_PC", "val_F1", "val_ACC"]}
    logger.log_posttrain(best_iou, final_metrics)
    logger.flush()
    print(f"=> Log saved to {log_path}")


if __name__ == "__main__":
    main()
