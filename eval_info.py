import argparse
import torch
import re
from datetime import datetime
from torchinfo import summary

default_model = "U_Net_re"

# 1. 配置命令行参数获取 model_name 和权重路径
parser = argparse.ArgumentParser(description="Evaluate model info")
parser.add_argument('--model', type=str, default=default_model, help=f'模型名称 (默认: {default_model})') # 输入.pth权重的名字即可调用，如U_Net_re_model_2026-07-02
parser.add_argument('--path', type=str, default=None, help='手动指定 .pth 权重路径 (可选)')
parser.add_argument('--img_size', type=int, default=256, help='img size of per batch')
parser.add_argument('--num_classes', type=int, default=1, help='seg num_classes')
args = parser.parse_args()

def get_time_format():
    """
    获取当前时间的自定义格式字符串。
    格式形如: 2026-01-01_01_02_03
    """
    return datetime.now().strftime("%Y-%m-%d_%H_%M_%S")

def get_model(args, parser=None):
    
  # if re.match("CMUNet", args.model) != None: # 小心前后缀影响；用re.match匹配字符串的开始
    if "CMUNet" in args.model:
        from src.network.conv_based.CMUNet import CMUNet
        model = CMUNet(output_ch=args.num_classes).cuda()
        
    elif "CMUNeXt" in args.model:
        from src.network.conv_based.CMUNeXt import cmunext
        model = cmunext(num_classes=args.num_classes).cuda()
    
    elif "U_Net_re" in args.model: # 小心前后缀影响
        from src.network.conv_based.U_Net_re import U_Net
        model = U_Net(ch_out=args.num_classes).cuda()

    elif "U_Net" in args.model:
        from src.network.conv_based.U_Net import U_Net
        model = U_Net(output_ch=args.num_classes).cuda()
        
    elif "AttU_Net" in args.model:
        from src.network.conv_based.AttU_Net import AttU_Net
        model = AttU_Net(output_ch=args.num_classes).cuda()
        
    elif "UNext" in args.model:
        from src.network.conv_based.UNeXt import UNext
        model = UNext(output_ch=args.num_classes).cuda()
        
    elif "UNetplus" in args.model:
        from src.network.conv_based.UNetplus import ResNet34UnetPlus
        model = ResNet34UnetPlus(num_class=args.num_classes).cuda()
        
    elif "UNet3plus" in args.model:
        from src.network.conv_based.UNet3plus import UNet3plus
        model = UNet3plus(n_classes=args.num_classes).cuda()

    elif "Mobile_U_ViT" in args.model:
        from src.network.hybrid_based.Mobile_U_ViT import mobileuvit
        model = mobileuvit(out_channel=args.num_classes).cuda()
        
    else:
        # 只有在运行 Transformer 系列模型（如 TransUnet, SwinUnet, MedT）时，才会导入这个包
        from src.network.transfomer_based.transformer_based_network import get_transformer_based_model
        model = get_transformer_based_model(parser=parser, model_name=args.model, img_size=args.img_size,
                                            num_classes=args.num_classes, in_ch=3).cuda()
    model_path = f"./checkpoint/{args.model}.pth"
    return model, model_path

# 2. 加载模型和模型路径
model, model_path = get_model(args=args)
if args.path is not None:
    model_path = args.path

# 3. 读取 .pth 文件中的权重字典
print(f"正在为模型 [{args.model}] 加载权重文件: {model_path} ...")
state_dict = torch.load(model_path, map_location="cuda")

# 4. 把权重填入骨架中
model.load_state_dict(state_dict)

# 5. 评估
model = model.cuda()
model.eval()

summary(model, input_size=(1, 3, args.img_size, args.img_size))