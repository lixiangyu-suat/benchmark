from argparse import Namespace  # 引入原生解包神器

def build_model(configs, parser=None):
    """
    使用数据驱动的 dict 映射构建模型（已移除 xxx_based 路径）
    """
    # 【自动提取核心模型名】即使传入 "U_Net_model_2026_07_04" 也能安全切出 "U_Net"
    raw_model_name = configs.model
    model_name = raw_model_name.split('_model_')[0] if '_model_' in raw_model_name else raw_model_name

    num_classes = configs.num_classes
    img_size = configs.img_size
    parser = parser

    # ============ 纯净的模型定义（移除了原作者的分类夹层） ============
    def get_unet():
        from src.network.U_Net import U_Net
        return U_Net(output_ch=num_classes)
        
    def get_unet_re():
        from src.network.U_Net_re import U_Net
        return U_Net(ch_out=num_classes)
        
    def get_unet_plus():
        from src.network.UNetplus import ResNet34UnetPlus
        return ResNet34UnetPlus(num_class=num_classes)
        
    def get_unet3_plus():
        from src.network.UNet3plus.UNet3plus import UNet3plus
        return UNet3plus(n_classes=num_classes)
        
    def get_unext():
        from src.network.UNeXt import UNext
        return UNext(output_ch=num_classes)
        
    def get_attu_net():
        from src.network.AttU_Net import AttU_Net
        return AttU_Net(output_ch=num_classes)
        
    def get_cmunet():
        from src.network.CMUNet import CMUNet
        return CMUNet(output_ch=num_classes)
        
    def get_cmunext():
        from src.network.CMUNeXt import cmunext
        return cmunext(num_classes=num_classes)

    def get_mobile_uvit():
        from src.network.Mobile_U_ViT import mobileuvit
        return mobileuvit(out_channel=num_classes)

    def get_med_t():
        from ..network.medicalT.axialnet import MedT # pylance自动推导路径!
        return MedT(img_size=img_size, imgchan=3, num_classes=num_classes)
    
    def get_transunet():
        from ..network.transUnet.transunet import TransUnet
        return TransUnet(img_ch=3, output_ch=num_classes)

    def get_swinunet():
        from src.network.swinUnet.vision_transformer import SwinUnet
        from src.network.swinUnet.config import get_config

        """
        【get_config(Namespace(**configs['model']))流转说明】

        1. config (原生字典)
        - 来源：从 YAML 直接加载的数据源。
        - 形态：纯正的 Python dict，只认键值对（如 config['model']['cfg']）。

        2. Namespace (伪装者)
        - 作用：将字典转为对象属性（把 ['cfg'] 变成 .cfg）。
        - 目的：捏造一个和 argparse 输出一模一样的假对象，去“骗”底层代码。

        3. swin_config (YACS 配置树)
        - 形态：CV 领域特有的庞大、层层嵌套的 YACS 配置对象。
        - 机制：拿着 Namespace 递来的图纸路径，在内部自动“生长”展开。
        - 结果：生成带有 .MODEL.SWIN.PATCH_SIZE 这种复杂节点的大树。
        """
        swin_config = get_config(Namespace(**configs['model']))
        
        return SwinUnet(swin_config, img_size=img_size, num_classes=num_classes)
    
    # 模型大字典：核心映射表
    MODEL_REGISTRY = {
        "U_Net": get_unet,
        "U_Net_re": get_unet_re,
        "UNetplus": get_unet_plus,
        "UNet3plus": get_unet3_plus,
        "UNext": get_unext,
        "AttU_Net": get_attu_net,
        "CMUNet": get_cmunet,
        "CMUNeXt": get_cmunext,
        "Mobile_U_ViT": get_mobile_uvit,
        "TransUnet": get_transunet,
        "SwinUnet": get_swinunet,
        "MedT": get_med_t,
    }

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"[!] 找不到模型: {model_name} (原始输入: {raw_model_name})。请检查拼写或在 builder.py 中注册。")

    print(f"==> 正在构建模型骨架: {model_name}")
    model = MODEL_REGISTRY[model_name]()
    return model.cuda()