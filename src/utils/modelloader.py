# src/network/builder.py

def build_model(args, parser=None):
    """
    使用数据驱动的 dict 映射构建模型（已移除 xxx_based 路径）
    """
    # 【自动提取核心模型名】即使传入 "U_Net_model_2026_07_04" 也能安全切出 "U_Net"
    raw_model_name = args.model
    model_name = raw_model_name.split('_model_')[0] if '_model_' in raw_model_name else raw_model_name

    num_classes = args.num_classes
    img_size = args.img_size
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

    def get_transformer_model():
        # Transformer系列内部可能会套娃，这里根据你展平后的实际路径调整
        from src.network.transformer_based_network import get_transformer_based_model
        return get_transformer_based_model(
            parser=parser, 
            model_name=model_name, 
            img_size=img_size, 
            num_classes=num_classes, 
            in_ch=3
        )
    
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
        #"SwinUnet": get_swinunet,
        "MedT": get_med_t,
    }

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"[!] 找不到模型: {model_name} (原始输入: {raw_model_name})。请检查拼写或在 builder.py 中注册。")

    print(f"==> 正在构建模型骨架: {model_name}")
    model = MODEL_REGISTRY[model_name]()
    return model.cuda()