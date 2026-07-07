import os
import argparse
import torch # type: ignore

from torch.utils.data import DataLoader
import src.utils.losses as losses
from src.utils.util import AverageMeter
from src.utils.metrics import iou_score
from src.utils.modelloader import build_model
from src.utils.dataloader import get_val_transform, MedicalDataSets
from src.utils.get_yaml_config import yaml_config

from torchvision.utils import save_image

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/eval.yaml', help='path to config file')
parser.add_argument('--ckpt', type=str, default='UNetplus_model_2026-07-04_23_17_55', help='checkpoint in ./checkpoint')
sys_args = parser.parse_args()

# yaml2dict
config_dict = yaml_config(sys_args.cfg)

batch_size = config_dict['eval']['batch_size']
img_size = config_dict['eval']['img_size']
data = config_dict['data']


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config_dict, sys_args.ckpt, device, eval=True)
    val_transform = get_val_transform(img_size)

    db_val = MedicalDataSets(base_dir=data['base_dir'], split="val", transform=val_transform, val_file_dir=data['val_file_dir'])
    val_loader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    criterion = losses.__dict__['BCEDiceLoss']().to(device)

    save_dir="validation_results"
    
    """执行验证，并且每隔十张图像保存一次预测结果到PNG文件"""
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_dice = 0.0
    val_rvd = 0.0
    os.makedirs(save_dir, exist_ok=True)  

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(val_loader):
            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            img_batch, label_batch = img_batch.to(device), label_batch.to(device)
            outputs = model(img_batch)
            loss = criterion(outputs, label_batch)

            val_loss += loss.item()

            iou, dice, rvd, _, _, _, _ = iou_score(outputs, label_batch)
            val_iou += iou
            val_dice += dice
            if rvd<1:
                val_rvd += rvd
            # 每隔十张图像保存一次预测结果
            if i_batch % 1 == 0:
                # 将模型输出转换为二值图像
                outputs = torch.sigmoid(outputs)
                outputs[outputs > 0.5] = 1
                outputs[outputs <= 0.5] = 0
                output_images = outputs.cpu().data
                
                # 保存图像
                for idx, img in enumerate(output_images):
                    save_path = os.path.join(save_dir, f"batch_{i_batch}_img_{idx}.png")
                    # 使用save_image从torchvision，或者使用其他方法将张量转换为图像并保存
                    save_image(img, save_path)

    val_loss /= len(val_loader)
    val_iou /= len(val_loader)
    val_dice /= len(val_loader)
    val_rvd /= len(val_loader)
    print(f'验证损失: {val_loss:.4f}, 验证IoU: {val_iou:.4f}, 验证dice：{val_dice:.4f}, 验证rvd：{val_rvd:.4f}')
