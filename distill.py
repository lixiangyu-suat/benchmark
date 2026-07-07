import os
import argparse
import random
import torch # type: ignore
import torch.optim as optim # type: ignore

import src.utils.losses as losses
from src.utils.util import AverageMeter
from src.utils.metrics import iou_score
from src.utils.seeds import seed_torch
from src.utils.timing import get_time_format
from src.utils.modelloader import build_model
from src.utils.get_yaml_config import yaml_config
from src.utils.dataloader import get_data, get_train_transform, DistillationDataset, MedicalDataSets

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def teacher(config_dict, teacher_model):
    device = next(teacher_model.parameters()).device
    teacher_model.eval()
    
    # 在训练集上学习
    val_file_dir = config_dict['train']['train_file_dir']
    # 构建数据集（使用验证集 transform，通常无需数据增强）
    from src.utils.dataloader import get_val_transform, MedicalDataSets
    val_transform = get_val_transform(config_dict['eval']['img_size'])
    db_val = MedicalDataSets(
        base_dir=config_dict['data']['base_dir'],
        split="val",  # 或者根据你的需求，这里最好允许传入自定义目录
        transform=val_transform,
        val_file_dir=val_file_dir,
        # 如果你修改了 MedicalDataSets，可以加参数 return_name=True
    )
    val_loader = DataLoader(db_val, batch_size=config_dict['eval']['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)
    
    # 创建保存目录
    save_dir = "teacher_probs"
    os.makedirs(save_dir, exist_ok=True)
    
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            case_names = batch['name']  # 必须存在
            outputs = teacher_model(images)
            probs = torch.sigmoid(outputs)  # 概率图，形状 [B, 1, H, W]
            
            # 逐个保存
            for i, name in enumerate(case_names):
                prob_tensor = probs[i].cpu()  # [1, H, W]
                save_path = os.path.join(save_dir, f"{name}.pt")
                torch.save(prob_tensor, save_path)
    
    return save_dir  # 返回保存路径


def student(config_dict, student_model, teacher_prob_dir):
    device = next(student_model.parameters()).device
    student_model.train()
    
    # 训练时使用增强 transform
    train_transform = get_train_transform(config_dict['train']['img_size'])
    train_file_dir = config_dict['data']['train_file_dir']
    # 构建基础数据集
    base_train = MedicalDataSets(
        base_dir=config_dict['data']['base_dir'],
        split='train',
        transform=train_transform,
        val_file_dir=train_file_dir
    )
    # 包装为蒸馏数据集
    distill_dataset = DistillationDataset(base_train, teacher_prob_dir)
    train_loader = DataLoader(distill_dataset,
                              batch_size=config_dict['train']['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    
    # 定义损失
    hard_loss_fn = losses.__dict__['BCEDiceLoss']().to(device)
    distill_loss_fn = torch.nn.MSELoss()
    
    optimizer = optim.Adam(student_model.parameters(), lr=config_dict['train']['base_lr'])
    
    # 训练循环（参考 main.py 风格，加入学习率调整等）
    max_epoch = config_dict['train']['epoch']
    for epoch in range(1, max_epoch+1):
        avg_loss = 0.0
        for batch in train_loader:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            teacher_probs = batch['teacher_prob'].to(device)
            
            optimizer.zero_grad()
            outputs = student_model(images)
            student_probs = torch.sigmoid(outputs)
            
            hard_loss = hard_loss_fn(outputs, labels)
            distill_loss = distill_loss_fn(student_probs, teacher_probs)
            alpha = 0.5   # 可调
            loss = hard_loss + alpha * distill_loss
            
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
        
        avg_loss /= len(train_loader)
        print(f"Epoch {epoch}/{max_epoch}, Average Loss: {avg_loss:.4f}")
        # 可以添加验证逻辑，这里暂略


"""
4. student 函数未保存模型
训练结束后没有保存学生模型的 checkpoint，这会导致训练结果丢失。参考 main.py，应保存最佳模型。

5. 学习率调度缺失
main.py 中使用了动态学习率调整（lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9），你的 student 函数没有，可能影响收敛。建议加入或直接使用 torch.optim.lr_scheduler。

6. teacher 函数每次运行覆盖之前的结果
save_dir = "teacher_probs" 固定，如果多次运行不同教师，会覆盖。建议加入教师名字标识，例如 f"teacher_probs_{teacher_model_name}"。
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/train.yaml', help='path to config file')
    parser.add_argument('--student_model_ckpt', default='UNetplus', help='student_model_checkpoint in ./checkpoint')
    parser.add_argument('--teacher_model_ckpt', type=str, default='UNetplus_model_2026-07-04_23_17_55', help='teacher_model_checkpoint in ./checkpoint')
    sys_args = parser.parse_args()

    # yaml2dict
    config_dict = yaml_config(sys_args.cfg)
    seed = config_dict['data']['seed']
    seed_torch(seed)

    teacher_model_name = sys_args.teacher_model_ckpt
    student_model_name = sys_args.student_model_ckpt

    teacher_model = build_model(config_dict, teacher_model_name, device, eval=True)
    student_model = build_model(config_dict, student_model_name, device, eval=False)

    # 老师蒸馏
    teacher_prob_dir = teacher(config_dict, teacher_model)

    # 学生训练
    student(config_dict, student_model, teacher_prob_dir)


if __name__ == "__main__":
    main()