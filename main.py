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
from src.utils.dataloader import get_data


parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/train.yaml', help='path to config file')
parser.add_argument('--ckpt', type=str, default='UNetplus_model_2026-07-04_23_17_55', help='checkpoint in ./checkpoint')
sys_args = parser.parse_args()

# yaml2dict
config_dict = yaml_config(sys_args.cfg)
seed = config_dict['data']['seed']

model_name = sys_args.ckpt

seed_torch(seed)

model_name_hash = get_time_format() + "_" + str(random.randint(1, 9999))
tmp_file_path = 'checkpoint/{}_model_{}_training.pth'.format(model_name, model_name_hash)


def main(config):
    # 提取超参数
    base_lr = config['train']['base_lr']
    max_epoch = config['train']['epoch']
    train_file_dir = config['data']['train_file_dir']
    val_file_dir = config['data']['val_file_dir']

    trainloader, valloader = get_data(config)
    
    # 把整本字典传给 builder，让它自己去查需要的参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config, config_dict["model"]["name"], device)

    print("train file dir:{} val file dir:{}".format(train_file_dir, val_file_dir))

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    criterion = losses.__dict__['BCEDiceLoss']().cuda()

    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    iter_num = 0
    max_iterations = len(trainloader) * max_epoch

    for epoch_num in range(1, max_epoch+1):
        model.train()
        avg_meters = {'loss': AverageMeter(),
                      'iou': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'val_SE': AverageMeter(),
                      'val_PC': AverageMeter(),
                      'val_F1': AverageMeter(),
                      'val_ACC': AverageMeter()}

        for i_batch, sampled_batch in enumerate(trainloader): #手动装载一个batch批次的data

            img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            img_batch, label_batch = img_batch.cuda(), label_batch.cuda()

            outputs = model(img_batch)

            loss = criterion(outputs, label_batch) #打分
            iou, dice, _, _, _, _, _ = iou_score(outputs, label_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num = iter_num + 1 # 需要放前面，迭代完一次之后立马+1，才能正确计算lr
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9 # 严重问题：第一轮训练结束后，学习率赋值了初始base_lr，观察发现学习率的迭代是落后一轮的
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
            
            avg_meters['loss'].update(loss.item(), img_batch.size(0)) # 通过AverageMeter自动更新acc等评分
            avg_meters['iou'].update(iou, img_batch.size(0))

        model.eval()
        with torch.no_grad():
            for i_batch, sampled_batch in enumerate(valloader):
                img_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                img_batch, label_batch = img_batch.cuda(), label_batch.cuda()
                output = model(img_batch)
                loss = criterion(output, label_batch)
                iou, _, SE, PC, F1, _, ACC = iou_score(output, label_batch)
                avg_meters['val_loss'].update(loss.item(), img_batch.size(0))
                avg_meters['val_iou'].update(iou, img_batch.size(0))
                avg_meters['val_SE'].update(SE, img_batch.size(0))
                avg_meters['val_PC'].update(PC, img_batch.size(0))
                avg_meters['val_F1'].update(F1, img_batch.size(0))
                avg_meters['val_ACC'].update(ACC, img_batch.size(0))

        print('epoch [%d/%d]  train_loss : %.4f, train_iou: %.4f - val_loss %.4f - val_iou %.4f - val_SE %.4f - '
              'val_PC %.4f - val_F1 %.4f - val_ACC %.4f '
            % (epoch_num, max_epoch, avg_meters['loss'].avg, avg_meters['iou'].avg,
               avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_SE'].avg,
               avg_meters['val_PC'].avg, avg_meters['val_F1'].avg, avg_meters['val_ACC'].avg))

        if avg_meters['val_iou'].avg > best_iou:
            if not os.path.isdir("./checkpoint"):
                os.makedirs("./checkpoint")
            torch.save(model.state_dict(), tmp_file_path)
            best_iou = avg_meters['val_iou'].avg
            print("=> saved best model")

    # 把临时文件保存为需要的文件
    if os.path.exists(tmp_file_path):
        os.rename(tmp_file_path, "./checkpoint/{}_model_{}.pth".format(model_name, get_time_format()))
        return "Training Finished!"
    else:
        return "Training Finished, but no best ideal model!"


if __name__ == "__main__":
    # 统一将字典传入主函数
    main(config_dict)