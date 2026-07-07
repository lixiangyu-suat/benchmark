import os
import cv2
import torch
from albumentations.augmentations import transforms # type: ignore
from albumentations.core.composition import Compose # type: ignore
from albumentations import RandomRotate90, Resize # type: ignore
from torch.utils.data import Dataset # type: ignore
from torch.utils.data import DataLoader # type: ignore

def get_val_transform(img_size):
    return Compose([
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])

def get_train_transform(img_size):
    Compose([
        RandomRotate90(),
        transforms.Flip(),
        Resize(img_size, img_size),
        transforms.Normalize(),
    ])

class DistillationDataset(Dataset):
    """包装 MedicalDataSets，额外加载教师软概率"""
    def __init__(self, base_dataset, teacher_prob_dir):
        self.base_dataset = base_dataset   # 这是一个 MedicalDataSets 实例
        self.teacher_prob_dir = teacher_prob_dir

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        sample = self.base_dataset[idx]   # 返回 {'image', 'label', 'name', 'idx'}
        case_name = sample['name']
        # 加载教师软概率
        prob_path = os.path.join(self.teacher_prob_dir, f"{case_name}.pt")
        teacher_prob = torch.load(prob_path)   # 形状 [1, H, W]
        # 注意：如果教师预测时使用了不同的 resize，这里可能需要 resize 到当前图像尺寸
        # 但一般保持一致，所以直接添加
        sample['teacher_prob'] = teacher_prob
        return sample
    

class MedicalDataSets(Dataset):
    def __init__(
            self,
            base_dir=None,
            split="train",
            transform=None,
            train_file_dir="train.txt",
            val_file_dir="val.txt",
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.train_list = []
        self.semi_list = []

        if self.split == "train":
            with open(os.path.join(self._base_dir, train_file_dir), "r") as f1:
                self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        elif self.split == "val":
            with open(os.path.join(self._base_dir, val_file_dir), "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("total {}  {} samples".format(len(self.sample_list), self.split))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        case = self.sample_list[idx]
        case_name = os.path.splitext(os.path.basename(case))[0]

        image = cv2.imread(os.path.join(self._base_dir, 'images', case + '.png')) # OpenCV 读图, 减少精度损失
        label = \
            cv2.imread(os.path.join(self._base_dir, 'masks', '0', case + '.png'), cv2.IMREAD_GRAYSCALE)[
                ..., None]

        augmented = self.transform(image=image, mask=label) # Albumentations, 减少精度损失
        image = augmented['image']
        label = augmented['mask']

        image = image.astype('float32') / 255 #平替 transforms.ToTensor(),像素值归一化
        image = image.transpose(2, 0, 1)

        label = label.astype('float32') / 255 #平替 transforms.ToTensor(),像素值归一化
        label = label.transpose(2, 0, 1)

        sample = {"image": image,
                  "label": label, 
                  "idx": idx,  
                  "name": case_name
                }
        return sample


def get_data(config, eval = False):
    # 用查字典的方式，先把变量拿出来，这样就不需要改动底下的逻辑了
    if (not eval):
        img_size = config['train']['img_size']
    else:
        img_size = config['eval']['img_size']
    model_name = config['model']['name']
    base_dir = config['data']['base_dir']
    train_file_dir = config['data']['train_file_dir']
    val_file_dir = config['data']['val_file_dir']
    batch_size = config['train']['batch_size']

    if model_name == "SwinUnet":
        img_size = 224

    train_transform = get_val_transform(img_size)

    val_transform = get_val_transform(img_size)

    
    db_train = MedicalDataSets(base_dir=base_dir, split="train",
                            transform=train_transform, train_file_dir=train_file_dir, val_file_dir=val_file_dir)
    db_val = MedicalDataSets(base_dir=base_dir, split="val", transform=val_transform,
                          train_file_dir=train_file_dir, val_file_dir=val_file_dir)
    print("train num:{}, val num:{}".format(len(db_train), len(db_val)))

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=False)
    valloader = DataLoader(db_val, batch_size=batch_size, shuffle=False, num_workers=4)

    if not eval:
        return trainloader, valloader
    else:
        return valloader