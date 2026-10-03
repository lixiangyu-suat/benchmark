#!/usr/bin/env bash

# 1.备用源
source -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 安装 PyTorch, albumentations 1.2.0
pip install pytorch torchvision torchaudio albumentations==1.2.0
pip install tqdm

# 3. 验证安装结果
python -c "
import torch, sklearn, albumentations
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available:  {torch.cuda.is_available()}')
print(f'scikit-learn:    {sklearn.__version__}')
print(f'albumentations:  {albumentations.__version__}')
"
