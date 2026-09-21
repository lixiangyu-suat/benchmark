#!/usr/bin/env bash

# 1.备用源
# source -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 安装 PyTorch 1.13.0 及 CUDA 11.7 工具链
conda install -y pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia

# 3. 安装 albumentations 1.2.0 
pip install albumentations==1.2.0

# 5. 验证安装结果
python -c "
import torch, sklearn, albumentations
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available:  {torch.cuda.is_available()}')
print(f'scikit-learn:    {sklearn.__version__}')
print(f'albumentations:  {albumentations.__version__}')
"
