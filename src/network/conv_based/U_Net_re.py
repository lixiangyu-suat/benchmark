import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=False), # kernel_size = padding*2 + 1 (ch_in==ch_out时)
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.conv(x)
        return x
    

class Up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False), # BatchNorm2d有减法算子，不需要可学习的偏置b了
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.up(x)
        return x

    
class U_Net(nn.Module):
    def __init__(self, img_ch=3, ch_out=1):
        super(U_Net, self).__init__()
        self.conv1 = ConvBlock(ch_in=img_ch, ch_out=64)
        self.conv2 = ConvBlock(ch_in=64, ch_out=128)
        self.conv3 = ConvBlock(ch_in=128, ch_out=256)
        self.conv4 = ConvBlock(ch_in=256, ch_out=512)
        self.conv5 = ConvBlock(ch_in=512, ch_out=1024)

        self.downsampling = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_conv5 = Up_conv(ch_in=1024, ch_out=512)
        self.re_conv5 = ConvBlock(ch_in=1024, ch_out=512)
        
        self.up_conv4 = Up_conv(ch_in=512, ch_out=256)
        self.re_conv4 = ConvBlock(ch_in=512, ch_out=256)

        self.up_conv3 = Up_conv(ch_in=256, ch_out=128)
        self.re_conv3 = ConvBlock(ch_in=256, ch_out=128)

        self.up_conv2 = Up_conv(ch_in=128, ch_out=64)
        self.re_conv2 = ConvBlock(ch_in=128, ch_out=64)

        self.up_conv1 = nn.Conv2d(64, ch_out, kernel_size=1, stride=1, padding=0) # 这里图像尺寸不变

    def forward(self, x0):
        x1 = self.conv1(x0)
        x2 = self.conv2(self.downsampling(x1))
        x3 = self.conv3(self.downsampling(x2))
        x4 = self.conv4(self.downsampling(x3))
        x5 = self.conv5(self.downsampling(x4))

        d5 = x5

        t4 = torch.cat((x4, self.up_conv5(d5)), dim=1) # 这里 Shape:(batch, channel, Height, Width), 分别对应(dim0, dim1, dim2, dim3).
        d4 = self.re_conv5(t4)
        
        t3 = torch.cat((x3, self.up_conv4(d4)), dim=1)
        d3 = self.re_conv4(t3)

        t2 = torch.cat((x2, self.up_conv3(d3)), dim=1) # 先上采样卷积，再cat拼接，再两次卷积
        d2 = self.re_conv3(t2)

        t1 = torch.cat((x1, self.up_conv2(d2)), dim=1)
        d1 = self.re_conv2(t1)

        d0 = self.up_conv1(d1) # 这里图像尺寸不变
        return d0
