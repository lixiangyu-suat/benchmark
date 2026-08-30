import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


def upsize(x, scale_factor=2):
    x = F.interpolate(x, scale_factor=scale_factor, mode='nearest')
    return x


class DecoderBlock(nn.Module):
    def __init__(self,
                 in_channels=512,
                 out_channels=256,
                 kernel_size=3,
                 is_deconv=False,
                 ):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(out_channels // 2)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(out_channels // 2)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = torch.cat(x, 1)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias=False)  # verify bias false
        self.bn = nn.BatchNorm2d(out_planes,
                                 eps=0.001,  # value found in tensorflow
                                 momentum=0.1,  # default pytorch value
                                 affine=True)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ResNet34UnetPlus(nn.Module):
    def __init__(self,
                 num_channels=3,
                 num_class=1,
                 is_deconv=False,
                 decoder_kernel_size=3,
                 ):
        super().__init__()

        filters = [64, 128, 256, 512, 1024]
        resnet = models.resnet34(pretrained=False)
        self.base_size = 512
        self.crop_size = 512
        self._up_kwargs = {'mode': 'bilinear', 'align_corners': True}

        self.mix = nn.Parameter(torch.FloatTensor(6))
        self.mix.data.fill_(1)

        if num_channels == 3:
            self.firstconv = resnet.conv1
        else:
            self.firstconv = nn.Conv2d(num_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        
        # 手动定义第 5 阶段：将 512 通道下采样到 1024 通道
        # 方案：使用 stride=2 的卷积或借用 torchvision 的 BasicBlock
        # self.encoder5 = nn.Sequential(
        self.encoder5 = resnet.layer5

        
        # Decoder
        self.decoder0_1 = DecoderBlock(in_channels=64 + 64,
                                       out_channels=64,
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)

        self.decoder1_1 = DecoderBlock(in_channels=128 + 64,
                                       out_channels=64,
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        self.decoder0_2 = DecoderBlock(in_channels=64 + 64 + 64,
                                       out_channels=64,
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)

        self.decoder2_1 = DecoderBlock(in_channels=128 + 256,
                                       out_channels=128,
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        self.decoder1_2 = DecoderBlock(in_channels=64 + 64 + 128,
                                       out_channels=128,
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        self.decoder0_3 = DecoderBlock(in_channels=64 + 64 + 64 + 128,
                                       out_channels=128,
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)

        self.decoder3_1 = DecoderBlock(in_channels=512 + 256,
                                       out_channels=256,
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        self.decoder2_2 = DecoderBlock(in_channels=128 + 128 + 256,
                                       out_channels=256,
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        self.decoder1_3 = DecoderBlock(in_channels=64 + 64 + 128 + 256,
                                       out_channels=256,
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        self.decoder0_4 = DecoderBlock(in_channels=64 + 64 + 64 + 128 + 256,
                                       out_channels=256,
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)

        self.decoder4_1 = DecoderBlock(in_channels=1024 + 512,
                                        out_channels=512,
                                        kernel_size=decoder_kernel_size,
                                        is_deconv=is_deconv)
        self.decoder3_2 = DecoderBlock(in_channels=256 + 256 + 512,
                                        out_channels=512,
                                        kernel_size=decoder_kernel_size,
                                        is_deconv=is_deconv)
        self.decoder2_3 = DecoderBlock(in_channels=128 + 128 + 256 + 512,
                                        out_channels=512,
                                        kernel_size=decoder_kernel_size,
                                        is_deconv=is_deconv)
        self.decoder1_4 = DecoderBlock(in_channels=64 + 64 + 128 + 256 + 512,
                                        out_channels=512,
                                        kernel_size=decoder_kernel_size,
                                        is_deconv=is_deconv)
        self.decoder0_5 = DecoderBlock(in_channels=64 + 64 + 64 + 128 + 256 + 512,
                                       out_channels=512,
                                       kernel_size=decoder_kernel_size,
                                       is_deconv=is_deconv)
        
        self.logit1 = nn.Conv2d(64, num_class, kernel_size=1)
        self.logit2 = nn.Conv2d(64, num_class, kernel_size=1)
        self.logit3 = nn.Conv2d(128, num_class, kernel_size=1)
        self.logit4 = nn.Conv2d(256, num_class, kernel_size=1)
        self.logit5 = nn.Conv2d(512, num_class, kernel_size=1)

    def require_encoder_grad(self, requires_grad):
        blocks = [self.firstconv,
                  self.encoder1,
                  self.encoder2,
                  self.encoder3,
                  self.encoder4,
                  self.encoder5]

        for block in blocks:
            for p in block.parameters():
                p.requires_grad = requires_grad

    def forward(self, x):
        _, _, H, W = x.shape
        # stem
        x = self.firstconv(x)  # subsample
        x = self.firstbn(x)
        x_ = self.firstrelu(x)

        # Encoder
        x = self.firstmaxpool(x_)  # 64
        e1 = self.encoder1(x)  # 64
        e2 = self.encoder2(e1)  # 128
        e3 = self.encoder3(e2)  # 256
        e4 = self.encoder4(e3)  # 512
        e5 = self.encoder5(e4)  # 1024
        # --------Unet Plus Plus Decoder------------

        x0_0 = x_
        x1_0 = e1
        x0_1 = self.decoder0_1([x0_0, upsize(x1_0)])  # 256 256

        x2_0 = e2
        x1_1 = self.decoder1_1([x1_0, upsize(x2_0)])
        x0_2 = self.decoder0_2([x0_0, x0_1, upsize(x1_1)])

        x3_0 = e3
        x2_1 = self.decoder2_1([x2_0, upsize(x3_0)])
        x1_2 = self.decoder1_2([x1_0, x1_1, upsize(x2_1)])
        x0_3 = self.decoder0_3([x0_0, x0_1, x0_2, upsize(x1_2)])

        x4_0 = e4
        x3_1 = self.decoder3_1([x3_0, upsize(x4_0)])
        x2_2 = self.decoder2_2([x2_0, x2_1, upsize(x3_1)])
        x1_3 = self.decoder1_3([x1_0, x1_1, x1_2, upsize(x2_2)])
        x0_4 = self.decoder0_4([x0_0, x0_1, x0_2, x0_3, upsize(x1_3)])

        x5_0 = e5
        x4_1 = self.decoder4_1([x4_0, upsize(x5_0)])
        x3_2 = self.decoder3_2([x3_0, x3_1, upsize(x4_1)])
        x2_3 = self.decoder2_3([x2_0, x2_1, x2_2, upsize(x3_2)])
        x1_4 = self.decoder1_4([x1_0, x1_1, x1_2, x1_3, upsize(x2_3)])
        x0_5 = self.decoder0_5([x0_0, x0_1, x0_2, x0_3, x0_4, upsize(x1_4)])

        logit1 = self.logit1(x0_1)
        logit2 = self.logit2(x0_2)
        logit3 = self.logit3(x0_3)
        logit4 = self.logit4(x0_4)
        logit5 = self.logit5(x0_5)
        logit = self.mix[1] * logit1 + self.mix[2] * logit2 + self.mix[3] * logit3 + self.mix[4] * logit4 + self.mix[5] * logit5
        logit = F.interpolate(logit, size=(H, W), mode='bilinear', align_corners=False)
        return logit
