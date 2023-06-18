import os.path

import torch.nn as nn
import torch
import math
from torch.utils import model_zoo
from PIL import Image
import numpy as np
import torch.nn.functional as F


def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class DecpNetwork(nn.Module):
    def __init__(self):
        super(DecpNetwork, self).__init__()
        self.VggEnc = VggEnc()
        self.cod_vgg_dec = VggDec()
        self.sod_vgg_dec = VggDec()

    def forward(self, input_1, img_type):
        fea = self.VggEnc(input_1)
        if img_type == 'sod':
            dbd1 = self.sod_vgg_dec(fea)
        elif img_type == 'cod':
            dbd1 = self.cod_vgg_dec(fea)
        else:
            dbd1 = None
        return dbd1


class VggEnc(nn.Module):
    def __init__(self):
        super(VggEnc, self).__init__()
        self.base1 = Base1()  # encoder

    def forward(self, input_1):
        s1, s2, s3, s4, s_m, s_p = self.base1(input_1)
        return [s1, s2, s3, s4, s_m, s_p]


class Decp(nn.Module):
    def __init__(self):
        super(Decp, self).__init__()
        self.decp_m = nn.Conv2d(512, 512, 1, 1)
        self.decp_p = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Linear(512, 512),
            nn.ReLU()
        )

    def forward(self, in_fea):
        s_m = self.decp_m(in_fea)
        s_p = self.decp_p(in_fea)
        return s_m, s_p


class VggDec(nn.Module):
    def __init__(self):
        super(VggDec, self).__init__()

        self.base22 = Base22()  # decoder block2
        self.base23 = Base23()  # decoder block3
        self.base24 = Base24()  # decoder block4
        self.mix_block1 = MixBlock(
            latent_dim=512,
            input_channels=512,
            filters=512,
            upsample=False
        )
        self.mix_block2 = MixBlock(
            latent_dim=512,
            input_channels=512,
            filters=512,
            upsample=False
        )
        self.output_dbd4 = OutputMask(chennel=64)
        self.output_dbd3 = OutputMask(chennel=128)
        self.output_dbd2 = OutputMask(chennel=256)

    def forward(self, s):
        fd1 = self.mix_block1(s[4], s[5])
        fd1 = F.interpolate(fd1, scale_factor=2, mode='bilinear')
        fd1 = self.mix_block2(fd1, s[5])
        fd1 = F.interpolate(fd1, scale_factor=2, mode='bilinear')
        fd2 = self.base22(fd1)
        fd3 = self.base23(fd2)
        fd4 = self.base24(fd3)
        fd4_pred = self.output_dbd4(fd4)
        fd3_pred = self.output_dbd3(fd3)
        fd2_pred = self.output_dbd2(fd2)
        fd2_pred = F.interpolate(
            fd2_pred,
            scale_factor=4,
            mode='bilinear'
        )  # (bs, 1, 88, 88) -> (bs, 1, 352, 352)
        fd3_pred = F.interpolate(
            fd3_pred,
            scale_factor=2,
            mode='bilinear'
        )  # (bs, 1, 176, 176) -> (bs, 1, 352, 352)

        return fd2_pred, fd3_pred, fd4_pred


class VggDecImg(nn.Module):
    def __init__(self):
        super(VggDecImg, self).__init__()

        self.base22 = Base22()
        self.base23 = Base23()
        self.base24 = Base24()
        # self.decp = Decp()
        self.mix_block1 = MixBlock(
            latent_dim=512,
            input_channels=512,
            filters=512,
            upsample=False
        )
        self.mix_block2 = MixBlock(
            latent_dim=512,
            input_channels=512,
            filters=512,
            upsample=False
        )

        self.conv1 = nn.Sequential(

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True)
        )

        self.conv2 = nn.Sequential(

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True)
        )

        self.conv3 = nn.Sequential(

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0),
            nn.Sigmoid()
        )

        self.decy1 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
        )
        self.decy2 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=512),
            nn.ReLU(True),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(True)
        )

    def forward(self, s, m, p):
        s1, s2, s3, s4, s_m, s_p = s[0], s[1], s[2], s[3], s[4], s[5]
        s_m = torch.cat([s_m, m[4]], 1)
        s_p = torch.cat([s_p, p[5]], 1)
        s_m = self.decy1(s_m)
        s_p = self.decy2(s_p)

        fd1 = self.mix_block1(s_m, s_p)
        fd1 = F.interpolate(fd1, scale_factor=2, mode='bilinear')
        fd1 = self.mix_block2(fd1, s_p)

        fd1 = torch.cat([fd1, s3], 1)
        fd1 = self.conv1(fd1)

        fd1 = torch.cat([fd1, s2], 1)
        fd1 = self.conv2(fd1)

        fd1 = torch.cat([fd1, s1], 1)
        fd1 = self.conv3(fd1)

        return fd1


class Base1(nn.Module):
    def __init__(self):
        super(Base1, self).__init__()
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv1_1_2 = BaseConv(3, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv1_2_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_1_2 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv2_2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_1_2 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_2_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv3_3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_1_2 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_1_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_2_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv5_3_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.decp = Decp()

    def forward(self, x):
        x = self.conv1_1_2(x)
        x = self.conv1_2_2(x)
        x = self.maxpool(x)
        x = self.conv2_1_2(x)
        x = self.conv2_2_2(x)
        s1 = x
        x = self.maxpool(x)
        x = self.conv3_1_2(x)
        x = self.conv3_2_2(x)
        x = self.conv3_3_2(x)
        s2 = x

        x = self.maxpool(x)
        x = self.conv4_1_2(x)
        x = self.conv4_2_2(x)
        x = self.conv4_3_2(x)
        s3 = x
        s_m, s_p = self.decp(s3)
        # print(x.size())
        x = self.maxpool(s_m)
        x = self.conv5_1_2(x)
        x = self.conv5_2_2(x)
        x = self.conv5_3_2(x)
        s4 = x
        s_m = s4

        return s1, s2, s3, s4, s_m, s_p


class Base22(nn.Module):
    def __init__(self):
        super(Base22, self).__init__()
        self.conv3_2 = BaseConv(512, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fd1):
        x = fd1
        # x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # x = torch.cat([x, s2], 1)
        x = self.conv3_2(x)
        x = self.conv4_2(x)
        fd2 = x

        return fd2


class Base23(nn.Module):
    def __init__(self):
        super(Base23, self).__init__()
        self.conv5_2 = BaseConv(256, 256, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.conv6_2 = BaseConv(256, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv7_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fd2):
        x = fd2
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        # x = torch.cat([x, s1], 1)
        x = self.conv5_2(x)
        x = self.conv6_2(x)
        x = self.conv7_2(x)
        fd3 = x

        return fd3


class Base24(nn.Module):
    def __init__(self):
        super(Base24, self).__init__()
        self.conv_out_base_1 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=True)
        self.conv_out_base_2 = BaseConv(128, 64, 3, 1, activation=nn.ReLU(), use_bn=True)

    def forward(self, fd3):
        x = fd3
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        x = self.conv_out_base_1(x)
        x = self.conv_out_base_2(x)
        fd4 = x

        return fd4


class OutputMask(nn.Module):
    def __init__(self, chennel):
        super(OutputMask, self).__init__()
        self.conv_out_base_3 = BaseConv(chennel, 1, 1, 1, activation=nn.Sigmoid(), use_bn=False)

    def forward(self, fd4):
        x = fd4
        x = self.conv_out_base_3(x)
        dbd1 = x

        return dbd1


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel // 2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


class MixBlock(nn.Module):
    '''
    Modified StyleGAN block
    based on https://github.com/mahmoudnafifi/HistoGAN
    '''
    def __init__(self, latent_dim, input_channels, filters, upsample=True,
                 upsample_rgb=True, rgba=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear',
                                    align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()

    def forward(self, x, istyle, latent=None):
        if self.upsample is not None:
            x = self.upsample(x)

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x)

        if latent is not None:
            x = x + latent
        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x)

        return x


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1,
                 dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in',
                                nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + 1e-8)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

