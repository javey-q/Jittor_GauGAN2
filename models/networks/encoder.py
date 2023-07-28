"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from jittor import init
from jittor import nn
import numpy as np
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer


from models.networks.base_module import *

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(
            nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(
            nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(
            nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(
            nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(
                nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)

        self.actvn = nn.LeakyReLU(0.2)
        self.opt = opt

    def execute(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = nn.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        if self.opt.crop_size >= 256:
            x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar



config_input_shape = (384, 512)
config_num_stage = 4
Base_channels_for_Dec = 64
Base_channels_for_latent = 8

# segmentation map is resized to match the resolution of the
# corresponding feature map using nearest-neighbor downsampling
class SegmentationEncoder(nn.Module):
    def __init__(self,
                 in_channels=1,
                 base_channel=Base_channels_for_latent,
                 num_stage=config_num_stage,  # Maximum # channels for Latent // Base # channels for Latent
                 segmap_shape=config_input_shape,
                 ):
        super(SegmentationEncoder, self).__init__()
        self.ms_out_dims = []
        sh, sw = segmap_shape
        downsamples, embeds, convs = [], [], []
        custom_channel = base_channel // 2
        for i in range(num_stage):
            if i != (num_stage - 1):
                sh, sw = sh//2 , sw//2
                downsamples.append(DownSample(size=(sh,sw)))

            fused = True if sw >= 128 else False

            embeds.append(ConvBlock(in_channels, custom_channel, 3, 1, downsample=False, fused=fused))
            convs.append(ConvBlock(custom_channel, 2*custom_channel, 3, 1, downsample=True, fused=fused))

            custom_channel = 2*custom_channel
            self.ms_out_dims.append(custom_channel)

        self.downsamples = nn.ModuleList(downsamples)
        self.embeds = nn.ModuleList(embeds)
        self.convs = nn.ModuleList(convs)

        self.out_dim = custom_channel

    def execute(self, seg):
        outputs = [seg]  # ?
        for i, (conv, embed) in enumerate(zip(self.convs, self.embeds)):
            if i == 0:
                x = conv(embed(seg))
                x_d = self.downsamples[i](seg)

            elif i > 0:

                x = conv(embed(x_d) + x)

                if i != (len(self.downsamples)):
                    x_d = self.downsamples[i](x_d)
            outputs.append(x)

        return outputs


class StyleResblock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 learned_shortcut=True,
                 fused=True,
                 ):
        super(StyleResblock, self).__init__()

        # FIXME  middle_channels
        middle_channels = out_channels // 2

        self.learned_shortcut = learned_shortcut

        self.norm_0 = nn.InstanceNorm2d(in_channels, affine=False)
        self.norm_1 = nn.InstanceNorm2d(middle_channels, affine=False)

        self.conv_0 = ConvBlock(in_channels, middle_channels, 3, 1, downsample=False, fused=fused)
        self.conv_1 = ConvBlock(middle_channels, out_channels, 3, 1, downsample=True,
                                fused=fused)  # RESNET DOWNSAMPLING IS CON V1X1 AND NORM = IDENTITY

        if self.learned_shortcut:
            self.norm_s = nn.InstanceNorm2d(in_channels, affine=False)
            self.conv_s = ConvBlock(in_channels, out_channels, 3, 1, downsample=True, fused=fused)

    def execute(self, style):
        x_s = self.shortcut(style)

        x = self.conv_0(self.norm_0(style))
        x = self.conv_1(self.norm_1(x))

        out = x_s + x

        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x))
        else:
            x_s = x
        return x_s
class StyleEncoder(nn.Module):
    def __init__(self,
                 out_channels=512,
                 in_channels=3,
                 num_stage=config_num_stage,
                 middle_channels=None,
                 ):
        super(StyleEncoder, self).__init__()
        modules = []
        if middle_channels is None:
            middle_channels = [out_channels // (2 ** (num_stage - i)) for i in range(1, num_stage+1)] # 64/ 128 / 256 / 512

        for middle_channel in middle_channels:
            modules.append(StyleResblock(in_channels,
                                         middle_channel))

            in_channels = middle_channel

        self.resblocks = nn.ModuleList(modules)

        self.out_dim = sum(middle_channels)

    def execute(self, x):
        mean, log_var = [], []
        for resblock in self.resblocks:
            x = resblock(x)
            x_mean, x_log_var = self.calc_mean_var(x)  # 128, 128 / 256, 256 / out_channels, out_channels
            mean.append(x_mean)
            log_var.append(x_log_var)

        mu = jt.concat(mean, dim=1)
        logvar = jt.concat(log_var, dim=1)

        return x, [mu, logvar]

    def calc_mean_var(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        feat_size = feat.size()
        assert (len(feat_size) == 4)
        N, C = feat_size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_log_var = jt.log(feat_var).view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_log_var



if __name__ == '__main__':
    input = jt.randn(1, 1, 384, 512)
    seg_encoder = SegmentationEncoder()
    outputs = seg_encoder(input)
    # print(seg_encoder.named_modules())
    for output in outputs:
        print(output.shape)
    # input = jt.randn(1, 3, 384, 512)
    # style_encoder = StyleEncoder()
    # x, style_features = style_encoder(input)
    # # print(style_encoder.named_modules())
    # mu, logvar = style_features[0], style_features[1]
    # print(mu.shape)
    # print(logvar.shape)