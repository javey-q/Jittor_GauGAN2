"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from jittor import init
from jittor import nn
import numpy as np

import util.util as util
from models.networks.encoder import SegmentationEncoder, StyleEncoder
from models.networks.base_module import EqualLinear, EqualConv2d, ConvBlock


class MPD(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # self.linear = nn.utils.spectral_norm(EqualConv2d(in_channels, 1, 1, padding=0).conv ,'weight_orig')
        self.linear = EqualConv2d(in_channels, 1, kernel_size=1, padding=0)

    # fix
    def execute(self, x, y = None):
        out = self.linear(x)  # [N,1,H,W]

        if y is not None:
            for y_k in y:
                # y_k = torch.sum(y_k*x, dim=1, keepdim=True) # [N,C,H,W] -> [N,1,H,W]
                y_k = (y_k * x).mean(dim=1, keepdim=True)

                out += y_k
        return out  # [N,1,H,W]

class Resblock(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 learned_shortcut = True,
                 fused = True,
                  ):
        super(Resblock, self).__init__()

        # FIXME  middle_channels
        middle_channels =  out_channels // 2

        self.learned_shortcut = learned_shortcut

        self.norm_0 = nn.InstanceNorm2d(in_channels, affine=False)
        self.norm_1 = nn.InstanceNorm2d(middle_channels, affine=False)

        self.conv0 = ConvBlock(in_channels, middle_channels, 3, 1, downsample=False, fused=fused)
        self.conv1 = ConvBlock(middle_channels, out_channels, 3, 1, downsample=True, fused=fused) # RESNET DOWNSAMPLING IS CON V1X1 AND NORM = IDENTITY

        if self.learned_shortcut:
            self.norm_s = nn.InstanceNorm2d(in_channels, affine=False)
            # FIXME k p
            self.convs = ConvBlock(in_channels, out_channels, 1, 0, downsample=True, fused=fused)

    def execute(self, x):
        x_s = self.shortcut(x)

        x = self.conv0(self.norm_0(x))
        x = self.conv1(self.norm_1(x))

        out = x_s + x

        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.convs(self.norm_s(x))
        else:
            x_s = x
        return x_s


class ImageEncoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 middle_channel=64,
                 num_stage=4,
                 image_shape=(384, 512)
                 ):
        super(ImageEncoder, self).__init__()
        self.ms_out_dims = []

        modules = []

        sw, sh = image_shape
        for i in range(num_stage):
            sh, sw = sh // 2, sw // 2
            fused = True if sh >= 128 else False
            modules.append(Resblock(in_channels,
                                    middle_channel, fused=fused))
            in_channels = middle_channel

            self.ms_out_dims.append(middle_channel)  # 64 / 128 / 256 / 512

            if i != num_stage - 1:
                middle_channel *= 2

        self.resblocks = nn.ModuleList(modules)
        self.out_dim = middle_channel

    def execute(self, x):
        outputs = []
        for resblock in self.resblocks:
            x = resblock(x)
            outputs.append(x)

        return outputs

# class D_get_logits(nn.Module):
#     def __init__(self, linear):
#         super().__init__()
#
#         self.linear = linear
#
#     def forward(self,img_embed, text_embed):
#         # img_embed [N, C< H, W]
#         # text_embed [N, C]
#         img_embed = F.adaptive_avg_pool2d(img_embed, (1,1)) # .squeeze(-1).squeeze(-1) -> view
#         img_embed = img_embed.view(img_embed.size(0),-1)
#         img_embed = self.linear(img_embed)
#
#         return img_embed, text_embed

config_input_shape = (384, 512)
config_num_stage = 4 # 2 **（stage + 3) = image_size
Base_channels_for_latent = 8
Base_channels_for_Dec = 64

class PoE_Discriminator(nn.Module):
    def __init__(self, opt, num_stage=config_num_stage, base_channel=Base_channels_for_Dec,
                 base_channel_latent=Base_channels_for_latent):
        super().__init__()

        self.opt = opt
        self.input_shape = (round(opt.crop_size / opt.aspect_ratio), opt.crop_size)

        self.seg_encoder = SegmentationEncoder(self.opt.semantic_nc,
                                               base_channel_latent,
                                               num_stage,
                                               self.input_shape)
        # self.style_encoder = StyleEncoder()

        self.image_encoder = ImageEncoder(in_channels=3,
                 middle_channel=base_channel,
                 num_stage=num_stage,
                 image_shape=self.input_shape)

        d_channel_list = [base_channel * (2 ** i) for i in range(num_stage-1)] + [512]  # 64, 128, 512, 1024
        self.num_stage = num_stage

        mpd_list, style_list, seg_list = [], [], []

        # multi scale
        for d_channel, dim in zip(d_channel_list, getattr(self.seg_encoder, 'ms_out_dims')):
            mpd_list.append(MPD(d_channel))

            # D_style = EqualLinear(getattr(self.style_encoder, 'out_dim') * 2, d_channel)  # concat logvar, mean
            # style_list.append(D_style)

            D_seg = nn.Sequential(ConvBlock(dim, d_channel, 1, 0, downsample=False)
                                  # ,nn.AdaptiveAvgPool2d((1, 1)),nn.Flatten()
                                  )
            seg_list.append(D_seg)

        self.mpd_list = nn.ModuleList(mpd_list)
        # self.style_list = nn.ModuleList(style_list)
        self.seg_list = nn.ModuleList(seg_list)


    def execute(self, x, segment):  # x, y1,y2, y3, y4 #  text, seg, sketch, style
        logits = []

        img_outputs = self.image_encoder(x)

        seg_outputs = self.seg_encoder(segment)[1:]

        # style_vector = jt.concat(self.style_encoder(style)[1], dim=1).squeeze(3).squeeze(2)  # concat log_var vetor and mean vector

        in_h, in_w = self.input_shape
        for i in range(self.num_stage):
            scale_modality_output = []
            in_h, in_w = in_h // 2, in_w // 2
            seg_mlp_output = self.seg_list[i](seg_outputs[i])
            scale_modality_output.append(seg_mlp_output)

            # style_output = self.style_list[i](style_vector).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, in_h,
            #                                                                                    in_w)
            # scale_modality_output.append(style_output)

            img_output = img_outputs[i]
            logit = self.mpd_list[i](img_output, scale_modality_output)

            logits.append(logit)

        return logits  # Dx, Dy for contrastive loss


if __name__ == '__main__':
    from options.train_options import  TrainOptions
    opt =  TrainOptions().parse()
    image = jt.randn(1, 3, 384, 512)
    segment = jt.randn(1, 1, 384, 512)
    # style = jt.randn(1, 3, 384, 512)
    discriminator = PoE_Discriminator(opt)
    outputs = discriminator(image, segment)
    # for output in outputs:
    #     print(output.shape)