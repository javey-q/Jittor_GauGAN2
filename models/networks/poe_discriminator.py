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

    def forward(self, x, y: List = None):
        out = self.linear(x)  # [N,1,H,W]

        if y is not None:
            for y_k in y:
                # y_k = torch.sum(y_k*x, dim=1, keepdim=True) # [N,C,H,W] -> [N,1,H,W]

                y_k = torch.mean(y_k * x, dim=1, keepdim=True)

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

        middle_channels = min(in_channels,out_channels)

        self.learned_shortcut = learned_shortcut

        self.norm_0 = nn.InstanceNorm2d(in_channels, affine=False)
        self.norm_1 = nn.InstanceNorm2d(middle_channels, affine=False)

        self.conv0 = ConvBlock(in_channels, middle_channels, 3, 1, downsample=False, fused=fused)
        self.conv1 = ConvBlock(middle_channels, out_channels, 3, 1, downsample=True, fused=fused) # RESNET DOWNSAMPLING IS CON V1X1 AND NORM = IDENTITY

        if self.learned_shortcut:
            self.norm_s = nn.InstanceNorm2d(in_channels, affine=False)
            self.convs = ConvBlock(in_channels, out_channels, 3, 1, downsample=True, fused=fused)

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

config_input_shape = (384, 512)
num_stage = 4 # 2 **（stage + 3) = image_size
Base_channels_for_latent = 4
Base_channels_for_Dec = 64

class ImageEncoder(nn.Module):
    def __init__(self,
                 in_channels=3,
                 middle_channel=Base_channels_for_latent,
                 num_stage=num_stage,
                 image_shape=config_input_shape
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

            self.ms_out_dims.append(middle_channel)

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

class D_get_logits(nn.Module):
    def __init__(self, linear):
        super().__init__()

        self.linear = linear

    def forward(self,img_embed, text_embed):
        # img_embed [N, C< H, W]
        # text_embed [N, C]
        img_embed = F.adaptive_avg_pool2d(img_embed, (1,1)) # .squeeze(-1).squeeze(-1) -> view
        img_embed = img_embed.view(img_embed.size(0),-1)
        img_embed = self.linear(img_embed)

        return img_embed, text_embed


class Discriminator(nn.Module):
    def __init__(self, image_channel=Base_channels_for_Dec, input_shape=config_input_shape, num_stage=num_stage):
        super().__init__()

        self.input_shape = input_shape

        self.seg_encoder = SegmentationEncoder()
        self.style_encoder = StyleEncoder()

        self.image_encoder = ImageEncoder()

        d_channel_list = [image_channel * (2 ** i) for i in range(num_stage)]  # 64, 128, 512, 1024
        self.stage = num_stage

        mpd_list, text_list, style_list, seg_list, skt_list = [], [], [], [], []

        # multi scale
        for d_channel, dim in zip(d_channel_list, getattr(self.seg_encoder, 'ms_out_dims')):
            mpd_list.append(MPD(d_channel))
            t, s = self.vector_y_heads(d_channel)
            text_list.append(t)
            style_list.append(s)

            seg, skt = self.spatial_y_heads(dim, d_channel)
            seg_list.append(seg)
            skt_list.append(skt)

        self.mpd_list = nn.ModuleList(mpd_list)
        self.text_list = nn.ModuleList(text_list)
        self.style_list = nn.ModuleList(style_list)
        self.seg_list = nn.ModuleList(seg_list)
        self.skt_list = nn.ModuleList(skt_list)

        contrastive_linear = EqualLinear(getattr(self.image_encoder, 'out_dim'), getattr(self.text_encoder, 'out_dim'))
        self.img_cond_dnet = D_get_logits(contrastive_linear)

    def vector_y_heads(self, d_channel):
        D_text = EqualLinear(getattr(self.text_encoder, 'out_dim'), d_channel)
        D_style = EqualLinear(getattr(self.style_encoder, 'out_dim') * 2, d_channel)  # concat logvar, mean

        return D_text, D_style

    def spatial_y_heads(self, dim, d_channel):
        D_seg = nn.Sequential(ConvBlock(dim, d_channel, 1, 0, downsample=False)
                              # ,nn.AdaptiveAvgPool2d((1, 1)),nn.Flatten()
                              )
        D_sketch = nn.Sequential(ConvBlock(dim, d_channel, 1, 0, downsample=False)
                                 # ,nn.AdaptiveAvgPool2d((1, 1)),nn.Flatten()
                                 )

        return D_seg, D_sketch

    def forward(self, x, modality_dict):  # x, y1,y2, y3, y4 #  text, seg, sketch, style
        logits = []

        img_outputs = self.image_encoder(x)

        if 'text' in modality_dict.keys():
            text_vector = self.text_encoder(modality_dict['text'])  # replicate -> repeat
        if 'seg_maps' in modality_dict.keys():
            seg_outputs = self.seg_encoder(modality_dict['seg_maps'])[1:]
        if 'sketch_maps' in modality_dict.keys():
            skt_outputs = self.sketch_encoder(modality_dict['sketch_maps'])[1:]  # delete sketch
        if 'style' in modality_dict.keys():
            style_vector = jt.concat(self.style_encoder(modality_dict['style'])[1],
                                     dim=1).squeeze()  # concat log_var vetor and mean vector

        input_shape = self.input_shape
        for i in range(self.stage):
            scale_modality_output = []
            input_shape = input_shape // 2
            if 'seg_maps' in modality_dict.keys():
                seg_output = self.seg_list[i](seg_outputs[i])
                scale_modality_output.append(seg_output)
            if 'sketch_maps' in modality_dict.keys():
                skt_output = self.skt_list[i](skt_outputs[i])
                scale_modality_output.append(skt_output)
            if 'text' in modality_dict.keys():
                text_output = self.text_list[i](text_vector).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, input_shape,
                                                                                                input_shape)
                scale_modality_output.append(text_output)
            if 'style' in modality_dict.keys():
                style_output = self.style_list[i](style_vector).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, input_shape,
                                                                                                   input_shape)
                scale_modality_output.append(style_output)

            img_output = img_outputs[i]
            logit = self.mpd_list[i](img_output, scale_modality_output)

            logits.append(logit)

        contrastive_input = [img_output, text_vector] if 'text_output' in locals().keys() else None

        return logits, contrastive_input  # Dx, Dy for contrastive loss
