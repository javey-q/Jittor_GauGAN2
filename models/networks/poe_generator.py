"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from jittor import init
from jittor import nn

from models.networks.encoder import SegmentationEncoder, StyleEncoder
from models.networks.poe_module import GlobalPoeNet, LocalPoeNet
from models.networks.base_module import EqualLinear, ConstantInput, EqualConv2d, ConvBlock



class MlpHead(nn.Module):
    def __init__(self, in_channels, out_channels=512, n_mlp=4, seperate=False,
                 spatial=False):  # out_channels mean code dim
        super(MlpHead, self).__init__()

        self.seperate = seperate
        self.spatial = spatial
        self.out_channels = out_channels

        hidden_dim = in_channels // 4

        mlp_head = []
        for _ in range(n_mlp - 1):
            mlp_head.append(EqualLinear(in_channels, hidden_dim))
            mlp_head.append(nn.LeakyReLU(0.2))
            in_channels = hidden_dim
        if self.seperate:
            self.fc_mu = EqualLinear(hidden_dim, out_channels)
            self.fc_var = EqualLinear(hidden_dim, out_channels)
        else:
            mlp_head.append(EqualLinear(hidden_dim, out_channels * 2))
        self.mlp_head = nn.Sequential(*mlp_head)

        self.adapool = nn.AdaptiveAvgPool2d((1, 1))

    def execute(self, x):
        if self.spatial:
            x = self.adapool(x)

        if self.seperate:

            mu = self.fc_mu(self.mlp_head(x[0].squeeze(3).squeeze(2)))
            logvar = self.fc_var(self.mlp_head(x[1].squeeze(3).squeeze(2)))
        else:
            x = x.view(x.size(0), -1)
            x = self.mlp_head(x)
            mu = x[:, :self.out_channels]
            logvar = x[:, self.out_channels:]

        return [mu, logvar]


class G_ResBlock(nn.Module):
    def __init__(self, pre_out_channel, spatial_channel, w_channel, fused=False, initial=False):
        super(G_ResBlock, self).__init__()

        if initial:
            self.upsample = nn.Identity()
            upsample = False
        else:
            self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
            upsample = True

        out_channel = pre_out_channel // 2  # 1024 -> 512

        self.conv1 = ConvBlock(pre_out_channel, out_channel, 3, 1, upsample=upsample, fused=fused)
        self.conv2_1 = ConvBlock(pre_out_channel, out_channel, 3, 1, upsample=upsample, fused=fused)
        self.conv2_2 = ConvBlock(out_channel, out_channel, 3, 1, upsample=False, fused=fused)

        self.lg_adain = LGAdaIN(out_channel, spatial_channel, w_channel)
        self.local_poe = LocalPoeNet(pre_out_channel, spatial_channel)

    def execute(self, pre_output, w, spatial_feats):

        # 0. upsample
        up_pre_output = self.upsample(pre_output)

        # 1. conv ( to residual)
        # FIXME pre_output -> up_pre_output
        out_1 = self.conv1(pre_output)

        # 2. conv ( to lgadain)
        # FIXME pre_output -> up_pre_output
        h_k = self.conv2_1(pre_output)

        # 3. local poe spatial_feats
        z_k, kl_input = self.local_poe(up_pre_output, spatial_feats)

        h_k = self.lg_adain(h_k, z_k, w)
        h_k = self.conv2_2(h_k)
        out_2 = self.lg_adain(h_k, z_k, w)

        out = out_1 + out_2

        return out, kl_input


class LGAdaIN(nn.Module):
    def __init__(self, in_channels, z_channels, w_channels):
        super(LGAdaIN, self).__init__()

        self.param_free_norm = nn.InstanceNorm2d(in_channels, affine=False)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        hidden_dim = 128
        kernel_size = 3
        padding = kernel_size // 2

        # FIXME  kernel_size padding
        self.conv_shared = ConvBlock(z_channels, hidden_dim, kernel_size, padding)
        self.conv_gamma = EqualConv2d(hidden_dim, in_channels, kernel_size, padding=padding)
        self.conv_beta = EqualConv2d(hidden_dim, in_channels, kernel_size, padding=padding)

        self.conv_gamma.conv.bias.data = jt.ones_like(self.conv_gamma.conv.bias.data)
        self.conv_beta.conv.bias.data = jt.zeros_like(self.conv_beta.conv.bias.data)
        # pw = ks // 2
        # self.conv_gamma = nn.Conv2d(nhidden, in_channels, kernel_size=ks, padding=pw)
        # self.conv_beta = nn.Conv2d(nhidden, in_channels, kernel_size=ks, padding=pw)

        self.mlp = EqualLinear(w_channels, in_channels * 2)
        self.mlp.linear.bias.data[:in_channels] = 1
        self.mlp.linear.bias.data[in_channels:] = 0

    def execute(self, h_k, z_k, w):
        norm = self.param_free_norm(h_k)

        local_actv = self.conv_shared(z_k)
        l_gamma = self.conv_gamma(local_actv)
        l_beta = self.conv_beta(local_actv)
        out = norm * l_gamma + l_beta

        global_feat = self.mlp(w).unsqueeze(2).unsqueeze(3)
        g_gamma, g_beta = global_feat.chunk(2, 1)

        out = g_gamma * out + g_beta

        return out

config_input_shape = (384, 512)
num_stage = 4 # 2 **（stage + 3) = image_size
Base_channels_for_latent = 4
Base_channels_for_Dec = 64
input_type = 'PoE_prior'  # PoE_prior  Constant

class Generator(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')

        return parser

    def __init__(self, opt, num_stage=num_stage, base_channel_dec=Base_channels_for_Dec,
                 base_channel_latent=Base_channels_for_latent, input_type=input_type, latent_dim = 512):
        super().__init__()
        self.opt = opt

        self.latent_dim = latent_dim
        self.base_channel_dec = base_channel_dec
        self.base_channel_latent = base_channel_latent
        self.num_stage = num_stage
        self.input_type = input_type

        # encoder
        input_shape = (round(opt.crop_size / opt.aspect_ratio), opt.crop_size)
        self.seg_encoder = SegmentationEncoder(1, base_channel_latent, num_stage, input_shape)
        self.seg_mlp_head = MlpHead(getattr(self.seg_encoder,'out_dim'), spatial=True)

        self.style_encoder = StyleEncoder(num_stage=num_stage)
        self.style_mlp_head = MlpHead(getattr(self.style_encoder, 'out_dim'), seperate=True)

        # define new decoder
        self.global_poe = GlobalPoeNet(code_dim=latent_dim)
        prior_mu = jt.zeros(self.latent_dim)
        prior_logvar = jt.log(jt.ones(self.latent_dim)) # mu, std

        # self.register_buffer('prior_mu',prior_mu)
        # self.register_buffer('prior_logvar',prior_logvar)
        setattr(self, 'prior_mu', prior_mu.stop_grad())
        setattr(self, 'prior_logvar', prior_logvar.stop_grad())

        self.maximum_channel, self.first_h , self.first_w = self.compute_latent_vector_size(opt)
        print(self.maximum_channel, self.first_h , self.first_w)
        if self.input_type  == 'PoE_prior':
            self.input_fc = nn.Linear(latent_dim,  self.maximum_channel * self.first_h * self.first_w)

        pre_out_channel = self.maximum_channel
        spatial_channel = 2 ** (self.num_stage) * self.base_channel_latent
        decoder_network = [G_ResBlock(self.maximum_channel, spatial_channel, latent_dim, fused=False, initial=True)]
        # for _, channel in zip(range(stage), reversed(getattr(self.seg_encoder, 'ms_out_dims'))):
        for i in range(num_stage):
            pre_out_channel = pre_out_channel // 2
            spatial_channel = spatial_channel // 2
            if i == (num_stage - 1):
                spatial_channel = 1
            decoder_network.append(G_ResBlock(pre_out_channel, spatial_channel, latent_dim, fused=False, initial=False))
        self.decoder = nn.ModuleList(decoder_network)

        self.out_layer = EqualConv2d(pre_out_channel // 2, 3, 1, padding=0)

    def compute_latent_vector_size(self, opt):
        fc = 2 ** (self.num_stage) * self.base_channel_dec
        fw = opt.crop_size // (2**self.num_stage)
        fh = round(fw / opt.aspect_ratio)

        return fc, fh, fw

    def execute(self, segment, style):
        B = segment.size(0)

        modality_stats = []
        #
        segment_features = self.seg_encoder(segment) # features

        seg_stats = self.seg_mlp_head(segment_features[-1])  # [(b, 512), (b, 512)]
        modality_stats.append(seg_stats)

        _, style_features = self.style_encoder(style)  # style_features : mu, std (b,  894) 894=512+256+128

        style_stats = self.style_mlp_head(style_features)  #  [(b, 512), (b, 512)]
        modality_stats.append(style_stats)

        # Global PoE_Net
        prior_stats = [self.prior_mu.expand(B, self.latent_dim), self.prior_logvar.expand(B,self.latent_dim)]
        modality_embed_list = [prior_stats, *modality_stats]
        w = self.global_poe(modality_embed_list)

        if self.input_type  == 'Constant':
            pre_output = jt.randn((B, self.maximum_channel, self.first_h , self.first_w))
        elif self.input_type  == 'PoE_prior':
            pre_output = self.input_fc(w)
            pre_output = pre_output.view(B, self.maximum_channel, self.first_h , self.first_w)


        kl_inputs = {'style':style_stats, 'segment':seg_stats, 'segment_multi':[]}
        for i, resblock in enumerate(self.decoder):
            spatial_idx = - i - 1
            style_z = self.reparameterize(*style_stats)
            pre_output, kl_input = resblock(pre_output, style_z, [segment_features[spatial_idx]])
            kl_inputs['segment_multi'].append(kl_input[0])

        out = self.out_layer(pre_output)
        return out, kl_inputs

    def reparameterize(self, mu, logvar):

        std = jt.exp(0.5 * logvar)
        eps = jt.randn_like(std)
        return eps.multiply(std).add(mu)



if __name__ == '__main__':
    from options.train_options import  TrainOptions
    opt =  TrainOptions().parse()
    segment = jt.randn(1, 1, 384, 512)
    style = jt.randn(1, 3, 384, 512)
    generator = Generator(opt)
    output, kl_inputs = generator(segment, style)
    print(output.shape)



