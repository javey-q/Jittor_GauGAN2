import jittor as jt
from jittor import init
from jittor import nn

from models.networks.base_module import PixelNorm, EqualLinear, ConvBlock


# update exp log
class PoE(nn.Module):
    def __init__(self):
        super(PoE, self).__init__()

    def execute(self, mu_list, logvar_list, eps=1e-8):
        # mu : N x
        T_sum = 0
        mu_T_sum = 0
        for mu, logvar in zip(mu_list, logvar_list):
            var = logvar.exp() + eps
            T = 1 / (var + eps)

            T_sum += T
            mu_T_sum += mu * T

        mu = mu_T_sum / T_sum
        var = 1 / T_sum
        logvar = jt.log(var + eps)

        return mu, logvar


class GlobalPoeNet(nn.Module):
    def __init__(self, n_mlp_mapping=2, code_dim=512
                 ):
        super(GlobalPoeNet, self).__init__()

        self.global_poe = PoE()
        self.code_dim = code_dim

        mapping_network = [PixelNorm()]

        for i in range(n_mlp_mapping):
            mapping_network.append(EqualLinear(code_dim, code_dim))
            mapping_network.append(nn.LeakyReLU(0.2))

        self.mapping_network = nn.Sequential(*mapping_network)

    def execute(self, modality_embed_list):
        mu_list, logvar_list = list(zip(*modality_embed_list))  # tuple : (3)

        # apply tanh
        logvar_list = list(logvar_list)
        mu_list = list(mu_list)
        for i, logvar in enumerate(logvar_list):
            if i == 0:
                logvar_list[i] = logvar_tanh(logvar, theta=1)
            else:
                logvar_list[i] = logvar_tanh(logvar, theta=10)

        mu, logvar = self.global_poe(mu_list, logvar_list)

        z_0 = self.reparameterize(mu, logvar)

        w = self.mapping_network(z_0)

        return w

    def reparameterize(self, mu, logvar):

        # std = torch.exp(logvar).square()
        # eps = torch.randn_like(std)
        # FIXME
        std = jt.exp(0.5 * logvar)
        eps = jt.randn_like(std)

        return eps.multiply(std).add(mu)

def logvar_tanh(logvar, theta):
    limit_log_var = theta * jt.tanh((logvar / theta))
    return limit_log_var

class LocalPoeNet(nn.Module):
    def __init__(self, pre_out_channel, spatial_channel
                 ):
        super(LocalPoeNet, self).__init__()
        self.local_poe = PoE()
        self.pre_cnn = CNNHead(pre_out_channel, spatial_channel)
        self.spatial_1_cnn = CNNHead(pre_out_channel + spatial_channel, spatial_channel)
        self.spatial_2_cnn = CNNHead(pre_out_channel + spatial_channel, spatial_channel)

        # if you want to use more spatial modality, add that modality cnn head

    def execute(self, up_pre_out, spatial_feats=None):  # seg_feat, sketch_feat

        pre_stats = self.pre_cnn(up_pre_out)
        pre_stats[1] = logvar_tanh(pre_stats[1], theta=1)

        if spatial_feats == None:
            z_k = self.reparameterize(pre_stats[0], pre_stats[1])

            return z_k, [pre_stats, pre_stats]

        spatial_stats = []
        for i, spatial_feat in enumerate(spatial_feats):
            spt_stat = getattr(self, f'spatial_{i + 1}_cnn')(jt.concat((up_pre_out, spatial_feat), dim=1))
            spt_stat[1] = logvar_tanh(spt_stat[1], theta=10)
            spatial_stats.append(spt_stat)


        modality_embed_list = [pre_stats, *spatial_stats]
        mu_list, logvar_list = list(zip(*modality_embed_list))

        # apply tanh to poe layer
        mu, logvar = self.local_poe(list(mu_list), list(logvar_list))

        z_k = self.reparameterize(mu, logvar)

        # FIXME
        return z_k, spatial_stats  # p' , p(z|y)

    def reparameterize(self, mu, logvar):

        std = jt.pow(jt.exp(logvar), 2)
        eps = jt.randn_like(std)

        return eps * std + mu

class CNNHead(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(CNNHead,self).__init__()

        self.out_channels = out_channels

        hideen_dim = in_channels//4

        kernel_size_list = [1, 3, 3]

        cnn_head = []
        for kernel_size in kernel_size_list:
            if kernel_size == 1:
                padding = 0
            elif kernel_size == 3:
                padding = 1
            cnn_head.append(ConvBlock(in_channels,hideen_dim,kernel_size,padding=padding))
            in_channels = hideen_dim


        cnn_head.append(ConvBlock(hideen_dim, out_channels*2, 1, 0))
        self.cnn_head = nn.Sequential(*cnn_head)


    def execute(self,x):
        # x = x.squeeze()  ???
        x = self.cnn_head(x)
        mu = x[:, :self.out_channels,:,:]
        logvar = x[:, self.out_channels:,:,:]

        return [mu, logvar]