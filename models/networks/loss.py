"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
from jittor import init
from jittor import nn
from jittor import distributions
from models.networks.architecture import VGG19
from jittor import models
import numpy as np
from skimage.exposure import match_histograms


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
                 tensor=jt.float32, opt=None):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.zero_tensor = None
        self.Tensor = tensor
        self.gan_mode = gan_mode
        self.opt = opt
        if gan_mode == 'ls':
            pass
        elif gan_mode == 'original':
            pass
        elif gan_mode == 'w':
            pass
        elif gan_mode == 'hinge':
            pass
        else:
            raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            if self.real_label_tensor is None:
                self.real_label_tensor = self.Tensor(1).fill_(self.real_label)
                self.real_label_tensor.requires_grad_(False)
            return self.real_label_tensor.expand_as(input)
        else:
            if self.fake_label_tensor is None:
                self.fake_label_tensor = self.Tensor(1).fill_(self.fake_label)
                self.fake_label_tensor.requires_grad_(False)
            return self.fake_label_tensor.expand_as(input)

    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(0.)
            # self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)

    def loss(self, input, target_is_real, for_discriminator=True):
        if self.gan_mode == 'original':  # cross entropy loss
            target_tensor = self.get_target_tensor(input, target_is_real)
            loss = nn.binary_cross_entropy_with_logits(input, target_tensor)
            return loss
        elif self.gan_mode == 'ls':
            target_tensor = self.get_target_tensor(input, target_is_real)
            return nn.mse_loss(input, target_tensor)
        elif self.gan_mode == 'hinge':
            if for_discriminator:
                if target_is_real:
                    minval = jt.minimum(input - 1, self.get_zero_tensor(input))
                    loss = -jt.mean(minval)
                else:
                    minval = jt.minimum(-input - 1,
                                        self.get_zero_tensor(input))
                    loss = -jt.mean(minval)
            else:
                assert target_is_real, "The generator's hinge loss must be aiming for real"
                loss = -jt.mean(input)
            return loss
        else:
            # wgan
            if target_is_real:
                return -input.mean()
            else:
                return input.mean()

    def __call__(self, input, target_is_real, for_discriminator=True):
        # computing loss is a bit complicated because |input| may not be
        # a tensor, but list of tensors in case of multiscale discriminator
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    pred_i = pred_i[-1]
                # print(pred_i.shape)
                loss_tensor = self.loss(
                    pred_i, target_is_real, for_discriminator)
                bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
                new_loss = jt.mean(loss_tensor.view(bs, -1), dim=1)
                loss += new_loss
            return loss / len(input)
        else:
            return self.loss(input, target_is_real, for_discriminator)


# Perceptual loss that uses a pretrained VGG network
class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = VGG19()
        self.criterion = nn.L1Loss()
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def execute(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * \
                    self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


# KL Divergence loss used in VAE with an image encoder
class KLDLoss(nn.Module):
    def execute(self, mu, logvar):
        return -0.5 * jt.sum(1 + logvar - mu.pow(2) - logvar.exp())


class VGG_contrastive(nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        self.vgg19 = models.vgg19(pretrained=True).features

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def execute(self, x):
        x = self.vgg19(x)
        x = x.view(x.size(0), -1)
        return x


class ContrastiveLoss(nn.Module):
    def __init__(self, gpu_ids, temperature=0.3):
        super(ContrastiveLoss, self).__init__()
        self.vgg = VGG_contrastive()
        self.temperature = temperature

    def execute(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        x_norm = x_vgg / x_vgg.norm(dim=-1, keepdim=True)
        y_norm = x_vgg / x_vgg.norm(dim=-1, keepdim=True)

        logits_x = x_norm @ y_norm.t() / self.temperature
        logits_y = y_norm @ x_norm.t() / self.temperature

        labels = jt.arange(len(logits_x))
        loss_x_term = nn.cross_entropy_loss(logits_x, labels)
        loss_y_term = nn.cross_entropy_loss(logits_y, labels)
        loss = (loss_x_term + loss_y_term) / 2

        return loss


def getHistMatched(imgs: jt.Var, refs: jt.Var):
    assert imgs.shape == refs.shape
    bs, c, h, w = imgs.shape
    matched_list = []
    imgs, refs = imgs.clamp(0, 1).data, refs.clamp(0, 1).data
    # print(imgs.shape)
    for i in range(bs):
        img, ref = imgs[i, :, :, :], refs[i, :, :, :]
        img = np.array((img + 1) / 2.0 * 255.0, dtype=np.uint8).transpose((1, 2, 0))
        ref = np.array((ref + 1) / 2.0 * 255.0, dtype=np.uint8).transpose((1, 2, 0))
        matched = match_histograms(img, ref, channel_axis=-1)
        matched = matched.transpose((2, 0, 1)) / 255
        matched = matched.reshape(1, *matched.shape)
        matched_list.append(matched)
    matcheds = jt.array(np.concatenate(matched_list, axis=0), dtype="float32").stop_grad()
    return matcheds


class HistLoss(nn.Module):
    def __init__(self):
        super(HistLoss, self).__init__()
        self.l1_loss = nn.L1Loss()

    def execute(self, imgs, refs):
        assert imgs.shape == refs.shape
        matcheds = getHistMatched(imgs, refs)
        loss = self.l1_loss(imgs, matcheds)
        return loss

def kl_divergence(mu1, sigma_1, mu2, sigma_2):
    # mu1, sigma_1, mu2, sigma_2 = nn.avg_pool2d(mu1, (1,1)), nn.avg_pool2d(sigma_1, (1,1),), \
    #                              nn.avg_pool2d(mu2, (1,1)), nn.avg_pool2d(sigma_2, (1,1))
    mu1, sigma_1, mu2, sigma_2 = mu1.view(mu1.size(0),-1), sigma_1.view(mu1.size(0),-1), \
                                 mu2.view(mu1.size(0),-1), sigma_2.view(mu1.size(0),-1)

    cov1 = jt.stack([jt.diag(sigma) for sigma in sigma_1.exp()])
    print(cov1)
    mvn1 = distributions.Normal(mu1, cov1)
    cov2 = jt.stack([jt.diag(sigma) for sigma in sigma_2.exp()])
    print(cov2)
    mvn2 = distributions.Normal(mu2, cov2)

    return distributions.kl_divergence(mvn1, mvn2).sum()

class SoftHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (jt.arange(bins).float() + 0.5)
        self.centers.stop_grad()

    def execute(self, x):
        x = jt.unsqueeze(x, 0) - jt.unsqueeze(self.centers, 1)
        x = jt.sigmoid(self.sigma * (x + self.delta/2)) - jt.sigmoid(self.sigma * (x - self.delta/2))
        x = x.sum(dim=1)
        # y = x.sum()
        # x = x / (x.sum() + 0.0001)
        return x

    def execute_1(self, x):
        x = jt.unsqueeze(x, 0) - jt.unsqueeze(self.centers, 1)
        x = jt.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        x = x.sum(dim=-1)
        # x = x / (x.sum() + 0.00001)
        return x

def hist_loss(imgs, refs, normalize=True):
    assert imgs.shape == refs.shape
    N, C, H, W = imgs.shape
    bit = 256
    soft_hist = SoftHistogram(bins=bit, min=0, max=1, sigma=400)
    loss = []
    if normalize:
        imgs = (imgs + 1) / 2.0
        refs = (refs + 1) / 2.0
    imgs = imgs.reshape(N, C, -1)
    refs = refs.reshape(N, C, -1)
    for n in range(N):
        img = imgs[n]
        ref = refs[n]
        for c in range(C):
            img_hist = soft_hist(img[c])
            ref_hist = soft_hist(ref[c])
            loss.append(nn.L1Loss()(img_hist, ref_hist))
        loss = sum(loss) / (N * H * W)
    return loss



if __name__ == '__main__':
    # ContrastiveLoss
    # contrastive_loss = ContrastiveLoss(gpu_ids=0)
    # input_x = jt.randn(2, 3, 384, 512)
    # input_y = jt.randn(2, 3, 384, 512)
    # output = contrastive_loss(input_x, input_y)
    # print(output)

    # criterionGAN
    # from poe_generator import PoE_Generator
    # from poe_discriminator import PoE_Discriminator
    # from options.train_options import  TrainOptions
    # opt =  TrainOptions().parse()
    # criterionGAN = GANLoss(
    #     opt.gan_mode, tensor=jt.float32, opt=opt)
    # segment = jt.randn(1, 1, 384, 512)
    # style = jt.randn(1, 3, 384, 512)
    # generator = PoE_Generator(opt)
    # fake_image, kl_inputs = generator(segment, style)
    # discriminator = PoE_Discriminator(opt)
    # pred_fake = discriminator(fake_image, segment)
    # lossGAN = criterionGAN(
    #     pred_fake, True, for_discriminator=False)

    # hist_loss
    from options.train_options import TrainOptions
    from data import find_dataset_using_name
    opt = TrainOptions().parse()
    print(opt)
    dataset = find_dataset_using_name(opt.dataset_mode)
    instance = dataset()
    instance.initialize(opt)
    dataloader = instance.set_attrs(
        batch_size=opt.batchSize,
        shuffle=False,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain
    )
    for i, inputs in enumerate(dataloader):
        label = inputs['label']
        image = inputs['image']
        style = inputs['style']
        path = inputs['path']
        trans_image = getHistMatched(image, style)
        _loss = hist_loss(trans_image, style)
        print('loss: ', _loss)
        # print(label.shape)
        # print(image.shape)
        # print(style.shape)
        if i>5:
            break