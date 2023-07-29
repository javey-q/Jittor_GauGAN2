"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import jittor as jt
from jittor import init
from jittor import nn
import jittor.transform as transform
import models.networks as networks
import util.util as util
from models.networks.poe_generator import PoE_Generator
from models.networks.poe_discriminator import PoE_Discriminator
import warnings
warnings.filterwarnings("ignore")


class PoEGAN_Model(nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = jt.float32
        self.ByteTensor = jt.float32

        self.netG, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            if not opt.no_contrast_loss:
                self.criterionContrast= networks.ContrastiveLoss(self.opt.gpu_ids)
            self.KLDLoss = networks.KLDLoss()
            if opt.use_hist:
                self.HistLoss = networks.HistLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def execute(self, data, mode):
        input_semantics, real_image, style_image = self.preprocess_input(data)
        # print("input_semantics: " , input_semantics.shape)
        # print("real_image: ", real_image.shape)
        # exit(0)
        if mode == 'generator':
            g_loss, generated = self.compute_generator_loss(
                input_semantics, real_image, style_image)
            return g_loss, generated
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                input_semantics, real_image, style_image)
            return d_loss
        elif mode == 'inference':
            with jt.no_grad():
                fake_image, _, _ = self.generate_fake(input_semantics, style_image)
            return fake_image
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())

        if opt.isTrain:
            D_params = list(self.netD.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = jt.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = jt.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = PoE_Generator(opt)
        netD = PoE_Discriminator(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)

        return netG, netD

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # change data types
        data['label'] = data['label'].long()

        # create one-hot label map
        label_map = data['label']
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label else self.opt.label_nc
        input_label = jt.zeros((bs, nc, h, w), dtype=self.FloatTensor)
        input_semantics = input_label.scatter_(1, label_map, jt.float32(1.0))

        if self.opt.dataset_mode == 'Jittor':
            return input_semantics, data['image'], data['style']
        else:
            return input_semantics, data['image'], None

    def compute_generator_loss(self, input_semantics, real_image, style_image):
        G_losses = {}

        fake_image, KLD_loss, Hist_loss = self.generate_fake(
            input_semantics, style_image, compute_kld_loss=True, compute_hist_loss=self.opt.use_hist)

        G_losses['KLD'] = KLD_loss
        if self.opt.use_hist:
            G_losses['Hist'] = Hist_loss

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        G_losses['GAN'] = self.criterionGAN(
            pred_fake, True, for_discriminator=False)

        if not self.opt.no_contrast_loss:
            G_losses['Contrast'] = self.criterionContrast(
                fake_image, real_image) * self.opt.lambda_contrast

        return G_losses, fake_image

    def compute_discriminator_loss(self, input_semantics, real_image, style_image):
        D_losses = {}
        with jt.no_grad():
            fake_image, _, _ = self.generate_fake(input_semantics, style_image)
            # fake_image = fake_image.detach()
            # fake_image.requires_grad_()

        pred_fake, pred_real = self.discriminate(
            input_semantics, fake_image, real_image)

        D_losses['D_Fake'] = self.criterionGAN(
            pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(
            pred_real, True, for_discriminator=True)

        return D_losses

    def generate_fake(self, input_semantics, style_image, compute_kld_loss=False, compute_hist_loss=False):
        Hist_loss = None
        KLD_loss = None
        fake_image, kl_inputs = self.netG(input_semantics, style_image)
        if compute_kld_loss:
            kl_style = kl_inputs['style']
            KLD_loss = self.KLDLoss(kl_style[0], kl_style[1]) * self.opt.lambda_kld_style
            kl_segment = kl_inputs['segment']
            KLD_loss += self.KLDLoss(kl_segment[0], kl_segment[1]) * self.opt.lambda_kld_segment
        if compute_hist_loss:
            Hist_loss = self.HistLoss(fake_image, style_image) * self.opt.lambda_hist


        return fake_image, KLD_loss, Hist_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        # fake_concat = jt.concat([input_semantics, fake_image], dim=1)
        # real_concat = jt.concat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        image_concat = jt.concat([fake_image, real_image], dim=0)
        segment_concat = jt.concat([input_semantics, input_semantics], dim=0)


        discriminator_out = self.netD(image_concat, segment_concat)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                # fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                # real.append([tensor[tensor.size(0) // 2:] for tensor in p])
                fake.append(p[:p.size(0) // 2])
                real.append(p[p.size(0) // 2:])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]
        return fake, real

    def get_edges(self, t):
        edge = self.ByteTensor(t.size()).zero_()
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (
            t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-
                                  1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (
            t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1,
                                  :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = jt.exp(0.5 * logvar)
        eps = jt.randn_like(std)
        return eps.multiply(std).add(mu)

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

