import torch
from .base_model import BaseModel
import networks
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime



class XFcGANModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='instance', netG='unet_256', dataset_mode='yuv')
        parser.set_defaults(where_add='input', nz=0)
        if is_train:
            parser.set_defaults(gan_mode='lsgan', lambda_l1=100.0)

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """


        self.train_mode = "normal"
        BaseModel.__init__(self, opt)

        if self.isTrain:
            self.model_names = ['En', 'D']
        else:  # during test time, only load G
            self.model_names = ['En']
        # define networks (both generator and discriminator)

        self.netEn = networks.define_for_contrast()





        self.netD = networks.define_D(opt.input_nc*2, opt.ndf, netD=opt.netD2, norm=opt.norm, nl=opt.nl,
                                      init_type=opt.init_type, init_gain=opt.init_gain, num_Ds=opt.num_Ds, gpu_ids=self.gpu_ids)



        if self.isTrain:
            # define loss functions
            #if not opt.no_ganFeat_loss:
            #    self.criterionGAN = networks.GANLoss_hd(gan_mode=opt.gan_mode).to(self.device)
            #else:
            #    self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            #self.criterionGAN = networks.GANLoss_hd(gan_mode=opt.gan_mode).to(self.device)
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionSSIM_mod = networks.modified_SSIM_Loss2()
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
            self.criterionSSIM = networks.SSIM()  # SSIM 结构相似性损失作为图像的重构损失
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionTV = networks.TVLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_En = torch.optim.Adam(self.netEn.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_En)

            self.optimizers.append(self.optimizer_D)



    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if 'F' in input.keys():
            self.real_F = input['F'].to(self.device)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def test(self):
        with torch.no_grad():
            # concat_AB = torch.cat([self.real_A,self.real_B],1)

            self.cat_AB = torch.cat([self.real_A, self.real_B], 1)
            self.dict_AB, self.lat_AB = self.netEn(self.cat_AB)

            self.fake_lat = self.lat_AB
            self.fake_F = self.netDe(self.fake_lat, self.dict_AB)
            #cat_dict,z = self.encode(concat_AB)
            #self.fake_F = self.netG(concat_AB)

            return self.real_A, self.fake_F, self.real_B

    def test2(self,real_A,real_B):
        with torch.no_grad():
            # concat_AB = torch.cat([self.real_A,self.real_B],1)


            self.fake_F = self.netEn(real_A, real_B)

            #cat_dict,z = self.encode(concat_AB)
            #self.fake_F = self.netG(concat_AB)

            return real_A, self.fake_F, real_B

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""


        self.fake_F = self.netEn(self.real_A, self.real_B)



    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake1 = self.netD(torch.cat([self.real_A,self.fake_F.detach()],1))

        self.loss_D_fake1,_  = self.criterionGAN(pred_fake1, False)

        pred_fake2 = self.netD(torch.cat([self.fake_F.detach(),self.real_B],1))

        self.loss_D_fake2,_  = self.criterionGAN(pred_fake2, False)

        self.loss_D_fake = self.loss_D_fake1+self.loss_D_fake2

        # Real
        #real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(torch.cat([self.real_A,self.real_B],1))
        self.loss_D_real,_  = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = self.loss_D_fake + self.loss_D_real


        self.loss_D.backward()





    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        #fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake1 = self.netD(torch.cat([self.real_A, self.fake_F.detach()], 1))

        self.loss_G_GAN1, _ = self.criterionGAN(pred_fake1, True)

        pred_fake2 = self.netD(torch.cat([self.fake_F.detach(), self.real_B], 1))

        self.loss_G_GAN2, _ = self.criterionGAN(pred_fake2, True)

        self.loss_D_fake = self.loss_D_fake1 + self.loss_D_fake2

        self.loss_G_GAN=self.loss_G_GAN1+self.loss_G_GAN2





        self.loss_SSIM = 0
        if self.opt.lambda_SSIM > 0.0:
           self.loss_SSIM = self.criterionSSIM_mod(self.real_A, self.real_B, self.fake_F) * self.opt.lambda_SSIM






        self.loss_G = self.loss_G_GAN + self.loss_SSIM

        self.loss_G.backward(retain_graph=True)


    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)

        self.set_requires_grad(self.netD, True)  # enable backprop for D
        #self.set_requires_grad(self.netD, False)
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()
        self.optimizer_D.step()  # update D's weights

        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_En.zero_grad()  # set G's gradients to zero

        self.backward_G()  # calculate graidents for G
        self.optimizer_En.step()  # udpate G's weights


        self.optimizer_En.zero_grad()  # set G's gradients to zero

        self.optimizer_En.step()  # udpate G's weights


class LaplacianConv(nn.Module):
    # 仅有一个参数，通道，用于自定义算子模板的通道
    def __init__(self, channels=1):
        super().__init__()
        self.channels = channels
        kernel = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)  # 扩展到3个维度
        kernel = np.repeat(kernel, self.channels, axis=0)  # 3个通道都是同一个算子
        self.weight = nn.Parameter(data=kernel, requires_grad=False)  # 不允许求导更新参数，保持常量

    def __call__(self, x):
        # 第一个参数为输入，由于这里只是测试，随便打开的一张图只有3个维度，pytorch需要处理4个维度的样本，因此需要添加一个样本数量的维度
        # padding2是为了能够照顾到边角的元素
        self.weight.data = self.weight.data.to(x.device)
        x = F.conv2d(x, self.weight, padding=1, groups=self.channels)
        return x


