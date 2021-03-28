import torch
import torch.nn as nn
from torch.optim import lr_scheduler

import functools


# https://github.com/donydchen/ganimation_replicate/blob/a2ca194ec422aa2145404ce414dfc218c3a42f5c/model/model_utils.py

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(
        optimizer,
        lr_policy,
        epoch_count,
        niter,
        niter_decay,
        lr_decay_iters
    ):
    if lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + epoch_count - niter) / float(niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super().__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class Generator(nn.Module):
    def __init__(
            self,
            img_nc,
            aus_nc,
            ngf=64,
            norm_layer=nn.BatchNorm2d,
            use_dropout=False,
            n_blocks=6,
            padding_type='zero'
    ):
        super().__init__()
        self.input_nc = img_nc + aus_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.Conv2d(self.input_nc, ngf, kernel_size=7, stride=1,
                           padding=3, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=4,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                  use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=4,
                                         stride=2, padding=1, bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        self.model = nn.Sequential(*model)

        color_top = []
        color_top += [nn.Conv2d(ngf, img_nc, kernel_size=7, stride=1, padding=3, bias=False),
                      nn.Tanh()]
        self.color_top = nn.Sequential(*color_top)
        au_top = []
        au_top += [nn.Conv2d(ngf, 1, kernel_size=7, stride=1, padding=3, bias=False),
                   nn.Sigmoid()]
        self.au_top = nn.Sequential(*au_top)

    def forward(self, img, au):
        sparse_au = au.unsqueeze(2).unsqueeze(3)
        sparse_au = sparse_au.expand(sparse_au.size(0), sparse_au.size(1), img.size(2), img.size(3))
        self.input_img_au = torch.cat([img, sparse_au], dim=1)

        embed_features = self.model(self.input_img_au)

        return self.color_top(embed_features), self.au_top(embed_features), embed_features


class Discriminator(nn.Module):
    def __init__(
            self,
            input_nc,
            aus_nc,
            image_size=128,
            ndf=64,
            n_layers=6,
            norm_layer=nn.BatchNorm2d
    ):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.01, True)
        ]

        cur_dim = ndf
        for n in range(1, n_layers):
            sequence += [
                nn.Conv2d(cur_dim, 2 * cur_dim,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                nn.LeakyReLU(0.01, True)
            ]
            cur_dim = 2 * cur_dim

        self.model = nn.Sequential(*sequence)

        self.dis_top = nn.Conv2d(cur_dim, 1, kernel_size=kw - 1, stride=1, padding=padw, bias=False)

        k_size = int(image_size / (2 ** n_layers))
        self.aus_top = nn.Conv2d(cur_dim, aus_nc, kernel_size=k_size, stride=1, bias=False)

    def forward(self, img):
        embed_features = self.model(img)
        pred_map = self.dis_top(embed_features)
        pred_aus = self.aus_top(embed_features)
        return pred_map.squeeze(), pred_aus.squeeze()


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]


class GANLoss(nn.Module):
    def __init__(self, gan_type='wgan-gp', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_type = gan_type
        if self.gan_type == 'wgan-gp':
            self.loss = lambda x, y: -torch.mean(x) if y else torch.mean(x)
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'gan':
            self.loss = nn.BCELoss()
        else:
            raise NotImplementedError('GAN loss type [%s] is not found' % gan_type)

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        if self.gan_type == 'wgan-gp':
            target_tensor = target_is_real
        else:
            target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)