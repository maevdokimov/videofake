import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam

from src.deepfake.models.ganimation_modules import Generator, Discriminator, TVLoss, GANLoss, get_scheduler


class Ganimation(pl.LightningModule):
    def __init__(
            self,
            generator_config,
            discriminator_config,
            scheduler_config,
            train_config,
    ):
        super().__init__()
        ##########
        self.save_hyperparameters()
        self.automatic_optimization = False
        ##########

        self.generator = Generator(**generator_config)
        self.discriminator = Discriminator(**discriminator_config)

        self.scheduler_config = scheduler_config
        self.train_config = train_config

        self.criterionTV = TVLoss()
        self.criterionGAN = GANLoss(gan_type=train_config['gan_type'])
        self.criterionL1 = nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss().to(self.device)

    def configure_optimizers(self):
        opt_gen = Adam(self.generator.parameters(), lr=self.train_config['lr'],
                       betas=(self.train_config['beta1'], 0.999))
        opt_dis = Adam(self.discriminator.parameters(), lr=self.train_config['lr'],
                       betas=(self.train_config['beta1'], 0.999))
        gen_scheduler = get_scheduler(opt_gen, **self.scheduler_config)
        dis_scheduler = get_scheduler(opt_dis, **self.scheduler_config)
        return [opt_gen, opt_dis], [gen_scheduler, dis_scheduler]

    def training_step(self, batch, batch_idx):
        opt_gen, opt_dis = self.optimizers()

        src_imgs, tar_imgs, src_aus, tar_aus = batch
        result = self.forward(src_imgs, tar_aus, tar_imgs, src_aus)

        with opt_dis.toggle_model():
            opt_dis.zero_grad()
            dis_loss = self.generator_closure(result)
            opt_dis.step()

        gen_loss = None
        if batch_idx % self.train_config['train_gen_iter'] == 0:
            with opt_gen.toggle_model():
                opt_gen.zero_grad()
                gen_loss = self.generator_closure(result)
                opt_gen.step()

        self.log_dict({'g_loss': gen_loss, 'd_loss': dis_loss}, prog_bar=True)

    def forward(self, src_imgs, tar_aus, tar_imgs=None, src_aus=None):
        color_mask, aus_mask, embed = self.generator(src_imgs, tar_aus)
        fake_img = aus_mask * src_imgs + (1 - aus_mask) * color_mask

        if tar_imgs is None != src_aus is None:
            raise ValueError('tar_imgs or src_aus not given')

        result_map = {
            'src_imgs': src_imgs,
            'tar_aus': tar_aus,
            'color_mask': color_mask,
            'aus_mask': aus_mask,
            'embed': embed,
            'fake_img': fake_img,
        }

        if tar_imgs is not None:
            rec_color_mask, rec_aus_mask, rec_embed = self.generator(fake_img, src_aus)
            rec_real_img = rec_aus_mask * fake_img + (1 - rec_aus_mask) * rec_color_mask
            result_map['rec_color_mask'] = rec_color_mask
            result_map['rec_aus_mask'] = rec_aus_mask
            result_map['rec_embed'] = rec_embed
            result_map['rec_real_img'] = rec_real_img
            result_map['tar_imgs'] = tar_imgs
            result_map['src_aus'] = src_aus
        return result_map

    def generator_closure(self, result):
        pred_fake, pred_fake_aus = self.discriminator(result['fake_image'])
        loss_gen_gan = self.criterionGAN(pred_fake, True)
        loss_gen_fake_aus = self.criterionMSE(pred_fake_aus, result['tar_aus'])

        loss_gen_rec = self.criterionL1(result['rec_real_img'], result['src_img'])

        loss_gen_mask_real_aus = torch.mean(result['aus_mask'])
        loss_gen_mask_fake_aus = torch.mean(result['rec_aus_mask'])
        loss_gen_smooth_real_aus = self.criterionTV(result['aus_mask'])
        loss_gen_smooth_fake_aus = self.criterionTV(result['rec_aus_mask'])

        loss_gen = self.train_config['lambda_dis'] * loss_gen_gan \
                   + self.train_config['lambda_aus'] * loss_gen_fake_aus \
                   + self.train_config['lambda_rec'] * loss_gen_rec \
                   + self.train_config['lambda_mask'] * (loss_gen_mask_real_aus + loss_gen_mask_fake_aus) \
                   + self.train_config['lambda_tv'] * (loss_gen_smooth_real_aus + loss_gen_smooth_fake_aus)

        self.manual_backward(loss_gen)
        return loss_gen

    def discriminator_closure(self, result):
        pred_real, pred_real_aus = self.discriminator(result['src_imgs'])
        loss_dis_real = self.criterionGAN(pred_real, True)
        loss_dis_real_aus = self.criterionMSE(pred_real_aus, result['src_aus'])

        pred_fake, _ = self.net_dis(result['fake_img'].detach())
        loss_dis_fake = self.criterionGAN(pred_fake, False)

        loss_dis = self.train_config['lambda_dis'] * (loss_dis_fake + loss_dis_real) \
                   + self.train_config['lambda_aus'] * loss_dis_real_aus

        if self.train_config['gan_type'] == 'wgan-gp':
            loss_dis_gp = self.gradient_penalty(self.src_img, self.fake_img)
            loss_dis += self.train_config['lambda_wgan_gp'] * loss_dis_gp

        self.manual_backward(loss_dis)

    def gradient_penalty(self, input_img, generate_img):
        # interpolate sample
        alpha = torch.rand(input_img.size(0), 1, 1, 1).to(self.device)
        inter_img = (alpha * input_img.data + (1 - alpha) * generate_img.data).requires_grad_(True)
        inter_img_prob, _ = self.net_dis(inter_img)

        # computer gradient penalty: x: inter_img, y: inter_img_prob
        # (L2_norm(dy/dx) - 1)**2
        dydx = torch.autograd.grad(outputs=inter_img_prob,
                                   inputs=inter_img,
                                   grad_outputs=torch.ones(inter_img_prob.size()).to(self.device),
                                   retain_graph=True,
                                   create_graph=True,
                                   only_inputs=True)[0]
        dydx = dydx.view(dydx.size(0), -1)
        dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
        return torch.mean((dydx_l2norm - 1) ** 2)
