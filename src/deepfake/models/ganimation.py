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
            lr,
            beta1,
            gan_type
    ):
        super().__init__()
        self.generator = Generator(**generator_config)
        self.discriminator = Discriminator(**discriminator_config)

        self.scheduler_config = scheduler_config

        self.lr = lr
        self.beta1 = beta1
        self.gan_type = gan_type

        self.criterionTV = TVLoss()
        self.criterionGAN = GANLoss(gan_type=gan_type)
        self.criterionL1 = nn.L1Loss()
        self.criterionMSE = torch.nn.MSELoss().to(self.device)

    def configure_optimizers(self):
        opt_gen = Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        opt_dis = Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        gen_scheduler = get_scheduler(opt_gen, **self.scheduler_config)
        dis_scheduler = get_scheduler(opt_dis, **self.scheduler_config)
        return [opt_gen, opt_dis], [gen_scheduler, dis_scheduler]
