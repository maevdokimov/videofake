import torch
import torch.nn as nn


class DeepfakeVAE(nn.Module):
    def __init__(
            self,
            image_size,
            embedding_size,
    ):
        super().__init__()
        self.nonlinearity = nn.ReLU()

        self.image_size = image_size
        self.embedding_size = embedding_size

        self.conv_block = self.encode_module()

        self.linear_mu = nn.Linear(256 * int(image_size / 8) ** 2, embedding_size)
        self.linear_logsigma = nn.Linear(256 * int(image_size / 8) ** 2, embedding_size)

        self.reconstruction_mu_1 = nn.Linear(embedding_size, 256 * int(image_size / 8) ** 2)
        self.reconstruction_logsigma_1 = nn.Linear(embedding_size, 256 * int(image_size / 8) ** 2)
        self.reconstruction_mu_2 = nn.Linear(embedding_size, 256 * int(image_size / 8) ** 2)
        self.reconstruction_logsigma_2 = nn.Linear(embedding_size, 256 * int(image_size / 8) ** 2)

        self.deconv_mu_1 = self.deconv_module()
        self.deconv_logsigma_1 = self.deconv_module()
        self.deconv_mu_2 = self.deconv_module()
        self.deconv_logsigma_2 = self.deconv_module()

    def encode_module(self):
        conv_block = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            self.nonlinearity,
            nn.Conv2d(64, 64, 3, padding=1),
            self.nonlinearity,
            nn.Conv2d(64, 64, 3, padding=1),
            self.nonlinearity,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, 3, padding=1),
            self.nonlinearity,
            nn.Conv2d(128, 128, 3, padding=1),
            self.nonlinearity,
            nn.Conv2d(128, 128, 3, padding=1),
            self.nonlinearity,
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, 3, padding=1),
            self.nonlinearity,
            nn.Conv2d(256, 256, 3, padding=1),
            self.nonlinearity,
            nn.Conv2d(256, 256, 3, padding=1),
            self.nonlinearity,
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
        )
        return conv_block

    def deconv_module(self):
        module = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2),
            self.nonlinearity,
            nn.Conv2d(256, 256, 3, padding=1),
            self.nonlinearity,
            nn.Conv2d(256, 256, 3, padding=1),
            self.nonlinearity,
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            self.nonlinearity,
            nn.Conv2d(128, 128, 3, padding=1),
            self.nonlinearity,
            nn.Conv2d(128, 128, 3, padding=1),
            self.nonlinearity,
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            self.nonlinearity,
            nn.Conv2d(64, 64, 3, padding=1),
            self.nonlinearity,
            nn.Conv2d(64, 3, 3, padding=1)
        )
        return module

    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            std = logsigma.exp()
            eps = torch.randn(*std.shape).cuda() if next(self.parameters()).is_cuda else torch.randn(*std.shape)
            return eps.mul(std).add(mu)
        else:
            return mu

    def encode(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = self.conv_block(x)

        return self.linear_mu(x), self.linear_logsigma(x)

    def decode(self, x, block_num):
        if block_num == 1:
            reconstruction_mu, reconstruction_logsigma = self.reconstruction_mu_1, self.reconstruction_logsigma_1
            deconv_mu, deconv_logsigma = self.deconv_mu_1, self.deconv_logsigma_1
        elif block_num == 2:
            reconstruction_mu, reconstruction_logsigma = self.reconstruction_mu_2, self.reconstruction_logsigma_2
            deconv_mu, deconv_logsigma = self.deconv_mu_2, self.deconv_logsigma_2
        else:
            raise ValueError("Incorrect block_num")

        mu, logsigma = reconstruction_mu(x), reconstruction_logsigma(x)
        mu = mu.view(mu.shape[0], 256, int(self.image_size / 8), int(self.image_size / 8))
        logsigma = logsigma.view(mu.shape[0], 256, int(self.image_size / 8), int(self.image_size / 8))
        mu, logsigma = deconv_mu(mu), deconv_logsigma(logsigma)
        return mu, logsigma

    def forward(self, x, block_num):
        latent_mu, latent_logsigma = self.encode(x)
        latent_sample = self.gaussian_sampler(latent_mu, latent_logsigma)
        reconstruction_mu, reconstruction_logsigma = self.decode(latent_sample, block_num)

        return reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma


if __name__ == '__main__':
    t = torch.randn(4, 3, 128, 128).cuda()
    model = DeepfakeVAE(128, 512).cuda()
    res = model(t, 1)
    print([elem.shape for elem in res])
    import time
    time.sleep(3)
