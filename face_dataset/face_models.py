import torch
import torch.nn as nn
from copy import deepcopy


class VAE(nn.Module):
    def __init__(self, image_size, embedding_size):
        super().__init__()
        self.input_size = image_size * image_size * 3
        self.intermediate_size = int(image_size * image_size * 3 / 2)

        self.fc1 = nn.Linear(self.input_size, self.intermediate_size)
        self.fc2_1 = nn.Linear(self.intermediate_size, embedding_size)
        self.fc2_2 = nn.Linear(self.intermediate_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, self.intermediate_size)
        self.fc4_1 = nn.Linear(self.intermediate_size, self.input_size)
        self.fc4_2 = nn.Linear(self.intermediate_size, self.input_size)

    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            std = logsigma.exp()
            eps = torch.randn(*std.shape).cuda() if next(self.parameters()).is_cuda else torch.randn(*std.shape)
            return eps.mul(std).add(mu)
        else:
            return mu

    def encode(self, x):
        h1 = self.fc1(x)
        return self.fc2_1(h1), self.fc2_2(h1)

    def decode(self, x):
        h3 = self.fc3(x)
        return self.fc4_1(h3), self.fc4_2(h3)

    def forward(self, x):
        flattened_view_x = x.view(-1, self.input_size)

        latent_mu, latent_logsigma = self.encode(flattened_view_x)
        latent_sample = self.gaussian_sampler(latent_mu, latent_logsigma)
        reconstruction_mu, reconstruction_logsigma = self.decode(latent_sample)

        return reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma


class ConvolutionalVAE(nn.Module):
    def __init__(
            self,
            image_size,
            num_filters,
            embedding_size,
            kernel_size=3,
            pool_kernel_size=2,
            nonlinearity='relu',
    ):
        super().__init__()
        if nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        else:
            raise ValueError(f'Unknown nonlinearity {nonlinearity}')
        self.image_size = image_size
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.pool_kernel_size = pool_kernel_size

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size, padding=1),
            self.nonlinearity,
            nn.Conv2d(num_filters, 2 * num_filters, kernel_size, padding=1),
            self.nonlinearity,
            nn.Conv2d(2 * num_filters, 2 * num_filters, kernel_size, padding=1),
            self.nonlinearity,
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(2 * num_filters, 2 * num_filters, kernel_size, padding=1),
            self.nonlinearity,
            nn.Conv2d(2 * num_filters, 4 * num_filters, kernel_size, padding=1),
            self.nonlinearity,
            nn.Conv2d(4 * num_filters, 4 * num_filters, kernel_size, padding=1),
            self.nonlinearity,
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(4 * num_filters, 4 * num_filters, kernel_size, padding=1),
            self.nonlinearity,
            nn.Conv2d(4 * num_filters, 6 * num_filters, kernel_size, padding=1),
            self.nonlinearity,
            nn.Conv2d(6 * num_filters, 6 * num_filters, kernel_size, padding=1),
            self.nonlinearity,
            nn.MaxPool2d(kernel_size=pool_kernel_size)
        )
        self.compressed_size = int(image_size / 8)
        self.out_filters = 6 * num_filters

        out_shape = int(self.out_filters * (self.compressed_size ** 2))

        self.linear_mu = nn.Linear(out_shape, embedding_size)
        self.linear_logsigma = nn.Linear(out_shape, embedding_size)

        self.reconstruction_mu = nn.Linear(embedding_size, out_shape)
        self.reconstruction_logsigma = nn.Linear(embedding_size, out_shape)

        self.deconv_mu = self.deconv_module()
        self.deconv_logsigma = self.deconv_module()

    def deconv_module(self):
        module = nn.Sequential(
            nn.ConvTranspose2d(6 * self.num_filters, 6 * self.num_filters, self.pool_kernel_size, stride=2),
            self.nonlinearity,
            nn.Conv2d(6 * self.num_filters, 4 * self.num_filters, self.kernel_size, padding=1),
            self.nonlinearity,
            nn.Conv2d(4 * self.num_filters, 4 * self.num_filters, self.kernel_size, padding=1),
            self.nonlinearity,
            nn.ConvTranspose2d(4 * self.num_filters, 4 * self.num_filters, self.pool_kernel_size, stride=2),
            self.nonlinearity,
            nn.Conv2d(4 * self.num_filters, 2 * self.num_filters, self.kernel_size, padding=1),
            self.nonlinearity,
            nn.Conv2d(2 * self.num_filters, 2 * self.num_filters, self.kernel_size, padding=1),
            nn.ConvTranspose2d(2 * self.num_filters, 2 * self.num_filters, self.pool_kernel_size, stride=2),
            self.nonlinearity,
            nn.Conv2d(2 * self.num_filters, self.num_filters, self.kernel_size, padding=1),
            self.nonlinearity,
            nn.Conv2d(self.num_filters, 3, self.kernel_size, padding=1)
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
        h1 = self.conv1(x)
        h1 = self.conv2(h1)
        h1 = self.conv3(h1).view(x.shape[0], -1)

        return self.linear_mu(h1), self.linear_logsigma(h1)

    def decode(self, x):
        mu, logsigma = self.reconstruction_mu(x), self.reconstruction_logsigma(x)
        mu = mu.view(mu.shape[0], self.out_filters, self.compressed_size, self.compressed_size)
        logsigma = logsigma.view(mu.shape[0], self.out_filters, self.compressed_size, self.compressed_size)
        mu, logsigma = self.deconv_mu(mu), self.deconv_logsigma(logsigma)
        return mu, logsigma

    def forward(self, x):
        latent_mu, latent_logsigma = self.encode(x)
        latent_sample = self.gaussian_sampler(latent_mu, latent_logsigma)
        reconstruction_mu, reconstruction_logsigma = self.decode(latent_sample)

        return reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma


if __name__ == '__main__':
    t = torch.randn(16, 3, 128, 128).cuda()
    model = ConvolutionalVAE(128, 32, 512).cuda()
    res = model(t)
    print([tens.shape for tens in res])
    # model = model.cuda()
    # t = t.cuda()
