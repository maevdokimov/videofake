import torch
import torch.nn as nn


def init_wights(layer):
    if type(layer) == nn.Conv2d or type(layer) == nn.ConvTranspose2d:
        nn.init.normal_(layer.weight.data, 0.0, 0.02)
    elif type(layer) == nn.BatchNorm2d:
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv_block = nn.Sequential(
            nn.ConvTranspose2d(100, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh(),
        )
        self.deconv_block.apply(init_wights)

    def forward(self, input):
        return self.deconv_block(input)


# class Discriminator(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv_block = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=0, bias=False),
#             nn.Sigmoid(),
#         )
#         self.conv_block.apply(init_wights)
#
#     def forward(self, input):
#         return self.conv_block(input)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )
        self.conv_block.apply(init_wights)

    def forward(self, input):
        return self.conv_block(input)


if __name__ == '__main__':
    z = torch.randn(32, 100, 1, 1)
    gen = Generator()
    discr = Discriminator()
    assert gen(z).shape == (32, 3, 64, 64)
    assert discr(gen(z)).squeeze().shape == (32,)
