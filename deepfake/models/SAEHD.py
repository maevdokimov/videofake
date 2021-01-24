import torch
import torch.nn as nn
import torch.nn.functional as F

from deepfake.models.utils import same_pad


class Downscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride)
        self.nonlinearity = nn.LeakyReLU(0.1)

    def forward(self, x):
        x = same_pad(x, self.kernel_size, self.stride)
        return self.nonlinearity(self.conv1(x))


class DownscaleBlock(nn.Module):
    def __init__(self, in_ch, ch, n_downscales, kernel_size):
        super().__init__()
        self.downs = []

        out_ch = ch
        for i in range(n_downscales):
            self.downs.append(Downscale(in_ch, out_ch, kernel_size=kernel_size))
            in_ch, out_ch = out_ch, out_ch * 2

    def cuda(self, **kwargs):
        super().cuda()
        for i in range(len(self.downs)):
            self.downs[i] = self.downs[i].cuda()
        return self

    def forward(self, x):
        for down in self.downs:
            x = down(x)
        return x


class Upscale(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(in_ch, out_ch*4, kernel_size=kernel_size)
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.depth_to_space = nn.PixelShuffle(2)

    def forward(self, x):
        x = same_pad(x, self.kernel_size, 1)
        x = self.conv1(x)
        x = self.nonlinearity(x)
        return self.depth_to_space(x)


class ResidualBlock(nn.Module):
    def __init__(self, ch, kernel_size=3):
        super().__init__()
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(ch, ch, kernel_size=kernel_size)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=kernel_size)
        self.nonlinearity = nn.LeakyReLU(0.2)

    def forward(self, inp):
        x = inp
        x = same_pad(x, self.kernel_size, 1)
        x = self.conv1(x)
        x = self.nonlinearity(x)
        x = same_pad(x, self.kernel_size, 1)
        x = self.conv2(x)
        return self.nonlinearity(x + inp)


class Encoder(nn.Module):
    def __init__(self, in_ch, e_ch):
        super().__init__()
        self.in_ch = in_ch
        self.e_ch = e_ch

        self.down = DownscaleBlock(self.in_ch, self.e_ch, n_downscales=4, kernel_size=5)

    def cuda(self, **kwargs):
        super().cuda()
        self.down = self.down.cuda()
        return self

    def forward(self, x):
        return torch.flatten(self.down(x), start_dim=1)

    def get_out_shape(self, resolution):
        return self.e_ch * 8, resolution // (2**4), resolution // (2**4)


class Inter(nn.Module):
    def __init__(self, in_ch, ae_ch, ae_out_ch, lowest_dense_res=4):
        super().__init__()
        self.in_ch, self.ae_ch, self.ae_out_ch = in_ch, ae_ch, ae_out_ch
        self.lowest_dense_res = lowest_dense_res

        self.dense1 = nn.Linear(in_ch, ae_ch)
        self.dense2 = nn.Linear(ae_ch, lowest_dense_res * lowest_dense_res * ae_out_ch)
        self.upscale = Upscale(ae_out_ch, ae_out_ch)

    def forward(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = torch.reshape(x, [-1, self.ae_out_ch, self.lowest_dense_res, self.lowest_dense_res])
        return self.upscale(x)

    def get_out_ch(self):
        return self.ae_out_ch


class Decoder(nn.Module):
    def __init__(self, in_ch, d_ch, d_mask_ch):
        super().__init__()
        self.upscale0 = Upscale(in_ch, d_ch * 8, kernel_size=3)
        self.upscale1 = Upscale(d_ch * 8, d_ch * 4, kernel_size=3)
        self.upscale2 = Upscale(d_ch * 4, d_ch * 2, kernel_size=3)

        self.res0 = ResidualBlock(d_ch * 8, kernel_size=3)
        self.res1 = ResidualBlock(d_ch * 4, kernel_size=3)
        self.res2 = ResidualBlock(d_ch * 2, kernel_size=3)

        self.out_conv = nn.Conv2d(d_ch * 2, 3, kernel_size=1)

        self.upscalem0 = Upscale(in_ch, d_mask_ch * 8, kernel_size=3)
        self.upscalem1 = Upscale(d_mask_ch * 8, d_mask_ch * 4, kernel_size=3)
        self.upscalem2 = Upscale(d_mask_ch * 4, d_mask_ch * 2, kernel_size=3)
        self.out_convm = nn.Conv2d(d_mask_ch * 2, 1, kernel_size=1)

    def forward(self, inp):
        z = inp

        x = self.upscale0(z)
        x = self.res0(x)
        x = self.upscale1(x)
        x = self.res1(x)
        x = self.upscale2(x)
        x = self.res2(x)
        x = F.sigmoid(self.out_conv(x))

        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
        m = F.sigmoid(self.out_convm(m))

        return x, m
