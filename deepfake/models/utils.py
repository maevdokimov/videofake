import torch
import torch.nn.functional as F
import numpy as np


def same_pad(x: torch.tensor, kernel_size: int, stride: int):
    in_height, in_width = x.shape[-2:]

    if in_height % stride == 0:
        pad_along_height = max(kernel_size - stride, 0)
    else:
        pad_along_height = max(kernel_size - (in_height % stride), 0)
    if in_width % stride == 0:
        pad_along_width = max(kernel_size - stride, 0)
    else:
        pad_along_width = max(kernel_size - (in_width % stride), 0)

    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    return F.pad(x, [pad_left, pad_right, pad_top, pad_bottom])


def gaussian_blur(input, radius=2.):
    def gaussian(x, mu, sigma):
        return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))

    def make_kernel(sigma):
        kernel_size = max(3, int(2 * 2 * sigma + 1))
        mean = np.floor(0.5 * kernel_size)
        kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
        np_kernel = np.outer(kernel_1d, kernel_1d).astype(np.float32)
        kernel = np_kernel / np.sum(np_kernel)
        return kernel, kernel_size

    gauss_kernel, kernel_size = make_kernel(radius)
    padding = kernel_size//2
    if padding != 0:
        padding = [padding, padding, padding, padding]
    else:
        padding = [0, 0]
    nb_channels = input.shape[1]
    k = torch.from_numpy(gauss_kernel).view(1, 1, kernel_size, kernel_size).repeat(nb_channels, 1, 1, 1)
    if input.is_cuda:
        k = k.cuda()

    x = input
    x = F.pad(x, padding)
    x = F.conv2d(x, k, groups=nb_channels)
    return x


def dssim(img1, img2, max_val, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    if img1.is_cuda != img2.is_cuda:
        raise ValueError('Images dtype is not consistent')

    filter_size = max(1, filter_size)

    kernel = np.arange(0, filter_size, dtype=np.float32)
    kernel -= (filter_size - 1 ) / 2.0
    kernel = kernel**2
    kernel *= ( -0.5 / (filter_sigma**2) )
    kernel = np.reshape (kernel, (1,-1)) + np.reshape(kernel, (-1,1) )
    kernel = torch.from_numpy(kernel).view(1, -1)
    kernel = F.softmax(kernel, dim=1).view(1, 1, filter_size, filter_size).repeat(3, 1, 1, 1)
    if img1.is_cuda:
        kernel = kernel.cuda()

    def reducer(x):
        return F.conv2d(x, kernel, groups=3)

    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2

    mean0 = reducer(img1)
    mean1 = reducer(img2)
    num0 = mean0 * mean1 * 2.0
    den0 = torch.square(mean0) + torch.square(mean1)
    luminance = (num0 + c1) / (den0 + c1)

    num1 = reducer(img1 * img2) * 2.0
    den1 = reducer(torch.square(img1) + torch.square(img2))
    c2 *= 1.0
    cs = (num1 - num0 + c2) / (den1 - den0 + c2)

    lcs = luminance * cs
    ssim_val = lcs.view(lcs.shape[0], lcs.shape[1], -1).mean(2)
    dssim = (1.0 - ssim_val ) / 2.0

    return dssim


if __name__ == '__main__':
    mask_path = './data/faces/dst_masked/0.npy'
    from skimage.transform import resize
    mask = np.load(mask_path)
    mask = torch.from_numpy(np.expand_dims(resize(mask, [64,64]).transpose([2, 0, 1]), axis=0))
    mask2d = gaussian_blur(mask).numpy().squeeze()
