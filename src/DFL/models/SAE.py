import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from pathlib import Path

from src.deepfake.models.utils import same_pad
from src.DFL.training.init_model import PairedDataset
from src.deepfake.models.utils import gaussian_blur, dssim


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
        self.downs = nn.ModuleList([])

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
        x = torch.sigmoid(self.out_conv(x))

        m = self.upscalem0(z)
        m = self.upscalem1(m)
        m = self.upscalem2(m)
        m = torch.sigmoid(self.out_convm(m))

        return x, m


class SAEModel(pl.LightningModule):
    def __init__(
            self,
            resolution: int = 64,
            encoder_dims: int = 64,
            inter_dims: int = 128,
            decoder_dims: int = 64,
            decoder_mask_dims: int = 22,
            lr: float = 5e-5,
            batch_size: int = 8,
            num_workers: int = 8,
            src_path: Path = None,
            dst_path: Path = None,
            out_path: Path = None,
            out_freq: int = 10,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.resolution = resolution
        self.encoder_dims = encoder_dims
        self.inter_dims = inter_dims
        self.decoder_dims = decoder_dims
        self.decoder_mask_dims = decoder_mask_dims

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.src_path = src_path
        self.dst_path = dst_path
        self.out_path = out_path

        self.lr = lr
        self.encoder = Encoder(3, encoder_dims)
        encoder_out_shape = self.encoder.get_out_shape(resolution)
        self.inter = Inter(np.prod(encoder_out_shape), inter_dims, inter_dims, lowest_dense_res=resolution // 16)
        inter_out_ch = self.inter.get_out_ch()
        self.decoder_src = Decoder(inter_out_ch, decoder_dims, decoder_mask_dims)
        self.decoder_dst = Decoder(inter_out_ch, decoder_dims, decoder_mask_dims)

        self.loader = None
        self.opt = None

        self.out_freq = out_freq
        self.loss_storage = []

    def forward(self, x):
        latent_repr = self.inter(self.encoder(x))
        return self.decoder_src(latent_repr), self.decoder_dst(latent_repr)

    def forward_src(self, x):
        latent_repr = self.inter(self.encoder(x))
        return self.decoder_src(latent_repr)

    def forward_dst(self, x):
        latent_repr = self.inter(self.encoder(x))
        return self.decoder_dst(latent_repr)

    def training_step(self, batch, batch_idx):
        (src_batch, srcm_batch), (dst_batch, dstm_batch) = batch
        src_src, src_srcm = self.decoder_src(self.inter(self.encoder(src_batch)))
        dst_dst, dst_dstm = self.decoder_dst(self.inter(self.encoder(dst_batch)))

        target_srcm_blur = gaussian_blur(srcm_batch, max(1, self.resolution // 32))
        target_srcm_blur = torch.clamp(target_srcm_blur, 0, 0.5) * 2
        target_dstm_blur = gaussian_blur(dstm_batch, max(1, self.resolution // 32))
        target_dstm_blur = torch.clamp(target_dstm_blur, 0, 0.5) * 2

        target_src_masked_opt = src_batch * target_srcm_blur
        target_dst_masked_opt = dst_batch * target_dstm_blur

        pred_src_src_masked_opt = src_src * target_srcm_blur
        pred_dst_dst_masked_opt = dst_dst * target_dstm_blur

        src_loss = torch.mean(10 * dssim(target_src_masked_opt, pred_src_src_masked_opt, max_val=1.0,
                                         filter_size=int(self.resolution / 11.6)), dim=1)

        nbatch = src_batch.shape[0]
        flattened_masked = (target_src_masked_opt - pred_src_src_masked_opt).view(nbatch, -1)
        src_loss += torch.mean(10 * torch.square(flattened_masked), dim=1)

        flattened_masks = (srcm_batch - src_srcm).view(nbatch, -1)
        src_loss += torch.mean(10 * torch.square(flattened_masks), dim=1)
        src_loss = torch.mean(src_loss)

        dst_loss = torch.mean(10 * dssim(target_dst_masked_opt, pred_dst_dst_masked_opt, max_val=1.0,
                                         filter_size=int(self.resolution / 11.6)), dim=1)

        nbatch = dst_batch.shape[0]
        flattened_masked = (target_dst_masked_opt - pred_dst_dst_masked_opt).view(nbatch, -1)
        dst_loss += torch.mean(10 * torch.square(flattened_masked), dim=1)

        flattened_masks = (dstm_batch - dst_dstm).view(nbatch, -1)
        dst_loss += torch.mean(10 * torch.square(flattened_masks), dim=1)
        dst_loss = torch.mean(dst_loss)

        loss = src_loss + dst_loss
        return {'loss': loss, 'src_loss': src_loss.detach(), 'dst_loss': dst_loss.detach()}

    def training_epoch_end(self, outputs):
        curr_loss = [d['loss'].item() for d in outputs]
        self.loss_storage.extend(curr_loss)
        self.loader.dataset.resample()

        if self.current_epoch % self.out_freq == 0 and self.current_epoch != 0:
            if self.trainer.precision == 16:
                raise NotImplementedError("Incorrect evaluation on 16-bit")

            dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

            (src_batch, srcm_batch), (dst_batch, dstm_batch) = next(iter(self.loader))
            src_batch, srcm_batch = src_batch.type(dtype), srcm_batch.type(dtype)
            dst_batch, dstm_batch = dst_batch.type(dtype), dstm_batch.type(dtype)

            with torch.no_grad():
                src_src, src_srcm = self.decoder_src(self.inter(self.encoder(src_batch)))
                dst_dst, dst_dstm = self.decoder_dst(self.inter(self.encoder(dst_batch)))
                src_dst, src_dstm = self.decoder_dst(self.inter(self.encoder(src_batch)))

            result_unmasked = torch.cat([src_batch, src_src, dst_batch, dst_dst, src_dst])
            result_unmasked = make_grid(result_unmasked, nrow=src_batch.shape[0]).cpu().detach().numpy()
            result_masked = torch.cat(
                [src_batch, src_src * src_srcm, dst_batch, dst_dst * dst_dstm, src_dst * src_dstm])
            result_masked = make_grid(result_masked, nrow=src_batch.shape[0]).cpu().detach().numpy()

            output_path = self.out_path / str(self.current_epoch)
            output_path.mkdir()
            plt.imsave(str(output_path / 'unmasked.jpg'), np.transpose(result_unmasked, (1, 2, 0)))
            plt.imsave(str(output_path / 'masked.jpg'), np.transpose(result_masked, (1, 2, 0)))

            plt.plot(self.loss_storage)
            plt.savefig(str(output_path / 'src_loss.jpg'))
            plt.clf()

    def train_dataloader(self):
        dataset = PairedDataset(self.src_path, self.dst_path, self.resolution)
        self.loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                 num_workers=self.num_workers, drop_last=True)
        return self.loader

    def configure_optimizers(self):
        self.opt = Adam(self.parameters(), lr=self.lr)
        return self.opt

    def load_old_checkpoint(self, model_path: Path):
        checkpoint = torch.load(str(model_path))

        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.inter.load_state_dict(checkpoint['inter_state_dict'])
        self.decoder_src.load_state_dict(checkpoint['decoder_src_state_dict'])
        self.decoder_dst.load_state_dict(checkpoint['decoder_dst_state_dict'])
