from pathlib import Path
from itertools import chain
import argparse
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from deepfake.training.init_model import init_model, init_data, default_options
from deepfake.models.utils import gaussian_blur, dssim


def checkpoint_model(encoder, inter, decoder_src, decoder_dst, opt, epoch, output_path):
    output_path = output_path / f'{epoch}_ckpt'
    output_path.mkdir()
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'inter_state_dict': inter.state_dict(),
        'decoder_src_state_dict': decoder_src.state_dict(),
        'decoder_dst_state_dict': decoder_dst.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
    }, str(output_path / 'checkpoint.pt'))


def save_output(src_loss_total, dst_loss_total, encoder, inter, decoder_src, decoder_dst,
                src_loader, dst_loader, use_cuda, output_path, epoch):
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    src_batch, srcm_batch = next(iter(src_loader))
    dst_batch, dstm_batch = next(iter(dst_loader))
    src_batch, srcm_batch = src_batch.type(dtype), srcm_batch.type(dtype)
    dst_batch, dstm_batch = dst_batch.type(dtype), dstm_batch.type(dtype)

    src_src, src_srcm = decoder_src(inter(encoder(src_batch)))
    dst_dst, dst_dstm = decoder_dst(inter(encoder(dst_batch)))
    src_dst, src_dstm = decoder_dst(inter(encoder(src_batch)))

    result_unmasked = torch.cat([src_batch, src_src, dst_batch, dst_dst, src_dst])
    result_unmasked = make_grid(result_unmasked, nrow=src_batch.shape[0]).cpu().detach().numpy()
    result_masked = torch.cat([src_batch, src_src * src_srcm, dst_batch, dst_dst * dst_dstm, src_dst * src_dstm])
    result_masked = make_grid(result_masked, nrow=src_batch.shape[0]).cpu().detach().numpy()

    output_path = output_path / str(epoch)
    output_path.mkdir()
    plt.imsave(str(output_path / 'unmasked.jpg'), np.transpose(result_unmasked, (1, 2, 0)))
    plt.imsave(str(output_path / 'masked.jpg'), np.transpose(result_masked, (1, 2, 0)))

    plt.plot(list(chain(*src_loss_total)))
    plt.savefig(str(output_path / 'src_loss.jpg'))
    plt.clf()
    plt.plot(list(chain(*dst_loss_total)))
    plt.savefig(str(output_path / 'dst_loss.jpg'))
    plt.clf()


def training_loop(encoder, inter, decoder_src, decoder_dst, opt,
                  src_loader, dst_loader, use_cuda, resolution,
                  output_path, num_epochs=1000, out_freq=10, checkpoint_freq=50):
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    src_loss_total, dst_loss_total = [], []

    for i in range(num_epochs):
        if i % checkpoint_freq == 0 and i != 0:
            checkpoint_model(encoder, inter, decoder_src, decoder_dst, opt, i, output_path)

        if i % out_freq == 0 and i != 0:
            save_output(src_loss_total, dst_loss_total, encoder, inter, decoder_src, decoder_dst,
                        src_loader, dst_loader, use_cuda, output_path, i)

        src_loss_epoch, dst_loss_epoch = [], []
        for (src_batch, srcm_batch), (dst_batch, dstm_batch) in tqdm(zip(src_loader, dst_loader)):
            opt.zero_grad()

            src_batch, srcm_batch = src_batch.type(dtype), srcm_batch.type(dtype)
            dst_batch, dstm_batch = dst_batch.type(dtype), dstm_batch.type(dtype)

            src_src, src_srcm = decoder_src(inter(encoder(src_batch)))
            dst_dst, dst_dstm = decoder_dst(inter(encoder(dst_batch)))

            target_srcm_blur = gaussian_blur(srcm_batch, max(1, resolution // 32))
            target_srcm_blur = torch.clamp(target_srcm_blur, 0, 0.5) * 2
            target_dstm_blur = gaussian_blur(dstm_batch, max(1, resolution // 32))
            target_dstm_blur = torch.clamp(target_dstm_blur, 0, 0.5) * 2

            target_src_masked_opt = src_batch * target_srcm_blur
            target_dst_masked_opt = dst_batch * target_dstm_blur

            pred_src_src_masked_opt = src_src * target_srcm_blur
            pred_dst_dst_masked_opt = dst_dst * target_dstm_blur

            src_loss = torch.mean(10 * dssim(target_src_masked_opt, pred_src_src_masked_opt, max_val=1.0,
                                             filter_size=int(resolution/11.6)), dim=1)

            nbatch = src_batch.shape[0]
            flattened_masked = (target_src_masked_opt - pred_src_src_masked_opt).view(nbatch, -1)
            src_loss += torch.mean(10 * torch.square(flattened_masked), dim=1)

            flattened_masks = (srcm_batch - src_srcm).view(nbatch, -1)
            src_loss += torch.mean(10 * torch.square(flattened_masks), dim=1)
            src_loss = torch.mean(src_loss)

            dst_loss = torch.mean(10 * dssim(target_dst_masked_opt, pred_dst_dst_masked_opt, max_val=1.0,
                                             filter_size=int(resolution / 11.6)), dim=1)

            nbatch = dst_batch.shape[0]
            flattened_masked = (target_dst_masked_opt - pred_dst_dst_masked_opt).view(nbatch, -1)
            dst_loss += torch.mean(10 * torch.square(flattened_masked), dim=1)

            flattened_masks = (dstm_batch - dst_dstm).view(nbatch, -1)
            dst_loss += torch.mean(10 * torch.square(flattened_masks), dim=1)
            dst_loss = torch.mean(dst_loss)

            loss = src_loss + dst_loss
            loss.backward()
            opt.step()

            src_loss_epoch.append(src_loss.item())
            dst_loss_epoch.append(dst_loss.item())
        src_loss_total.append(src_loss_epoch)
        dst_loss_total.append(dst_loss_epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-src', '--data_src', required=True)
    parser.add_argument('--data-dst', '--data_dst', required=True)
    parser.add_argument('--batch-size', '--batch_size', required=True)
    parser.add_argument('--image-output', '--image_output', required=True)
    args = parser.parse_args()

    attr = init_model('SAE', default_options)
    src_data = init_data(Path(args.data_src), default_options['resolution'], int(args.batch_size))
    dst_data = init_data(Path(args.data_dst), default_options['resolution'], int(args.batch_size))

    out_path = Path(args.image_output)
    if not out_path.exists():
        raise ValueError(f'Path does not exists {out_path}')
    if len(list(out_path.iterdir())) != 0:
        raise ValueError(f'Out path is not empty {out_path}')
    training_loop(*attr, src_data, dst_data, default_options['gpu'], default_options['resolution'], out_path)
