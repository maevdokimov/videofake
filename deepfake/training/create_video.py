import imageio
from pathlib import Path
from PIL import Image
from skimage.transform import resize
import numpy as np
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Union, List

from deepfake.models.SAEHD import Encoder, Inter, Decoder
from deepfake.training.init_model import default_options


ModelType = Union[Path, List[nn.Module]]


def create_gif(src_path: Path, output_path: Path, model_path: ModelType, options, apply_mask: bool = False):
    if not src_path.exists() or not src_path.is_dir():
        raise ValueError(f"No such folder {src_path}")
    folder_items = list(src_path.iterdir())
    if len(folder_items) == 0:
        raise ValueError(f'Input folder is empty')

    param_dict = options['SAE']
    resolution = options['resolution']

    if isinstance(model_path, Path):
        if not model_path.exists():
            raise ValueError(f'No such file {model_path}')

        checkpoint = torch.load(str(model_path))

        encoder = Encoder(3, param_dict['encoder_dims']).cuda()
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        encoder.eval()
        encoder_out_shape = encoder.get_out_shape(resolution)
        inter = Inter(np.prod(encoder_out_shape), param_dict['inter_dims'],
                      param_dict['inter_dims'], lowest_dense_res=resolution // 16).cuda()
        inter.load_state_dict(checkpoint['inter_state_dict'])
        inter.eval()
        decoder_src = Decoder(inter.get_out_ch(), param_dict['decoder_dims'], param_dict['decoder_mask_dims']).cuda()
        decoder_src.load_state_dict(checkpoint['decoder_src_state_dict'])
        decoder_src.eval()
        decoder_dst = Decoder(inter.get_out_ch(), param_dict['decoder_dims'], param_dict['decoder_mask_dims']).cuda()
        decoder_dst.load_state_dict(checkpoint['decoder_dst_state_dict'])
        decoder_dst.eval()
    else:
        encoder, inter, decoder_src, decoder_dst = model_path

    consistent_keys, consistent_imgs = [], []
    for p in folder_items:
        stem, ext = p.stem, p.suffix
        if ext in ['.png', '.jpg']:
            consistent_keys.append(int(stem))
            consistent_imgs.append(resize(np.asarray(Image.open(p)), (resolution, resolution, 3)))

    consistent_keys, consistent_imgs = np.array(consistent_keys), np.array(consistent_imgs)
    ordered_imgs = consistent_imgs[np.argsort(consistent_keys)]

    images = []
    for i in tqdm(range(ordered_imgs.shape[0])):
        source_img = torch.from_numpy(ordered_imgs[i].transpose([2, 0, 1])).unsqueeze(0).type(torch.FloatTensor)
        out_img, out_img_mask = decoder_dst(inter(encoder(source_img)))
        if apply_mask:
            out_img = out_img * out_img_mask
        out_img = out_img.squeeze().detach().numpy()
        out_img = np.transpose(out_img, [1, 2, 0])
        out_img = np.concatenate([out_img, ordered_imgs[i]], axis=0)
        images.append(out_img)

    with imageio.get_writer(str(output_path), mode='I') as writer:
        for image in images:
            writer.append_data(image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-path', '--src_path', required=True)
    parser.add_argument('--output-path', '--output_path', required=True)
    parser.add_argument('--model-path', '--model_path', required=True)
    args = parser.parse_args()

    create_gif(Path(args.src_path), Path(args.output_path), Path(args.model_path), default_options)
