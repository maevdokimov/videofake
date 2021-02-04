import imageio
from pathlib import Path
from PIL import Image
from skimage.transform import resize
import numpy as np
import argparse
import torch
from tqdm import tqdm
from typing import Union
import pytorch_lightning as pl

from deepfake.models.SAE import SAEModel


ModelType = Union[Path, pl.LightningModule]


def create_gif(
        src_path: Path,
        output_path: Path,
        model: ModelType,
        apply_mask: bool = False,
        old_checkpoint: bool = False,
) -> None:
    if not src_path.exists() or not src_path.is_dir():
        raise ValueError(f"No such folder {src_path}")
    folder_items = list(src_path.iterdir())
    if len(folder_items) == 0:
        raise ValueError(f'Input folder is empty')

    if output_path.exists():
        raise ValueError(f"{output_path} already exist")

    if isinstance(model, Path):
        if not model.exists():
            raise ValueError(f'No such file {model}')

        if old_checkpoint:
            mp = model
            model = SAEModel().cuda()
            model.load_old_checkpoint(mp)
        else:
            model = SAEModel.load_from_checkpoint(str(model)).cuda()

    resolution = model.resolution

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
        source_img = torch.from_numpy(ordered_imgs[i].transpose([2, 0, 1])).unsqueeze(0).type(torch.cuda.FloatTensor)
        out_img, out_img_mask = model.forward_src(source_img)
        if apply_mask:
            out_img = out_img * out_img_mask
        out_img = out_img.cpu().squeeze().detach().numpy()
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

    create_gif(Path(args.src_path), Path(args.output_path), Path(args.model_path))
