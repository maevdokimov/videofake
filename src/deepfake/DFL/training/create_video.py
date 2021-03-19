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
from shutil import copyfile
import cv2

from src.deepfake.DFL.models.SAE import SAEModel
from src.deepfake.DFL.training.merge_frame import merge_frame

ModelType = Union[Path, pl.LightningModule]


def create_vid(
        orig_images: Path,
        meta_path: Path,
        output_img_path: Path,
        output_vid_path: Path,
        model_path: Path,
        fps: int,
):
    d = merge_frames(orig_images, meta_path, output_img_path, model_path)
    image_paths = [d[i] for i in sorted(list(d.keys()))]
    shape = cv2.imread(str(image_paths[0])).shape

    video = cv2.VideoWriter(str(output_vid_path), cv2.VideoWriter_fourcc(*'DIVX'), fps, (shape[:2:-1]))
    for path in image_paths:
        img = cv2.imread(str(path))
        print(img.shape)
        video.write(img)

    cv2.destroyAllWindows()
    video.release()


def merge_frames(
        orig_images: Path,
        meta_path: Path,
        output_path: Path,
        model_path: Path,
):
    curr_suffix = None

    d = {}
    for p in orig_images.iterdir():
        if curr_suffix is None:
            curr_suffix = p.suffix
        stem = int(p.stem)
        d[stem] = [p, False]
    for p in meta_path.iterdir():
        splited_stem = p.stem.split('_')
        if len(splited_stem) == 1: continue
        stem = splited_stem[0]
        meta = np.load(str(p), allow_pickle=True)
        if meta == np.array([0]): continue
        d[int(stem)] = [[d[int(stem)][0], p], True]

    model = SAEModel()
    model.load_old_checkpoint(model_path)

    result = {}

    for i, (item, flag) in tqdm(d.items()):
        if not flag:
            curr_out_path = output_path / f'{i}{item.suffix}'
            copyfile(str(item), str(curr_out_path))
        else:
            curr_out_path = output_path / f'{i}{item[0].suffix}'
            orig, meta = item
            merge_frame(orig, meta, model, curr_out_path)

        result[i] = curr_out_path
    return result


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
    with torch.no_grad():
        for i in tqdm(range(ordered_imgs.shape[0])):
            source_img = torch.from_numpy(ordered_imgs[i].transpose([2, 0, 1]))\
                .unsqueeze(0).type_as(next(model.parameters()))
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
    parser.add_argument('--src-images', '--src_images', required=True)
    parser.add_argument('--meta-path', '--meta_path', required=True)
    parser.add_argument('--output-img', '--output_img', required=True)
    parser.add_argument('--output-vid', '--output_vid', required=True)
    parser.add_argument('--model-path', '--model_path', required=True)
    parser.add_argument('--fps', required=True)
    args = parser.parse_args()

    create_vid(Path(args.src_images), Path(args.meta_path), Path(args.output_img),
               Path(args.output_vid), Path(args.model_path), int(args.fps))
