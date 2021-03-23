from pathlib import Path
import numpy as np
import pytorch_lightning as pl
import torch
import cv2

from src.DFL.process_face import get_image_hull_mask
from src.DFL.models.SAE import SAEModel


def _scale_array(arr, clip=True):
    if clip:
        return np.clip(arr, 0, 255)

    mn = arr.min()
    mx = arr.max()
    scale_range = (max([mn, 0]), min([mx, 255]))

    if mn < scale_range[0] or mx > scale_range[1]:
        return (scale_range[1] - scale_range[0]) * (arr - mn) / (mx - mn) + scale_range[0]

    return arr


def lab_image_stats(image):
    # compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (lMean, lStd) = (l.mean(), l.std())
    (aMean, aStd) = (a.mean(), a.std())
    (bMean, bStd) = (b.mean(), b.std())

    # return the color statistics
    return lMean, lStd, aMean, aStd, bMean, bStd


def reinhard_color_transfer(target, source, clip=False, preserve_paper=False, source_mask=None, target_mask=None):
    # convert the images from the RGB to L*ab* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    target = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)

    # compute color statistics for the source and target images
    src_input = source if source_mask is None else source * source_mask
    tgt_input = target if target_mask is None else target * target_mask
    (lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = lab_image_stats(src_input)
    (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = lab_image_stats(tgt_input)

    # subtract the means from the target image
    (l, a, b) = cv2.split(target)
    l -= lMeanTar
    a -= aMeanTar
    b -= bMeanTar

    if preserve_paper:
        # scale by the standard deviations using paper proposed factor
        l = (lStdTar / lStdSrc) * l
        a = (aStdTar / aStdSrc) * a
        b = (bStdTar / bStdSrc) * b
    else:
        # scale by the standard deviations using reciprocal of paper proposed factor
        l = (lStdSrc / lStdTar) * l
        a = (aStdSrc / aStdTar) * a
        b = (bStdSrc / bStdTar) * b

    # add in the source mean
    l += lMeanSrc
    a += aMeanSrc
    b += bMeanSrc

    # clip/scale the pixel intensities to [0, 255] if they fall
    # outside this range
    l = _scale_array(l, clip=clip)
    a = _scale_array(a, clip=clip)
    b = _scale_array(b, clip=clip)

    # merge the channels together and convert back to the RGB color
    # space, being sure to utilize the 8-bit unsigned integer data
    # type
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # return the color transferred image
    return transfer


def merge_frame(
        orig_image: Path,
        meta_path: Path,
        model: pl.LightningModule,
        out_path: Path,
):
    if not orig_image.exists():
        raise ValueError(f"Image path {orig_image} does not exist")
    if not meta_path.exists():
        raise ValueError(f"File {meta_path} does not exist")

    meta = np.load(str(meta_path), allow_pickle=True)
    offset_x, offset_y = meta[0][0], meta[0][1]
    landmarks = meta[1].astype('int32')
    landmarks[:, 0] += offset_x
    landmarks[:, 1] += offset_y

    resolution = model.resolution

    img = cv2.imread(str(orig_image))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face_img = img[meta[0][1]:meta[0][1] + meta[0][3], meta[0][0]:meta[0][0] + meta[0][2], :]
    face_img = cv2.resize(face_img, (resolution, resolution))

    # Unused without rotation mat
    # ----------------------------
    img_face_mask = get_image_hull_mask(img.shape, landmarks)
    img_face_mask = np.clip(img_face_mask, 0, 1)
    # ----------------------------

    feed_img = torch.from_numpy(face_img.transpose([2, 0, 1])).unsqueeze(0).type_as(next(model.parameters())) / 255.
    with torch.no_grad():
        src_dst, src_dstm = model.forward_src(feed_img)
        src_dst, src_dstm = src_dst.squeeze(0), src_dstm.squeeze(0)
        src_dstm = torch.clip(src_dstm, 0, 1)
        src_dstm[src_dstm < 1./255.] = 0.

    src_dst, src_dstm = src_dst.cpu().numpy().transpose([1, 2, 0]), src_dstm.cpu().numpy().transpose([1, 2, 0])

    resized_dstm = cv2.resize(src_dstm, (meta[0][2], meta[0][3]), interpolation=cv2.INTER_CUBIC)
    full_dstm = np.zeros(img.shape[:2] + (1,)).astype(resized_dstm.dtype)
    full_dstm[meta[0][1]:meta[0][1] + meta[0][3], meta[0][0]:meta[0][0] + meta[0][2], :] = np.expand_dims(resized_dstm, axis=-1)

    maxregion = np.argwhere(full_dstm >= 0.1)
    if maxregion.size != 0:
        miny, minx = maxregion.min(axis=0)[:2]
        maxy, maxx = maxregion.max(axis=0)[:2]
        lenx = maxx - minx
        leny = maxy - miny
        if min(lenx, leny) >= 4:
            flat_mask = src_dstm.copy()
            flat_mask[flat_mask > 0] = 1.

            # src_dst = reinhard_color_transfer(np.clip(src_dst*flat_mask*255, 0, 255).astype(np.uint8),
            #                                   np.clip(face_img*flat_mask*255, 0, 255).astype(np.uint8))
            # src_dst = np.clip(src_dst.astype(np.float32) / 255.0, 0.0, 1.0)

            resized_dst = cv2.resize(src_dst, (meta[0][2], meta[0][3]), interpolation=cv2.INTER_CUBIC)
            full_dst = np.zeros(img.shape[:2] + (3,)).astype(resized_dst.dtype)
            full_dst[meta[0][1]:meta[0][1] + meta[0][3], meta[0][0]:meta[0][0] + meta[0][2], :] = resized_dst
            full_dst = np.clip(full_dst, 0., 1.)

            out_img = (img / 255.) * (1 - full_dstm) + full_dst * full_dstm
        else:
            raise ValueError(f'Mask for image {orig_image} is too thin')
    else:
        raise ValueError(f'No mask for {orig_image}')

    out_img = (out_img * 255).astype('uint8')
    out_img = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path), out_img)


if __name__ == '__main__':
    orig_image = Path('/home/maxim/python/videofake/data/faces/dst_1/266.jpg')
    meta_path = Path('/home/maxim/python/videofake/data/faces/dst_1_masked/266_meta.npy')

    model = SAEModel()
    model.load_old_checkpoint(Path('/home/maxim/python/videofake/data/new_output/550_ckpt/checkpoint.pt'))

    merge_frame(orig_image, meta_path, model, Path('/home/maxim/python/videofake/data/gif/111.jpg'))
