from face_alignment import FaceAlignment, LandmarksType
import cv2
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import warnings


def expand_eyebrows(lmrks, eyebrows_expand_mod=1.0):
    if len(lmrks) != 68:
        raise Exception('works only with 68 landmarks')
    lmrks = np.array( lmrks.copy(), dtype=np.int )

    # #nose
    ml_pnt = (lmrks[36] + lmrks[0]) // 2
    mr_pnt = (lmrks[16] + lmrks[45]) // 2

    # mid points between the mid points and eye
    ql_pnt = (lmrks[36] + ml_pnt) // 2
    qr_pnt = (lmrks[45] + mr_pnt) // 2

    # Top of the eye arrays
    bot_l = np.array((ql_pnt, lmrks[36], lmrks[37], lmrks[38], lmrks[39]))
    bot_r = np.array((lmrks[42], lmrks[43], lmrks[44], lmrks[45], qr_pnt))

    # Eyebrow arrays
    top_l = lmrks[17:22]
    top_r = lmrks[22:27]

    # Adjust eyebrow arrays
    lmrks[17:22] = top_l + eyebrows_expand_mod * 0.5 * (top_l - bot_l)
    lmrks[22:27] = top_r + eyebrows_expand_mod * 0.5 * (top_r - bot_r)
    return lmrks


def get_image_hull_mask (image_shape, image_landmarks, eyebrows_expand_mod=1.0 ):
    hull_mask = np.zeros(image_shape[0:2]+(1,),dtype=np.float32)

    lmrks = expand_eyebrows(image_landmarks, eyebrows_expand_mod)

    r_jaw = (lmrks[0:9], lmrks[17:18])
    l_jaw = (lmrks[8:17], lmrks[26:27])
    r_cheek = (lmrks[17:20], lmrks[8:9])
    l_cheek = (lmrks[24:27], lmrks[8:9])
    nose_ridge = (lmrks[19:25], lmrks[8:9],)
    r_eye = (lmrks[17:22], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    l_eye = (lmrks[22:27], lmrks[27:28], lmrks[31:36], lmrks[8:9])
    nose = (lmrks[27:31], lmrks[31:36])
    parts = [r_jaw, l_jaw, r_cheek, l_cheek, nose_ridge, r_eye, l_eye, nose]

    for item in parts:
        merged = np.concatenate(item)
        cv2.fillConvexPoly(hull_mask, cv2.convexHull(merged), (1,) )

    return hull_mask


def xyxy_to_xywh(bbox, numpy=True):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    if numpy:
        return np.array([int(elem) for elem in [bbox[0], bbox[1], w, h]])
    return [int(elem) for elem in [bbox[0], bbox[1], w, h]]


def process_frame(face_model: FaceAlignment, image_path: Path, out_path: Path, only_meta: bool = True):
    img = cv2.imread(str(image_path))
    detected_faces = face_model.face_detector.detect_from_image(img.copy())
    if len(detected_faces) > 1:
        warnings.warn(f"Multiple faces detected on image {image_path}. Make sure there is only one face in each image.")
        detected_faces = [sorted(detected_faces, key=lambda face: face[-1])[-1]]
    if len(detected_faces) == 0:
        image_name = image_path.absolute().stem
        out_path_data = out_path / f'{image_name}_meta.npy'
        np.save(str(out_path_data), np.array([0]), allow_pickle=True)
        return
    x, y, w, h = xyxy_to_xywh(detected_faces[0][:-1])
    img = img[y:y + h, x:x + w, :]

    detected_faces[0][:-1] = np.array([0, 0, w, h])
    landmarks = face_model.get_landmarks_from_image(img, detected_faces)[0]
    mask = np.clip(get_image_hull_mask(img.shape, landmarks), 0, 1)

    image_name = image_path.absolute().stem
    ext = 'jpg' if 'jpg' in image_path.suffixes else 'png'
    out_path_img = out_path / f'{image_name}.{ext}'
    out_path_mask = out_path / f'{image_name}.npy'
    out_path_data = out_path / f'{image_name}_meta.npy'
    if not only_meta:
        cv2.imwrite(str(out_path_img), img)
        np.save(str(out_path_mask), mask)
    np.save(str(out_path_data), np.array([np.array([x, y, w, h]), landmarks]), allow_pickle=True)


def process_folder(input_path, output_path):
    input_path, output_path = Path(input_path), Path(output_path)

    if not input_path.exists():
        raise ValueError(f'No such folder {input_path}')
    if not output_path.exists():
        raise ValueError(f'No such folder {output_path}')

    fa = FaceAlignment(LandmarksType._2D, flip_input=False)
    for img_path in tqdm(input_path.iterdir()):
        process_frame(fa, img_path, output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    process_folder(args.source, args.output)
