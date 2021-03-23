from face_alignment import FaceAlignment, LandmarksType
import cv2
from pathlib import Path
import numpy as np
import argparse
from tqdm import tqdm
import warnings

from src.DFL.landmarks_utils import get_transform_mat, transform_points


def process_frame(face_model: FaceAlignment, image_path: Path, out_path: Path, image_size: int):
    img = cv2.imread(str(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    detected_faces = face_model.face_detector.detect_from_image(img.copy())
    if len(detected_faces) > 1:
        warnings.warn(f"Multiple faces detected on image {image_path}. Make sure there is only one face in each image.")
        # Taking box with best score
        detected_faces = [sorted(detected_faces, key=lambda face: face[-1])[-1]]
    if len(detected_faces) == 0:
        image_name = image_path.absolute().stem
        out_path_data = out_path / f'{image_name}_meta.npy'
        np.save(str(out_path_data), np.array([0]), allow_pickle=True)
        return

    landmarks = face_model.get_landmarks_from_image(img, detected_faces)[0]
    image_to_face_mat = get_transform_mat(landmarks, image_size)

    face_image = cv2.warpAffine(img, image_to_face_mat, (image_size, image_size), cv2.INTER_LANCZOS4)
    face_image_landmarks = transform_points(landmarks, image_to_face_mat)

    image_name = image_path.absolute().stem
    ext = 'jpg' if 'jpg' in image_path.suffixes else 'png'
    out_path_img = out_path / f'{image_name}.{ext}'
    out_path_data = out_path / f'{image_name}_meta.npy'

    face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_path_img), face_image)
    np.save(str(out_path_data), np.array([face_image_landmarks, detected_faces[0][:-1],
                                          landmarks, image_to_face_mat]), allow_pickle=True)


def process_folder(input_path, output_path, image_size):
    input_path, output_path = Path(input_path), Path(output_path)

    if not input_path.exists():
        raise ValueError(f'No such folder {input_path}')
    if not output_path.exists():
        raise ValueError(f'No such folder {output_path}')

    fa = FaceAlignment(LandmarksType._2D, flip_input=False)
    for img_path in tqdm(input_path.iterdir()):
        process_frame(fa, img_path, output_path, image_size)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-path', required=True)
    parser.add_argument('--output-path', required=True)
    parser.add_argument('--image-size', required=True)
    args = parser.parse_args()

    process_folder(args.source_path, args.output_path, int(args.image_size))
