from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import cv2
import insightface
import numpy as np

import pickle
from tqdm import tqdm
import warnings
from pathlib import Path
from typing import List, Union, Dict


class FaceDataset(Dataset):
    def __init__(
            self,
            img_paths: List[Path],
            result_size: int,
            random_horizontal_flip: bool,
            emb_dict: Dict[str, np.ndarray]
    ):
        super().__init__()
        self.img_paths = img_paths
        self.result_size = result_size
        self.random_horizontal_flip = random_horizontal_flip
        self.emb_dict = emb_dict

        self.transform = transforms.Compose([
            transforms.Resize([result_size, result_size], InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    @staticmethod
    def _load_image(path: Path):
        img = cv2.imread(str(path))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self._load_image(self.img_paths[idx])
        if img.shape[0] < self.result_size or img.shape[1] < self.result_size:
            warnings.warn(f"Image {self.img_paths[idx]} has shape {img.shape}")

        emb = self.emb_dict[str(self.img_paths[idx])]

        return self.transform(img), emb


def test_pretrained(img_path: Path):
    img = cv2.imread(str(img_path))
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0)

    faces = model.get(img)
    for idx, face in enumerate(faces):
        print("Face [%d]:" % idx)
        print("\tage:%d" % face.age)
        gender = 'Male'
        if face.gender == 0:
            gender = 'Female'
        print("\tgender:%s" % gender)
        print("\tembedding shape:%s" % face.embedding.shape)
        print("\tbbox:%s" % (face.bbox.astype(np.int).flatten()))
        print("\tlandmark:%s" % (face.landmark.astype(np.int).flatten()))
        print("\tdetection score:%s" % face.det_score)
        print("")


def prepare_embeddings(paths: List[Path], pickle_path: Union[Path, None]):
    emb_dict = {}
    model = insightface.app.FaceAnalysis()
    model.prepare(ctx_id=0)

    paths = [img_path for p in paths for img_path in p.iterdir()]
    for img_path in tqdm(paths):
        img = cv2.imread(str(img_path))
        faces = model.get(img)
        if len(faces) == 0:
            warnings.warn(f"No faces detected on image {img_path}")
            continue
        if len(faces) > 1:
            warnings.warn(f"Multiple faces detected on image {img_path}")
        faces = sorted(faces, key=lambda x: x.det_score, reverse=True)
        emb_dict[str(img_path)] = faces[0].embedding

    if pickle_path is not None:
        with open(pickle_path, 'wb') as file:
            pickle.dump(emb_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
    return emb_dict


if __name__ == '__main__':
    paths = [Path('/home/maxim/python/deepfake-tools/data/data512x512'),
             Path('/home/maxim/python/deepfake-tools/data/resized')]
    paths_union = [img_path for p in paths for img_path in p.iterdir()]
    emb_dict = prepare_embeddings(paths, None)

    dataset = FaceDataset(paths_union, 256, True, emb_dict)
    print(dataset[0])
