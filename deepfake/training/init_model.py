from pathlib import Path
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset
import random
from collections import defaultdict
import cv2


default_param = {
    'encoder_dims': 64,
    'inter_dims': 128,
    'decoder_dims': 64,
    'decoder_mask_dims': 22
}

default_options = {
    'SAE': default_param,
    'resolution': 64,
    'optimizer': Adam,
    'lr': 5e-5,
    'gpu': True
}

# -----------------------------------------------------------------
# -----------------------------------------------------------------
# -----------------------------------------------------------------

PossibleImageExt = ['.png', '.jpg']


class PairedDataset(Dataset):
    def __init__(self, src_path: Path, dst_path: Path, resolution):
        super().__init__()
        if not src_path.exists():
            raise ValueError(f'Src path {src_path} does not exist')
        if not dst_path.exists():
            raise ValueError(f'Dst path {dst_path} does not exist')

        self.src_path = src_path
        self.dst_path = dst_path
        self.resolution = resolution

        self.src_idx, self.src_len = self.index_folder(src_path)
        self.dst_idx, self.dst_len = self.index_folder(dst_path)
        self.idx_map = None
        self.resample()

    def index_folder(self, path: Path):
        index = defaultdict(lambda: [None, None])
        for p in path.iterdir():
            stem, ext = p.stem, p.suffix
            if ext in PossibleImageExt:
                index[int(stem)][0] = p
            elif ext == '.npy':
                index[int(stem)][1] = p
            else:
                raise NotImplementedError(f'Unknown extension {ext}')

        return index, len(list(filter(lambda x: index[x][0] is not None and index[x][1] is not None, index.keys())))

    def resample(self):
        if self.src_len == self.dst_len:
            self.idx_map = None
            return self.idx_map
        max_len = max(self.src_len, self.dst_len)
        idx_list = list(range(max_len))
        random.shuffle(idx_list)
        self.idx_map = {i: idx for i, idx in zip(range(max_len), idx_list)}
        return self.idx_map

    def preprocess_pair(self, img_path: Path, mask_path: Path):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
        img = cv2.resize(img, (self.resolution, self.resolution))
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        mask = cv2.resize(np.load(str(mask_path)), (self.resolution, self.resolution))
        mask = np.expand_dims(mask, axis=2)
        mask = torch.from_numpy(mask.transpose([2, 0, 1])).float()

        return img, mask

    def __len__(self):
        return min(self.src_len, self.dst_len)

    def __getitem__(self, idx):
        return self.preprocess_pair(*self.src_idx[idx]), self.preprocess_pair(*self.dst_idx[self.idx_map[idx]])


if __name__ == '__main__':
    pass
