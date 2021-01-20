import torch
from torch.utils.data import Dataset

from numbers import Number
from typing import List, Union
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random


class FacesDataset(Dataset):
    def __init__(
            self,
            path: Union[List[str], str],
            subsample_rate: Union[List[Number], Number],
            image_size,
            random_seed
    ):
        super().__init__()
        self.images = self.load_images(path, subsample_rate, image_size, random_seed)

    @staticmethod
    def load_images(path, subsample_rate, image_size, random_seed):
        # Align path and subsample_rate
        if isinstance(subsample_rate, Number):
            if isinstance(path, list):
                subsample_rate = [subsample_rate] * len(path)
            else:
                subsample_rate = [subsample_rate]
        if isinstance(path, str):
            path = [path]

        images = []
        for path, coef in zip(path, subsample_rate):
            file_names = [os.path.join(path, f) for f in os.listdir(path)
                          if os.path.isfile(os.path.join(path, f))]
            random.Random(random_seed).shuffle(file_names)
            file_names = file_names[::int(1 / coef)]
            images.extend([np.asarray(Image.open(file).resize([image_size, image_size])).transpose([2, 0, 1]) / 255.
                           for file in file_names])
        random.Random(random_seed).shuffle(images)
        return torch.tensor(images)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]


def plot_gallery(images, n_row, n_col):
    plt.figure(figsize=(1.5 * n_col, 1.7 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].transpose([1, 2, 0]), vmin=-1, vmax=1, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.show()


def plot_results(model, num_pairs, dataset, image_size, dtype):
    model.eval()
    for j in range(num_pairs):
        idx = random.randrange(0, len(dataset))
        with torch.no_grad():
            input = dataset[idx].type(dtype)
            reconstruction_mu, _, _, _ = model(input)
        plot_gallery([dataset[idx].numpy(), reconstruction_mu.view(3, image_size, image_size).data.cpu().numpy()],
                     n_row=1, n_col=2)


def morph_images(model, dataset, left_img_idx, right_img_idx, image_size, dtype):
    model.eval()
    with torch.no_grad():
        left_input, right_input = dataset[left_img_idx].type(dtype), dataset[right_img_idx].type(dtype)
        _, _, left_embed, _ = model(left_input)
        _, _, right_embed, _ = model(right_input)
        morphed_img, _ = model.decode((left_embed + right_embed) / 2)
    plot_gallery([dataset[left_img_idx].numpy(),
                  morphed_img.view(3, image_size, image_size).data.cpu().numpy(),
                  dataset[right_img_idx].numpy()], n_row=1, n_col=3)
