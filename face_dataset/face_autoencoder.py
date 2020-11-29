import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from face_dataset.face_models import VAE, ConvolutionalVAE

from typing import List
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
from time import time

IMAGE_SIZE = 128
LOG_SQRT_2PI = np.log(np.sqrt(2 * np.pi))

USE_CUDA = torch.cuda.is_available()
DTYPE = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

SEED = 0xDEADF00D
TEST_SIZE = .2


class MyFaceDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.images = self.load_images(data_path)

    @staticmethod
    def load_images(data_path):
        file_names = [os.path.join(data_path, f) for f in os.listdir(data_path)
                      if os.path.isfile(os.path.join(data_path, f))]
        images = [np.asarray(Image.open(file).resize([IMAGE_SIZE, IMAGE_SIZE])).transpose([2, 0, 1]) / 255.
                  for file in file_names]
        random.Random(SEED).shuffle(images)
        return torch.tensor(images)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]


class FacesDataset(Dataset):
    def __init__(self, paths: List[str], thinning_coefs: List[float]):
        super().__init__()
        self.images = self.load_images(paths, thinning_coefs)

    @staticmethod
    def load_images(paths, thinning_coefs):
        images = []
        for path, coef in zip(paths, thinning_coefs):
            file_names = [os.path.join(path, f) for f in os.listdir(path)
                          if os.path.isfile(os.path.join(path, f))]
            file_names = file_names[::int(1 / coef)]
            images.extend([np.asarray(Image.open(file).resize([IMAGE_SIZE, IMAGE_SIZE])).transpose([2, 0, 1]) / 255.
                           for file in file_names])
        random.Random(SEED).shuffle(images)
        return torch.tensor(images)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]


#####################################################################################
############################ Computing loss #########################################
#####################################################################################
def KL_divergence(mu, logsigma):
    divergence = 2 * logsigma - torch.square(mu) - torch.exp(2 * logsigma) + 1
    return -torch.sum(divergence, dim=1) / 2


def log_likelihood(x, mu, logsigma):
    mean_delta = torch.square(mu - x)
    blob = torch.div(mean_delta, 2 * torch.exp(2 * logsigma))
    return torch.sum(-blob - logsigma - LOG_SQRT_2PI, dim=1)


def loss_vae(x, mu_gen, logsigma_gen, mu_z, logsigma_z):
    assert mu_gen.shape == logsigma_gen.shape
    assert mu_z.shape == logsigma_z.shape

    if len(mu_gen.shape) > 2:
        mu_gen = mu_gen.view(mu_gen.shape[0], -1)
        logsigma_gen = logsigma_gen.view(logsigma_gen.shape[0], -1)
    kl = KL_divergence(mu_z, logsigma_z)
    likelihood = log_likelihood(x, mu_gen, logsigma_gen)
    loss = torch.mean(kl - likelihood)
    return loss
#####################################################################################


def train_model(model, train_data, val_data, batch_size, num_epochs, lr):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=5)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                            drop_last=False, num_workers=5)
    model = model.cuda() if USE_CUDA else model
    opt = Adam(model.parameters(), lr=lr)

    start_time = time()
    train_loss, val_loss = [], []
    for i in range(num_epochs):
        tmp_train_loss, tmp_test_loss = [], []

        model.train()
        for batch in train_loader:
            opt.zero_grad()
            cuda_batch = batch.type(DTYPE)
            result = model(cuda_batch)
            loss = loss_vae(cuda_batch.view(-1, 3 * IMAGE_SIZE ** 2), *result)
            loss.backward()
            opt.step()
            tmp_train_loss.append(-loss.item())

        model.eval()
        for batch in val_loader:
            with torch.no_grad():
                cuda_batch = batch.type(DTYPE)
                result = model(cuda_batch)
                loss = loss_vae(cuda_batch.view(-1, 3 * IMAGE_SIZE ** 2), *result)
                tmp_test_loss.append(-loss.item())

        train_loss.append(np.mean(tmp_train_loss))
        val_loss.append(np.mean(tmp_test_loss))
        print(f'[INFO] Iter: {i}, Train loss: {train_loss[-1]}, Val loss: {val_loss[-1]}')

    print(f'Training took {time() - start_time} seconds.')
    # Plot train loss
    fig = plt.figure()
    plt.plot(train_loss)
    fig.suptitle('Train loss')
    plt.xlabel('epoch')
    plt.ylabel('lower bound')
    plt.show()

    # Plot val loss
    fig = plt.figure()
    plt.plot(val_loss)
    fig.suptitle('Val loss')
    plt.xlabel('epoch')
    plt.ylabel('lower bound')
    plt.show()


def plot_gallery(images, n_row, n_col):
    plt.figure(figsize=(1.5 * n_col, 1.7 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].transpose([1, 2, 0]), vmin=-1, vmax=1, interpolation='nearest')
        plt.xticks(())
        plt.yticks(())
    plt.show()


def plot_results(model, num_pairs, dataset):
    model.eval()
    for j in range(num_pairs):
        idx = random.randrange(0, len(dataset))
        with torch.no_grad():
            input = dataset[idx].type(DTYPE)
            reconstruction_mu, _, _, _ = model(input)
        plot_gallery([dataset[idx].numpy(), reconstruction_mu.view(3, IMAGE_SIZE, IMAGE_SIZE).data.cpu().numpy()],
                     n_row=1, n_col=2)


def morph_images(model, dataset, left_img_idx, right_img_idx):
    model.eval()
    with torch.no_grad():
        left_input, right_input = dataset[left_img_idx].type(DTYPE), dataset[right_img_idx].type(DTYPE)
        _, _, left_embed, _ = model(left_input)
        _, _, right_embed, _ = model(right_input)
        morphed_img, _ = model.decode((left_embed + right_embed) / 2)
    plot_gallery([dataset[left_img_idx].numpy(),
                  morphed_img.view(3, IMAGE_SIZE, IMAGE_SIZE).data.cpu().numpy(),
                  dataset[right_img_idx].numpy()], n_row=1, n_col=3)


if __name__ == '__main__':
    torch.manual_seed(SEED)

    data_path = 'data/faces'
    den_data_path = 'data/den_faces_2'
    dataset = FacesDataset([data_path, den_data_path], [1, 0.5])
    train_size, test_size = len(dataset) - int(len(dataset) * TEST_SIZE), int(len(dataset) * TEST_SIZE)
    train_data, val_data = random_split(dataset, [train_size, test_size],
                                        generator=torch.Generator().manual_seed(SEED))

    train_params = {
        'num_epochs': 20,
        'batch_size': 8,
        'lr': 0.00007,
    }
    hparams = {
        'embedding_size': 512,
        'num_filters': 32,
    }

    model = ConvolutionalVAE(IMAGE_SIZE, **hparams)
    train_model(model, train_data, val_data, **train_params)

    plot_results(model, 5, val_data)
    morph_images(model, val_data, 10, 20)
