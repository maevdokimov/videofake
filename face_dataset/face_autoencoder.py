import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

IMAGE_SIZE = 32
LOG_SQRT_2PI = np.log(np.sqrt(2 * np.pi))

USE_CUDA = torch.cuda.is_available()
DTYPE = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor


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
        return torch.tensor(images)

    def __len__(self):
        return self.images.shape[0]

    def __getitem__(self, idx):
        return self.images[idx]


class VAE(nn.Module):
    def __init__(self, image_size, embedding_size):
        super().__init__()
        self.input_size = image_size * image_size * 3
        self.intermediate_size = int(image_size * image_size * 3 / 2)

        self.fc1 = nn.Linear(self.input_size, self.intermediate_size)
        self.fc2_1 = nn.Linear(self.intermediate_size, embedding_size)
        self.fc2_2 = nn.Linear(self.intermediate_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, self.intermediate_size)
        self.fc4_1 = nn.Linear(self.intermediate_size, self.input_size)
        self.fc4_2 = nn.Linear(self.intermediate_size, self.input_size)

    def gaussian_sampler(self, mu, logsigma):
        if self.training:
            std = logsigma.exp()
            eps = torch.randn(*std.shape).type(DTYPE)
            return eps.mul(std).add(mu)
        else:
            return mu

    def encode(self, x):
        h1 = self.fc1(x)
        return self.fc2_1(h1), self.fc2_2(h1)

    def decode(self, x):
        h3 = self.fc3(x)
        return self.fc4_1(h3), self.fc4_2(h3)

    def forward(self, x):
        flattened_view_x = x.view(-1, self.input_size)

        latent_mu, latent_logsigma = self.encode(flattened_view_x)
        latent_sample = self.gaussian_sampler(latent_mu, latent_logsigma)
        reconstruction_mu, reconstruction_logsigma = self.decode(latent_sample)

        return reconstruction_mu, reconstruction_logsigma, latent_mu, latent_logsigma


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
    kl = KL_divergence(mu_z, logsigma_z)
    likelihood = log_likelihood(x, mu_gen, logsigma_gen)
    loss = torch.mean(kl - likelihood)
    return loss
#####################################################################################


def train_model(model, dataset, batch_size, num_epochs, lr):
    face_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             drop_last=False, num_workers=5)
    model = model.cuda() if USE_CUDA else model
    opt = Adam(model.parameters(), lr=lr)

    train_loss, test_loss = [], []

    for i in range(num_epochs):
        tmp_train_loss, tmp_test_loss = [], []

        model.train()
        for batch in face_loader:
            opt.zero_grad()
            cuda_batch = batch.type(DTYPE)
            result = model(cuda_batch)
            loss = loss_vae(cuda_batch.view(-1, 3 * IMAGE_SIZE ** 2), *result)
            loss.backward()
            opt.step()
            tmp_train_loss.append(loss.item())

        model.eval()
        for batch in face_loader:
            with torch.no_grad():
                cuda_batch = batch.type(DTYPE)
                result = model(cuda_batch)
                loss = loss_vae(cuda_batch.view(-1, 3 * IMAGE_SIZE ** 2), *result)
                tmp_test_loss.append(loss.item())

        train_loss.append(np.mean(tmp_train_loss))
        test_loss.append(np.mean(tmp_test_loss))
        print(f'[INFO] Iter: {i}, Train loss: {train_loss[-1]}, Val loss: {test_loss[-1]}')


def plot_gallery(images, n_row, n_col):
    plt.figure(figsize=(1.5 * n_col, 1.7 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].transpose([1, 2, 0]), cmap=plt.cm.gray, vmin=-1, vmax=1, interpolation='nearest')
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
    data_path = '/home/maxim/python/videofake/data/face_videos/faces'
    dataset = MyFaceDataset(data_path)

    train_params = {
        'num_epochs': 200,
        'batch_size': 16,
        'lr': 0.000003,
    }
    hparams = {
        'embedding_size': 512,
    }

    model = VAE(IMAGE_SIZE, **hparams)
    train_model(model, dataset, **train_params)

    plot_results(model, 5, dataset)
    morph_images(model, dataset, 10, 20)
