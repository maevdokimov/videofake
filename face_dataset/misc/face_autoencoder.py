import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
from face_dataset.misc.face_models import ConvolutionalVAE
from face_dataset.misc.utils import FacesDataset, plot_results

import numpy as np
import matplotlib.pyplot as plt
from time import time

IMAGE_SIZE = 128
LOG_SQRT_2PI = np.log(np.sqrt(2 * np.pi))

USE_CUDA = torch.cuda.is_available()
DTYPE = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

SEED = 0xDEADF00D
TEST_SIZE = .2


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


def construct_dataset(path, sample_rate):
    dataset = FacesDataset(path, sample_rate, IMAGE_SIZE, SEED)

    train_size, test_size = len(dataset) - int(len(dataset) * TEST_SIZE), int(len(dataset) * TEST_SIZE)
    train_data, val_data = random_split(dataset, [train_size, test_size],
                                        generator=torch.Generator().manual_seed(SEED))
    return train_data, val_data


if __name__ == '__main__':
    torch.manual_seed(SEED)

    data_path = 'data/faces'
    den_data_path = 'data/den_faces'
    my_train_data, my_val_data = construct_dataset(data_path, 1)
    den_train_data, den_val_data = construct_dataset(den_data_path, .5)

    train_params = {
        'num_epochs': 20,
        'batch_size': 8,
        'lr': 0.00007,
    }
    hparams = {
        'embedding_size': 512,
        'num_filters': 32,
    }

    my_model = ConvolutionalVAE(IMAGE_SIZE, **hparams)
    den_model = ConvolutionalVAE(IMAGE_SIZE, **hparams)

    print(f'[INFO] Training model on my face dataset')
    train_model(my_model, my_train_data, my_val_data, **train_params)
    print(f'[INFO] Training model on den face dataset')
    train_model(den_model, den_train_data, den_val_data, **train_params)

    plot_results(my_model, 3, my_val_data, IMAGE_SIZE, DTYPE)
    plot_results(den_model, 3, den_val_data, IMAGE_SIZE, DTYPE)

