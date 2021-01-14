import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from torch.optim import Adam

from deepfake.misc.models.ConvolutionalVAE import DeepfakeVAE
from face_dataset.face_autoencoder import loss_vae, construct_dataset
from face_dataset.utils import FacesDataset, plot_gallery

import time
import matplotlib.pyplot as plt

IMAGE_SIZE = 128
LOG_SQRT_2PI = np.log(np.sqrt(2 * np.pi))

USE_CUDA = torch.cuda.is_available()
DTYPE = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

SEED = 0xDEADF00D
TEST_SIZE = .2


def train_model(model, train_data_1, val_data_1,
                train_data_2, val_data_2, batch_size, num_epochs, lr):
    # Train loaders
    train_loader_1 = DataLoader(train_data_1, batch_size=batch_size, shuffle=True,
                                drop_last=False, num_workers=5)
    train_loader_2 = DataLoader(train_data_2, batch_size=batch_size, shuffle=True,
                                drop_last=False, num_workers=5)

    # Val loaders
    val_loader_1 = DataLoader(val_data_1, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=5)
    val_loader_2 = DataLoader(val_data_2, batch_size=batch_size, shuffle=True,
                              drop_last=False, num_workers=5)

    model = model.cuda() if USE_CUDA else model
    opt = Adam(model.parameters(), lr=lr)

    start_time = time.time()
    train_loss_1, train_loss_2 = [], []
    val_loss_1, val_loss_2 = [], []
    for i in range(num_epochs):
        tmp_train_loss_1, tmp_train_loss_2 = [], []
        tmp_val_loss_1, tmp_val_loss_2 = [], []

        model.train()
        for batch_1, batch_2 in zip(train_loader_1, train_loader_2):
            opt.zero_grad()
            batch_1 = batch_1.type(DTYPE)
            batch_2 = batch_2.type(DTYPE)

            result = model(batch_1, 1)
            loss = loss_vae(batch_1.view(-1, 3 * IMAGE_SIZE ** 2), *result)
            loss.backward()
            tmp_train_loss_1.append(-loss.item())
            result = model(batch_2, 2)
            loss = loss_vae(batch_2.view(-1, 3 * IMAGE_SIZE ** 2), *result)
            loss.backward()
            tmp_train_loss_2.append(-loss.item())

            opt.step()

        model.eval()
        for batch_1, batch_2 in zip(val_loader_1, val_loader_2):
            with torch.no_grad():
                batch_1 = batch_1.type(DTYPE)
                batch_2 = batch_2.type(DTYPE)

                result = model(batch_1, 1)
                loss = loss_vae(batch_1.view(-1, 3 * IMAGE_SIZE ** 2), *result)
                tmp_val_loss_1.append(-loss.item())
                result = model(batch_2, 2)
                loss = loss_vae(batch_2.view(-1, 3 * IMAGE_SIZE ** 2), *result)
                tmp_val_loss_2.append(-loss.item())

        train_loss_1.append(np.mean(tmp_train_loss_1))
        train_loss_2.append(np.mean(tmp_train_loss_2))
        val_loss_1.append(np.mean(tmp_val_loss_1))
        val_loss_2.append(np.mean(tmp_val_loss_2))
        print(f'[INFO] Iter: {i}, Train 1 loss: {train_loss_1[-1]}, Val 1 loss: {val_loss_1[-1]}')
        print(f'[INFO] Iter: {i}, Train 2 loss: {train_loss_2[-1]}, Val 2 loss: {val_loss_2[-1]}')

    print(f'Training took {time.time() - start_time} seconds.')


def apply_deepfake(model, image_1, image_2):
    with torch.no_grad():
        img_1_embed, _ = model.encode(image_1.type(DTYPE))
        img_2_embed, _ = model.encode(image_2.type(DTYPE))
        fake_img_1, _ = model.decode(img_1_embed, 2)
        fake_img_2, _ = model.decode(img_2_embed, 1)

    plot_gallery([image_1.data.cpu().numpy(), fake_img_1.squeeze().data.cpu().numpy()], n_col=2, n_row=1)
    plot_gallery([image_2.data.cpu().numpy(), fake_img_2.squeeze().data.cpu().numpy()], n_col=2, n_row=1)


if __name__ == '__main__':
    torch.manual_seed(SEED)

    data_path_1 = 'data/faces'
    data_path_2 = 'data/den_faces'
    train_data_1, val_data_1 = construct_dataset(data_path_1, 1)
    train_data_2, val_data_2 = construct_dataset(data_path_2, .5)

    train_params = {
        'num_epochs': 20,
        'batch_size': 8,
        'lr': 0.0001,
    }
    hparams = {
        'embedding_size': 512,
    }

    model = DeepfakeVAE(IMAGE_SIZE, **hparams)
    train_model(model, train_data_1, val_data_1,
                train_data_2, val_data_2, **train_params)

    apply_deepfake(model, val_data_1[0], val_data_2[2])
