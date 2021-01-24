from itertools import chain
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.transform import resize

from deepfake.models.SAEHD import Encoder, Inter, Decoder


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


class MaskedDataset(Dataset):
    def __init__(self, img_mask_pairs, resolution):
        super().__init__()
        self.resolution = resolution

        self.faces, self.masks = self.construct_tensor(img_mask_pairs)

    def construct_tensor(self, img_mask_pairs):
        faces = torch.zeros(len(img_mask_pairs), 3, self.resolution, self.resolution)
        masks = torch.zeros(len(img_mask_pairs), 1, self.resolution, self.resolution)
        for i, (face, mask) in enumerate(img_mask_pairs):
            faces[i] = torch.from_numpy(face.transpose(2, 0, 1)).type(torch.FloatTensor)
            masks[i] = torch.from_numpy(mask.transpose(2, 0, 1)).type(torch.FloatTensor)

        return faces, masks

    def __len__(self):
        return self.faces.shape[0]

    def __getitem__(self, idx):
        return self.faces[idx], self.masks[idx]


def init_model(model_name, options):
    if model_name == 'SAE':
        param_dict = options[model_name]
        resolution = options['resolution']
        use_cuda = options['gpu']

        encoder = Encoder(3, param_dict['encoder_dims'])
        encoder_out_shape = encoder.get_out_shape(resolution)
        inter = Inter(np.prod(encoder_out_shape), param_dict['inter_dims'],
                      param_dict['inter_dims'], lowest_dense_res=resolution//16)
        inter_out_ch = inter.get_out_ch()
        decoder_src = Decoder(inter_out_ch, param_dict['decoder_dims'], param_dict['decoder_mask_dims'])
        decoder_dst = Decoder(inter_out_ch, param_dict['decoder_dims'], param_dict['decoder_mask_dims'])
        if use_cuda:
            encoder = encoder.cuda()
            inter = inter.cuda()
            decoder_src = decoder_src.cuda()
            decoder_dst = decoder_dst.cuda()

        opt = options['optimizer'](list(chain(*[encoder.parameters(), inter.parameters(), decoder_src.parameters(),
                                                decoder_dst.parameters()])), lr=options['lr'])

        return encoder, inter, decoder_src, decoder_dst, opt

    else:
        raise ValueError(f"Unknown model name {model_name}")


def init_data(data_path: Path, image_size: int, batch_size: int, num_workers: int = 0):
    d = defaultdict(lambda: [None, None])

    for p in data_path.iterdir():
        stem, ext = p.stem, p.suffix
        if ext in ['.png', '.jpg']:
            d[stem][0] = resize(np.asarray(Image.open(p)), (image_size, image_size, 3))
        elif ext == '.npy':
            d[stem][1] = resize(np.load(str(p)), (64, 64, 1))

    img_mask_pairs = [[img, mask] for img, mask in d.values()]
    dataset = MaskedDataset(img_mask_pairs, resolution=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)


if __name__ == '__main__':
    pass
