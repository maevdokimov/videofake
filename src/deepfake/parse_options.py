from src.deepfake.models.ganimation_modules import get_norm_layer

from pathlib import Path
import yaml
from typing import Dict


class MappingCFG:
    generator = {
        'img_nc': 'img_nc',
        'aus_nc': 'aus_nc',
        'ngf': 'ngf',
        'norm_layer': 'norm_layer',
        'use_dropout': 'use_dropout',
        'n_blocks': 'n_blocks',
        'padding_type': 'padding_type',
    }
    discriminator = {
        'input_nc': 'img_nc',
        'aus_nc': 'aus_nc',
        'image_size': 'image_size',
        'ndf': 'ndf',
        'n_layers': 'n_layers',
        'norm_layer': 'norm_layer',
    }


class ModelConfig:
    def __init__(self, cfg_path: Path):
        self.cfg_path = cfg_path
        self.cfg = self._load_config(cfg_path)

        self.cfg['norm_layer'] = get_norm_layer(self.cfg['norm_layer'])

    @staticmethod
    def _load_config(cfg_path: Path):
        if not cfg_path.exists():
            raise ValueError(f'No such config {cfg_path}')

        with open(str(cfg_path), 'r') as stream:
            return yaml.safe_load(stream)

    def get_config(self):
        return self.cfg

    def get_cfg_by_names(self, names: Dict[str, str]):
        return {tar_name: self.cfg[src_name] for src_name, tar_name in names.items()}
