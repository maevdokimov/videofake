from pathlib import Path
import yaml


class ModelConfig:
    def __init__(self, cfg_path: Path):
        self.cfg_path = cfg_path
        self.cfg = self._load_config(cfg_path)

    @staticmethod
    def _load_config(cfg_path: Path):
        if not cfg_path.exists():
            raise ValueError(f'No such config {cfg_path}')

        with open(cfg_path, 'r') as stream:
            return yaml.safe_load(stream)

    def get_config(self):
        return self.cfg
