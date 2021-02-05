from pathlib import Path
import argparse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from deepfake.models.SAE import SAEModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-src', '--data_src', required=True)
    parser.add_argument('--data-dst', '--data_dst', required=True)
    parser.add_argument('--batch-size', '--batch_size', required=True)
    parser.add_argument('--image-output', '--image_output', required=True)
    args = parser.parse_args()

    out_path = Path(args.image_output)
    if not out_path.exists():
        raise ValueError(f'Path does not exists {out_path}')
    if len(list(out_path.iterdir())) != 0:
        raise ValueError(f'Out path is not empty {out_path}')

    (out_path / 'checkpoints').mkdir()
    checkpoint_callback = ModelCheckpoint(dirpath=str(out_path / 'checkpoints'), period=20)

    model = SAEModel(src_path=Path(args.data_src), dst_path=Path(args.data_dst),
                     out_path=out_path, batch_size=int(args.batch_size))
    trainer = pl.Trainer(gpus=-1, max_epochs=1000, callbacks=[checkpoint_callback], logger=False)
    trainer.fit(model)
