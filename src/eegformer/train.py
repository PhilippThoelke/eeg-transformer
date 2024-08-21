from glob import glob

import pytorch_lightning as pl

from eegformer.data import get_dataloader, load_data, normalize
from eegformer.module import LightningModule


def train(ckpt_path=None):
    subjs_train = range(1, 96)
    subjs_val = range(96, 106)
    # subjs 106-109 are reserved for testing

    x_train, pos_train, sfreq, _ = load_data(subjs_train)
    x_val, pos_val, _, _ = load_data(subjs_val)

    # robust normalization of x
    x_train, x_val = normalize(x_train, x_val)

    if ckpt_path:
        paths = glob(ckpt_path)
        if len(paths) > 1:
            print(f"Multiple checkpoints found. Using {paths[0]}")
        ckpt_path = paths[0]
        module = LightningModule.load_from_checkpoint(ckpt_path)
    else:
        module = LightningModule(epoch_size=int(sfreq // 10))

    trainer = pl.Trainer(
        precision="16-mixed",
        gradient_clip_val=1,
        gradient_clip_algorithm="norm",
        max_epochs=-1,
    )

    trainer.fit(
        module,
        get_dataloader(x_train, pos_train),
        get_dataloader(x_val, pos_val, stage="val"),
    )


if __name__ == "__main__":
    train()
