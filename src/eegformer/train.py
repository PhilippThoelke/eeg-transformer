from lightning.pytorch.cli import LightningCLI

from eegformer.data import DataModule
from eegformer.module import LightningModule

if __name__ == "__main__":
    cli = LightningCLI(
        LightningModule,
        DataModule,
        trainer_defaults=dict(
            max_epochs=-1,
            precision="16-mixed",
            gradient_clip_val=1.0,
        ),
    )
