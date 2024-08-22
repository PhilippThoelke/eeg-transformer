from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.loggers import TensorBoardLogger

from eegformer.data import DataModule
from eegformer.module import LightningModule

if __name__ == "__main__":
    LightningCLI(
        LightningModule,
        DataModule,
        trainer_defaults=dict(
            max_epochs=-1,
            precision="16-mixed",
            gradient_clip_val=5.0,
            callbacks=[
                LearningRateMonitor(logging_interval="step"),
                ModelCheckpoint(monitor="loss/val", filename="{epoch}-{step}-{val_loss:.2f}", save_last=True),
            ],
            logger=TensorBoardLogger("", default_hp_metric=False),
        ),
    )
