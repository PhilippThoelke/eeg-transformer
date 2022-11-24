from abc import ABC, abstractmethod
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from eegt.model import EEGEncoder


class LightningModule(pl.LightningModule, ABC):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # transformer encoder
        self.model = EEGEncoder(
            self.hparams.embedding_dim,
            self.hparams.num_layers,
            self.hparams.token_size,
            nheads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
        )

    @abstractmethod
    def step(self, batch, batch_idx, training_stage):
        pass

    def forward(self, x, ch_pos, mask=None):
        """
        Perform a forward pass of the EEG Transformer.

        Letters in the shape descriptions stand for
        N - batch size, S - EEG samples, C - channels, D - classes, L - latent dimension

        Parameters:
            x (torch.Tensor): raw EEG epochs with shape (N, S, C)
            ch_pos (torch.Tensor): channel positions in 3D space normalized to the unit cube (N, C, 3)
            mask (torch.Tensor, optional): boolean mask to hide padded channels from the attention mechanism (N, C)

        Returns:
            z (torch.Tensor): latent representation (N, L)
        """
        # reshape x from (N, S, C) to (C, N, S)
        x = x.permute(2, 0, 1)
        # reshape ch_pos from (N, C, 3) to (C, N, 3)
        ch_pos = ch_pos.permute(1, 0, 2)

        # apply encoder model
        return self.model(x, ch_pos, mask)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, training_stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, training_stage="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, training_stage="test")

    def configure_optimizers(self):
        # optimizer
        opt = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # learning rate scheduler
        opt.param_groups[0]["initial_lr"] = self.hparams.learning_rate
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=2, T_mult=2)
        return dict(optimizer=opt, lr_scheduler=scheduler, monitor="train_loss")

    def training_epoch_end(self, *args, **kwargs):
        # log the learning rate
        optimizers = self.optimizers()
        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for i, opt in enumerate(optimizers):
            name = "lr" if len(optimizers) == 1 else f"lr_{i}"
            self.log(name, opt.param_groups[0]["lr"])

    def optimizer_step(self, *args, **kwargs):
        # learning rate warmup
        if self.global_step < self.hparams.warmup_steps:
            scheduler = self.lr_schedulers()
            scheduler.base_lrs[0] = self.hparams.learning_rate * (
                (self.global_step + 1) / self.hparams.warmup_steps
            )

        # continue with the optimizer step normally
        return super().optimizer_step(*args, **kwargs)
