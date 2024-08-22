from typing import Union

import lightning.pytorch as pl
import torch
from torch import nn

from eegformer.model import Decoder, Encoder
from eegformer.utils import mask_channels, plot_gradients


class LightningModule(pl.LightningModule):
    def __init__(
        self,
        epoch_size: int = 16,
        lr: float = 1e-3,
        lr_warmup: int = 4000,
        kl_warmup: int = 12000,
        mask_rate: float = 0.5,
        noise_scale: float = 0.1,
        weight_decay: float = 0,
        debug: Union[bool, int] = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore="debug")
        self.debug = debug

        self.encoder = Encoder(epoch_size)
        self.decoder = Decoder.from_encoder(self.encoder)
        self.loss = nn.MSELoss()

    def forward(self, x, pos, reparametrize=True):
        return self.encoder(x, pos, reparametrize)

    def step(self, batch, batch_idx, stage):
        x, pos = batch
        x = self.encoder.to_epochs(x)

        if self.training:
            # add noise to the channel positions
            pos = pos + torch.randn_like(pos) * pos.std() * self.hparams.noise_scale

        # hide some spatial channels
        x_masked, pos_masked = mask_channels(x, pos, rate=self.hparams.mask_rate, dim=1)

        # encode the data up to the last epoch
        mu, logvar = self(x_masked[:, :, :-1], pos_masked, reparametrize=False)
        z = self.encoder.reparametrize(mu, logvar)

        # reconstruct raw data for all spatial channels
        x_recon = self.decoder(z, pos)

        # compute autoregressive loss
        loss = self.loss(x_recon, x[:, :, 1:])

        # KL divergence loss
        kl_div = (1 + logvar - mu**2 - logvar.exp()).mean() * -0.5
        kl_weight = min(1.0, self.global_step / self.hparams.kl_warmup)

        # log loss
        self.log(f"loss/{stage}", loss, prog_bar=True)
        self.log(f"kl_div/{stage}", kl_div, prog_bar=False)
        return loss + kl_div * kl_weight

    def on_after_backward(self) -> None:
        if (
            self.debug
            and self.global_step % self.trainer.num_training_batches == 0
            and self.current_epoch % int(self.debug) == 0
        ):
            plot_gradients(self)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.75)
        return {"optimizer": opt, "lr_scheduler": sch, "monitor": "loss/val"}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # perform lr warmup
        if self.trainer.global_step < self.hparams.lr_warmup:
            lr_scale = min(1.0, (self.trainer.global_step + 1) / self.hparams.lr_warmup)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.lr

        return super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
