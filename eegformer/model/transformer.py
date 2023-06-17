import math

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch import nn
from xformers.factory import xFormer, xFormerConfig

from eegformer.utils import MLP3DPositionalEmbedding


class Transformer(pl.LightningModule):
    def __init__(
        self,
        learning_rate=1e-3,
        weight_decay=0.01,
        num_classes=10,
        dim=320,
        n_layer=3,
        n_head=5,
        pdropout=0.0,
        hidden_layer_multiplier=4,
        warmup_steps=100,
        lr_decay_steps=5000,
        z_transform=True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 3D position embedding
        self.pos_embed = MLP3DPositionalEmbedding(dim, add_class_token=True)

        # configure encoder model
        xformer_config = [
            {
                "block_type": "encoder",
                "num_layers": n_layer,
                "dim_model": dim,
                "residual_norm_style": "pre",
                "multi_head_config": {
                    "num_heads": n_head,
                    "residual_dropout": pdropout,
                    "attention": {
                        "name": "scaled_dot_product",
                        "dropout": pdropout,
                        "causal": False,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dropout": pdropout,
                    "activation": "gelu",
                    "hidden_layer_multiplier": hidden_layer_multiplier,
                },
            }
        ]

        config = xFormerConfig(xformer_config)
        self.model = xFormer.from_config(config)

        # classifier head
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    @staticmethod
    def linear_warmup_cosine_decay(warmup_steps, total_steps):
        """
        Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps
        """

        def fn(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return fn

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                self.linear_warmup_cosine_decay(self.hparams.warmup_steps, self.hparams.lr_decay_steps),
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]

    def forward(self, x, ch_pos):
        if self.hparams.z_transform:
            x = (x - x.mean(dim=-1, keepdims=True)) / (x.std(dim=-1, keepdims=True) + 1e-8)

        x = self.pos_embed(x, ch_pos)
        x = self.model(x)

        # extract the class token
        x = x[:, 0]

        x = self.ln(x)
        x = self.head(x)
        return x

    def training_step(self, batch, _):
        signal, ch_pos, y = batch
        y_hat = self(signal, ch_pos)

        loss = F.cross_entropy(y_hat, y)

        self.logger.log_metrics(
            {
                "train_loss": loss.mean(),
                "learning_rate": self.lr_schedulers().get_last_lr()[0],
            },
            step=self.global_step,
        )
        return loss

    def evaluate(self, batch, stage=None):
        signal, ch_pos, y = batch
        y_hat = self(signal, ch_pos)

        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=-1) == y).float().mean()

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, _):
        self.evaluate(batch, "val")

    def test_step(self, batch, _):
        self.evaluate(batch, "test")
