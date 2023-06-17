import math
from enum import Enum

import lightning.pytorch as pl
import torch
from torch import nn
from torchmetrics import Accuracy
from xformers.factory import xFormer, xFormerConfig


class Transformer(pl.LightningModule):
    def __init__(
        self,
        learning_rate=5e-4,
        weight_decay=0.03,
        num_classes=10,
        dim=320,
        n_layer=3,
        n_head=5,
        pdropout=0.0,
        hidden_layer_multiplier=4,
        warmup_steps=100,
        lr_decay_steps=10000,
    ):
        super().__init__()
        self.save_hyperparameters()

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
                    "name": "FusedMLP",
                    "dropout": pdropout,
                    "activation": "gelu",
                    "hidden_layer_multiplier": hidden_layer_multiplier,
                },
                "position_encoding_config": {
                    "name": "mlp-3d",
                    "dim_model": dim,
                    "add_class_token": True,
                },
            }
        ]

        config = xFormerConfig(xformer_config)
        self.model = xFormer.from_config(config)
        print(self.model)

        # The classifier head
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_accuracy = Accuracy("binary")

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
        x = self.model(x, ch_pos)
        x = self.ln(x)

        # extract the class token
        x = x[:, 0]

        x = self.head(x)
        return x

    def training_step(self, batch, _):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)

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
        loss = self.criterion(y_hat, y)
        acc = self.val_accuracy(y_hat, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, _):
        self.evaluate(batch, "val")

    def test_step(self, batch, _):
        self.evaluate(batch, "test")
