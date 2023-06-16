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
        steps,
        learning_rate=5e-4,
        weight_decay=0.03,
        num_classes=10,
        patch_size=2,
        dim=384,
        n_layer=6,
        n_head=6,
        resid_pdrop=0.0,
        attn_pdrop=0.0,
        mlp_pdrop=0.0,
        hidden_layer_multiplier=4,
    ):
        super().__init__()
        self.save_hyperparameters()
        num_patches = (image_size // patch_size) ** 2

        xformer_config = [
            {
                "block_type": "encoder",
                "num_layers": n_layer,
                "dim_model": dim,
                "residual_norm_style": "pre",
                "multi_head_config": {
                    "num_heads": n_head,
                    "residual_dropout": resid_pdrop,
                    "attention": {
                        "name": "scaled_dot_product",
                        "dropout": attn_pdrop,
                        "causal": False,
                    },
                },
                "feedforward_config": {
                    "name": "FusedMLP",
                    "dropout": mlp_pdrop,
                    "activation": "gelu",
                    "hidden_layer_multiplier": hidden_layer_multiplier,
                },
                "position_encoding_config": {
                    "name": "learnable",
                    "seq_len": num_patches,
                    "dim_model": dim,
                    "add_class_token": True,
                },
                "patch_embedding_config": {
                    "in_channels": 3,
                    "out_channels": dim,
                    "kernel_size": patch_size,
                    "stride": patch_size,
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
        self.val_accuracy = Accuracy()

    @staticmethod
    def linear_warmup_cosine_decay(warmup_steps, total_steps):
        """
        Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps
        """
        def fn(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return fn

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            betas=self.hparams.betas,
            weight_decay=self.hparams.weight_decay,
        )

        warmup_steps = int(self.hparams.linear_warmup_ratio * self.hparams.steps)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                self.linear_warmup_cosine_decay(warmup_steps, self.hparams.steps),
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]

    def forward(self, x):
        x = self.model(x)
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
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = self.val_accuracy(y_hat, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, _):
        self.evaluate(batch, "val")

    def test_step(self, batch, _):
        self.evaluate(batch, "test")
