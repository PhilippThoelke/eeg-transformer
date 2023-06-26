import math
from typing import Optional

import lightning.pytorch as pl
import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn
from xformers.factory import xFormer, xFormerConfig

from eegformer.models.utils import MLP3DPositionalEmbedding
from eegformer.utils import subsample_signal_batch


class Transformer(pl.LightningModule):
    """
    Transformer model for EEG classification.

    TODO: offload model configuration and instantiation to a separate class
    """

    def __init__(
        self,
        model_dim: int = 128,
        input_dim: int = 320,
        n_layer: int = 5,
        n_head: int = 5,
        hidden_layer_multiplier: int = 4,
        learning_rate: float = 1e-3,
        warmup_steps: int = 500,
        lr_decay_steps: int = 10000,
        lr_cycle_steps: int = 5000,
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        similarity_subsamples: int = 0,
        similarity_loss_weight: float = 0.1,
        raw_batchnorm: bool = True,
        num_classes: int = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        assert model_dim % n_head == 0, "model_dim must be divisible by n_head"

        # initialize raw signal batchnorm
        if raw_batchnorm:
            self.raw_norm = nn.BatchNorm1d(input_dim)

        # raw signal embedding
        self.signal_embed = nn.Linear(input_dim, model_dim)
        # 3D position embedding
        self.pos_embed = MLP3DPositionalEmbedding(model_dim, add_class_token=True, dropout=dropout)

        # configure encoder model
        xformer_config = [
            {
                "block_type": "encoder",
                "num_layers": n_layer,
                "dim_model": model_dim,
                "residual_norm_style": "pre",
                "multi_head_config": {
                    "num_heads": n_head,
                    "residual_dropout": dropout,
                    "attention": {
                        "name": "scaled_dot_product",
                        "dropout": dropout,
                        "causal": False,
                    },
                },
                "feedforward_config": {
                    "name": "MLP",
                    "dropout": dropout,
                    "activation": "gelu",
                    "hidden_layer_multiplier": hidden_layer_multiplier,
                },
            }
        ]
        config = xFormerConfig(xformer_config)
        self.model = xFormer.from_config(config)

        # classifier head
        self.ln = nn.LayerNorm(model_dim)
        self.head = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.GELU(),
            nn.Linear(model_dim // 2, num_classes),
        )

        # initialize class weights
        self.class_weights = None

    @staticmethod
    def linear_warmup_cosine_decay(warmup_steps, total_steps, cycle_steps=None):
        """
        Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps and optional cycles.

        ### Args
            - `warmup_steps` (int): number of warmup steps
            - `total_steps` (int): total number of steps to decay to 0
            - `cycle_steps` (int): number of steps per cycle (default: None)

        ### Returns
            - `fn` (function): function that takes in a step and returns a learning rate factor
        """

        def fn(step: int) -> float:
            # linear warmup
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))

            # cosine decay to 0 after warmup
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            decay = 0.5 * (1.0 + math.cos(math.pi * progress))

            if cycle_steps is None:
                return decay

            # cosine decay to 0 after warmup, with cycles
            cycle_progress = float(step - warmup_steps) / float(max(1, cycle_steps - warmup_steps))
            return decay * 0.5 * (1.0 + math.cos(math.pi * 2.0 * cycle_progress))

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
                self.linear_warmup_cosine_decay(
                    self.hparams.warmup_steps, self.hparams.lr_decay_steps, cycle_steps=self.hparams.lr_cycle_steps
                ),
            ),
            "interval": "step",
        }

        return [optimizer], [scheduler]

    def forward(
        self, x: torch.Tensor, ch_pos: torch.Tensor, mask: Optional[torch.Tensor] = None, return_latent: bool = False
    ):
        """
        Forward pass of the model.

        ### Args
            - `x` (Tensor): tensor of raw EEG signals (batch, channels, time)
            - `ch_pos` (Tensor): tensor of channel positions (batch, channels, 3)
            - `mask` (Tensor): optional attention mask (batch, channels)
            - `return_latent` (bool): whether to return the latent representation

        ### Returns
            Tensor or Tuple[Tensor, Tensor]: logits and latent representation if `return_latent` is True, otherwise just logits
        """
        # apply batchnorm to the raw signal
        if self.hparams.raw_batchnorm:
            x = self.raw_norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        # embed the raw signal
        x = self.signal_embed(x)
        x, mask = self.pos_embed(x, ch_pos, mask=mask)

        # forward pass of the Transformer
        z = self.model(x, encoder_input_mask=mask)
        # extract the normalized class token
        z = self.ln(z[:, 0])
        # classifier head
        y = self.head(z)

        if return_latent:
            return y, z
        return y

    def evaluate(self, batch, stage=None):
        # split the signal into multiple chunks and expand the batch
        if self.hparams.similarity_subsamples > 1:
            batch = subsample_signal_batch(batch, self.hparams)

        signal, ch_pos, mask, y = batch

        # compute loss
        y_hat, z = self(signal, ch_pos, mask=mask, return_latent=True)
        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)

        # compute similarity regularization loss
        similarity_loss = 0.0
        if self.hparams.similarity_subsamples > 1:
            # split the latent representation into its subsamples
            zs = z.split(z.size(0) // self.hparams.similarity_subsamples)

            # compute pairwise cosine similarity
            idxs = np.triu_indices(len(zs), k=1)
            for i, j in zip(*idxs):
                similarity_loss -= F.cosine_similarity(zs[i], zs[j]).mean()
            similarity_loss /= len(idxs[0])

        # log metrics
        if stage and not self.trainer.sanity_checking:
            acc = (y_hat.argmax(dim=-1) == y).float().mean()

            self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

            if self.hparams.similarity_subsamples > 1:
                self.log(f"{stage}/similarity_loss", similarity_loss, on_step=False, on_epoch=True)

            self.accumulate_labels(y, y_hat.argmax(dim=-1), stage)
        if stage == "train":
            self.log("learning_rate", self.lr_schedulers().get_last_lr()[0], on_step=True, on_epoch=False)
        return loss + similarity_loss * self.hparams.similarity_loss_weight

    def training_step(self, batch, _):
        return self.evaluate(batch, "train")

    def validation_step(self, batch, _):
        self.evaluate(batch, "val")

    def test_step(self, batch, _):
        self.evaluate(batch, "test")

    def on_fit_start(self):
        # acquire class weights from the datamodule
        if self.class_weights is None:
            self.class_weights = self.trainer.datamodule.class_weights.to(self.device)

    def on_train_epoch_start(self):
        self.initialize_labels()

    def on_train_epoch_end(self):
        self.log_confusion_matrix("train")

    def on_validation_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        self.log_confusion_matrix("val")

    def on_test_epoch_end(self):
        self.log_confusion_matrix("test")

    def initialize_labels(self):
        # initialize containers for true and predicted labels
        self.true_labels = {}
        self.predicted_labels = {}

    def accumulate_labels(self, y_true, y_pred, stage):
        # accumulate true labels
        if stage not in self.true_labels:
            self.true_labels[stage] = []
        self.true_labels[stage].extend(y_true.cpu().numpy())
        # accumulate predicted labels
        if stage not in self.predicted_labels:
            self.predicted_labels[stage] = []
        self.predicted_labels[stage].extend(y_pred.cpu().numpy())

    def log_confusion_matrix(self, stage):
        if not hasattr(self.logger.experiment, "log"):
            pl.utilities.rank_zero_warn("Cannot log confusion matrix, no experiment logger found.")
            return

        # get class names
        class_names = [self.trainer.datamodule.train_data.idx2label(i) for i in range(self.hparams.num_classes)]
        if class_names[0] is NotImplemented:
            class_names = [str(i) for i in range(self.hparams.num_classes)]

        # compute confusion matrix
        cm = confusion_matrix(self.true_labels[stage], self.predicted_labels[stage], normalize="true")

        # visualize confusion matrix
        plt.figure()
        plt.imshow(cm, cmap="Reds")
        plt.xticks(range(self.hparams.num_classes), class_names, rotation=60, ha="right")
        plt.yticks(range(self.hparams.num_classes), class_names)
        plt.title(f"Confusion matrix ({stage})")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()

        # log confusion matrix
        self.logger.experiment.log({f"confusion_matrix_{stage}": plt})
        plt.close()
