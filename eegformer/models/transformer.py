import math

import lightning.pytorch as pl
import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch import nn
from xformers.factory import xFormer, xFormerConfig

from eegformer.models.utils import MLP3DPositionalEmbedding


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
        weight_decay: float = 0.0,
        dropout: float = 0.0,
        num_classes: int = None,
        z_transform: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # raw signal embedding
        self.signal_embed = nn.Linear(input_dim, model_dim)
        # 3D position embedding
        self.pos_embed = MLP3DPositionalEmbedding(model_dim, add_class_token=True)

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
        self.head = nn.Linear(model_dim, self.hparams.num_classes)

        # initialize class weights
        self.class_weights = None

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

        x = self.signal_embed(x)
        x = self.pos_embed(x, ch_pos)
        x = self.model(x)

        # extract the class token
        x = x[:, 0]

        x = self.ln(x)
        x = self.head(x)
        return x

    def evaluate(self, batch, stage=None):
        signal, ch_pos, y = batch
        y_hat = self(signal, ch_pos)

        loss = F.cross_entropy(y_hat, y, weight=self.class_weights)

        if stage and not self.trainer.sanity_checking:
            acc = (y_hat.argmax(dim=-1) == y).float().mean()

            self.log(f"{stage}/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{stage}/acc", acc, on_step=False, on_epoch=True, prog_bar=True)
            self.accumulate_labels(y, y_hat.argmax(dim=-1), stage)
        if stage == "train":
            self.log("learning_rate", self.lr_schedulers().get_last_lr()[0], on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, _):
        return self.evaluate(batch, "train")

    def validation_step(self, batch, _):
        self.evaluate(batch, "val")

    def test_step(self, batch, _):
        self.evaluate(batch, "test")

    def on_train_epoch_start(self):
        self.initialize_labels()

        if self.class_weights is None:
            self.class_weights = self.trainer.datamodule.class_weights("train").to(self.device)

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
