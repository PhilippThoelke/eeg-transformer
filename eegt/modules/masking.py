from distutils import util
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
from eegt.modules import base
from eegt.augmentation import augmentations


def add_arguments(parser):
    parser.add_argument(
        "--fourier-loss",
        default=False,
        type=lambda x: bool(util.strtobool(x)),
        help="whether to compute loss in the frequency or time domain",
    )
    parser.add_argument(
        "--augmentation-prob",
        default=0.2,
        type=float,
        help="probability to apply data augmentation to the current batch",
    )
    parser.add_argument(
        "--dataset-loss-weight",
        default=0.5,
        type=float,
        help="weighting for adversarial dataset loss (0 to disable)",
    )


def collate_decorator(collate_fn, args, training=False):
    if not training:
        return collate_fn

    def augment(batch):
        x, ch_pos, mask, condition, subject, dataset = collate_fn(batch)

        if torch.rand((1,)).item() < args.augmentation_prob:
            # apply data augmentation to the current batch
            idx = torch.randint(0, len(augmentations), (1,)).item()
            x, ch_pos, mask = augmentations[idx](x, ch_pos, mask)

        # return augmented batch
        return x, ch_pos, mask, condition, subject, dataset

    return augment


class LightningModule(base.LightningModule):
    def __init__(self, hparams, model=None):
        super().__init__(hparams, model)
        use_dataset_loss = (
            self.hparams.dataset_loss_weight > 0
            and len(self.hparams.dataset_weights) > 1
        )
        if use_dataset_loss:
            self.register_buffer(
                "dataset_weights", torch.tensor(self.hparams.dataset_weights)
            )

        # prediction network
        target_size = self.hparams.token_size
        if self.hparams.fourier_loss:
            target_size = self.hparams.token_size * 2
        self.mask_predictor = nn.Sequential(
            nn.Linear(self.hparams.embedding_dim * 2, self.hparams.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hparams.embedding_dim * 2, target_size),
        )

        if use_dataset_loss:
            # dataset prediction network
            self.dataset_predictor = nn.Sequential(
                nn.Linear(self.hparams.embedding_dim, self.hparams.embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(
                    self.hparams.embedding_dim // 2, len(self.hparams.dataset_weights)
                ),
            )

    def step(self, batch, batch_idx, training_stage):
        x, ch_pos, mask, condition, subject, dataset = batch

        # mask one channel from each sample
        masked_channel_idxs = (
            torch.rand(mask.size(0), device=mask.device) * mask.sum(dim=1)
        ).long()
        index = torch.arange(mask.size(0), device=mask.device)
        mask[(index, masked_channel_idxs)] = False

        # retrieve masked channel position and samples
        masked_ch_pos = ch_pos[(index, masked_channel_idxs)]
        target = x[(index, slice(None), masked_channel_idxs)]

        # forward pass
        z = self(x, ch_pos, mask)

        # precict masked channel
        z_pos = torch.cat([self.model.pe.encoder(masked_ch_pos), z], dim=1)
        pred = self.mask_predictor(z_pos)

        # compute loss
        if self.hparams.fourier_loss:
            pred_real, pred_imag = pred.split(self.hparams.token_size, dim=-1)
            target = torch.fft.fft(target)
            real = F.mse_loss(pred_real, target.real)
            imag = F.mse_loss(pred_imag, target.imag)
            loss = real + imag
        else:
            loss = F.mse_loss(pred, target)
        self.log(f"{training_stage}_loss", loss)

        # apply dataset prediction network and reverse gradients
        if hasattr(self, "dataset_predictor"):
            y_dset = self.dataset_predictor(-z + (2 * z).detach())
            dataset_loss = F.cross_entropy(y_dset, dataset, self.dataset_weights)
            self.log(f"{training_stage}_dataset_loss", dataset_loss)
            self.log(
                f"{training_stage}_dataset_acc",
                (y_dset.argmax(dim=1) == dataset).float().mean(),
            )
            return loss + dataset_loss * self.hparams.dataset_loss_weight
        return loss
