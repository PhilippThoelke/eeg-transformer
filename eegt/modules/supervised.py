import torch
from torch import nn
import torch.nn.functional as F
from eegt.modules import base
from eegt.augmentation import augmentations


def add_arguments(parser):
    parser.add_argument(
        "--augmentation-prob",
        default=0,
        type=float,
        help="probability to apply data augmentation to the current batch",
    )
    parser.add_argument(
        "--dataset-loss-weight",
        default=0.5,
        type=float,
        help="weighting for adversarial dataset loss (0 to disable)",
    )


def collate_decorator(collate_fn, args):
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
    def __init__(self, hparams):
        super().__init__(hparams)
        # store weights loss weighting
        self.register_buffer("class_weights", torch.tensor(self.hparams.class_weights))
        if self.hparams.dataset_loss_weight > 0:
            self.register_buffer(
                "dataset_weights", torch.tensor(self.hparams.dataset_weights)
            )

        # output network
        self.output_network = nn.Sequential(
            nn.Linear(self.hparams.embedding_dim, self.hparams.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hparams.embedding_dim // 2, len(self.class_weights)),
        )

        if self.hparams.dataset_loss_weight > 0:
            # dataset prediction network
            self.dataset_predictor = nn.Sequential(
                nn.Linear(self.hparams.embedding_dim, self.hparams.embedding_dim // 2),
                nn.ReLU(),
                nn.Linear(
                    self.hparams.embedding_dim // 2, len(self.hparams.dataset_weights)
                ),
            )

    def forward(self, x, ch_pos, mask=None, return_latent=False):
        """
        Perform a forward pass of the EEG Transformer.

        Letters in the shape descriptions stand for
        N - batch size, S - EEG samples, C - channels, D - classes, L - latent dimension

        Parameters:
            x (torch.Tensor): raw EEG epochs with shape (N, S, C)
            ch_pos (torch.Tensor): channel positions in 3D space normalized to the unit cube (N, C, 3)
            mask (torch.Tensor, optional): boolean mask to hide padded channels from the attention mechanism (N, C)
            return_latent (bool, optional): if True, also return the latent representation

        Returns:
            y (torch.Tensor): logit class predictions (N, D)
            z (torch.Tensor): latent representation (N, L), only if return_latent is True
        """
        # reshape x from (N, S, C) to (C, N, S)
        x = x.permute(2, 0, 1)
        # reshape ch_pos from (N, C, 3) to (C, N, 3)
        ch_pos = ch_pos.permute(1, 0, 2)

        # apply encoder model
        z = self.model(x, ch_pos, mask)
        y = self.output_network(z)
        if return_latent:
            return y, z
        return y

    def step(self, batch, batch_idx, training_stage):
        x, ch_pos, mask, condition, subject, dataset = batch

        # forward pass
        y, z = self(x, ch_pos, mask, return_latent=True)

        # compute cross entropy loss
        loss = F.cross_entropy(y, condition, self.class_weights)
        self.log(f"{training_stage}_loss", loss)

        # compute accuracy
        acc = (y.argmax(dim=1) == condition).float().mean()
        self.log(f"{training_stage}_acc", acc)

        if self.hparams.dataset_loss_weight > 0:
            # apply dataset prediction network and reverse gradients
            y_dset = self.dataset_predictor(-z + (2 * z).detach())
            dataset_loss = F.cross_entropy(y_dset, dataset, self.dataset_weights)
            self.log(f"{training_stage}_dataset_loss", dataset_loss)
            self.log(
                f"{training_stage}_dataset_acc",
                (y_dset.argmax(dim=1) == dataset).float().mean(),
            )
            return loss + dataset_loss * self.hparams.dataset_loss_weight
        return loss
