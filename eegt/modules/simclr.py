import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from eegt.model import EEGEncoder
from eegt.augmentation import augmentations


def collate_decorator(collate_fn, args):
    def augment(batch, eps=1e-7):
        x, ch_pos, mask, condition, subject, dataset = collate_fn(batch)

        # standardize signal channel-wise
        mean, std = x.mean(dim=(0, 1), keepdims=True), x.std(dim=(0, 1), keepdims=True)
        x = (x - mean) / (std + eps)

        # augment data
        x_all, ch_pos_all, mask_all = [], [], []
        for k in range(2):
            x_aug, ch_pos_aug, mask_aug = x.clone(), ch_pos.clone(), mask.clone()
            perm = torch.randperm(len(augmentations))
            for j in perm[: args.num_augmentations]:
                x_aug, ch_pos_aug, mask_aug = augmentations[j](
                    x_aug, ch_pos_aug, mask_aug
                )
            x_all.append(x_aug)
            ch_pos_all.append(ch_pos_aug)
            mask_all.append(mask_aug)
        x = torch.cat(x_all)
        ch_pos = torch.cat(ch_pos_all)
        mask = torch.cat(mask_all)
        condition = torch.cat([condition, condition])
        subject = torch.cat([subject, subject])
        dataset = torch.cat([dataset, dataset])

        # return augmented batch
        return x, ch_pos, mask, condition, subject, dataset

    return augment


class LightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.register_buffer(
            "dataset_weights", torch.tensor(self.hparams.dataset_weights)
        )

        # transformer encoder
        self.model = EEGEncoder(
            self.hparams.embedding_dim,
            self.hparams.num_layers,
            self.hparams.token_size,
            nheads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
        )

        # projection head
        self.projection = nn.Sequential(
            nn.Linear(self.hparams.embedding_dim, self.hparams.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.embedding_dim, self.hparams.embedding_dim),
        )

        # dataset prediction network
        self.dataset_predictor = nn.Sequential(
            nn.Linear(self.hparams.embedding_dim, self.hparams.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(
                self.hparams.embedding_dim // 2, len(self.hparams.dataset_weights)
            ),
        )

    def forward(self, x, ch_pos, mask=None):
        """
        Perform a forward pass of the EEG Transformer.

        Letters in the shape descriptions stand for
        N - batch size, S - EEG samples, C - channels, L - latent dimension

        Parameters:
            x (torch.Tensor): raw EEG epochs with shape (N, S, C)
            ch_pos (torch.Tensor): channel positions in 3D space normalized to the unit cube (N, C, 3)
            mask (torch.Tensor, optional): boolean mask to hide padded channels from the attention mechanism (N, C)

        Returns:
            z (torch.Tensor): latent representation of input samples (N, L)
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

    def step(self, batch, batch_idx, training_stage):
        x, ch_pos, mask, condition, subject, dataset = batch
        initial_bs = x.size(0) // 2

        # forward pass
        z = self(x, ch_pos, mask)

        # apply dataset prediction network and reverse gradients
        y_pred = self.dataset_predictor(-z + (2 * z).detach())
        dataset_loss = F.cross_entropy(y_pred, dataset, self.dataset_weights)
        self.log(f"{training_stage}_dataset_loss", dataset_loss)
        self.log(
            f"{training_stage}_dataset_acc",
            (y_pred.argmax(dim=1) == dataset).float().mean(),
        )

        # apply projection head
        z_proj = self.projection(z)

        # compute pairwise cosine similarity
        similarity = F.cosine_similarity(
            z_proj.unsqueeze(1), z_proj.unsqueeze(0), dim=2
        )

        # get nominator from positive samples
        positives = torch.cat(
            [torch.diag(similarity, initial_bs), torch.diag(similarity, -initial_bs)]
        )
        nominator = (positives / self.hparams.temperature).exp()
        # get denominator from negative samples
        mask = (~torch.eye(z_proj.size(0), dtype=bool, device=z_proj.device)).float()
        denominator = mask * (similarity / self.hparams.temperature).exp()

        # compute final loss
        all_losses = -(nominator / denominator.sum(dim=1)).log()
        loss = all_losses.sum() / z_proj.size(0)
        self.log(f"{training_stage}_loss", loss)
        return loss + dataset_loss

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
