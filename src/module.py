import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from augmentation import augmentations


class TransformerModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.register_buffer(
            "dataset_weights", torch.tensor(self.hparams.dataset_weights)
        )

        # transformer encoder
        self.encoder = EEGEncoder(
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
        return self.encoder(x, ch_pos, mask)

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
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=self.hparams.lr_decay)
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
            optimizers = self.optimizers()
            if not isinstance(optimizers, list):
                optimizers = [optimizers]

            for opt in optimizers:
                opt.param_groups[0]["lr"] = self.hparams.learning_rate * (
                    (self.global_step + 1) / self.hparams.warmup_steps
                )
        # continue with the optimizer step normally
        return super().optimizer_step(*args, **kwargs)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, nheads, dropout=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.nheads = nheads
        self.qkv = nn.Linear(embedding_dim, embedding_dim * 3)
        self.out = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale_factor = math.sqrt(embedding_dim / nheads)

    def forward(self, x, mask=None):
        # transform x into queries, keys and values
        q, k, v = self.qkv(x).split(self.embedding_dim, dim=-1)
        q = q.reshape(q.size(0), q.size(1) * self.nheads, -1).permute(1, 0, 2)
        k = k.reshape(k.size(0), k.size(1) * self.nheads, -1).permute(1, 2, 0)
        # compute attention weights
        q = q / self.scale_factor
        if mask is not None:
            mask = mask.repeat_interleave(self.nheads, dim=0).float()
            # set masked tokens to 0
            q = q * mask.unsqueeze(2)
            k = k * mask.unsqueeze(1)
            # dot product between queries and keys
            mask = mask.unsqueeze(2) @ mask.unsqueeze(1)
            attn = torch.baddbmm(1 - mask, q, k, beta=-1e9)
        else:
            # dot product between queries and keys
            attn = torch.bmm(q, k)
        attn = torch.softmax(attn, dim=-1)
        # rescale values by attention weights
        v = v.reshape(v.size(0), v.size(1) * self.nheads, -1).permute(1, 0, 2)
        v = attn @ v
        # apply output projection
        v = v.permute(1, 0, 2).reshape(x.shape)
        o = self.out(v)
        # return output and attention weights
        attn = attn.reshape(x.size(1), self.nheads, x.size(0), x.size(0))
        return self.dropout(o), attn


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embedding_dim, nheads=8, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.mha = MultiHeadAttention(embedding_dim, nheads, dropout=dropout)
        self.ff = nn.Sequential(
            nn.Linear(embedding_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, embedding_dim),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.mha(self.norm1(x), mask)[0] + x
        x = self.ff(self.norm2(x)) + x
        return self.dropout(x)


class EEGEncoder(nn.Module):
    def __init__(self, embedding_dim, num_layers, token_size, nheads=8, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(token_size, embedding_dim)
        self.pe = PositionalEncoding(embedding_dim, dropout=dropout)
        self.register_parameter("class_token", nn.Parameter(torch.randn(embedding_dim)))

        # create encoder layers
        self.encoder_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embedding_dim,
                    nheads=nheads,
                    dim_feedforward=embedding_dim * 2,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.output_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, ch_pos, mask=None):
        """
        Perform a forward pass of the EEG Transformer Encoder.

        Letters in the shape descriptions stand for
        N - batch size, S - EEG samples, C - channels, L - latent dimension

        Parameters:
            x (torch.Tensor): raw EEG epochs with shape (C, N, S)
            ch_pos (torch.Tensor): channel positions in 3D space normalized to the unit cube (C, N, 3)
            mask (torch.Tensor, optional): boolean mask to hide padded channels from the attention mechanism (N, C)

        Returns:
            z (torch.Tensor): latent representation of input samples (N, L)
        """
        # linear projection into embedding dimension
        x = self.in_proj(x)
        # add positional encoding
        x = self.pe(x, ch_pos)
        # prepend class token to the sequence
        x = torch.cat([self.class_token[None, None].repeat(1, x.size(1), 1), x], dim=0)
        if mask is not None:
            add_row = torch.ones(mask.size(0), 1, dtype=mask.dtype, device=mask.device)
            mask = torch.cat([add_row, mask], dim=1)

        for layer in self.encoder_layers:
            x = layer(x, mask)
        # return the class token only
        return self.dropout(self.output_norm(x[0]))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoder = nn.Sequential(
            nn.Linear(3, d_model * 2),
            nn.Tanh(),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh(),
        )

    def forward(self, x, ch_pos):
        return self.dropout(x + self.encoder(ch_pos))
