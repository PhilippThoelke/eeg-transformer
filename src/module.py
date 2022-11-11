import math
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt


class TransformerModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.register_buffer("class_weights", torch.tensor(self.hparams.class_weights))

        # transformer encoder
        self.encoder = EEGEncoder(
            self.hparams.embedding_dim,
            self.hparams.num_layers,
            self.hparams.token_size,
            nheads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
        )

        # output network
        self.outnet = nn.Sequential(
            nn.Linear(self.hparams.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, len(self.hparams.conditions)),
        )

        self.confusion_matrices = {}

    def forward(self, x, ch_pos, mask=None, return_logits=False, eps=1e-7):
        """
        Perform a forward pass of the EEG Transformer.

        Letters in the shape descriptions stand for
        N - batch size, S - EEG samples, C - channels, O - output classes

        Parameters:
            x (torch.Tensor): raw EEG epochs with shape (N, S, C)
            ch_pos (torch.Tensor): channel positions in 3D space normalized to the unit cube (N, C, 3)
            mask (torch.Tensor, optional): boolean mask to hide padded channels from the attention mechanism (N, C)
            return_logits (bool, optional): if True, return unnormalized logits (otherwise softmax scores are returned)
            eps (float, optional): small epsilon to avoid division by zero when normalizing the data

        Returns:
            y (torch.Tensor): logits or softmax scores depending on the return_logits parameter (N, O)
        """
        # standardize channels
        mean, std = x.mean(dim=(0, 1), keepdims=True), x.std(dim=(0, 1), keepdims=True)
        x = (x - mean) / (std + eps)

        # reshape x from (N, S, C) to (C, N, S)
        x = x.permute(2, 0, 1)
        # reshape ch_pos from (N, C, 3) to (C, N, 3)
        ch_pos = ch_pos.permute(1, 0, 2)

        # apply encoder model
        x = self.encoder(x, ch_pos, mask)
        # apply output model
        y = self.outnet(x)

        if return_logits:
            return y
        return y.softmax(dim=-1)

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, training_stage="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, training_stage="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, training_stage="test")

    def step(self, batch, batch_idx, training_stage):
        x, ch_pos, mask, condition, subject = batch

        # regularization
        if training_stage == "train":
            # EEG noise regularization
            if self.hparams.eeg_noise > 0:
                noise = torch.randn_like(x) * x.std()
                x = x + noise * self.hparams.eeg_noise
            # channel position noise regularization
            if self.hparams.channel_noise > 0:
                noise = torch.randn_like(ch_pos) * ch_pos.std()
                ch_pos = ch_pos + noise * self.hparams.channel_noise
            # token dropout via masking
            if self.hparams.token_dropout > 0:
                if mask is None:
                    # initialize mask
                    mask = torch.ones(
                        x.size(0), x.size(2), dtype=torch.bool, device=x.device
                    )
                dropout_mask = (
                    torch.rand(mask.shape, device=mask.device)
                    < self.hparams.token_dropout
                )
                mask[dropout_mask] = False

        # forward pass
        logits = self(x, ch_pos, mask, return_logits=True)

        # loss
        loss = F.cross_entropy(logits, condition, self.class_weights)
        self.log(f"{training_stage}_loss", loss)

        # accuracy
        acc = (logits.argmax(dim=-1) == condition).float().mean()
        self.log(f"{training_stage}_acc", acc)

        # accumulate confusion matrices
        cm = confusion_matrix(
            condition.cpu(),
            logits.argmax(dim=-1).cpu(),
            labels=range(len(self.hparams.conditions)),
        )
        if training_stage not in self.confusion_matrices:
            self.confusion_matrices[training_stage] = cm
        else:
            self.confusion_matrices[training_stage] += cm

        return loss

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

    def validation_epoch_end(self, outputs):
        for stage, cm in self.confusion_matrices.items():
            # normalize confusion matrix
            counts = cm.sum(axis=1, keepdims=True)
            cm = cm / np.where(counts, counts, 1)

            # create confusion matrix plot
            conditions = self.hparams.conditions
            fig, ax = plt.subplots()
            ax.imshow(cm)
            ax.set_xticks(range(len(conditions)), conditions, rotation=90)
            ax.set_yticks(range(len(conditions)), conditions)
            ax.set_xlabel("prediction")
            ax.set_ylabel("ground truth")
            ax.xaxis.set_label_position("top")
            ax.yaxis.set_label_position("right")
            fig.tight_layout()

            # log the confusion matrix
            self.logger.experiment.add_figure(f"{stage}_cm", fig)
        self.confusion_matrices = {}

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
