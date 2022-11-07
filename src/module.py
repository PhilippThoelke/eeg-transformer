import math
import torch
from torch import nn, optim
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix


class TransformerModule(pl.LightningModule):
    def __init__(self, hparams, mean=0, std=1):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.register_buffer("class_weights", torch.tensor(self.hparams.class_weights))
        self.register_buffer("mean", torch.scalar_tensor(mean))
        self.register_buffer("std", torch.scalar_tensor(std))

        if self.hparams.used_data_length is None:
            self.sample_length = self.hparams.epoch_length // self.hparams.num_tokens
        else:
            self.sample_length = (
                self.hparams.used_data_length // self.hparams.num_tokens
            )

        self.norm = nn.BatchNorm1d(
            self.hparams.num_channels - len(self.hparams.ignore_channels)
        )

        # transformer encoder
        self.encoder = EEGEncoder(
            self.hparams.embedding_dim,
            self.hparams.num_layers,
            self.sample_length,
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

    def forward(self, x, return_logits=False):
        # add a batch dimension if required
        if x.ndim == 2:
            x = x.unsqueeze(0)

        # crop sequence to be divisible by the desired number of tokens
        cut_length = self.hparams.num_tokens * self.sample_length
        if cut_length < x.size(1):
            offset = 0
            if self.training:
                offset = torch.randint(0, x.size(1) - cut_length, (1,), device=x.device)
            x = x[:, offset : offset + cut_length]

        # potentially drop some channels
        if len(self.hparams.ignore_channels) > 0:
            ch_mask = torch.ones(x.size(2), dtype=torch.bool)
            ch_mask.scatter_(0, torch.tensor(self.hparams.ignore_channels), False)
            x = x[..., ch_mask]

        # standardize data
        x = (x - self.mean) / self.std
        x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        # reshape x from (B x time x elec) to (B x token x signal x channel)
        x = x.view(x.size(0), self.hparams.num_tokens, self.sample_length, x.size(2))

        # randomly reorder tokens
        if self.training and self.hparams.shuffle_tokens != "none":
            for i in range(x.size(0)):
                if self.hparams.shuffle_tokens in ["temporal", "all"]:
                    x[i] = x[i, torch.randperm(x.size(1), device=x.device)]
                if self.hparams.shuffle_tokens in ["channels", "all"]:
                    x[i] = x[i, :, :, torch.randperm(x.size(3), device=x.device)]

        # reshape x from (B x token x signal x channel) to (token x B x window_length)
        x = x.permute(0, 3, 1, 2).reshape(x.size(0), -1, self.sample_length)
        x = x.permute(1, 0, 2)

        # dropout entire tokens
        if self.training and self.hparams.token_dropout > 0:
            mask = torch.rand(x.shape[:2], device=x.device) < self.hparams.token_dropout
            x[mask] = 0
            x = x * (1 / (1 - self.hparams.token_dropout))

        # apply encoder model
        x = self.encoder(x)
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
        x, condition, stage, subject = batch
        logits = self(x, return_logits=True)

        # loss
        loss = F.cross_entropy(logits, condition, self.class_weights)
        self.log(f"{training_stage}_loss", loss)

        # accuracy
        acc = (logits.argmax(dim=-1) == condition).float().mean()
        self.log(f"{training_stage}_acc", acc)

        # accumulate confusion matrices
        cm = confusion_matrix(condition.cpu(), logits.argmax(dim=-1).cpu())
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
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.995, last_epoch=30)
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
            cm = cm / cm.sum()
            self.logger.experiment.add_image(f"{stage}_cm", cm, dataformats="HW")
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

    def forward(self, x):
        q, k, v = self.qkv(x).split(self.embedding_dim, dim=-1)
        q = q.reshape(q.size(0), q.size(1) * self.nheads, -1).permute(1, 0, 2)
        k = k.reshape(k.size(0), k.size(1) * self.nheads, -1).permute(1, 2, 0)
        attn = torch.softmax((q @ k) / self.scale_factor, dim=-1)
        v = v.reshape(v.size(0), v.size(1) * self.nheads, -1).permute(1, 0, 2)
        o = attn @ v
        o = o.permute(1, 0, 2).reshape(x.shape)
        o = self.out(o)
        o = self.dropout(o)
        return o, attn.reshape(x.size(1), self.nheads, x.size(0), x.size(0))


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

    def forward(self, x):
        x = self.mha(self.norm1(x))[0] + x
        x = self.ff(self.norm2(x)) + x
        return self.dropout(x)


class EEGEncoder(nn.Module):
    def __init__(self, embedding_dim, num_layers, sample_length, nheads=8, dropout=0.1):
        super().__init__()
        self.in_proj = nn.Linear(sample_length, embedding_dim)
        self.pe = PositionalEncoding(embedding_dim, dropout=dropout)
        self.register_parameter("class_token", nn.Parameter(torch.randn(embedding_dim)))

        # create encoder layers
        self.encoder_layers = nn.Sequential(
            *[
                TransformerEncoderLayer(
                    embedding_dim,
                    nheads=nheads,
                    dim_feedforward=embedding_dim * 2,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # linear projection into embedding dimension
        x = self.in_proj(x)
        # add positional encoding
        x = self.pe(x)
        # prepend class token to the sequence
        x = torch.cat([self.class_token[None, None].repeat(1, x.size(1), 1), x], dim=0)
        # pass sequence through the transformer and extract class tokens
        x = self.encoder_layers(x)[0]
        return self.dropout(self.norm(x))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encodings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        x = x + self.encodings(torch.arange(x.size(0), device=x.device)).unsqueeze(1)
        return self.dropout(x)
