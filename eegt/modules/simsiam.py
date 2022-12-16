import torch
from torch import nn
import torch.nn.functional as F
from eegt.augmentation import augmentations
from eegt.modules import base


def add_arguments(parser):
    parser.add_argument(
        "--num-augmentations",
        default=2,
        type=int,
        help="number of data augmentation steps during pretraining",
    )
    parser.add_argument(
        "--augmentation-indices",
        default=[-1],
        nargs="+",
        type=int,
        help="indices of augmentation functions to use (-1 to use all)",
    )
    parser.add_argument(
        "--dataset-loss-weight",
        default=0,
        type=float,
        help="weighting for adversarial dataset loss (0 to disable)",
    )
    parser.add_argument(
        "--latent-regularization",
        default=0,
        type=float,
        help="weight of l1 norm on the latent space (0 to disable)",
    )


def collate_decorator(collate_fn, args, training=False):
    def augment(batch, eps=1e-7):
        x, ch_pos, mask, condition, subject, dataset = collate_fn(batch)

        # define augmentation indices
        augmentation_indices = args.augmentation_indices
        if len(augmentation_indices) == 1 and augmentation_indices[0] == -1:
            augmentation_indices = list(range(len(augmentations)))

        # augment data
        x_all, ch_pos_all, mask_all = [], [], []
        for _ in range(2):
            x_aug, ch_pos_aug, mask_aug = x.clone(), ch_pos.clone(), mask.clone()
            perm = torch.randperm(len(augmentation_indices))
            for j in perm[: args.num_augmentations]:
                augmentation_fn = augmentations[augmentation_indices[j]]
                x_aug, ch_pos_aug, mask_aug = augmentation_fn(
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


class EncoderProjection(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        embed_dim = encoder.class_token.shape[0]
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim, affine=False),
        )

    def forward(self, *args, **kwargs):
        z = self.encoder(*args, **kwargs)
        return self.projection(z)


class LightningModule(base.LightningModule):
    def __init__(self, hparams, model=None, **kwargs):
        super().__init__(hparams, model, **kwargs)

        # add a projection head to the base model
        self.model = EncoderProjection(self.model)

        # define dataset weights
        if (
            self.hparams.dataset_loss_weight > 0
            and "dataset_weights" in kwargs
            and len(kwargs["dataset_weights"]) > 1
        ):
            self.dataset_weights = torch.tensor(kwargs["dataset_weights"])

        embed_dim = self.hparams.embedding_dim
        # predictor network
        self.predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        if hasattr(self, "dataset_weights"):
            # dataset prediction network
            self.dataset_predictor = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.ReLU(),
                nn.Linear(embed_dim // 2, len(self.dataset_weights)),
            )

    def step(self, batch, batch_idx, training_stage):
        x, ch_pos, mask, condition, subject, dataset = batch
        initial_bs = x.size(0) // 2

        # forward pass
        z = self(x, ch_pos, mask)
        z1, z2 = z.detach().split(initial_bs, dim=0)

        # apply predictor network
        p = self.predictor(z)
        p1, p2 = p.split(initial_bs, dim=0)

        # compute loss
        loss = (
            -(F.cosine_similarity(p1, z2).mean() + F.cosine_similarity(p2, z1).mean())
            * 0.5
        )
        self.log(f"{training_stage}_loss", loss)

        # regularize the latent space
        if self.hparams.latent_regularization > 0:
            loss = loss + z.norm(p=1, dim=1).mean() * self.hparams.latent_regularization

        if hasattr(self, "dataset_predictor"):
            # apply dataset prediction network and reverse gradients
            y_pred = self.dataset_predictor(-z + (2 * z).detach())
            dataset_loss = F.cross_entropy(y_pred, dataset, self.dataset_weights)
            self.log(f"{training_stage}_dataset_loss", dataset_loss)
            self.log(
                f"{training_stage}_dataset_acc",
                (y_pred.argmax(dim=1) == dataset).float().mean(),
            )
            loss = loss + dataset_loss * self.hparams.dataset_loss_weight
        return loss
