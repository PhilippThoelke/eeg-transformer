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
        "--temperature",
        default=0.5,
        type=float,
        help="temperature parameter of the SimCLR method",
    )
    parser.add_argument(
        "--dataset-loss-weight",
        default=0,
        type=float,
        help="weighting for adversarial dataset loss (0 to disable)",
    )


def collate_decorator(collate_fn, args):
    def augment(batch, eps=1e-7):
        x, ch_pos, mask, condition, subject, dataset = collate_fn(batch)

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


class LightningModule(base.LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        if self.hparams.dataset_loss_weight > 0:
            # store weights for loss weighting
            self.register_buffer(
                "dataset_weights", torch.tensor(self.hparams.dataset_weights)
            )

        # projection head
        self.projection = nn.Sequential(
            nn.Linear(self.hparams.embedding_dim, self.hparams.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.hparams.embedding_dim, self.hparams.embedding_dim),
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

    def step(self, batch, batch_idx, training_stage):
        x, ch_pos, mask, condition, subject, dataset = batch
        initial_bs = x.size(0) // 2

        # forward pass
        z = self(x, ch_pos, mask)

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

        if self.hparams.dataset_loss_weight > 0:
            # apply dataset prediction network and reverse gradients
            y_pred = self.dataset_predictor(-z + (2 * z).detach())
            dataset_loss = F.cross_entropy(y_pred, dataset, self.dataset_weights)
            self.log(f"{training_stage}_dataset_loss", dataset_loss)
            self.log(
                f"{training_stage}_dataset_acc",
                (y_pred.argmax(dim=1) == dataset).float().mean(),
            )
            return loss + dataset_loss * self.hparams.dataset_loss_weight
        return loss