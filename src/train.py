import argparse
from functools import reduce
from os import makedirs, path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from dataset import RawDataset
from module import TransformerModule
from augmentation import augmentations


def split_data(data, val_subject_ratio):
    dataset_names = set([name.split("-")[0] for name in data.subject_mapping])

    # iterate over all datasets
    train_idxs, val_idxs = [], []
    for name in dataset_names:
        dset_mask = data.subject_mapping.str.startswith(name)[data.subject_ids]
        subj_ids = data.subject_ids[dset_mask]

        # split the data by subjects
        unique_subj_ids = np.unique(subj_ids)
        num_val_subjs = max(int(len(unique_subj_ids) * val_subject_ratio), 1)
        val_subjs = np.random.choice(unique_subj_ids, num_val_subjs, replace=False)
        val_mask = reduce(np.bitwise_or, [data.subject_ids == i for i in val_subjs])
        # get train/val indices
        train_idxs.append(np.where(~val_mask & dset_mask)[0])
        val_idxs.append(np.where(val_mask)[0])
    return np.concatenate(train_idxs), np.concatenate(val_idxs)


def data_augmentation(collate_fn, args):
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


def main(args):
    # load data
    data = RawDataset(args)
    idx_train, idx_val = split_data(data, args.val_subject_ratio)

    # train subset
    train_data = Subset(data, idx_train)
    train_dl = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=data_augmentation(RawDataset.collate, args),
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,
    )
    # fetch dataset weights depending on their size
    args.dataset_weights = data.dataset_weights(idx_train)
    # store the size of a single token
    args.token_size = train_data[0][0].shape[0]

    # val subset
    val_data = Subset(data, idx_val)
    val_dl = DataLoader(
        val_data,
        batch_size=args.batch_size,
        collate_fn=data_augmentation(RawDataset.collate, args),
        num_workers=8,
        prefetch_factor=4,
    )

    # define model
    module = TransformerModule(args)

    # define trainer instance
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=args.max_epochs,
        callbacks=[
            pl.callbacks.EarlyStopping(
                "val_loss", patience=args.early_stopping_patience, mode="min"
            ),
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min"),
        ],
        accumulate_grad_batches=args.gradient_accumulation,
    )

    # store train val splits
    makedirs(trainer.log_dir, exist_ok=True)
    splits = dict(
        train_idx=idx_train,
        val_idx=idx_val,
        train_subjects=data.id2subject(np.unique(data.subject_ids[idx_train])),
        val_subjects=data.id2subject(np.unique(data.subject_ids[idx_val])),
    )
    torch.save(splits, path.join(trainer.log_dir, "splits.pt"))

    # train model
    trainer.fit(model=module, train_dataloaders=train_dl, val_dataloaders=val_dl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="path to the memory mapped data file",
    )
    parser.add_argument(
        "--label-path",
        type=str,
        required=True,
        help="path to the csv file containing labels",
    )
    parser.add_argument(
        "--learning-rate",
        default=5e-3,
        type=float,
        help="base learning rate",
    )
    parser.add_argument(
        "--lr-decay",
        default=0.995,
        type=float,
        help="learning rate decay factor",
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=100,
        type=int,
        help="number of epochs to continue training if val loss doesn't improve anymore",
    )
    parser.add_argument(
        "--batch-size",
        default=64,
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--val-subject-ratio",
        default=0.15,
        type=float,
        help="ratio of subjects to be used for validation",
    )
    parser.add_argument(
        "--embedding-dim",
        default=128,
        type=int,
        help="dimension of tokens inside the transformer",
    )
    parser.add_argument(
        "--num-layers",
        default=5,
        type=int,
        help="number of encoder layers in the transformer",
    )
    parser.add_argument(
        "--num-heads",
        default=8,
        type=int,
        help="number of attention heads",
    )
    parser.add_argument(
        "--dropout",
        default=0.0,
        type=float,
        help="dropout ratio",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.3,
        type=float,
        help="weight decay",
    )
    parser.add_argument(
        "--warmup-steps",
        default=1000,
        type=int,
        help="number of steps for lr warmup",
    )
    parser.add_argument(
        "--max-epochs",
        default=1000,
        type=int,
        help="maximum number of epochs",
    )
    parser.add_argument(
        "--sample-rate",
        default=None,
        type=float,
        help="sampling frequency of the data",
    )
    parser.add_argument(
        "--notch-freq",
        default=None,
        type=float,
        help="frequency at which to apply a notch filter",
    )
    parser.add_argument(
        "--low-pass",
        default=None,
        type=float,
        help="frequency at which to apply a low pass filter",
    )
    parser.add_argument(
        "--high-pass",
        default=None,
        type=float,
        help="frequency at which to apply a high pass filter",
    )
    parser.add_argument(
        "--gradient-accumulation",
        default=10,
        type=int,
        help="number of gradient accumulation steps",
    )
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

    args = parser.parse_args()
    main(args)
