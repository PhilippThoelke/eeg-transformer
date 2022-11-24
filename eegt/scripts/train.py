import pkgutil
import importlib
import argparse
from functools import reduce
from os import makedirs, path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
import eegt
from eegt.dataset import RawDataset


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


def main(args):
    # load data
    data = RawDataset(args)
    idx_train, idx_val = split_data(data, args.val_subject_ratio)
    # fetch class and dataset weights
    args.dataset_weights = data.dataset_weights(idx_train)
    args.class_weights = data.class_weights(idx_train)
    # store the size of a single token
    args.token_size = data[0][0].shape[0]

    # instantiate PyTorch-Lightning module
    paradigm = importlib.import_module(f"eegt.modules.{args.training_paradigm}")
    module = paradigm.LightningModule(args)

    # prepare the data collate function
    collate_fn = RawDataset.collate
    if hasattr(paradigm, "collate_decorator"):
        collate_fn = paradigm.collate_decorator(collate_fn, args)

    # train subset
    train_data = Subset(data, idx_train)
    train_dl = DataLoader(
        train_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,
    )

    # val subset
    val_data = Subset(data, idx_val)
    val_dl = DataLoader(
        val_data,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=8,
        prefetch_factor=4,
    )

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

    def add_default_args(parser):
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
            default=5,
            type=int,
            help="number of gradient accumulation steps",
        )
        parser.add_argument(
            "--learning-rate",
            default=5e-4,
            type=float,
            help="base learning rate",
        )
        parser.add_argument(
            "--embedding-dim",
            default=128,
            type=int,
            help="dimension of tokens inside the transformer",
        )
        parser.add_argument(
            "--num-layers",
            default=3,
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

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        title="training paradigms",
        help="select a training paradigm for more detailed information",
        dest="training_paradigm",
    )

    paradigms = [
        pkg.name
        for pkg in pkgutil.walk_packages([eegt.__path__[0] + "/modules"])
        if pkg.name != "base"
    ]
    for name in paradigms:
        paradigm_parser = subparsers.add_parser(name)
        # add general training arguments
        add_default_args(paradigm_parser)
        # add arguments specific to the training paradigm
        paradigm = importlib.import_module(f"eegt.modules.{name}")
        if hasattr(paradigm, "add_arguments"):
            paradigm.add_arguments(paradigm_parser)

    args = parser.parse_args()
    main(args)
