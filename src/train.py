import argparse
from functools import reduce
from os import makedirs, path
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import pytorch_lightning as pl
from dataset import RawDataset
from module import TransformerModule


def split_data(data, val_subject_ratio):
    # split the data by subjects
    unique_subject_ids = np.unique(data.subject_ids)
    num_val_subjects = int(len(unique_subject_ids) * val_subject_ratio)
    subject_idxs = np.random.choice(unique_subject_ids, num_val_subjects, replace=False)
    val_mask = reduce(np.bitwise_or, [data.subject_ids == i for i in subject_idxs])
    # return train/val indices
    return np.where(~val_mask)[0], np.where(val_mask)[0]


def main(args):
    # load data
    data = RawDataset(
        args.data_path,
        args.label_path,
        args.epoch_length,
        args.num_channels,
        stages=args.stages,
        conditions=args.conditions,
        sample_rate=args.sample_rate,
        notch_freq=args.notch_freq,
        low_pass=args.low_pass,
        high_pass=args.high_pass,
    )
    idx_train, idx_val = split_data(data, args.val_subject_ratio)

    # make sure args contains a list of conditions and stages, not "all"
    args.conditions = data.condition_mapping.tolist()
    args.stages = data.stage_mapping.tolist()

    # train subset
    train_data = Subset(data, idx_train)
    train_dl = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=8,
        prefetch_factor=4,
    )
    # store training class weights for use inside the lightning module
    args.class_weights = data.class_weights(idx_train)

    # val subset
    val_data = Subset(data, idx_val)
    val_dl = DataLoader(
        val_data, batch_size=args.batch_size, num_workers=8, prefetch_factor=4
    )

    # compute data mean and std
    mean, std = 0, 1
    if args.standardize:
        result = [
            (sample[0].mean(), sample[0].std())
            for sample in tqdm(
                DataLoader(train_data, batch_size=256, num_workers=4),
                desc="extracting mean and standard deviation",
            )
        ]
        means, stds = zip(*result)
        mean, std = torch.tensor(means).mean(), torch.tensor(stds).mean()

    # define model
    module = TransformerModule(args, mean, std)

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
        "--epoch-length",
        type=int,
        required=True,
        help="number of samples in each epoch",
    )
    parser.add_argument(
        "--num-channels",
        type=int,
        required=True,
        help="number of channels in the raw EEG",
    )
    parser.add_argument(
        "--learning-rate",
        default=5e-3,
        type=float,
        help="base learning rate",
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=100,
        type=int,
        help="number of epochs to continue training if val loss doesn't improve anymore",
    )
    parser.add_argument(
        "--batch-size",
        default=32,
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
        "--num-tokens",
        default=4,
        type=int,
        help="number of temporal tokens the EEG is split into",
    )
    parser.add_argument(
        "--embedding-dim",
        default=64,
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
        default=0.3,
        type=float,
        help="dropout ratio",
    )
    parser.add_argument(
        "--token-dropout",
        default=0.1,
        type=float,
        help="dropout ratio for entire tokens",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.1,
        type=float,
        help="weight decay",
    )
    parser.add_argument(
        "--shuffle-tokens",
        default="none",
        type=str,
        choices=["none", "channels", "temporal", "all"],
        help="type of random reordering of tokens",
    )
    parser.add_argument(
        "--warmup-steps",
        default=500,
        type=int,
        help="number of steps for lr warmup",
    )
    parser.add_argument(
        "--max-epochs",
        default=300,
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
        "--ignore-channels",
        default=[],
        type=int,
        help="list of channel indices to ignore",
        nargs="+",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        nargs="+",
        help="list of stages to use",
    )
    parser.add_argument(
        "--conditions",
        default="all",
        type=str,
        help="list of conditions to use",
        nargs="+",
    )
    parser.add_argument(
        "--standardize",
        default=True,
        type=bool,
        help="whether to standardize the data using a global mean and std",
    )
    parser.add_argument(
        "--used-data-length",
        default=None,
        type=int,
        help="amount of samples to actually use from each epoch (default: use full epoch)",
    )

    args = parser.parse_args()
    main(args)
