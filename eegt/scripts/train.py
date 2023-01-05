import pkgutil
import importlib
from distutils import util
import argparse
from os import makedirs, path
import numpy as np
import torch
import pytorch_lightning as pl
import eegt
from eegt.modules.base import lr_schedules
from eegt import utils
from eegt.dataset import RawDataset


def main(args):
    # load data and split into train and validation set
    data = utils.get_dataset(args)
    idx_train, idx_val = utils.split_data(data, args.val_subject_ratio)
    # store the size of a single token
    args.token_size = data[0][0].shape[0]

    # fetch class and dataset weights
    lightning_module_kwargs = {}
    if hasattr(data, "dataset_weights"):
        lightning_module_kwargs["dataset_weights"] = data.dataset_weights(idx_train)
    if hasattr(data, "class_weights"):
        lightning_module_kwargs["class_weights"] = data.class_weights(idx_train)

    # handle model loading
    if args.load_model is not None:
        pl.utilities.rank_zero_info(f"Using pretrained encoder: {args.load_model}")
        lightning_module_kwargs["model"] = utils.load_model(args.load_model)

    # instantiate PyTorch-Lightning module
    paradigm = importlib.import_module(f"eegt.modules.{args.training_paradigm}")
    module = paradigm.LightningModule(args, **lightning_module_kwargs)

    # create dataloaders
    train_dl = utils.get_dataloader(args, data, indices=idx_train, training=True)
    val_dl = utils.get_dataloader(args, data, indices=idx_val, training=False)

    # define trainer instance
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_epochs=-1,
        callbacks=[
            pl.callbacks.EarlyStopping(
                "val_loss", patience=args.early_stopping_patience, mode="min"
            ),
            pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min"),
        ],
        accumulate_grad_batches=args.gradient_accumulation,
        limit_train_batches=args.train_batches,
        limit_val_batches=args.val_batches,
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
            "--train-batches",
            default=None,
            type=int,
            help="max number of training batches per epoch",
        )
        parser.add_argument(
            "--val-batches",
            default=None,
            type=int,
            help="max number of validation batches per epoch",
        )
        parser.add_argument(
            "--val-subject-ratio",
            default=0.15,
            type=float,
            help="ratio of subjects to be used for validation",
        )
        parser.add_argument(
            "--load-model",
            default=None,
            type=str,
            help="path to a checkpoint file (only loads the encoder model)",
        )
        parser.add_argument(
            "--freeze-steps",
            default=1000,
            type=int,
            help="number of steps before the pretrained model is trained (0 to disable freezing, -1 to never unfreeze)",
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
            "--num-workers",
            default=8,
            type=int,
            help="number of workers for loading the data",
        )
        parser.add_argument(
            "--weighted-sampler",
            default=True,
            type=lambda x: bool(util.strtobool(x)),
            help="if true, use a weighted random sampler to counteract imbalance",
        )
        parser.add_argument(
            "--gradient-accumulation",
            default=None,
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
            "--lr-schedule",
            default="cosine",
            type=str,
            choices=list(lr_schedules.keys()),
            help="learning rate schedule",
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
