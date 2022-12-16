from copy import deepcopy
from functools import partial
import importlib
from glob import glob
import numpy as np
from functools import reduce
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import pytorch_lightning as pl
from eegt.dataset import RawDataset, ConcatDataset


def load_model(path):
    return load_lightning_module(path).model


def load_lightning_module(path):
    paradigm_name = torch.load(path, map_location="cpu")["hyper_parameters"][
        "training_paradigm"
    ]
    paradigm = importlib.import_module(f"eegt.modules.{paradigm_name}")
    module = paradigm.LightningModule.load_from_checkpoint(path, map_location="cpu")
    return module


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


def get_dataset(hparams):
    """
    Instantiate a dataset and potentially concatenate multiple RawDatasets.

    args:
        hparams (argparse.Namespace): hyperparameters from the training script or checkpoint

    returns:
        an instance of RawDataset or ConcatDataset
    """
    is_data_glob = "*" in hparams.data_path
    is_label_glob = "*" in hparams.label_path

    if not is_data_glob and not is_label_glob:
        # no glob strings, return a single dataset
        return RawDataset(hparams)
    if is_data_glob and is_label_glob:
        # data and label path is a glob string
        dpaths = sorted(glob(hparams.data_path))
        lpaths = sorted(glob(hparams.label_path))
        assert len(dpaths) == len(lpaths), (
            f"Found {len(dpaths)} data paths and {len(lpaths)} label paths "
            "but they should be the same."
        )

        # load individual datasets
        datasets = []
        for dp, lp in zip(dpaths, lpaths):
            datasets.append(RawDataset(hparams, data_path=dp, label_path=lp))
        return ConcatDataset(datasets)

    # one of data or label path was a glob string
    raise RuntimeError(
        "Either both or none of data and label path should be a glob string. "
        f"Got data_path: {'glob' if is_data_glob else 'not glob'} and "
        f"label_path: {'glob' if is_label_glob else 'not glob'}"
    )


def get_dataloader(hparams, full_dataset, indices=None, training=False, **kwargs):
    """
    Instantiate a preconfigured DataLoader

    args:
        hparams (argparse.Namespace): hyperparameters from the training script or checkpoint
        full_dataset (RawDataset): the dataset object to be used
        indices (Tensor, optional): if set, use a subset of the data given by the indices
        training (bool, optional): whether to configure the DataLoader for training or validation
        **kwargs: additional arguments overwriting hparams fields

    returns:
        a fully configured dataloader
    """
    # update hparams with kwargs
    hparams = deepcopy(hparams)
    hparams.__dict__.update(kwargs)

    # use only a subset of the data
    if indices is not None:
        data = Subset(full_dataset, indices)
    else:
        data = full_dataset

    # prepare the data collate function
    collate_fn = RawDataset.collate
    # potentially wrap the collate function in a paradigm-specific way
    paradigm = importlib.import_module(f"eegt.modules.{hparams.training_paradigm}")
    if hasattr(paradigm, "collate_decorator"):
        collate_fn = paradigm.collate_decorator(collate_fn, hparams, training)
        assert collate_fn is not None, (
            f"{paradigm.__name__}.collate_decorator "
            "did not return a collate function"
        )

    # create sampler
    sampler = None
    if hparams.weighted_sampler:
        assert hasattr(full_dataset, "sample_weights"), (
            "The dataset doesn't implement the sample_weights function, "
            "please disable weighted sampling."
        )
        sampler = WeightedRandomSampler(full_dataset.sample_weights(indices), len(data))

    # the argument prefetch_factor is only allowed if num_workers > 0
    kwargs = {}
    if hparams.num_workers > 0:
        kwargs["prefetch_factor"] = 4

    # instantiate dataloader
    return DataLoader(
        data,
        batch_size=hparams.batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=training if sampler is None else False,
        num_workers=hparams.num_workers,
        **kwargs,
    )


class Attention:
    def __init__(self, model):
        if isinstance(model, pl.LightningModule):
            self.encoder_model = model.model
        else:
            self.encoder_model = model
        self.handles = []
        self.attn = []

    def get(self):
        return torch.stack(self.attn, dim=1)

    def __enter__(self):
        def attention_hook(module, input, output, attn):
            attn.append(output[1])

        for encoder_layer in self.encoder_model.encoder_layers:
            hook = partial(attention_hook, attn=self.attn)
            self.handles.append(encoder_layer.mha.register_forward_hook(hook))
        return self

    def __exit__(self, type, value, traceback):
        for handle in self.handles:
            handle.remove()


def rollout(attn, head_fuse="max", only_class=False):
    assert (
        attn.ndim == 5
    ), "expected attn to have 5 dimensions (batch x layers x heads x tokens x tokens)"
    assert head_fuse in [
        "max",
        "mean",
        "min",
    ], "expected head_fuse to be one of max, mean, min"
    fuse_fns = {"max": torch.max, "mean": torch.mean, "min": torch.min}

    attn = fuse_fns[head_fuse](attn, dim=2)
    if not isinstance(attn, torch.Tensor):
        attn = attn.values

    result = torch.eye(attn.size(-1))
    for layer in attn.permute(1, 0, 2, 3).flip(dims=(0,)):
        result = layer @ result

    if only_class:
        result = result[:, 0, 1:]
        return result / result.max()
    return result
