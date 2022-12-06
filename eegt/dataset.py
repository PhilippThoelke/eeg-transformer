from os.path import exists, expanduser
import numpy as np
import pandas as pd
import pickle
import codecs
import torch
from torch.utils.data import Dataset, default_collate
from mne.filter import notch_filter, filter_data
import warnings


class RawDataset(Dataset):
    def __init__(self, args={}, **kwargs):
        if not isinstance(args, dict):
            args = args.__dict__
        args.update(kwargs)
        assert (
            "data_path" in args and "label_path" in args
        ), "Arguments require at least data_path and label_path to be defined"

        self.data_path = expanduser(args.get("data_path"))
        self.label_path = expanduser(args.get("label_path"))
        self.sample_rate = args.get("sample_rate")
        self.notch_freq = args.get("notch_freq")
        self.low_pass = args.get("low_pass")
        self.high_pass = args.get("high_pass")

        if not exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        if not exists(self.label_path):
            raise FileNotFoundError(f"Label file not found: {self.label_path}")

        if self.notch_freq or self.low_pass or self.high_pass:
            assert self.sample_rate, (
                "Sample rate must be specified to run a notch, "
                "low pass or high pass filter"
            )

        # memory map the raw data
        self.data = np.memmap(self.data_path, mode="r", dtype=np.float32)
        # load metadata CSV
        self.metadata = pd.read_csv(self.label_path, index_col=0)

        # potentially drop some conditions
        conditions = args.get("conditions", "all")
        if isinstance(conditions, list) and len(conditions) == 1:
            conditions = conditions[0]
        if conditions != "all":
            if not isinstance(conditions, list):
                conditions = [conditions]
            mask = np.zeros(self.metadata.shape[0], dtype=bool)
            for cond in conditions:
                mask = mask | (self.metadata["condition"] == cond)
            self.metadata = self.metadata[mask]

        # create unique indices and mappings for subjects, conditions and datasets
        self.subject_ids, self.subject_mapping = self.metadata["subject"].factorize()
        self.condition_ids, self.condition_mapping = self.metadata[
            "condition"
        ].factorize()
        self.dataset_ids, self.dataset_mapping = (
            self.metadata["subject"].apply(lambda x: x.split("-")[0]).factorize()
        )

        # decode channel positions
        self.metadata["channel_pos"] = self.metadata["channel_pos_pickle"].apply(
            lambda x: pickle.loads(codecs.decode(x.encode(), "base64")).astype(
                np.float32
            )
        )

    def class_weights(self, indices=None):
        classes = self.condition_ids
        if indices is not None:
            # only select a subset
            classes = classes[indices]
        counts = np.unique(classes, return_counts=True)[1]
        # normalize class counts to be centered around one
        counts = counts / counts.mean()
        # invert normalized counts -> larger weight for underrepresented classes
        return (1 / counts).tolist()

    def dataset_weights(self, indices=None):
        datasets = self.dataset_ids
        if indices is not None:
            # only select a subset
            datasets = datasets[indices]
        counts = np.unique(datasets, return_counts=True)[1]
        # normalize dataset counts to be centered around one
        counts = counts / counts.mean()
        # invert normalized counts -> larger weight for underrepresented dataset
        return (1 / counts).tolist()

    def sample_weights(self, indices=None):
        # take dataset, subject and condition into account
        groups = self.metadata["subject"].str.cat(self.metadata["condition"], "-")
        if indices is not None:
            groups = groups.iloc[indices]
        group_ids = groups.factorize()[0]

        # compute sample-wise weight
        _, inv, counts = np.unique(group_ids, return_inverse=True, return_counts=True)
        return 1 / counts[inv]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # read the current sample
        metadata = self.metadata.iloc[idx]
        x = np.array(self.data[metadata["start_idx"] : metadata["stop_idx"]])
        x = x.reshape(-1, metadata["num_channels"])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            # notch filter
            if self.notch_freq is not None:
                x = notch_filter(
                    x.T.astype(float),
                    self.sample_rate,
                    np.arange(self.notch_freq, self.sample_rate // 2, self.notch_freq),
                    verbose="warning",
                ).T.astype(np.float32)

            # band pass filter
            if self.low_pass is not None or self.high_pass is not None:
                x = filter_data(
                    x.T.astype(float),
                    self.sample_rate,
                    self.high_pass,
                    self.low_pass,
                    verbose="warning",
                ).T.astype(np.float32)

        return [
            torch.from_numpy(x),
            torch.from_numpy(metadata["channel_pos"]),
            torch.ones(x.shape[1], dtype=torch.bool),
            self.condition_ids[idx],
            self.subject_ids[idx],
            self.dataset_ids[idx],
        ]

    def collate(batch):
        channel_counts = torch.tensor([sample[0].size(1) for sample in batch])
        if not (channel_counts[0] == channel_counts).all():
            max_channels = channel_counts.max()
            # pad samples to the same channel count
            for i in torch.where(channel_counts != max_channels)[0]:
                num_channels = batch[i][0].size(1)
                # pad raw data
                padding = torch.zeros(batch[i][0].size(0), max_channels - num_channels)
                batch[i][0] = torch.cat([batch[i][0], padding], dim=1)
                # pad channel positions
                padding = torch.zeros(max_channels - num_channels, 3)
                batch[i][1] = torch.cat([batch[i][1], padding], dim=0)
                # pad mask
                padding = torch.zeros(max_channels - num_channels, dtype=torch.bool)
                batch[i][2] = torch.cat([batch[i][2], padding])

        return default_collate(batch)

    def id2subject(self, subject_id):
        # mapping from index to subject identifier
        return self.subject_mapping[subject_id]

    def id2condition(self, condition_id):
        # mapping from index to condition identifier
        return self.condition_mapping[condition_id]

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(len={len(self)}, n_conditions={len(self.condition_mapping)}, "
            f"n_subjects={len(self.subject_mapping)}, n_datasets={len(self.dataset_mapping)})"
        )
