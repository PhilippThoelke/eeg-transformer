from os.path import exists
import numpy as np
import pandas as pd
import pickle
import codecs
import torch
from torch.utils.data import Dataset
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

        self.data_path = args.get("data_path")
        self.label_path = args.get("label_path")
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

        with open(self.label_path, "r") as f:
            # parse shape information
            header = next(f)
            key, value = header.strip("#\n").split("=")
            assert key == "shape"
            data_shape = tuple(map(int, value.strip("()").split(", ")))

            # load the labels
            label = pd.read_csv(f, index_col=0, dtype=str)

        # memory map the raw data
        self.data = np.memmap(
            self.data_path, mode="r", dtype=np.float32, shape=data_shape
        )

        assert self.data.shape[0] == label.shape[0], (
            f"Sample count in the data ({self.data.shape[0]}) and "
            f"label file ({label.shape[0]}) doesn't match"
        )

        self.nchannels = data_shape[2]

        # potentially drop some conditions
        conditions = args.get("conditions", "all")
        if isinstance(conditions, list) and len(conditions) == 1:
            conditions = conditions[0]
        if conditions != "all":
            if not isinstance(conditions, list):
                conditions = [conditions]
            mask = np.zeros(label.shape[0], dtype=bool)
            for cond in conditions:
                mask = mask | (label["condition"] == cond)
            label = label[mask]

        # save indices of sample with select channels and conditions
        self.indices = label.index.values

        # create unique indices and mappings for subjects and conditions
        self.subject_ids, self.subject_mapping = label["subject"].factorize()
        self.condition_ids, self.condition_mapping = label["condition"].factorize()

        # store channel positions
        self.channel_positions = [
            pickle.loads(codecs.decode(pkl.encode(), "base64"))
            for pkl in label["channel_pos_pickle"].values
        ]

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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # read the current sample
        x = np.array(self.data[self.indices[idx]]).astype(float)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)

            # notch filter
            if self.notch_freq is not None:
                x = notch_filter(
                    x.T,
                    self.sample_rate,
                    np.arange(self.notch_freq, self.sample_rate // 2, self.notch_freq),
                    verbose="warning",
                ).T

            # band pass filter
            if self.low_pass is not None or self.high_pass is not None:
                x = filter_data(
                    x.T,
                    self.sample_rate,
                    self.high_pass,
                    self.low_pass,
                    verbose="warning",
                ).T

        # create the channel mask
        channel_pos = torch.from_numpy(self.channel_positions[idx].astype(np.float32))
        mask = torch.zeros(self.nchannels, dtype=bool)
        mask[: channel_pos.size(0)] = True

        # pad the channel positions tensor
        channel_padding = self.nchannels - channel_pos.size(0)
        if channel_padding > 0:
            padding = torch.zeros(channel_padding, channel_pos.size(1))
            channel_pos = torch.cat([channel_pos, padding], dim=0)

        return (
            torch.from_numpy(x.astype(np.float32)),
            channel_pos,
            mask,
            self.condition_ids[idx],
            self.subject_ids[idx],
        )

    def id2subject(self, subject_id):
        # mapping from index to subject identifier
        return self.subject_mapping[subject_id]

    def id2condition(self, condition_id):
        # mapping from index to condition identifier
        return self.condition_mapping[condition_id]
