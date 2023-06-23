import os
from abc import ABC, abstractmethod, abstractstaticmethod
from os.path import abspath, exists, expanduser, join
from typing import List, Optional, Tuple

import numpy as np
import torch
import webdataset as wds
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

from eegformer.utils import PreprocessingConfig


class Dataset(wds.WebDataset, ABC):
    """
    Abstract base class for datasets based on WebDataset.

    The dataset is expected to provide three attributes per sample:
    - `signal.npy`: The raw signal as a NumPy array of shape `(n_channels, n_samples)`.
    - `ch_pos.npy`: The channel positions as a NumPy array of shape `(n_channels, 3)`.
    - `label.cls`: The class label as a single integer.

    It is additionally required to implement the following methods:
    - `subject_ids`: A static method providing a list of all subject IDs for the particular dataset.
    - `num_classes: A static method providing the number of classes for the particular dataset.
    - `list_shards`: Return a list of file names for the dataset shards, which are expected to be
        stored in `self.processed_path`. The file names should not contain the full path but only
        the file name.
    - `prepare_data`: Download the dataset, preprocess it and save to disk as a WebDataset
        (single tar file or multiple shards). The raw data should be stored in `self.raw_path`
        and the processed data in `self.processed_path`.

    You can optionally implement `label2idx` and `idx2label` to convert between class names and
    indices.

    ### Args
        - `root`: The root directory of the dataset.
        - `preprocessing`: The preprocessing configuration.
        - `subjects`: The subjects to use. If `None`, all subjects are used.
        - `compute_data_stats`: Whether to compute data statistics for the dataset, !!! currently
            very computationally expensive !!!
        - `compute_data_metrics`: Whether to compute data metrics (mean, std, class weights) for
            the dataset, !!! currently very computationally expensive !!!
    """

    def __init__(
        self,
        root: str,
        preprocessing: PreprocessingConfig = None,
        subjects: List[int] = None,
        compute_data_metrics: bool = False,
    ):
        self._root = abspath(expanduser(join(root, self.__class__.__name__)))
        self.preprocessing = PreprocessingConfig() if preprocessing is None else preprocessing
        self._subjects = subjects

        # get a list of shards to load
        shards = [join(self.processed_path, shard) for shard in self.list_shards()]

        # if the dataset is not yet prepared, prepare it
        if not all(map(exists, shards)):
            # make sure the directories structure exists
            if not exists(self._root):
                os.mkdir(self._root)
            if not exists(self.raw_path):
                os.mkdir(self.raw_path)
            if not exists(self.processed_path):
                os.mkdir(self.processed_path)

            # download and preprocess the dataset
            self.prepare_data()

        # initialize the WebDataset
        super().__init__(wds.SimpleShardList(shards))

        # process the dataset
        self.decode()
        self.to_tuple("signal.npy", "ch_pos.npy", "label.cls")
        self.map_tuple(torch.from_numpy, torch.from_numpy)

        # compute class weights
        self._class_weights = None
        self._signal_mean = None
        self._signal_std = None
        if compute_data_metrics:
            ##############################################################################
            # TODO: optimize this, it's currently very slow to iterate the whole dataset #
            ##############################################################################
            import joblib

            # iterate dataset and extract signal mean, std and labels
            ds = wds.WebDataset(shards).decode().to_tuple("signal.npy", "label.cls").batched(128)
            dl = wds.WebLoader(ds, batch_size=None, shuffle=False, num_workers=joblib.cpu_count())
            means, stds, labels = tuple(
                zip(*[(sig.mean(), sig.std(), lbl) for sig, lbl in tqdm(dl, desc="Computing dataset metrics")])
            )

            # compute mean and std
            self._signal_mean = np.mean(means)
            self._signal_std = np.mean(stds)

            # compute class weights
            labels = np.concatenate(labels)
            self._class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
            self._class_weights = torch.from_numpy(self._class_weights).float()

    @staticmethod
    def collate_fn(samples) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        assert isinstance(samples[0], (list, tuple)), f"samples must be a list or tuple, got {type(samples[0])}"
        signal, ch_pos, label = zip(*samples)

        # get the number of channels in each sample
        sizes = torch.tensor([s.size(0) for s in signal])
        if sizes.min() == sizes.max():
            # all samples have the same number of channels, we don't need to pad or create a mask
            signal = torch.stack(signal)
            ch_pos = torch.stack(ch_pos)
            label = torch.tensor(label)
            return signal, ch_pos, None, label

        # pad the samples to the same channel count and create a mask
        signal = torch.nn.utils.rnn.pad_sequence(signal, batch_first=True)
        ch_pos = torch.nn.utils.rnn.pad_sequence(ch_pos, batch_first=True)
        mask = torch.arange(signal.size(1)).expand(signal.size(0), -1) < sizes.unsqueeze(1)
        label = torch.tensor(label)
        return signal, ch_pos, mask, label

    @abstractstaticmethod
    def subject_ids() -> List[int]:
        """
        List of all subject IDs for the particular dataset.
        """
        pass

    @abstractstaticmethod
    def num_classes() -> int:
        """
        Number of classes in the dataset.
        """
        pass

    @abstractmethod
    def list_shards(self) -> List[str]:
        """
        Lists the shards of the dataset.
        """
        pass

    @abstractmethod
    def prepare_data(self):
        """
        Downloads the dataset, preprocesses it and saves it to disk as a WebDataset (single tar file or multiple shards).
        """
        pass

    def label2idx(self, label: str) -> int:
        """
        Converts a class name to an index.
        """
        return NotImplemented

    def idx2label(self, idx: int) -> str:
        """
        Converts an index to a class name.
        """
        return NotImplemented

    @property
    def signal_mean(self) -> float:
        """
        Mean of the signal for the dataset.
        """
        return self._signal_mean

    @property
    def signal_std(self) -> float:
        """
        Standard deviation of the signal for the dataset.
        """
        return self._signal_std

    @property
    def class_weights(self) -> torch.Tensor:
        """
        Class weights for the dataset.
        """
        return self._class_weights

    @property
    def subjects(self) -> List[int]:
        """
        List of the subjects loaded in this dataset.
        """
        return self._subjects

    @property
    def raw_path(self) -> str:
        """
        Directory containing the raw dataset.
        """
        return join(self._root, "raw")

    @property
    def processed_path(self) -> str:
        """
        Directory containing the processed dataset.
        """
        return join(self._root, "processed")
