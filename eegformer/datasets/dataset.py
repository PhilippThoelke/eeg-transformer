import os
from abc import ABC, abstractmethod, abstractstaticmethod
from os.path import abspath, exists, expanduser, join
from typing import List

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
        - `compute_class_weights`: Whether to compute class weights for the dataset, !!! currently
            very computationally expensive !!!
    """

    def __init__(
        self,
        root: str,
        preprocessing: PreprocessingConfig = None,
        subjects: List[int] = None,
        compute_class_weights: bool = False,
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
        if compute_class_weights:
            ##############################################################################
            # TODO: optimize this, it's currently very slow to iterate the whole dataset #
            ##############################################################################
            import joblib

            ds = wds.WebDataset(shards).to_tuple("label.cls").map_tuple(lambda x: int(x)).batched(128)
            dl = wds.WebLoader(ds, batch_size=None, shuffle=False, num_workers=joblib.cpu_count())
            labels = np.concatenate([lbl[0] for lbl in tqdm(dl, desc="Computing class weights")])
            self._class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
            self._class_weights = torch.from_numpy(self._class_weights).float()

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
