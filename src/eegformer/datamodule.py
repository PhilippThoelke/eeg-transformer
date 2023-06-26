import importlib
import inspect
from typing import List, Optional, Union

import lightning.pytorch as pl
import torch
from webdataset import WebLoader

from eegformer import augmentations
from eegformer.utils import PreprocessingConfig


def load_dataset_kwargs(dataclass: type, dataset_kwargs: dict, exclude: List[str]) -> dict:
    """
    Load the dataset keyword arguments and update them with the given dataset kwargs.

    #### Args
        - `dataclass` (type): The dataset class.
        - `dataset_kwargs` (dict): Keyword arguments used to instantiate the dataset.
        - `exclude` (List[str]): A list of keyword arguments to exclude from the default kwargs.

    #### Returns
        dict: A dictionary of dataset keyword arguments.
    """
    dataset_defaults = dict(inspect.signature(dataclass).parameters)
    dataset_defaults = {k: v.default for k, v in dataset_defaults.items() if v.default is not inspect.Parameter.empty}
    dataset_defaults = {k: v for k, v in dataset_defaults.items() if k not in exclude}
    dataset_defaults.update(dataset_kwargs)
    return dataset_defaults


def parse_subjects(subjects: Union[int, str], available: List[int] = []) -> List[int]:
    """
    Parse the subjects to be used in a split.

    #### Args
        - `subjects` (Union[int, str]): A number of subjects to be used in a split, or a string specifying
            the subjects to be used in a split.
        - `available` (List[int]): A list of subjects that are available for use in this split.

    #### Returns
        A list of subjects to be used in a split.
    """

    # return random subject split if subjects is an int
    if isinstance(subjects, int):
        if len(available) < subjects:
            raise ValueError(f"Only {len(available)} subjects available, {subjects} requested.")

        result = [available[i] for i in torch.randperm(len(available))]
        return result[:subjects]

    # return the chosen list of subjects if subjects is a string
    if "," in subjects:
        # recursively parse subjects
        return sum([parse_subjects(s) for s in subjects.split(",") if len(s.strip()) > 0], [])

    if "-" in subjects:
        start, end = subjects.split("-")
        result = list(range(int(start), int(end) + 1))
    else:
        result = [int(subjects)]

    unavailable_match = [i for i in result if i not in available]
    if len(unavailable_match) > 0:
        raise ValueError(f"Subjects {unavailable_match} have been used in another split.")
    return result


class DataModule(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for datasets using the webdataset package.

    ### Args
        - `dataclass` (Union[str, type]): The dataset class.
        - `dataset_kwargs` (dict): Keyword arguments used to instantiate the dataset.
        - `preprocessing` (PreprocessingConfig): The preprocessing configuration.
        - `augmentations` (Optional[BaseTransform]): Optional list of augmentations to be applied to the data.
        - `train_subjects` (Union[int, str]): Number of subjects in training split, or string specifying subject IDs.
        - `val_subjects` (Union[int, str]): Number of subjects in validation split, or string specifying subject IDs.
        - `test_subjects` (Union[int, str]): Number of subjects in test split, or string specifying subject IDs.
        - `batch_size` (int): The batch size.
        - `num_workers` (int): The number of workers for loading the data.
        - `root` (str): The root directory where the dataset will be stored.

    ### Examples:
        >>> from eegformer.dataset import PhysionetMotorImagery
        >>> data = PhysionetMotorImagery(train_subjects="1-20", val_subjects=5, test_subjects=2)
    """

    def __init__(
        self,
        dataclass: Union[str, type],
        dataset_kwargs: dict,
        preprocessing: Optional[PreprocessingConfig] = None,
        augmentations: Optional[augmentations.BaseTransform] = None,
        train_subjects: Union[int, str] = 80,
        val_subjects: Union[int, str] = 20,
        test_subjects: Union[int, str] = 3,
        batch_size: int = 512,
        num_workers: int = 16,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.batch_size = batch_size
        self.augmentations = augmentations
        self.num_workers = num_workers

        # configure dataset
        self.dataclass = dataclass
        if isinstance(dataclass, str):
            self.dataclass = getattr(importlib.import_module("eegformer.datasets"), dataclass.split(".")[-1])

        # configure dataset kwargs
        dataset_kwargs["preprocessing"] = preprocessing
        dataset_kwargs = load_dataset_kwargs(
            self.dataclass, dataset_kwargs, exclude=["subjects", "compute_class_weights"]
        )
        self.dataset_kwargs = dataset_kwargs

        # create lists of subjects for each split
        available_subjects = self.dataclass.subject_ids(**dataset_kwargs)
        self.train_subjects = parse_subjects(train_subjects, available=available_subjects)
        available_subjects = [s for s in available_subjects if s not in self.train_subjects]
        self.val_subjects = parse_subjects(val_subjects, available=available_subjects)
        available_subjects = [s for s in available_subjects if s not in self.val_subjects]
        self.test_subjects = parse_subjects(test_subjects, available=available_subjects)

    def prepare_data(self):
        """
        Instantiate the dataset once to download the data.
        """
        self.dataclass(
            subjects=self.train_subjects + self.val_subjects + self.test_subjects,
            **self.dataset_kwargs,
        )

    @property
    def num_classes(self) -> int:
        """
        Return the number of classes.
        """
        return self.dataclass.num_classes(**self.dataset_kwargs)

    @property
    def class_weights(self):
        """
        Return the training set class weights.
        """
        return self.train_data.class_weights

    def setup(self, stage: str):
        """
        Instantiate the dataset for each split.
        """
        if stage == "fit":
            self.train_data = self.dataclass(
                subjects=self.train_subjects, compute_class_weights=True, **self.dataset_kwargs
            )
            self.val_data = self.dataclass(subjects=self.val_subjects, **self.dataset_kwargs)
            self.test_data = self.dataclass(subjects=self.test_subjects, **self.dataset_kwargs)
        elif stage == "val":
            self.val_data = self.dataclass(subjects=self.val_subjects, **self.dataset_kwargs)
        elif stage == "test":
            self.test_data = self.dataclass(subjects=self.test_subjects, **self.dataset_kwargs)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self) -> WebLoader:
        """
        Return the training dataloader.
        """
        data = self.train_data.map(self.augmentations) if self.augmentations else self.train_data
        return WebLoader(
            data.shuffle(1000).batched(self.batch_size, collation_fn=self.train_data.collate_fn),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> WebLoader:
        """
        Return the training dataloader.
        """
        return WebLoader(
            self.val_data.batched(self.batch_size, collation_fn=self.val_data.collate_fn),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> WebLoader:
        """
        Return the training dataloader.
        """
        return WebLoader(
            self.test_data.batched(self.batch_size, collation_fn=self.test_data.collate_fn),
            batch_size=None,
            num_workers=self.num_workers,
            pin_memory=True,
        )
