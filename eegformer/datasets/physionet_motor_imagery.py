import os
from enum import Flag
from glob import glob
from os.path import abspath, basename, exists, expanduser, join
from typing import Callable, List, Optional, Union

import lightning.pytorch as pl
import mne
import numpy as np
import torch
from joblib import Parallel, delayed
from mne.datasets import eegbci
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from eegformer import augmentations
from eegformer.utils import PreprocessingConfig, TorchEpoch, get_channel_pos, preprocess


class PhysionetMotorImageryTask(Flag):
    BASELINE = 1
    MOTOR_EXECUTION_LEFT_RIGHT = 2
    MOTOR_EXECUTION_HANDS_FEET = 4
    MOTOR_IMAGERY_LEFT_RIGHT = 8
    MOTOR_IMAGERY_HANDS_FEET = 16
    ALL = (
        BASELINE
        | MOTOR_EXECUTION_LEFT_RIGHT
        | MOTOR_EXECUTION_HANDS_FEET
        | MOTOR_IMAGERY_LEFT_RIGHT
        | MOTOR_IMAGERY_HANDS_FEET
    )

    @staticmethod
    def get_runs(task) -> List[int]:
        runs = []
        if (task & PhysionetMotorImageryTask.BASELINE).value > 0:
            runs += [1, 2]
        if (task & PhysionetMotorImageryTask.MOTOR_EXECUTION_LEFT_RIGHT).value > 0:
            runs += [3, 7, 11]
        if (task & PhysionetMotorImageryTask.MOTOR_EXECUTION_HANDS_FEET).value > 0:
            runs += [5, 9, 13]
        if (task & PhysionetMotorImageryTask.MOTOR_IMAGERY_LEFT_RIGHT).value > 0:
            runs += [4, 8, 12]
        if (task & PhysionetMotorImageryTask.MOTOR_IMAGERY_HANDS_FEET).value > 0:
            runs += [6, 10, 14]
        return runs


# maps (run, annot) to a string label
LABEL_MAPPING = {
    (1, "T0"): "baseline_open",
    (2, "T0"): "baseline_closed",
    (3, "T1"): "execution_left",
    (3, "T2"): "execution_right",
    (4, "T1"): "imagery_left",
    (4, "T2"): "imagery_right",
    (5, "T1"): "execution_hands",
    (5, "T2"): "execution_feet",
    (6, "T1"): "imagery_hands",
    (6, "T2"): "imagery_feet",
    (7, "T1"): "execution_left",
    (7, "T2"): "execution_right",
    (8, "T1"): "imagery_left",
    (8, "T2"): "imagery_right",
    (9, "T1"): "execution_hands",
    (9, "T2"): "execution_feet",
    (10, "T1"): "imagery_hands",
    (10, "T2"): "imagery_feet",
    (11, "T1"): "execution_left",
    (11, "T2"): "execution_right",
    (12, "T1"): "imagery_left",
    (12, "T2"): "imagery_right",
    (13, "T1"): "execution_hands",
    (13, "T2"): "execution_feet",
    (14, "T1"): "imagery_hands",
    (14, "T2"): "imagery_feet",
}


def parse_subjects(subjects: Union[int, str], unavailable: List[int] = []) -> List[int]:
    """
    Parse the subjects to be used in a split.

    Args:
        subjects (Union[int, str]): A number of subjects to be used in a split, or a string specifying the subjects to be used in a split.
        unavailable: A list of subjects that have already been used in another split.

    Returns:
        A list of subjects to be used in a split.
    """

    # return random subject split if subjects is an int
    if isinstance(subjects, int):
        result = [i for i in (torch.randperm(109) + 1).tolist() if i not in unavailable]
        result = result[:subjects]

        if len(result) != subjects:
            raise ValueError(f"Only {len(result)} subjects available, {subjects} requested.")
        return result

    # return the chosen list of subjects if subjects is a string
    if "," in subjects:
        # recursively parse subjects
        return sum([parse_subjects(s) for s in subjects.split(",") if len(s.strip()) > 0], [])

    if "-" in subjects:
        start, end = subjects.split("-")
        result = list(range(int(start), int(end) + 1))
    else:
        result = [int(subjects)]

    unavailable_match = [i for i in result if i in unavailable]
    if len(unavailable_match) > 0:
        raise ValueError(f"Subjects {unavailable_match} have been used in another split.")
    return result


def preprocess_subject(sub: int, runs: List[int], path: str, config: PreprocessingConfig, force: bool = False) -> None:
    """
    Preprocess a subject. Stores the preprocessed data under `path`/processed.

    Args:
        sub (int): The subject ID.
        runs (List[int]): The runs to be used.
        path (str): The path to the dataset.
        config (PreprocessingConfig): The preprocessing configuration.
        force (bool): Whether to force preprocessing even if the file already exists.
    """
    for run in runs:
        # check if file already exists
        processed_path = join(path, "processed", f"sub-{sub}_run-{run}.pt")
        if not force and exists(processed_path):
            continue

        # (down)load raw data
        paths = eegbci.load_data(sub, run, path, update_path=False, verbose="ERROR")
        assert len(paths) == 1, f"Found more than one file for subject {sub} run {run}."

        raw = mne.io.read_raw(paths[0], verbose="ERROR", preload=True)
        eegbci.standardize(raw)
        raw = raw.set_eeg_reference("average", verbose="ERROR")  # reference to average
        raw = raw.drop_channels(["T9", "T10", "Iz"])  # drop reference channels
        raw.set_montage(mne.channels.make_standard_montage("brainproducts-RNP-BA-128"))  # set montage
        raw.info["line_freq"] = 60  # set power line frequency

        # preprocess
        raw = preprocess(raw, config)

        # crop raw according to annotations and assign labels
        samples = []
        for annot in raw.annotations:
            label_key = (run, annot["description"])

            # skip annotations that are not in the label mapping
            if label_key not in LABEL_MAPPING:
                continue

            # crop raw to annotation
            tmax = min(annot["onset"] + annot["duration"], raw.tmax)
            raw_segment = raw.copy().crop(tmin=annot["onset"], tmax=tmax)

            # assemble sample
            ch_pos = get_channel_pos(raw_segment)
            sfreq = raw_segment.info["sfreq"]
            label = LABEL_MAPPING[label_key]
            raw_segment = torch.from_numpy(raw_segment.get_data()).float()
            samples.append(TorchEpoch(raw_segment, ch_pos, sfreq, label))

        # save preprocessed data
        torch.save(samples, processed_path)


class PhysionetMotorImageryDataset(Dataset):
    def __init__(
        self, processed_files: List[str], epoch_length: float, overlap: float, transform: Optional[Callable] = None
    ):
        """
        Args:
            processed_files (List[str]): List of paths to preprocessed files.
            epoch_length (float): Length of epoch in seconds.
            overlap (float): Overlap between epochs in seconds.
            transform (Optional[Callable]): Optional transform to be applied to the data.
        """
        self.processed_files = processed_files
        self.epoch_length = epoch_length
        self.overlap = overlap
        self.transform = transform

        self.sample_index = []
        labels = []
        class_distribution = {}

        # index the data
        for processed_file in self.processed_files:
            samples = torch.load(processed_file)
            if len(samples) == 0:
                pl.utilities.rank_zero_warn(f"File {processed_file} contains no samples.")

            # add all possible epochs to the index
            for i, sample in enumerate(samples):
                duration = sample.signal.shape[1] / sample.sfreq
                num_epochs = int((duration - self.epoch_length) / (self.epoch_length - self.overlap)) + 1
                # compute epoch start times
                start_idxs = (np.arange(num_epochs) * (self.epoch_length - self.overlap) * sample.sfreq).astype(int)
                # add to index
                self.sample_index.extend(
                    list(
                        zip(
                            [processed_file] * num_epochs,  # epoch file path
                            [i] * num_epochs,  # sample index
                            start_idxs,  # epoch start time
                        )
                    )
                )
                labels.extend([sample.label] * num_epochs)
                # update class distribution
                if sample.label not in class_distribution:
                    class_distribution[sample.label] = num_epochs
                else:
                    class_distribution[sample.label] += num_epochs

        # create label mapping
        self.label_to_idx = {label: idx for idx, label in enumerate(sorted(set(labels)))}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        # compute class weights
        class_weights = compute_class_weight("balanced", classes=np.unique(labels), y=labels)
        self._class_weights = torch.from_numpy(class_weights).float()

    @property
    def class_weights(self):
        return self._class_weights

    def __len__(self):
        return len(self.sample_index)

    def __getitem__(self, idx):
        processed_file, sample_idx, start_idx = self.sample_index[idx]
        sample = torch.load(processed_file)[sample_idx]
        signal = sample.signal[:, start_idx : start_idx + int(self.epoch_length * sample.sfreq)]
        ch_pos = sample.ch_pos

        if self.transform is not None:
            signal, ch_pos = self.transform(signal, ch_pos)
        return signal, ch_pos, self.label_to_idx[sample.label]


class PhysionetMotorImagery(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the Physionet Motor Imagery dataset.

    Args:
        task (PhysionetMotorImageryTask): The task to be used in the dataset.
        preprocessing_config (PreprocessingConfig): The preprocessing configuration.
        augmentations (Optional[BaseTransform]): Optional list of augmentations to be applied to the data.
        train_subjects (Union[int, str]): Number of subjects in training split, or string specifying subject IDs.
        val_subjects (Union[int, str]): Number of subjects in validation split, or string specifying subject IDs.
        test_subjects (Union[int, str]): Number of subjects in test split, or string specifying subject IDs.
        epoch_length (float): Length of epoch in seconds.
        epoch_overlap (float): Overlap between epochs in seconds.
        batch_size (int): The batch size.
        num_workers (int): The number of workers for loading the data.
        root (str): The root directory where the dataset will be stored.
        force_preprocessing (bool): Whether to force preprocessing even if the file already exists.
        n_jobs (int): The number of jobs for downloading and preprocessing the data.

    Examples:
        >>> from eegformer.dataset import PhysionetMotorImagery
        >>> data = PhysionetMotorImagery(train_subjects="1-20", val_subjects=5, test_subjects=2)
    """

    def __init__(
        self,
        task: PhysionetMotorImageryTask = PhysionetMotorImageryTask.ALL,
        preprocessing_config: PreprocessingConfig = PreprocessingConfig(resample=160),
        augmentations: Optional[augmentations.BaseTransform] = None,
        train_subjects: Union[int, str] = 80,
        val_subjects: Union[int, str] = 20,
        test_subjects: Union[int, str] = 9,
        epoch_length: float = 2.0,
        epoch_overlap: float = 0.5,
        batch_size: int = 512,
        num_workers: int = 16,
        root: str = "data",
        force_preprocessing: bool = False,
        n_jobs: int = -1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.task = task
        self.preprocessing_config = preprocessing_config

        self.train_subjects = parse_subjects(train_subjects)
        self.val_subjects = parse_subjects(val_subjects, unavailable=self.train_subjects)
        self.test_subjects = parse_subjects(test_subjects, unavailable=self.train_subjects + self.val_subjects)

        self.epoch_length = epoch_length
        self.epoch_overlap = epoch_overlap
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.num_workers = num_workers
        self.root = join(abspath(expanduser(root)), "physionet-motor-imagery")
        self.force_preprocessing = force_preprocessing
        self.n_jobs = n_jobs

    def prepare_data(self):
        """
        Download the dataset, preprocess it, extract epochs and save them to `self.root`/processed.
        """
        # create results directory
        os.makedirs(join(self.root, "processed"), exist_ok=True)

        # collect run IDs for the current task
        runs = PhysionetMotorImageryTask.get_runs(self.task)

        # download and preprocess the dataset and store it under `self.root`/processed
        Parallel(n_jobs=self.n_jobs)(
            delayed(preprocess_subject)(sub, runs, self.root, self.preprocessing_config, force=self.force_preprocessing)
            for sub in tqdm(self.train_subjects + self.val_subjects + self.test_subjects, desc="Checking data")
        )

    @staticmethod
    def num_classes(**kwargs) -> int:
        if "task" not in kwargs:
            raise ValueError("Missing argument: 'task'")

        n = 0
        if (kwargs["task"] & PhysionetMotorImageryTask.BASELINE).value > 0:
            n += 2
        if (kwargs["task"] & PhysionetMotorImageryTask.MOTOR_EXECUTION_LEFT_RIGHT).value > 0:
            n += 2
        if (kwargs["task"] & PhysionetMotorImageryTask.MOTOR_EXECUTION_HANDS_FEET).value > 0:
            n += 2
        if (kwargs["task"] & PhysionetMotorImageryTask.MOTOR_IMAGERY_LEFT_RIGHT).value > 0:
            n += 2
        if (kwargs["task"] & PhysionetMotorImageryTask.MOTOR_IMAGERY_HANDS_FEET).value > 0:
            n += 2
        return n

    @property
    def class_names(self):
        return list(self.train_data.label_to_idx.keys())

    def class_weights(self, stage: str):
        """
        Return the class weights for the given stage.
        """
        if stage == "train":
            return self.train_data.class_weights
        elif stage == "val":
            return self.val_data.class_weights
        elif stage == "test":
            return self.test_data.class_weights
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def setup(self, stage: str):
        """
        Load the dataset from `self.root`/processed.
        """
        # collect run IDs for the current task
        runs = PhysionetMotorImageryTask.get_runs(self.task)

        if stage == "fit":
            train_files = self.list_processed_files(self.train_subjects, runs)
            val_files = self.list_processed_files(self.val_subjects, runs)
            test_files = self.list_processed_files(self.test_subjects, runs)

            self.train_data = PhysionetMotorImageryDataset(
                train_files, self.epoch_length, self.epoch_overlap, transform=self.augmentations
            )
            self.val_data = PhysionetMotorImageryDataset(val_files, self.epoch_length, self.epoch_overlap)
            self.test_data = PhysionetMotorImageryDataset(test_files, self.epoch_length, self.epoch_overlap)
        elif stage == "val":
            val_files = self.list_processed_files(self.val_subjects, runs)
            self.val_data = PhysionetMotorImageryDataset(val_files, self.epoch_length, self.epoch_overlap)
        elif stage == "test":
            test_files = self.list_processed_files(self.test_subjects, runs)
            self.test_data = PhysionetMotorImageryDataset(test_files, self.epoch_length, self.epoch_overlap)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self) -> DataLoader:
        """
        Return the training dataloader.
        """
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """
        Return the training dataloader.
        """
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """
        Return the training dataloader.
        """
        return DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def list_processed_files(self, subjects: List[int] = None, runs: List[int] = None) -> List[str]:
        """
        List all processed files for the given subjects and runs.

        Args:
            subjects (List[int], optional): List of subjects to include. Defaults to all subjects.
            runs (List[int], optional): List of runs to include. Defaults to all runs.

        Returns:
            List[str]: A list of file paths.
        """
        paths = []
        for path in glob(join(self.root, "processed", "*.pt")):
            sub, run = basename(path).split(".")[0].split("_")
            sub, run = int(sub.split("-")[1]), int(run.split("-")[1])
            if subjects is not None and sub not in subjects:
                continue
            if runs is not None and run not in runs:
                continue
            paths.append(path)
        return paths


if __name__ == "__main__":
    pl.seed_everything(42)
    data = PhysionetMotorImagery(
        task=PhysionetMotorImageryTask.BASELINE,
        train_subjects="1-2",
        val_subjects="3",
        test_subjects="4",
        force_preprocessing=True,
        root="~/data",
    )
    data.prepare_data()
    data.setup("fit")
