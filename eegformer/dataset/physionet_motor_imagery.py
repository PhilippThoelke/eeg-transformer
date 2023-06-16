import os
from enum import Flag
from os.path import abspath, expanduser, join
from typing import List, Union

import lightning.pytorch as pl
import mne
import torch
from joblib import Parallel, delayed
from mne.datasets import eegbci
from tqdm import tqdm

from eegformer.utils import PreprocessingConfig, preprocess


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
        return sum([parse_subjects(s) for s in subjects.split(",")], [])

    if "-" in subjects:
        start, end = subjects.split("-")
        result = list(range(int(start), int(end) + 1))
    else:
        result = [int(subjects)]

    unavailable_match = [i for i in result if i in unavailable]
    if len(unavailable_match) > 0:
        raise ValueError(f"Subjects {unavailable_match} have been used in another split.")
    return result


def preprocess_subject(sub: int, runs: List[int], path: str, config: PreprocessingConfig) -> None:
    """
    Preprocess a subject. Stores the preprocessed data under `path`/processed.

    Args:
        sub (int): The subject ID.
        runs (List[int]): The runs to be used.
        path (str): The path to the dataset.
        config (PreprocessingConfig): The preprocessing configuration.
    """
    segments = {}
    for run in runs:
        paths = eegbci.load_data(sub, run, path, update_path=False, verbose="ERROR")
        assert len(paths) == 1, f"Found more than one file for subject {sub} run {run}."

        raw = mne.io.read_raw(paths[0], verbose="ERROR", preload=True)
        eegbci.standardize(raw)
        raw = raw.set_eeg_reference("average")  # reference to average
        raw = raw.drop_channels(["T9", "T10", "Iz"])  # drop reference channels
        raw.set_montage(mne.channels.make_standard_montage("brainproducts-RNP-BA-128"))  # set montage
        raw.info["line_freq"] = 60  # set power line frequency

        # preprocess
        raw = preprocess(raw, config)

        # crop raw according to annotations and assign labels
        for annot in raw.annotations:
            raw_segment = raw.copy().crop(tmin=annot["onset"], tmax=annot["onset"] + annot["duration"])

            label = LABEL_MAPPING[(run, annot["description"])]
            if label in segments:
                segments[label].append(raw_segment)
            else:
                segments[label] = [raw_segment]

    # save preprocessed data
    # TODO


class PhysionetMotorImagery(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for the Physionet Motor Imagery dataset.

    Args:
        task (PhysionetMotorImageryTask): The task to be used in the dataset.
        train_subjects (Union[int, str]): Number of subjects in training split, or string specifying subject IDs.
        val_subjects (Union[int, str]): Number of subjects in validation split, or string specifying subject IDs.
        test_subjects (Union[int, str]): Number of subjects in test split, or string specifying subject IDs.
        root (str): The root directory where the dataset will be stored.
        n_jobs (int): The number of jobs for downloading and preprocessing the data.

    Examples:
        >>> from eegformer.dataset import PhysionetMotorImagery
        >>> data = PhysionetMotorImagery(train_subjects="1-20", val_subjects=5, test_subjects=2)
    """

    def __init__(
        self,
        task: PhysionetMotorImageryTask = PhysionetMotorImageryTask.ALL,
        preprocessing_config: PreprocessingConfig = PreprocessingConfig(),
        train_subjects: Union[int, str] = 80,
        val_subjects: Union[int, str] = 20,
        test_subjects: Union[int, str] = 9,
        root: str = "data",
        n_jobs: int = -1,
    ):
        super().__init__()
        self.task = task
        self.preprocessing_config = preprocessing_config

        self.train_subjects = parse_subjects(train_subjects)
        self.val_subjects = parse_subjects(val_subjects, unavailable=self.train_subjects)
        self.test_subjects = parse_subjects(test_subjects, unavailable=self.train_subjects + self.val_subjects)

        self.root = join(abspath(expanduser(root)), "physionet-motor-imagery")
        self.n_jobs = n_jobs

    def prepare_data(self):
        """
        Download the dataset, preprocess it, extract epochs and save them to `self.root`/processed.
        """
        # collect run IDs
        runs = []
        if (self.task & PhysionetMotorImageryTask.BASELINE).value > 0:
            runs += [1, 2]
        if (self.task & PhysionetMotorImageryTask.MOTOR_EXECUTION_LEFT_RIGHT).value > 0:
            runs += [3, 7, 11]
        if (self.task & PhysionetMotorImageryTask.MOTOR_EXECUTION_HANDS_FEET).value > 0:
            runs += [5, 9, 13]
        if (self.task & PhysionetMotorImageryTask.MOTOR_IMAGERY_LEFT_RIGHT).value > 0:
            runs += [4, 8, 12]
        if (self.task & PhysionetMotorImageryTask.MOTOR_IMAGERY_HANDS_FEET).value > 0:
            runs += [6, 10, 14]

        # create final directory
        os.makedirs(join(self.root, "processed"), exist_ok=True)

        # download and preprocess the dataset and store it under self.root/processed
        Parallel(n_jobs=self.n_jobs)(
            delayed(preprocess_subject)(sub, runs, self.root, self.preprocessing_config)
            for sub in tqdm(self.train_subjects + self.val_subjects + self.test_subjects, desc="Preprocessing")
        )


if __name__ == "__main__":
    pl.seed_everything(42)
    data = PhysionetMotorImagery(
        task=PhysionetMotorImageryTask.BASELINE,
        train_subjects="1-2",
        val_subjects="3",
        test_subjects="4",
        root="~/data",
        n_jobs=1,
    )
    data.prepare_data()
