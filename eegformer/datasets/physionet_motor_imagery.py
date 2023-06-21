from enum import Flag
from os.path import exists, join
from typing import List

import mne
import webdataset as wds
from joblib import Parallel, delayed
from mne.datasets import eegbci
from tqdm import tqdm

from eegformer.datasets import Dataset
from eegformer.utils import PreprocessingConfig, extract_ch_pos, preprocess


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
        """
        Returns a list of runs for the given task.

        ### Args:
            - `task` (PhysionetMotorImageryTask): The task at hand.

        ### Returns:
            List[int]: The list of runs.
        """
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

    @staticmethod
    def get_labels(task) -> List[str]:
        """
        Returns a list of labels for the given task.

        ### Args:
            - `task` (PhysionetMotorImageryTask): The task at hand.

        ### Returns:
            List[str]: The list of labels.
        """
        labels = []
        if (task & PhysionetMotorImageryTask.BASELINE).value > 0:
            labels += ["baseline_open", "baseline_closed"]
        if (task & PhysionetMotorImageryTask.MOTOR_EXECUTION_LEFT_RIGHT).value > 0:
            labels += ["execution_left", "execution_right"]
        if (task & PhysionetMotorImageryTask.MOTOR_EXECUTION_HANDS_FEET).value > 0:
            labels += ["execution_hands", "execution_feet"]
        if (task & PhysionetMotorImageryTask.MOTOR_IMAGERY_LEFT_RIGHT).value > 0:
            labels += ["imagery_left", "imagery_right"]
        if (task & PhysionetMotorImageryTask.MOTOR_IMAGERY_HANDS_FEET).value > 0:
            labels += ["imagery_hands", "imagery_feet"]
        return sorted(labels)


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


def shard_name(sub: int, task: PhysionetMotorImageryTask, config: PreprocessingConfig):
    return f"sub-{sub:03d}_task-{task.value:02d}_config-{hash(config)}.tar"


def preprocess_subject(
    sub: int,
    raw_path: str,
    processed_path: str,
    task: PhysionetMotorImageryTask,
    config: PreprocessingConfig,
    force: bool = False,
) -> str:
    """
    Preprocess data from a single subject and save individual epochs to a WebDataset shard.

    ### Args
        - `sub` (int): The subject ID.
        - `raw_path` (str): Path to the raw data.
        - `processed_path` (str): Path to the processed data.
        - `task` (PhysionetMotorImageryTask): The task to preprocess.
        - `config` (PreprocessingConfig): The preprocessing configuration.
        - `force` (bool): Whether to force preprocessing even if the file already exists.

    ### Returns
        str: Path to the saved WebDataset .tar shard.
    """
    current_shard = shard_name(sub, task, config)
    shard_path = join(processed_path, current_shard)
    if not force and exists(shard_path):
        # skip preprocessing if the file already exists
        return shard_path

    shard_writer = wds.TarWriter(shard_path, compress=True)

    epoch_idx = 0
    for run in PhysionetMotorImageryTask.get_runs(task):
        # download raw data
        paths = eegbci.load_data(sub, run, raw_path, update_path=False, verbose="ERROR")
        assert len(paths) == 1, f"Found more than one file for subject {sub} run {run}."

        raw = mne.io.read_raw(paths[0], verbose="ERROR", preload=True)
        eegbci.standardize(raw)
        raw = raw.set_eeg_reference("average", verbose="ERROR")  # reference to average
        raw = raw.drop_channels(["T9", "T10", "Iz"])  # drop reference channels
        raw.set_montage(mne.channels.make_standard_montage("brainproducts-RNP-BA-128"))  # set montage
        raw.info["line_freq"] = 60  # set power line frequency

        # crop raw according to annotations and assign labels
        for annot in raw.annotations:
            label_key = (run, annot["description"])

            # skip annotations that are not in the label mapping
            if label_key not in LABEL_MAPPING:
                continue

            # crop raw to annotation
            tmax = min(annot["onset"] + annot["duration"], raw.tmax)
            raw_segment = raw.copy().crop(tmin=annot["onset"], tmax=tmax)

            # preprocess the current segment
            epochs = preprocess(raw_segment, config)

            # skip if no epochs were extracted
            if len(epochs) == 0:
                continue

            # assemble sample
            ch_pos = extract_ch_pos(raw_segment)
            label = PhysionetMotorImageryTask.get_labels(task).index(LABEL_MAPPING[label_key])

            # save epochs
            for epoch in epochs:
                shard_writer.write(
                    {
                        "__key__": f"sample{epoch_idx:06d}",
                        "signal.npy": epoch,
                        "ch_pos.npy": ch_pos,
                        "label.cls": label,
                    }
                )
                epoch_idx += 1

    return shard_path


class PhysionetMotorImagery(Dataset):
    """
    The Physionet Motor Imagery dataset.

    ### Args
        - `root` (str): The path to the dataset.
        - `task` (PhysionetMotorImageryTask): The task to preprocess.
        - `preprocessing` (PreprocessingConfig): The preprocessing configuration.
        - `subjects` (List[int]): The subjects to use. If `None`, all subjects are used.
        - `exclude_problematic` (bool): Whether to exclude problematic subjects (88, 89, 92, 100, 104, 106).
        - `compute_class_weights` (bool): Whether to compute class weights for the dataset.
    """

    PROBLEMATIC_SUBJECTS = [88, 89, 92, 100, 104, 106]

    def __init__(
        self,
        root: str,
        task: PhysionetMotorImageryTask = PhysionetMotorImageryTask.ALL,
        preprocessing: PreprocessingConfig = None,
        subjects: List[int] = None,
        exclude_problematic: bool = True,
        compute_class_weights: bool = False,
    ):
        if isinstance(task, str):
            self.task = getattr(PhysionetMotorImageryTask, task)
        else:
            self.task = task

        # get a list of subjects and make sure they are not problematic
        subjects = subjects or PhysionetMotorImagery.subject_ids(exclude_problematic=exclude_problematic)
        if exclude_problematic and set(subjects) & set(PhysionetMotorImagery.PROBLEMATIC_SUBJECTS):
            raise ValueError(
                "You specified to exclude problematic subjects but asked for subjects "
                f"{set(subjects) & set(PhysionetMotorImagery.PROBLEMATIC_SUBJECTS)}, which are problematic."
            )

        # initialize the dataset
        super().__init__(root, preprocessing, subjects, compute_class_weights)

    @staticmethod
    def subject_ids(exclude_problematic: bool, **kwargs) -> List[int]:
        if exclude_problematic:
            return [sub for sub in range(1, 110) if sub not in PhysionetMotorImagery.PROBLEMATIC_SUBJECTS]
        return list(range(1, 110))

    @staticmethod
    def num_classes(task: PhysionetMotorImageryTask, **kwargs) -> int:
        if isinstance(task, str):
            task = getattr(PhysionetMotorImageryTask, task)
        return len(PhysionetMotorImageryTask.get_labels(task))

    def list_shards(self) -> List[str]:
        # list all shards for the given task
        return [shard_name(sub, self.task, self.preprocessing) for sub in self.subjects]

    def prepare_data(self):
        # download and preprocess the dataset and store epochs as WebDataset shards (one shard per subject)
        Parallel(n_jobs=self.preprocessing.n_jobs)(
            delayed(preprocess_subject)(sub, self.raw_path, self.processed_path, self.task, self.preprocessing)
            for sub in tqdm(self.subjects, desc="Preprocessing data")
        )

    def label2idx(self, label: str) -> int:
        return PhysionetMotorImageryTask.get_labels(self.task).index(label)

    def idx2label(self, idx: int) -> str:
        return PhysionetMotorImageryTask.get_labels(self.task)[idx]


if __name__ == "__main__":
    ds = PhysionetMotorImagery(
        "~/data",
        PhysionetMotorImageryTask.BASELINE,
        PreprocessingConfig(n_jobs=-1),
        # subjects=list(range(1, 6)),
    )

    print(ds)
    print(ds.class_weights)
