from enum import Flag
from os.path import exists, join
from typing import List, Union

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
    def parse_task(task: Union[str, "PhysionetMotorImageryTask"]) -> "PhysionetMotorImageryTask":
        """
        Parses a task from a string where multiple tasks can be combined with a `|`.

        ### Args
            - `task` (Union[str, PhysionetMotorImageryTask]): The task at hand.

        ### Returns
            PhysionetMotorImageryTask: The parsed task.
        """
        if isinstance(task, PhysionetMotorImageryTask):
            return task

        tasks = [getattr(PhysionetMotorImageryTask, t.strip()) for t in task.split("|")]
        result = tasks[0]
        for task in tasks[1:]:
            result |= task
        return result

    @staticmethod
    def get_runs(task: Union[str, "PhysionetMotorImageryTask"]) -> List[int]:
        """
        Returns a list of runs for the given task.

        ### Args
            - `task` (Union[str, PhysionetMotorImageryTask]): The task at hand.

        ### Returns
            List[int]: The list of runs.
        """
        task = PhysionetMotorImageryTask.parse_task(task)

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
    def get_labels(task: Union[str, "PhysionetMotorImageryTask"], combine_exec_imag: bool) -> List[str]:
        """
        Returns a list of labels for the given task.

        ### Args:
            - `task` (Union[str, PhysionetMotorImageryTask]): The task at hand.
            - `combine_exec_imag` (bool): Whether to combine execution and imagery labels.

        ### Returns:
            List[str]: The list of labels.
        """
        task = PhysionetMotorImageryTask.parse_task(task)

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

        if combine_exec_imag:
            labels = [label.replace("execution", "motor") for label in labels]
            labels = [label.replace("imagery", "motor") for label in labels]

        return sorted(set(labels))


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
    combine_exec_imag: bool,
    config: PreprocessingConfig,
) -> str:
    """
    Preprocess data from a single subject and save individual epochs to a WebDataset shard.

    ### Args
        - `sub` (int): The subject ID.
        - `raw_path` (str): Path to the raw data.
        - `processed_path` (str): Path to the processed data.
        - `task` (PhysionetMotorImageryTask): The task to preprocess.
        - `combine_exec_imag` (bool): Whether to combine execution and imagery labels.
        - `config` (PreprocessingConfig): The preprocessing configuration.

    ### Returns
        str: Path to the saved WebDataset .tar shard.
    """
    current_shard = shard_name(sub, task, config)
    shard_path = join(processed_path, current_shard)
    if not config.force and exists(shard_path):
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

            label = LABEL_MAPPING[label_key]
            if combine_exec_imag:
                label = label.replace("execution", "motor").replace("imagery", "motor")
            label = PhysionetMotorImageryTask.get_labels(task, combine_exec_imag).index(label)

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
        - `subjects` (List[int]): The subjects to use. If `None`, all subjects are used.
        - `exclude_problematic` (bool): Whether to exclude problematic subjects (88, 89, 92, 100, 104, 106).
        - `combine_exec_imag` (bool): Whether to combine execution and imagery tasks.
        - `**kwargs`: Additional arguments to pass to the `Dataset` constructor.
    """

    PROBLEMATIC_SUBJECTS = [88, 89, 92, 100, 104, 106]

    def __init__(
        self,
        root: str,
        task: PhysionetMotorImageryTask = PhysionetMotorImageryTask.ALL,
        subjects: List[int] = None,
        exclude_problematic: bool = True,
        combine_exec_imag: bool = False,
        **kwargs,
    ):
        self.task = PhysionetMotorImageryTask.parse_task(task)
        self.combine_exec_imag = combine_exec_imag

        # get a list of subjects and make sure they are not problematic
        subjects = subjects or PhysionetMotorImagery.subject_ids(exclude_problematic=exclude_problematic)
        if exclude_problematic and set(subjects) & set(PhysionetMotorImagery.PROBLEMATIC_SUBJECTS):
            raise ValueError(
                "You specified to exclude problematic subjects but asked for subjects "
                f"{set(subjects) & set(PhysionetMotorImagery.PROBLEMATIC_SUBJECTS)}, which are problematic."
            )

        # initialize the dataset
        super().__init__(root, subjects=subjects, **kwargs)

    @staticmethod
    def subject_ids(exclude_problematic: bool, **kwargs) -> List[int]:
        if exclude_problematic:
            return [sub for sub in range(1, 110) if sub not in PhysionetMotorImagery.PROBLEMATIC_SUBJECTS]
        return list(range(1, 110))

    @staticmethod
    def num_classes(task: Union[str, PhysionetMotorImageryTask], **kwargs) -> int:
        return len(PhysionetMotorImageryTask.get_labels(task, kwargs["combine_exec_imag"]))

    def list_shards(self) -> List[str]:
        # list all shards for the given task
        return [shard_name(sub, self.task, self.preprocessing) for sub in self.subjects]

    def prepare_data(self):
        # download and preprocess the dataset and store epochs as WebDataset shards (one shard per subject)
        Parallel(n_jobs=self.preprocessing.n_jobs)(
            delayed(preprocess_subject)(
                sub, self.raw_path, self.processed_path, self.task, self.combine_exec_imag, self.preprocessing
            )
            for sub in tqdm(self.subjects, desc="Preprocessing data")
        )

    def label2idx(self, label: str) -> int:
        return PhysionetMotorImageryTask.get_labels(self.task, self.combine_exec_imag).index(label)

    def idx2label(self, idx: int) -> str:
        return PhysionetMotorImageryTask.get_labels(self.task, self.combine_exec_imag)[idx]
