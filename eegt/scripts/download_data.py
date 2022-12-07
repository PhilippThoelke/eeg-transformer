import os
from os.path import join, exists
from abc import ABC, abstractmethod
import hashlib
import pickle
import codecs
from glob import glob
from tqdm import tqdm
import mne
from mne.io import read_raw
from mne.channels import make_standard_montage
from mne.channels.montage import transform_to_head
from mne.datasets import eegbci
import numpy as np
import pandas as pd
from braindecode.datasets import (
    BaseDataset,
    BaseConcatDataset,
    MOABBDataset,
)
import openneuro
from mne_bids import read_raw_bids, BIDSPath
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_fixed_length_windows,
)


def standardize(data, channelwise=True):
    axis = 1 if channelwise else None
    mean, std = data.mean(axis=axis, keepdims=True), data.std(axis=axis, keepdims=True)
    return (data - mean) / std


class ProcessedDataset(ABC):
    """
    Abstract dataset class used to standardize preprocessing for all inheriting datasets.
    All deriving classes have to implement the `instantiate` method, which should download
    the dataset files and return an instance of `braindecode.datasets.BaseDataset` or
    `braindecode.datasets.BaseConcatDataset`. Deriving classes should further set the
    `line_freq` and `subject_ids` attributes accordingly.

    Parameters
    ----------
    sfreq : desired sampling frequency after preprocessing
    filter_range : tuple of low and high frequency cutoffs
    standardize_channelwise : wheter to apply channel-wise or recording-wise standardization
    clip_stds : number of standard deviations beyond EEG data is clipped
    use_annotations: if True, drop data outside of annotations
    max_subjects : limits the number of subjects for debugging purposes
    kwargs : keyword arguments passed to the deriving dataset class
    """

    def __init__(
        self,
        sfreq=128,
        filter_range=(0.5, None),
        standardize_channelwise=True,
        clip_stds=5,
        use_annotations=True,
        max_subjects=-1,
        **kwargs,
    ):
        assert (
            hasattr(self, "line_freq") and self.line_freq is not None
        ), f"{type(self).__name__} doesn't implement the line_freq attribute."
        assert (
            hasattr(self, "subject_ids") and self.subject_ids is not None
        ), f"{type(self).__name__} doesn't implement the subject_ids attribute."

        self.sfreq = sfreq
        self.filter_range = filter_range
        self.standardize_channelwise = standardize_channelwise
        self.clip_stds = clip_stds
        self.use_annotations = use_annotations
        self.max_subjects = max_subjects
        self.instantiate_kwargs = kwargs

    def load_subject(self, subject_id):
        """subject_id is expected to be the id associated with the dataset, not an index"""
        raw_dset = self.instantiate(subject_id, **self.instantiate_kwargs)
        return self.preprocess(raw_dset)

    def idx2subj(self, idx):
        if idx >= len(self):
            raise IndexError()
        elif idx < -len(self):
            raise IndexError()
        return self.subject_ids[idx]

    @abstractmethod
    def instantiate(self, subject_id):
        """Implement this function to download and instantiate the dataset"""
        pass

    def label_transform(self, label, description, raw):
        """Overwrite this function to modify epoch labels"""
        return label

    def crop_by_annotations(self, raw):
        """Overwrite this function to apply non-standard epoching"""
        annots = raw.annotations
        start = (annots.onset * raw.info["sfreq"]).astype(int)
        end = ((annots.onset + annots.duration) * raw.info["sfreq"]).astype(int)

        result = []
        for s, e, lbl in zip(start, end, raw.annotations.description):
            if s >= 0 and e <= len(raw):
                # raw.copy().crop(...) and raw.get_data() cause excessive memory allocation
                cropped = mne.io.RawArray(raw._data[:, s:e], raw.info, copy="both")
                cropped._filenames = raw._filenames
                result.append((cropped, lbl))
        return result

    def prepare_annotations(self, raw):
        """Overwrite this function to prepare the annotations for epoching"""
        pass

    def preprocess(self, raw_dset):
        """Apply preprocessing and extract epochs"""
        if isinstance(raw_dset, BaseDataset):
            raw_dset = BaseConcatDataset([raw_dset])

        # define preprocessing steps
        preprocessors = []
        if self.line_freq is not None:
            preprocessors.append(
                Preprocessor(
                    "notch_filter",
                    freqs=np.arange(self.line_freq, self.sfreq / 2, self.line_freq),
                )
            )
        preprocessors.append(
            Preprocessor(
                "filter",
                l_freq=self.filter_range[0],
                h_freq=self.filter_range[1],
            )
        )
        preprocessors.append(Preprocessor("pick_types", eeg=True))
        preprocessors.append(Preprocessor("set_eeg_reference", ref_channels="average"))
        preprocessors.append(
            Preprocessor(standardize, channelwise=self.standardize_channelwise)
        )
        preprocessors.append(
            Preprocessor(np.clip, a_min=-self.clip_stds, a_max=self.clip_stds)
        )
        # apply preprocessing
        processed_dset = preprocess(raw_dset, preprocessors)

        epochs = []
        for curr in processed_dset.datasets:
            if self.use_annotations:
                self.prepare_annotations(curr.raw)
                raws = self.crop_by_annotations(curr.raw)
            else:
                raws = [(curr.raw, f"{self.name}-{curr.description['subject']}")]
            for raw_epoch, label in raws:
                # resample after epoching to avoid event jitter
                raw_epoch = raw_epoch.resample(self.sfreq)

                # update description object
                desc = curr.description.copy()
                if self.use_annotations:
                    desc["label"] = self.label_transform(label, desc, raw_epoch)
                else:
                    desc["label"] = label

                # add channel positions to description
                montage = transform_to_head(raw_epoch.get_montage())
                ch_pos = montage.get_positions()["ch_pos"]
                ch_pos = np.stack([ch_pos[ch_name] for ch_name in raw_epoch.ch_names])
                desc["ch_pos"] = ch_pos
                desc["dataset"] = self.name

                # append current epoch as a BaseDataset
                epochs.append(BaseDataset(raw_epoch, desc, target_name="label"))
        return epochs

    @property
    def name(self):
        return self.__class__.__name__

    def __len__(self):
        return self.max_subjects if self.max_subjects > 0 else len(self.subject_ids)


class PhysionetMI(ProcessedDataset):
    line_freq = 60
    subject_ids = list(range(1, 110))

    def instantiate(self, subject_id):
        raw_datasets = []
        for run in range(1, 15):
            raw = read_raw(eegbci.load_data(subject_id, run, update_path=True)[0])
            raw.rename_channels(
                dict(zip(raw.ch_names, PhysionetMI.format_channels(raw.ch_names)))
            )
            raw.set_montage(make_standard_montage("standard_1005"))

            if run == 1:
                # baseline: eyes open
                raw.annotations.rename(
                    dict(
                        zip(
                            raw.annotations.description,
                            ["rest-eyes_open"] * len(raw.annotations.description),
                        )
                    )
                )
            elif run == 2:
                # baseline: eyes closed
                raw.annotations.rename(
                    dict(
                        zip(
                            raw.annotations.description,
                            ["rest-eyes_closed"] * len(raw.annotations.description),
                        )
                    )
                )
            elif run in [3, 7, 11]:
                # motor execution: left hand vs right hand
                raw.annotations.rename(
                    dict(
                        T0="rest-between_trial",
                        T1="execution-left_hand",
                        T2="execution-right_hand",
                    )
                )
            elif run in [4, 8, 12]:
                # motor imagery: left hand vs right hand
                raw.annotations.rename(
                    dict(
                        T0="rest-between_trial",
                        T1="imagery-left_hand",
                        T2="imagery-right_hand",
                    )
                )
            elif run in [5, 9, 13]:
                # motor execution: hands vs feet
                raw.annotations.rename(
                    dict(
                        T0="rest-between_trial",
                        T1="execution-fists",
                        T2="execution-feet",
                    )
                )
            elif run in [6, 10, 14]:
                # motor imagery: hands vs feet
                raw.annotations.rename(
                    dict(T0="rest-between_trial", T1="imagery-fists", T2="imagery-feet")
                )
            raw_datasets.append(BaseDataset(raw))

        dset = BaseConcatDataset(raw_datasets)
        # create dataset description
        desc = pd.DataFrame([subject_id] * len(dset.description), columns=["subject"])
        dset.set_description(desc)
        return dset

    def format_channels(ch_names):
        names = []
        for name in ch_names:
            std_name = name.strip(".")
            std_name = std_name.upper()
            if std_name.endswith("Z"):
                std_name = std_name[:-1] + "z"
            if std_name.startswith("FP"):
                std_name = "Fp" + std_name[2:]
            names.append(std_name)
        return names


class Zhou2016(ProcessedDataset):
    line_freq = 50
    subject_ids = list(range(1, 5))

    def instantiate(self, subject_id):
        return MOABBDataset("Zhou2016", subject_id)

    def prepare_annotations(self, raw):
        keep_annots = ["left_hand", "right_hand", "feet"]
        delete_mask = [d not in keep_annots for d in raw.annotations.description]
        raw.annotations.delete(delete_mask)
        raw.annotations.duration[:] = 5


class MAMEM1(ProcessedDataset):
    line_freq = 50
    subject_ids = list(range(1, 11))

    def instantiate(self, subject_id):
        return MOABBDataset("MAMEM1", subject_id)

    def label_transform(self, label, description, raw):
        return "flickering_" + label + "Hz"


class RestingCognitive(ProcessedDataset):
    line_freq = 50
    subject_ids = list(range(1, 61))

    def instantiate(self, subject_id):
        subject_id_bids = f"{subject_id:02}"
        dataset_id = "ds004148"
        root = join(".", dataset_id)
        sub_dir = f"sub-{subject_id_bids}"
        # download current subject
        openneuro.download(dataset=dataset_id, target_dir=root, include=sub_dir)

        # add *_coordsystem.json files for every *_electrodes.tsv file
        # these are missing in the original dataset files
        electrode_paths = BIDSPath(
            subject=subject_id_bids, suffix="electrodes", root=root
        )
        for path in electrode_paths.match():
            path.suffix = "coordsystem"
            path.extension = ".json"
            with open(path.fpath, "w") as f:
                f.write(
                    '{\n"EEGCoordinateSystem":"Other",\n"EEGCoordinateUnits":"mm"\n}'
                )

        paths = BIDSPath(
            subject=subject_id_bids,
            datatype="eeg",
            suffix="eeg",
            extension="vhdr",
            root=root,
        )
        raw_datasets = []
        for path in paths.match():
            raw = read_raw_bids(path)
            # set annotations according to the task
            raw.set_annotations(
                mne.Annotations([raw.times.min()], [raw.times.max()], [path.task])
            )
            raw_datasets.append(BaseDataset(raw))

        # make sure all raws are consistent
        line_freqs = [d.raw.info["line_freq"] for d in raw_datasets]
        assert all(self.line_freq == lf for lf in line_freqs)

        dset = BaseConcatDataset(raw_datasets)
        # create dataset description
        desc = pd.DataFrame([subject_id] * len(dset.description), columns=["subject"])
        dset.set_description(desc)
        return dset


class SleepEpilepsy(ProcessedDataset):
    line_freq = 50
    subject_ids = list(range(1, 31))

    def instantiate(self, subject_id):
        subject_id_bids = f"{subject_id:02}"
        dataset_id = "ds003555"
        root = join(".", dataset_id)
        sub_dir = f"sub-{subject_id_bids}"
        # download current subject
        openneuro.download(dataset=dataset_id, target_dir=root, include=sub_dir)

        paths = BIDSPath(
            subject=subject_id_bids, datatype="eeg", suffix="eeg", root=root
        ).match()
        assert len(paths) == 1, f"Expected 1 EEG recording, found {len(paths)}"

        raw = read_raw_bids(paths[0])
        # set annotations according to the task
        raw.set_annotations(
            mne.Annotations([raw.times.min()], [raw.times.max()], [paths[0].task])
        )
        # drop T1 and T2 channels
        raw.drop_channels(["T1", "T2", "Cz2"], on_missing="ignore")
        # set montage
        raw.set_montage(make_standard_montage("standard_1020"))

        # create dataset
        dset = BaseDataset(raw)
        dset.set_description(dict(subject=subject_id))
        return dset


class GoNogo(ProcessedDataset):
    line_freq = 50
    subject_ids = list(range(2, 16))

    def instantiate(self, subject_id):
        subject_id_bids = f"{subject_id:03}"
        dataset_id = "ds002680"
        root = join(".", dataset_id)
        sub_dir = f"sub-{subject_id_bids}"
        # download current subject
        openneuro.download(dataset=dataset_id, target_dir=root, include=sub_dir)

        paths = BIDSPath(
            subject=subject_id_bids, datatype="eeg", suffix="eeg", root=root
        )

        raw_datasets = []
        for path in paths.match():
            raw = read_raw_bids(path)
            raw.set_channel_types({ch: "eeg" for ch in raw.info["ch_names"]})

            # set montage
            raw.set_montage(
                mne.channels.make_dig_montage(
                    {ch["ch_name"]: ch["loc"][:3] / 10 for ch in raw.info["chs"]}
                )
            )

            raw_datasets.append(BaseDataset(raw))

        # make sure all raws are consistent
        line_freqs = [d.raw.info["line_freq"] for d in raw_datasets]
        assert all(self.line_freq == lf for lf in line_freqs)

        dset = BaseConcatDataset(raw_datasets)
        # create dataset description
        desc = pd.DataFrame([subject_id] * len(dset.description), columns=["subject"])
        dset.set_description(desc)
        return dset

    def prepare_annotations(self, raw):
        raise RuntimeError(
            "Using annotations is currently not implemented for this dataset"
        )


class BrainInvaders(ProcessedDataset):
    line_freq = 50
    subject_ids = list(range(1, 65))

    def instantiate(self, subject_id):
        return MOABBDataset("bi2014a", subject_id)

    def prepare_annotations(self, raw):
        raise RuntimeError(
            "Using annotations is currently not implemented for this dataset"
        )


class DemonsP300(ProcessedDataset):
    line_freq = 50
    subject_ids = list(range(0, 60))

    def instantiate(self, subject_id):
        return MOABBDataset("DemonsP300", subject_id)

    def prepare_annotations(self, raw):
        raise RuntimeError(
            "Using annotations is currently not implemented for this dataset"
        )


class Shin2017B(ProcessedDataset):
    line_freq = 50
    subject_ids = list(range(1, 30))

    def instantiate(self, subject_id):
        return MOABBDataset("Shin2017B", subject_id, dataset_kwargs=dict(accept=True))


class Cho2017(ProcessedDataset):
    line_freq = 60
    subject_ids = list(range(1, 53))

    def instantiate(self, subject_id):
        return MOABBDataset("Cho2017", subject_id)


class Concentration(ProcessedDataset):
    line_freq = 60
    subject_ids = list(range(1, 21))

    def instantiate(self, subject_id):
        subject_id_bids = f"{subject_id:03}"
        dataset_id = "ds002691"
        root = join(".", dataset_id)
        sub_dir = f"sub-{subject_id_bids}"
        # download current subject
        openneuro.download(dataset=dataset_id, target_dir=root, include=sub_dir)

        paths = BIDSPath(
            subject=subject_id_bids, datatype="eeg", suffix="eeg", root=root
        )

        raw_datasets = []
        for path in paths.match():
            raw = read_raw_bids(path)

            # get template montage
            template_pos = mne.channels.make_standard_montage(
                "GSN-HydroCel-32"
            ).get_positions()["ch_pos"]
            # update channel positions
            raw.set_montage(
                mne.channels.make_dig_montage(
                    {
                        ch["ch_name"]: template_pos[ch["ch_name"]]
                        for ch in raw.info["chs"]
                    }
                )
            )

            raw_datasets.append(BaseDataset(raw))

        # make sure all raws are consistent
        line_freqs = [d.raw.info["line_freq"] for d in raw_datasets]
        assert all(self.line_freq == lf for lf in line_freqs)

        dset = BaseConcatDataset(raw_datasets)
        # create dataset description
        desc = pd.DataFrame([subject_id] * len(dset.description), columns=["subject"])
        dset.set_description(desc)
        return dset

    def prepare_annotations(self, raw):
        desc = np.array(
            list(map(lambda x: x.split("/")[0], raw.annotations.description))
        )

        # group blocks of the same annotation
        final_desc, final_onset, final_duration = [], [], []
        for i in range(len(desc)):
            val = desc[i]
            if len(final_desc) > 0 and final_desc[-1] == val:
                continue

            for j in range(i + 1, len(desc)):
                if desc[j] != val:
                    break
            final_desc.append(val)
            final_onset.append(raw.annotations.onset[i])
            final_duration.append(raw.annotations.onset[j] - raw.annotations.onset[i])

        raw.annotations.description = np.array(final_desc)
        raw.annotations.onset = np.array(final_onset)
        raw.annotations.duration = np.array(final_duration)


class TUHEEG(ProcessedDataset):
    """This class will not automatically download the TUH EEG Corpus.Please follow
    the instructions at https://isip.piconepress.com/projects/tuh_eeg/html/downloads.shtml#c_tueg"""

    line_freq = 60
    subject_ids = None
    ch_names_1020 = make_standard_montage("standard_1020").ch_names

    def __init__(self, path, *args, **kwargs):
        assert exists(path), (
            f"{path} does not exist, please correct the "
            "path argument or download the dataset"
        )
        self.path = path

        # indexing dataset
        subj_paths = glob(join(path, "*", "*"))
        self.subject2group = {
            p.split(os.sep)[-1]: p.split(os.sep)[-2]
            for p in subj_paths
            if len(glob(join(p, "**", "*.edf"), recursive=True)) > 0
        }
        self.subject_ids = sorted(list(self.subject2group.keys()))

        super().__init__(*args, **kwargs)

    def instantiate(self, subject_id):
        raw_paths = glob(
            join(self.path, self.subject2group[subject_id], subject_id, "**", "*.edf"),
            recursive=True,
        )

        raw_datasets = []
        for path in raw_paths:
            raw = read_raw(path)

            raw.rename_channels(TUHEEG.format_channel, allow_duplicates=True)
            raw.drop_channels([ch for ch in raw.ch_names if ch.startswith("unknown")])
            raw.set_montage(make_standard_montage("standard_1020"))

            raw_datasets.append(BaseDataset(raw))

        dset = BaseConcatDataset(raw_datasets)
        # create dataset description
        desc = pd.DataFrame([subject_id] * len(dset.description), columns=["subject"])
        dset.set_description(desc)
        return dset

    def prepare_annotations(self, raw):
        raise RuntimeError(
            "Using annotations is currently not implemented for this dataset"
        )

    def format_channel(ch):
        if not ch.startswith("EEG "):
            print("UNKNOWN CHANNEL", ch)
            return "unknown"
        ch = ch[4:]

        parts = ch.split("-")
        if len(parts) < 2:
            print("UNKNOWN FORMAT", ch)
            return "unknown"

        parts[0] = parts[0].replace("FP", "Fp").replace("Z", "z")

        if parts[0] not in TUHEEG.ch_names_1020:
            print("CHANNEL NOT IN 10/20 NAMES", parts[0])
            return "unknown"

        return parts[0]


if __name__ == "__main__":
    result_dir = "data/"
    epoch_length = 1
    epoch_overlap = 0.5
    sfreq = 128
    use_annotations = False

    # define datasets
    datasets = [
        PhysionetMI(sfreq=sfreq, use_annotations=use_annotations),
        Zhou2016(sfreq=sfreq, use_annotations=use_annotations),
        MAMEM1(sfreq=sfreq, use_annotations=use_annotations),
        RestingCognitive(sfreq=sfreq, use_annotations=use_annotations),
        SleepEpilepsy(sfreq=sfreq, use_annotations=use_annotations),
        GoNogo(sfreq=sfreq, use_annotations=use_annotations),
        BrainInvaders(sfreq=sfreq, use_annotations=use_annotations),
        DemonsP300(sfreq=sfreq, use_annotations=use_annotations),
        Shin2017B(sfreq=sfreq, use_annotations=use_annotations),
        Cho2017(sfreq=sfreq, use_annotations=use_annotations),
        Concentration(sfreq=sfreq, use_annotations=use_annotations),
        TUHEEG(path="data/tuh_eeg", sfreq=sfreq, use_annotations=use_annotations),
    ]

    # prepare processing the data
    mne.set_log_level("ERROR")
    arg_hash = f"{epoch_length}-{epoch_overlap}-{sfreq}-{use_annotations}"
    arg_hash = hashlib.md5(arg_hash.encode()).hexdigest()
    fname = "-".join(d.name for d in datasets) + "_" + arg_hash
    memmap_len = 0
    metadata = []
    stage = dict()

    if exists(join(result_dir, "raw-" + fname + ".dat")):
        print("Found existing dataset with the same name. Extending...")
        # get size of existing database
        memmap_len = np.memmap(
            join(result_dir, "raw-" + fname + ".dat"), dtype=np.float32, mode="r"
        ).shape[0]
        # load existing metadata file
        metadata = pd.read_csv(
            join(result_dir, "label-" + fname + ".csv"), index_col=0
        ).values.tolist()

        max_idx = max(row[4] for row in metadata)
        assert memmap_len >= max_idx, (
            f"The dataset is corrupted. The data file size ({memmap_len}) "
            f"is smaller than the largest index in the metadata file ({max_idx})."
        )

        if memmap_len > max_idx:
            print(
                f"Data file ({memmap_len}) is larger than the largest index in "
                f"the metadata file ({max_idx}). The data file will be truncated."
            )
            memmap_len = max_idx

    # iterate all data
    pbar = tqdm(total=sum(len(d) for d in datasets))
    for dset in datasets:
        for idx in range(len(dset)):
            subj_idx = dset.idx2subj(idx)
            stage["dataset"] = dset.name
            stage["subject"] = subj_idx

            if f"{dset.name}-{subj_idx}" in [row[0] for row in metadata]:
                # subject is already present in the metadata file
                pbar.update()
                continue

            # retrieve data and preprocess for the current subject
            stage["stage"] = "preprocessing"
            pbar.set_postfix(stage)
            subj = dset.load_subject(subj_idx)

            # drop recordings that are too short for a single window
            subj = [s for s in subj if len(s) >= int(sfreq * epoch_length)]

            # extract windows from epochs
            stage["stage"] = "windowing"
            pbar.set_postfix(stage)
            try:
                windows = create_fixed_length_windows(
                    BaseConcatDataset(subj),
                    window_size_samples=int(sfreq * epoch_length),
                    window_stride_samples=int(sfreq * epoch_overlap),
                    drop_last_window=True,
                    n_jobs=-1,
                )
            except:
                # windowing might have failed due to out of memory problems,
                # repeating with a single worker to save memory
                windows = create_fixed_length_windows(
                    BaseConcatDataset(subj),
                    window_size_samples=int(sfreq * epoch_length),
                    window_stride_samples=int(sfreq * epoch_overlap),
                    drop_last_window=True,
                    n_jobs=1,
                )

            # extract raw EEG and metadata
            stage["stage"] = "processing"
            pbar.set_postfix(stage)
            epochs, description = [], []
            for d in windows.datasets:
                # get raw EEG data
                epochs.extend(list(d.windows.get_data().astype(np.float32)))
                # repeat metadata for every window
                desc = [d.description] * len(d.windows)
                description.append(pd.concat(desc, axis=1).T)
            description = pd.concat(description).reset_index()
            description["subject_label"] = description["dataset"].str.cat(
                description["subject"].astype(str), sep="-"
            )
            description["ch_pos_pkl"] = description["ch_pos"].apply(
                lambda x: codecs.encode(pickle.dumps(x), "base64").decode()
            )

            # open the memory mapped data file, prepare storing the data
            flat_epochs_len = sum(map(np.size, epochs))
            file = np.memmap(
                join(result_dir, "raw-" + fname + ".dat"),
                mode="w+" if memmap_len == 0 else "r+",
                dtype=np.float32,
                offset=memmap_len * np.float32().nbytes,
                shape=flat_epochs_len,
            )

            # write epochs to disk
            stage["stage"] = "storing data"
            pbar.set_postfix(stage)

            offset = 0
            for i in range(len(epochs)):
                # write epoch to memory mapped NumPy file
                curr_epoch = epochs[i].T.reshape(-1)
                file[offset : offset + curr_epoch.size] = curr_epoch
                file.flush()

                # append a row in the metadata csv
                metadata.append(
                    [
                        description["subject_label"].iloc[i],  # subject label
                        description["label"].iloc[i],  # class label
                        epochs[i].shape[0],  # number of channels
                        memmap_len + offset,  # start index
                        memmap_len + offset + curr_epoch.size,  # stop index
                        description["ch_pos_pkl"].iloc[i],  # channel positions
                    ]
                )
                offset += curr_epoch.size
            # free epochs object
            del epochs

            # update memmap shape
            memmap_len += flat_epochs_len

            # write metadata to disk
            pd.DataFrame(
                metadata,
                columns=[
                    "subject",
                    "condition",
                    "num_channels",
                    "start_idx",
                    "stop_idx",
                    "channel_pos_pickle",
                ],
            ).to_csv(join(result_dir, "label-" + fname + ".csv"))
            pbar.update()
    pbar.close()
