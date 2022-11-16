from os.path import join
from abc import ABC, abstractmethod
import pickle
import codecs
from tqdm import tqdm
import mne
from mne.channels.montage import transform_to_head
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
    max_subjects : limits the number of subjects for debugging purposes
    kwargs : keyword arguments passed to the deriving dataset class
    """

    def __init__(
        self,
        sfreq=128,
        filter_range=(0.5, None),
        standardize_channelwise=True,
        clip_stds=5,
        max_subjects=-1,
        **kwargs,
    ):
        assert hasattr(
            self, "line_freq"
        ), f"{type(self).__name__} doesn't implement the line_freq attribute."
        assert hasattr(
            self, "subject_ids"
        ), f"{type(self).__name__} doesn't implement the subject_ids attribute."

        self.sfreq = sfreq
        self.filter_range = filter_range
        self.standardize_channelwise = standardize_channelwise
        self.clip_stds = clip_stds
        self.max_subjects = max_subjects
        self.instantiate_kwargs = kwargs

    def iter_subjects(self):
        subject_ids = self.subject_ids
        if self.max_subjects > 0:
            subject_ids = self.subject_ids[: self.max_subjects]
        for subject_id in subject_ids:
            raw_dset = self.instantiate(subject_id, **self.instantiate_kwargs)
            yield self.preprocess(raw_dset)

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
        processed_dset = preprocess(raw_dset, preprocessors)

        epochs = []
        for curr in processed_dset.datasets:
            self.prepare_annotations(curr.raw)
            raws = self.crop_by_annotations(curr.raw)
            for raw_epoch, label in raws:
                # resample after epoching to avoid event jitter
                raw_epoch = raw_epoch.resample(self.sfreq)

                # update description object
                desc = curr.description.copy()
                desc["label"] = self.label_transform(label, desc, raw_epoch)

                # add channel positions to description
                montage = transform_to_head(raw_epoch.get_montage())
                ch_pos = montage.get_positions()["ch_pos"]
                ch_pos = np.stack([ch_pos[ch_name] for ch_name in raw_epoch.ch_names])
                desc["ch_pos"] = ch_pos
                desc["dataset"] = self.__class__.__name__

                # append current epoch as a BaseDataset
                epochs.append(BaseDataset(raw_epoch, desc, target_name="label"))
        return epochs

    def __len__(self):
        return self.max_subjects if self.max_subjects > 0 else len(self.subject_ids)


class PhysionetMI(ProcessedDataset):
    line_freq = 60
    subject_ids = list(range(1, 110))

    def instantiate(self, subject_id):
        return MOABBDataset(
            "PhysionetMI",
            subject_id,
            dataset_kwargs=dict(imagined=True, executed=True),
        )

    def label_transform(self, label, description, raw):
        if label == "rest":
            return label

        run_idx = int(raw.filenames[0][-6:-4])
        if run_idx in [3, 5, 7, 9, 11, 13]:
            return "executed_" + label
        elif run_idx in [4, 6, 8, 10, 12, 14]:
            return "imagined_" + label
        else:
            raise ValueError(
                f"Unknown run index {run_idx}, expected to be between 3 and 14."
            )


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
    subject_ids = list(range(1, 12))

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


if __name__ == "__main__":
    result_dir = "data/"
    epoch_length = 2
    epoch_overlap = 0.3
    sfreq = 128

    # define datasets
    datasets = [
        PhysionetMI(sfreq=sfreq),
        Zhou2016(sfreq=sfreq),
        MAMEM1(sfreq=sfreq),
        RestingCognitive(sfreq=sfreq),
    ]

    # prepare processing the data
    mne.set_log_level("ERROR")
    fname = "-".join(d.__class__.__name__ for d in datasets)
    memmap_len = 0
    metadata = []
    pbar = tqdm(total=sum(len(d) for d in datasets))
    stage = dict()

    # iterate all data
    for dset in datasets:
        subj_iter = dset.iter_subjects()
        for subj_idx in range(len(dset)):
            stage["dataset"] = dset.__class__.__name__
            stage["subject"] = subj_idx

            # retrieve data and preprocess for the current subject
            stage["stage"] = "preprocessing"
            pbar.set_postfix(stage)
            subj = next(subj_iter)

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
                print(
                    f"ERROR during windowing: dataset {dset.__class__.__name__}"
                    f", subject {subj_idx}"
                )
                pbar.update()
                continue

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
