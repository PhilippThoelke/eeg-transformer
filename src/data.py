from os.path import join
from abc import ABC, abstractmethod
import pickle
import codecs
import mne
from mne.channels.montage import transform_to_head
import numpy as np
import pandas as pd
from braindecode.datasets import (
    BaseDataset,
    BaseConcatDataset,
    MOABBDataset,
)
from braindecode.preprocessing import (
    preprocess,
    Preprocessor,
    create_fixed_length_windows,
)

mne.set_log_level("ERROR")


def standardize(data, channelwise=True):
    axis = 1 if channelwise else None
    mean, std = data.mean(axis=axis, keepdims=True), data.std(axis=axis, keepdims=True)
    return (data - mean) / std


class ProcessedDataset(BaseConcatDataset, ABC):
    """
    Abstract dataset class used to standardize preprocessing for all inheriting datasets.
    All deriving classes have to implement the `instantiate` method, which should download
    the dataset files and return an instance of `braindecode.datasets.BaseDataset` or
    `braindecode.datasets.BaseConcatDataset`. Deriving classes should further set the
    `line_freq` attribute appropriately.

    Parameters
    ----------
    sfreq : desired sampling frequency after preprocessing
    filter_range : tuple of low and high frequency cutoffs
    kwargs : keyword arguments passed to the deriving dataset class
    """

    def __init__(
        self,
        sfreq=128,
        filter_range=(0.5, None),
        standardize_channelwise=True,
        clip_stds=5,
        **kwargs,
    ):
        assert hasattr(self, "line_freq"), (
            f"{type(self).__name__} doesn't implement the line_freq attribute, "
            "which is required for preprocessing."
        )
        self.sfreq = sfreq
        self.filter_range = filter_range
        self.standardize_channelwise = standardize_channelwise
        self.clip_stds = clip_stds

        # create the dataset
        raw_dset = self.instantiate(**kwargs)
        processed_dset = self.preprocess(raw_dset)

        # combine processed dataset epochs
        super(ProcessedDataset, self).__init__(processed_dset)

    @abstractmethod
    def instantiate(self):
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


class PhysionetMI(ProcessedDataset):
    line_freq = 60

    def instantiate(self, subject_ids=list(range(1, 110))):
        return MOABBDataset(
            "PhysionetMI",
            subject_ids,
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

    def instantiate(self, subject_ids=list(range(1, 5))):
        return MOABBDataset("Zhou2016", subject_ids)

    def prepare_annotations(self, raw):
        keep_annots = ["left_hand", "right_hand", "feet"]
        delete_mask = [d not in keep_annots for d in raw.annotations.description]
        raw.annotations.delete(delete_mask)
        raw.annotations.duration[:] = 5


class MAMEM1(ProcessedDataset):
    line_freq = 50

    def instantiate(self, subject_ids=list(range(1, 12))):
        return MOABBDataset("MAMEM1", subject_ids)

    def label_transform(self, label, description, raw):
        return "flickering_" + label + "Hz"


if __name__ == "__main__":
    result_dir = "data/"

    dset = BaseConcatDataset(
        [
            PhysionetMI(subject_ids=[1, 2]),
            Zhou2016(subject_ids=[1, 2]),
            # MAMEM1(subject_ids=[1, 2]),
        ]
    )
    sfreq = dset.datasets[0].raw.info["sfreq"]
    dset = create_fixed_length_windows(
        dset,
        window_size_samples=int(sfreq * 2),
        window_stride_samples=int(sfreq * 0.3),
        drop_last_window=True,
        n_jobs=-1,
    )

    epochs = sum(
        (list(d.windows.get_data()) for d in dset.datasets),
        [],
    )
    description = pd.concat(
        [pd.concat([d.description] * len(d.windows), axis=1).T for d in dset.datasets]
    ).reset_index()
    labels = description["label"].values
    subject_labels = (
        description["dataset"]
        .str.cat(description["subject"].astype(str), sep="-")
        .values
    )
    ch_pos = description["ch_pos"].values

    max_channels = max([c.shape[0] for c in ch_pos])
    shape = len(epochs), epochs[0].shape[1], max_channels
    
    fname = f"nsamp_{shape[0]}-eplen_{shape[1]}-nchan_{shape[2]}"

    print("\nSaving raw data...", end="")
    file = np.memmap(
        join(result_dir, "raw-" + fname + ".dat"),
        mode="w+",
        dtype=np.float32,
        shape=shape,
    )
    metadata = pd.DataFrame(
        index=np.arange(shape[0], dtype=int),
        columns=["subject", "condition", "channel_pos_pickle"],
    )

    for i in range(shape[0]):
        curr_epoch = epochs[i].T
        if curr_epoch.shape[1] < max_channels:
            padding = np.zeros((shape[1], max_channels - curr_epoch.shape[1]))
            curr_epoch = np.concatenate([curr_epoch, padding], axis=1)
        file[i] = curr_epoch
        file.flush()

        ch_pos_pickle = codecs.encode(pickle.dumps(ch_pos[i]), "base64").decode()
        metadata.iloc[i] = [subject_labels[i], labels[i], ch_pos_pickle]
    print("done")

    print("Saving metadata...", end="")
    with open(join(result_dir, "label-" + fname + ".csv"), "w") as f:
        f.write(f"#shape={shape}\n")
        metadata.to_csv(f)
    print("done")
