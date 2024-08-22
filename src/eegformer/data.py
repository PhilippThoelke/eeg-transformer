import os

import lightning.pytorch as pl
import numpy as np
import torch
from joblib import Parallel, delayed
from mne.channels import make_standard_montage
from mne.datasets import eegbci
from mne.io import read_raw
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

if "DATA_DIR" not in os.environ:
    raise ValueError("Please set the DATA_DIR environment variable to point to the EEGBCI dataset")

DATA_DIR = os.environ["DATA_DIR"]

PROBLEMATIC_SUBJECTS = [88, 89, 92, 100, 104, 106]


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        chunk_secs=4,
        overlap_secs=2.55,
        batch_size=128,
        num_workers=2,
        train_subjs=(1, 96),
        val_subjs=(96, 106),
        test_subjs=(106, 110),
        debug=False,
    ):
        super().__init__()

        if debug:
            train_subjs = (1, 2)
            val_subjs = (2, 3)
            test_subjs = (3, 4)

        self.save_hyperparameters(ignore="debug")

    def prepare_data(self):
        """Download the dataset."""
        subjs = (
            list(range(*self.hparams.train_subjs))
            + list(range(*self.hparams.val_subjs))
            + list(range(*self.hparams.test_subjs))
        )
        for subj in subjs:
            if subj in PROBLEMATIC_SUBJECTS:
                continue
            eegbci.load_data(subj, list(range(1, 15)), DATA_DIR, update_path=False)

    def setup(self, stage: str):
        """Instantiate the dataset for each split."""
        if stage == "fit":
            self.train_epochs, self.train_pos, self.sfreq, self.ch_names = load_data(
                range(*self.hparams.train_subjs), self.hparams.chunk_secs, self.hparams.overlap_secs
            )
            self.val_epochs, self.val_pos, _, _ = load_data(
                range(*self.hparams.val_subjs), self.hparams.chunk_secs, self.hparams.overlap_secs
            )
            self.test_epochs, self.test_pos, _, _ = load_data(
                range(*self.hparams.test_subjs), self.hparams.chunk_secs, self.hparams.overlap_secs
            )

            self.train_epochs, self.val_epochs, self.test_epochs = normalize(
                self.train_epochs, self.val_epochs, self.test_epochs
            )
        elif stage == "val":
            self.val_epochs, self.val_pos, self.sfreq, self.ch_names = load_data(
                range(*self.hparams.val_subjs), self.hparams.chunk_secs, self.hparams.overlap_secs
            )
            self.val_epochs = normalize(self.val_epochs)
        elif stage == "test":
            self.test_epochs, self.test_pos, self.sfreq, self.ch_names = load_data(
                range(*self.hparams.test_subjs), self.hparams.chunk_secs, self.hparams.overlap_secs
            )
            self.test_epochs = normalize(self.test_epochs)
        else:
            raise ValueError(f"Invalid stage: {stage}")

    def train_dataloader(self):
        return get_dataloader(
            self.train_epochs,
            self.train_pos,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return get_dataloader(
            self.val_epochs,
            self.val_pos,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return get_dataloader(
            self.test_epochs,
            self.test_pos,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )


def load_data(subjs, chunk_secs, overlap_secs):
    # remove problematic subjects
    subjs = [subj for subj in subjs if subj not in PROBLEMATIC_SUBJECTS]
    if len(subjs) == 0:
        raise ValueError("No valid subjects left after excluding problematic subjects")

    # get raw file paths
    paths = (eegbci.load_data(subj, list(range(1, 15)), DATA_DIR, update_path=False) for subj in subjs)
    paths = sum(paths, [])

    # load first file to retrieve metadata
    raw = load_recording(paths[0], get_raw=True)
    sfreq = raw.info["sfreq"]
    pos = raw._get_channel_positions().astype(np.float32)
    chunk_size = int(chunk_secs * sfreq)
    overlap_size = int(overlap_secs * sfreq)

    # extract epochs from all files
    epochs = Parallel(n_jobs=-1, backend="multiprocessing")(
        delayed(load_recording)(p, sfreq, pos, chunk_size, overlap_size) for p in tqdm(paths, desc="Loading data")
    )
    epochs = np.concatenate(epochs)

    return epochs, pos, sfreq, raw.info["ch_names"]


def get_dataloader(x, pos, **kwargs):
    if pos.ndim == 2:
        # NOTE: the DataLoader requires a batch dimension in all inputs
        pos = pos[None].repeat(len(x), 0)

    dataset = TensorDataset(
        torch.from_numpy(x),
        torch.from_numpy(pos),
    )
    return DataLoader(dataset, **kwargs)


def normalize(*xs, clip_percentile=95):
    # compute median and median absolute deviation from the first input
    med = np.median(xs[0])
    mad = np.median(np.abs(xs[0] - med))
    clip_val = np.percentile(np.abs((xs[0] - med) / mad), clip_percentile)

    xs = list(xs)
    for i in range(len(xs)):
        xs[i] = np.clip((xs[i] - med) / mad, -clip_val, clip_val)
    return tuple(xs) if len(xs) > 1 else xs[0]


def load_recording(path, sfreq=None, ch_pos=None, chunk_size=None, overlap_size=None, get_raw=False):
    assert get_raw or (
        sfreq is not None and ch_pos is not None and chunk_size is not None and overlap_size is not None
    ), "Specify either get_raw=True or provide all metadata"

    raw = read_raw(path, preload=True, verbose=False)
    eegbci.standardize(raw)
    raw.set_montage(make_standard_montage("standard_1005"), verbose=False)
    raw.set_eeg_reference("average", verbose=False)
    raw.notch_filter([60], verbose=False)
    raw.filter(1, None, verbose=False)

    # make sure the metadata matches the reference
    if sfreq is not None:
        assert raw.info["sfreq"] == sfreq, "Sampling frequency mismatch"
    if ch_pos is not None:
        assert np.allclose(raw._get_channel_positions(), ch_pos), "Channel positions mismatch"

    if get_raw:
        # simply return the raw object
        return raw

    # split the data into chunks
    signal = raw.get_data()
    chunks = []
    for i in range(0, signal.shape[1] - chunk_size, chunk_size - overlap_size):
        x = signal[:, i : i + chunk_size]
        chunks.append(x)
    return np.stack(chunks).astype(np.float32)
