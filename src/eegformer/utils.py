import hashlib
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mne
import numpy as np
import torch
from jsonargparse import Namespace
from mne.io import Raw
from torch import Tensor


@dataclass
class PreprocessingConfig:
    """
    Preprocessing configuration.

    ### Args:
        - `notch_filter` (bool): Whether to apply a notch filter (default: True).
        - `low_pass` (float): The low pass frequency (default: None).
        - `high_pass` (float): The high pass frequency (default: None).
        - `resample` (float): Frequency to resample to (default: None).
        - `epoch_length` (float): Length of epochs in seconds (default: 2.0).
        - `epoch_overlap` (float): Overlap of epochs in seconds (default: 0.5).
        - `force` (bool): Whether to force preprocessing even if the file already exists (default: False).
        - `n_jobs` (int): Number of jobs to use for parallel processing (default: -1).
    """

    notch_filter: bool = True
    low_pass: Optional[float] = None
    high_pass: Optional[float] = None
    resample: Optional[float] = None
    epoch_length: float = 2.0
    epoch_overlap: float = 0.5
    force: bool = False
    n_jobs: int = -1

    def __hash__(self) -> int:
        """
        Compute the hash of the configuration, only considering attributes that affect the output.
        """

        # select attributes and make sure they have their respective type
        values = (
            bool(self.notch_filter),
            float(self.low_pass) if self.low_pass is not None else None,
            float(self.high_pass) if self.high_pass is not None else None,
            float(self.resample) if self.resample is not None else None,
            float(self.epoch_length),
            float(self.epoch_overlap),
        )
        # return the has of the tuple of all values
        return int(hashlib.sha1(repr(values).encode("utf-8")).hexdigest(), 16) % (10**10)


def preprocess(raw: Raw, config: PreprocessingConfig) -> List[np.ndarray]:
    """
    Preprocess a raw MNE object.

    ### Args:
        - `raw` (mne.io.Raw): The raw MNE object.
        - `config` (PreprocessingConfig): The preprocessing configuration.

    ### Returns:
        List[np.ndarray]: The preprocessed raw epochs.
    """
    if (raw.tmax - raw.tmin) < config.epoch_length:
        # return empty list if the raw data is too short
        return []

    # apply notch filter
    if config.notch_filter:
        if raw.info["line_freq"] is None:
            raise ValueError("Line frequency is not set.")

        freqs = np.arange(raw.info["line_freq"], raw.info["sfreq"] / 2, raw.info["line_freq"])
        raw = raw.notch_filter(freqs, verbose="ERROR", n_jobs=config.n_jobs)

    # apply band pass filter
    if config.low_pass is not None or config.high_pass is not None:
        raw = raw.filter(config.low_pass, config.high_pass, verbose="ERROR", n_jobs=config.n_jobs)

    # resample
    if config.resample is not None and raw.info["sfreq"] != config.resample:
        raw = raw.resample(config.resample, verbose="ERROR", n_jobs=config.n_jobs)

    # split into epochs
    epochs = mne.make_fixed_length_epochs(
        raw, duration=config.epoch_length, overlap=config.epoch_overlap, verbose="ERROR"
    )

    # return preprocessed raw epochs
    return list(epochs.get_data().astype(np.float32))


def extract_ch_pos(raw: Raw) -> np.ndarray:
    """
    Get the channel positions from a raw MNE object.

    ### Args:
        - `raw` (mne.io.Raw): The raw MNE object.

    ### Returns:
        np.ndarray: The channel positions with shape (channels, 3).
    """
    montage = raw.get_montage()
    if montage is None:
        raise ValueError("Montage is not set.")

    # transform coordinates to head space
    montage.apply_trans(mne.channels.compute_native_head_t(montage))

    # get channel positions
    ch_pos = montage.get_positions()["ch_pos"]
    ch_pos = np.array([ch_pos[ch] for ch in raw.ch_names], dtype=np.float32)
    return ch_pos


def subsample_signal_batch(
    batch: Tuple[Tensor, Tensor, Optional[Tensor], Tensor], hparams: Namespace
) -> Tuple[Tensor, Tensor, Optional[Tensor], Tensor]:
    """
    Subsample the time dimension of a batch of signals and concatenate the resulting signals.

    ### Args
        - `batch` (Tuple[Tensor, Tensor, Optional[Tensor], Tensor]): The batch of signals.
        - `hparams` (Namespace): The hyperparameters, must contain `input_dim` and `similarity_subsamples`.

    ### Returns
        Tuple[Tensor, Tensor, Optional[Tensor], Tensor]: The subsampled batch.
    """
    signal, ch_pos, mask, y = batch

    # checks
    assert "input_dim" in hparams, "hparams must contain input_dim"
    assert "similarity_subsamples" in hparams, "hparams must contain similarity_subsamples"
    assert signal.size(2) > (
        hparams.input_dim + hparams.similarity_subsamples - 1
    ), "Signal must be longer than input_dim + similarity_subsamples - 1"

    # subsample the signal
    signals = []
    start_idxs = np.linspace(0, signal.size(2) - hparams.input_dim, hparams.similarity_subsamples, dtype=int)
    for i in start_idxs:
        signals.append(signal[:, :, i : i + hparams.input_dim])

    # concatenate and reassemble the batch
    signal = torch.cat(signals, dim=0)
    ch_pos = ch_pos.repeat(hparams.similarity_subsamples, 1, 1)
    if mask is not None:
        mask = mask.repeat(hparams.similarity_subsamples, 1)
    y = y.repeat(hparams.similarity_subsamples)
    return signal, ch_pos, mask, y
