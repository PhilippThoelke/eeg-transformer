import mne
import numpy as np
import torch
from mne.io import Raw


class TorchEpoch:
    """
    An epoch with a single label. Contains the raw signal, the channel positions and the label.

    Args:
        signal (torch.Tensor): The raw signal with shape (time, channels).
        ch_pos (torch.Tensor): The channel positions with shape (channels, 3).
        label str: The class label of the data.
    """

    def __init__(self, signal: torch.Tensor, ch_pos: torch.Tensor, label: torch.Tensor):
        self.signal = signal
        self.ch_pos = ch_pos
        self.label = label


class PreprocessingConfig:
    def __init__(self, notch_filter: bool = True, low_pass: float = None, high_pass: float = None) -> None:
        self.notch_filter = notch_filter
        self.low_pass = low_pass
        self.high_pass = high_pass


def preprocess(raw: Raw, config: PreprocessingConfig) -> Raw:
    """
    Preprocess a raw MNE object.

    Args:
        raw (mne.io.Raw): The raw MNE object.
        config (PreprocessingConfig): The preprocessing configuration.
    """
    # apply notch filter
    if config.notch_filter:
        if raw.info["line_freq"] is None:
            raise ValueError("Line frequency is not set.")

        freqs = np.arange(raw.info["line_freq"], raw.info["sfreq"] / 2, raw.info["line_freq"])
        raw = raw.notch_filter(freqs, verbose="ERROR")

    # apply band pass filter
    if config.low_pass is not None or config.high_pass is not None:
        raw = raw.filter(config.low_pass, config.high_pass, verbose="ERROR")

    # return preprocessed raw
    return raw


def get_channel_pos(raw: Raw) -> torch.Tensor:
    """
    Get the channel positions from a raw MNE object.

    Args:
        raw (mne.io.Raw): The raw MNE object.

    Returns:
        torch.Tensor: The channel positions.
    """
    montage = raw.get_montage()
    if montage is None:
        raise ValueError("Montage is not set.")

    # transform coordinates to head space
    montage.apply_trans(mne.channels.compute_native_head_t(montage))

    # get channel positions
    ch_pos = montage.get_positions()["ch_pos"]
    ch_pos = np.array([ch_pos[ch] for ch in raw.ch_names])
    return torch.from_numpy(ch_pos).float()
