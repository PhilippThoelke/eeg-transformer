import mne
import numpy as np
import torch
import torch.nn as nn
from mne.io import Raw


class MLP3DPositionalEmbedding(nn.Module):
    """
    MLP positional embedding for 3D data.

    Args:
        dim_model (int): The dimensionality of the model.
        add_class_token (bool): Whether to add a class token.
    """

    def __init__(self, dim_model: int, add_class_token: bool = True):
        super().__init__()
        self.dim_model = dim_model
        self.mlp = nn.Sequential(
            nn.Linear(3, dim_model // 2),
            nn.ReLU(),
            nn.Linear(dim_model // 2, dim_model),
        )
        self.class_token = torch.nn.Parameter(torch.zeros(dim_model)) if add_class_token else None

    def forward(self, x: torch.Tensor, ch_pos: torch.Tensor) -> torch.Tensor:
        """
        Embed the channel positions and optionally prepend a class token to the channel dimension.
        """
        # embed the channel positions
        out = x + self.mlp(ch_pos)

        # prepend class token
        if self.class_token is not None:
            clf_token = torch.ones(out.shape[0], 1, out.shape[-1], device=out.device) * self.class_token
            out = torch.cat([clf_token, out], dim=1)
        return out


class TorchEpoch:
    """
    An epoch with a single label. Contains the raw signal, the channel positions and the label.

    Args:
        signal (torch.Tensor): The raw signal with shape (channels, time).
        ch_pos (torch.Tensor): The channel positions with shape (channels, 3).
        sfreq (float): The sampling frequency of the signal.
        label str: The class label of the data.
    """

    def __init__(self, signal: torch.Tensor, ch_pos: torch.Tensor, sfreq: float, label: torch.Tensor):
        self.signal = signal
        self.ch_pos = ch_pos
        self.sfreq = sfreq
        self.label = label


class PreprocessingConfig:
    """
    The configuration for the preprocessing.

    Args:
        notch_filter (bool): Whether to apply a notch filter.
        low_pass (float): The low pass frequency.
        high_pass (float): The high pass frequency.
        resample (float): Frequency to resample to.
    """

    def __init__(
        self, notch_filter: bool = True, low_pass: float = None, high_pass: float = None, resample: float = None
    ) -> None:
        self.notch_filter = notch_filter
        self.low_pass = low_pass
        self.high_pass = high_pass
        self.resample = resample


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

    # resample
    if config.resample is not None:
        raw = raw.resample(config.resample, verbose="ERROR")

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
