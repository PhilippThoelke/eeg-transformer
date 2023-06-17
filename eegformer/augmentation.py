from abc import ABC
from typing import List, Optional, Tuple

import torch


class BaseTransform(ABC):
    """
    Base class for all transforms.
    """

    def __call__(self, signal: torch.Tensor, ch_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class Compose(BaseTransform):
    """
    Compose multiple transforms into one and optionally randomly select a random subset of transforms.

    Args:
        transforms (List[BaseTransform]): List of transforms to compose.
        max_transforms (Optional[int]): If not None, randomly select a number of transforms from the list.
    """

    def __init__(self, transforms: List[BaseTransform], max_transforms: Optional[int] = None):
        self.transforms = transforms

    def __call__(self, signal: torch.Tensor, ch_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for transform in self.transforms:
            signal, ch_pos = transform(signal, ch_pos)
        return signal, ch_pos


class RandomAmplitudeScaleShift(BaseTransform):
    """
    Randomly scale and shift the amplitude of the signal. The scale and shift are sampled from a normal distribution,
    the standard deviations are additionally multiplied by the signal standard deviation.

    Args:
        mean_shift (float): Mean of the random amplitude shift.
        std_shift (float): Standard deviation of the random amplitude shift.
        mean_scale (float): Mean of the random amplitude scale.
        std_scale (float): Standard deviation of the random amplitude scale.
        channelwise (bool): Whether to apply the same scale and shift to all channels.
    """

    def __init__(self, mean_shift=0.0, std_shift=0.1, mean_scale=1.0, std_scale=0.1, channelwise=True):
        self.mean_shift = mean_shift
        self.std_shift = std_shift
        self.mean_scale = mean_scale
        self.std_scale = std_scale
        self.channelwise = channelwise

    def __call__(self, signal: torch.Tensor, ch_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.channelwise:
            shift = torch.randn(signal.shape[0], 1) * signal.std() * self.std_shift + self.mean_shift
            scale = torch.randn(signal.shape[0], 1) * signal.std() * self.std_scale + self.mean_scale
        else:
            shift = torch.randn(1) * signal.std() * self.std_shift + self.mean_shift
            scale = torch.randn(1) * signal.std() * self.std_scale + self.mean_scale
        return signal * scale + shift, ch_pos


class RandomTimeShift(BaseTransform):
    """
    Randomly shift the signal in time by a number of samples. Time points that are shifted beyond the signal length
    are wrapped around to the beginning of the signal.

    Args:
        std (float): Standard deviation of the random time shift (in samples).
    """

    def __init__(self, std=10.0):
        self.std = std

    def __call__(self, signal: torch.Tensor, ch_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        shift = torch.randn(1) * self.std
        shift = int(shift.item())
        return torch.roll(signal, shift, dims=1), ch_pos


class GaussianNoiseSignal(BaseTransform):
    """
    Add Gaussian noise to the signal. The noise is scaled by the
    standard deviation of the signal.

    Args:
        mean (float): The mean of the Gaussian distribution.
        std (float): The standard deviation of the Gaussian distribution.
    """

    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, signal: torch.Tensor, ch_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return signal + (torch.randn_like(signal) * signal.std() * self.std + self.mean), ch_pos


class GaussianNoiseChannelPos(BaseTransform):
    """
    Add Gaussian noise to the channel positions. The noise is additionally scaled by
    the standard deviation of the channel positions.

    Args:
        mean (float): The mean of the Gaussian distribution.
        std (float): The standard deviation of the Gaussian distribution.
    """

    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, signal: torch.Tensor, ch_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return signal, ch_pos + (torch.randn_like(ch_pos) * ch_pos.std() * self.std + self.mean)


class FourierNoise(BaseTransform):
    """
    Add Fourier noise to the signal.

    Args:
        mean (float): The mean of the Gaussian distribution.
        std (float): The standard deviation of the Gaussian distribution.
    """

    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, signal: torch.Tensor, ch_pos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ft = torch.fft.fft(signal)
        ft = ft + (torch.randn_like(ft) * ft.std() * self.std + self.mean)
        return torch.fft.ifft(ft).real, ch_pos