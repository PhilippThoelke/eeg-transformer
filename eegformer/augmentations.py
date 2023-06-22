from abc import ABC
from typing import List, Optional, Tuple

import torch
from torch import Tensor


class BaseTransform(ABC):
    """
    Base class for all transforms.
    """

    def __call__(self, args: Tuple[Tensor, Tensor, int]) -> Tuple[Tensor, Tensor, int]:
        """
        Apply the transform to the input arguments.

        ### Args
            - `args` (Tuple[Tensor, Tensor, int]): A tuple of (signal, ch_pos, label).

        ### Returns
            Tuple[Tensor, Tensor, int]: The transformed tuple of (signal, ch_pos, label).
        """
        raise NotImplementedError


class Compose(BaseTransform):
    """
    Compose multiple transforms into one and optionally randomly select a random subset of transforms.

    ### Args
        - `transforms` (List[BaseTransform]): List of transforms to compose.
        - `max_transforms` (Optional[int]): If not None, randomly select a number of transforms from the list.
    """

    def __init__(self, transforms: List[BaseTransform], max_transforms: Optional[int] = None):
        self.transforms = transforms
        if max_transforms is None:
            max_transforms = len(transforms)
        self.max_transforms = min(max_transforms, len(transforms))

    def __call__(self, args: Tuple[Tensor, Tensor, int]) -> Tuple[Tensor, Tensor, int]:
        idxs = torch.randperm(len(self.transforms))[: self.max_transforms]
        for i in idxs:
            args = self.transforms[i](args)
        return args


class RandomAmplitudeScaleShift(BaseTransform):
    """
    Randomly scale and shift the amplitude of the signal. The scale and shift are sampled from a normal distribution,
    the standard deviations are additionally multiplied by the signal standard deviation.

    ### Args
        - `std_shift` (float): Standard deviation of the random amplitude shift.
        - `std_scale` (float): Standard deviation of the random amplitude scale.
        - `channelwise` (bool): Whether to apply the same scale and shift to all channels.
    """

    def __init__(self, std_shift=0.1, std_scale=0.1, channelwise=True):
        self.std_shift = std_shift
        self.std_scale = std_scale
        self.channelwise = channelwise

    def __call__(self, args: Tuple[Tensor, Tensor, int]) -> Tuple[Tensor, Tensor, int]:
        signal, ch_pos, label = args
        if self.channelwise:
            shift = torch.randn(signal.shape[0], 1) * signal.std() * self.std_shift
            scale = torch.randn(signal.shape[0], 1) * signal.std() * self.std_scale + 1
        else:
            shift = torch.randn(1) * signal.std() * self.std_shift
            scale = torch.randn(1) * signal.std() * self.std_scale + 1
        return signal * scale + shift, ch_pos, label


class RandomTimeShift(BaseTransform):
    """
    Randomly shift the signal in time by a number of samples. Time points that are shifted beyond the signal length
    are wrapped around to the beginning of the signal.

    ### Args
        - `std` (float): Standard deviation of the random time shift (in samples).
    """

    def __init__(self, std=3.0):
        self.std = std

    def __call__(self, args: Tuple[Tensor, Tensor, int]) -> Tuple[Tensor, Tensor, int]:
        signal, ch_pos, label = args
        shift = torch.randn(1) * self.std
        shift = int(shift.item())
        return torch.roll(signal, shift, dims=1), ch_pos, label


class GaussianNoiseSignal(BaseTransform):
    """
    Add Gaussian noise to the signal. The noise is scaled by the
    standard deviation of the signal.

    ### Args
        - `std` (float): The standard deviation of the Gaussian distribution.
    """

    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, args: Tuple[Tensor, Tensor, int]) -> Tuple[Tensor, Tensor, int]:
        signal, ch_pos, label = args
        return signal + (torch.randn_like(signal) * signal.std() * self.std), ch_pos, label


class GaussianNoiseChannelPos(BaseTransform):
    """
    Add Gaussian noise to the channel positions. The noise is additionally scaled by
    the standard deviation of the channel positions.

    ### Args
        - `std` (float): The standard deviation of the Gaussian distribution.
    """

    def __init__(self, std=0.08):
        self.std = std

    def __call__(self, args: Tuple[Tensor, Tensor, int]) -> Tuple[Tensor, Tensor, int]:
        signal, ch_pos, label = args
        return signal, ch_pos + (torch.randn_like(ch_pos) * ch_pos.std() * self.std), label


class FourierNoise(BaseTransform):
    """
    Add Fourier noise to the signal.

    ### Args
        - `std` (float): The standard deviation of the Gaussian distribution.
    """

    def __init__(self, std=0.5):
        self.std = std

    def __call__(self, args: Tuple[Tensor, Tensor, int]) -> Tuple[Tensor, Tensor, int]:
        signal, ch_pos, label = args
        ft = torch.fft.fft(signal, dim=1)
        ft = ft + (torch.randn_like(ft) * ft.std(dim=0, keepdims=True) * self.std)
        return torch.fft.ifft(ft, dim=1).real, ch_pos, label


class RandomPhase(BaseTransform):
    """
    Randomly shift the phase of the signal in the frequency domain.

    ### Args
        - `strength` (float): The strength of the randomization.
    """

    def __init__(self, strength=0.3):
        self.strength = strength

    def __call__(self, args: Tuple[Tensor, Tensor, int]) -> Tuple[Tensor, Tensor, int]:
        signal, ch_pos, label = args
        # transform the data into frequency domain
        ft = torch.fft.fft(signal.T, dim=0)
        # generate random rotations
        segment_size = ft.size(0) // 2 - 1 if ft.size(0) % 2 == 0 else (ft.size(0) - 1) // 2
        phase_segment = torch.rand(segment_size, 1) * torch.pi * 2j
        # combine random segments to match Fourier phase
        phase_segments = [
            torch.zeros(1, 1),
            phase_segment,
            torch.zeros(1 if ft.size(0) % 2 == 0 else 0, 1),
            -torch.flip(phase_segment, (0,)),
        ]
        random_phase = torch.cat(phase_segments, dim=0)
        # transform back into the time domain
        signal = torch.fft.ifft(ft * (random_phase * self.strength).exp(), dim=0).real.T
        return signal, ch_pos, label


class ChannelDropout(BaseTransform):
    """
    Randomly drop channels from the signal.

    ### Args
        - `p` (float): The probability of dropping a channel.
        - `min_channels` (int): The minimum number of channels to keep.
    """

    def __init__(self, p=0.2, min_channels=4):
        self.p = p
        self.min_channels = min_channels

    def __call__(self, args: Tuple[Tensor, Tensor, int]) -> Tuple[Tensor, Tensor, int]:
        signal, ch_pos, label = args
        # generate a mask of channels to keep
        keep_mask = torch.rand(signal.shape[0]) > self.p
        # make sure we keep at least `min_channels` channels
        if keep_mask.sum() < self.min_channels:
            keep_mask[torch.randperm(signal.shape[0])[: self.min_channels]] = True
        # return the data with the dropped channels
        return signal[keep_mask], ch_pos[keep_mask], label
