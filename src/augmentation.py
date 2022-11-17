import torch
from torchaudio.functional import resample as _resample


def gaussian_noise(x, ch_pos, mask, noise_scale=0.1):
    """Add Gaussian noise to the signal"""
    return x + torch.randn_like(x) * x.std() * noise_scale, ch_pos, mask


def gaussian_noise_channels(x, ch_pos, mask, noise_scale=0.1):
    """Add Gaussian noise to the channel positions"""
    return x, ch_pos + torch.randn_like(ch_pos) * ch_pos.std() * noise_scale, mask


def gaussian_noise_freq(x, ch_pos, mask, noise_scale=0.15):
    """Add Gaussian noise to the Fourier transformed signal and
    transform back into the temporal domain"""
    ft = torch.fft.fft(x, dim=1)
    noise = torch.randn_like(ft.real)
    noise = noise + noise * 1j
    ft = ft + noise * ft.std() * noise_scale
    return torch.fft.ifft(ft, dim=1).real, ch_pos, mask


def rescale_channels(x, ch_pos, mask, max_strength=0.3):
    """Scale the signal by a random scalar around 1"""
    scalers = torch.rand(x.size(0), 1, x.size(2), device=x.device)
    scalers = scalers * 2 * max_strength + 1 - max_strength
    mean = x.mean(dim=1, keepdims=True)
    return (x - mean) * scalers + mean, ch_pos, mask


def random_mean(x, ch_pos, mask, max_strength=0.2):
    """Add a random mean to the signal"""
    means = torch.rand(x.size(0), 1, x.size(2), device=x.device)
    means = means * 2 * max_strength - max_strength
    return x + means, ch_pos, mask


def shift_samples(x, ch_pos, mask, max_n=10):
    """Randomly shift the signal forwards of backwards in time"""
    shifts = torch.randint(1, max_n + 1, (x.size(0),), device=x.device)
    shifts[torch.rand(shifts.shape, device=x.device) < 0.5] *= -1
    for i, shift in enumerate(shifts):
        x[i] = torch.cat([x[i, -shift:], x[i, :-shift]], dim=0)
    return x, ch_pos, mask


def flip_sign(x, ch_pos, mask, prob=0.2):
    """Randomly flip the sign of the signal"""
    flip_mask = torch.rand(x.size(0), device=x.device) < prob
    x[flip_mask] = x[flip_mask] * -1
    return x, ch_pos, mask


def resample(x, ch_pos, mask, max_change_pct=30):
    """Downsample the signal to a lower frequency,
    follwed by upsampling to the original sampling rate"""
    x = x.permute(0, 2, 1).contiguous()
    freqs = torch.randint(100 - max_change_pct, 100, (x.size(0),))
    for i, freq in enumerate(freqs):
        freq = freq.item()
        x[i] = _resample(_resample(x[i], 100, freq), freq, 100)[:, : x.size(2)]
    return x.permute(0, 2, 1), ch_pos, mask


def temporal_dropout(x, ch_pos, mask, max_samples=64, prob=0.2):
    """Set a random section of a random subset of channels to 0"""
    mask = torch.rand(x.size(0), x.size(2), device=x.device) < prob
    sizes = torch.randint(1, max_samples + 1, (mask.sum(),), device=x.device)
    for b, c, size in zip(*torch.where(mask), sizes):
        start = torch.randint(0, x.size(1) - size, (1,)).item()
        x[b, start : start + size, c] = 0
    return x, ch_pos, mask


def channel_dropout(x, ch_pos, mask, prob=0.1):
    """Randomly mask out channels"""
    if mask is None:
        mask = torch.ones(x.size(0), x.size(2), dtype=torch.bool, device=x.device)
    dropout_mask = torch.rand(mask.shape, device=mask.device) < prob
    mask[dropout_mask] = False
    return x, ch_pos, mask


augmentations = [
    gaussian_noise,
    gaussian_noise_channels,
    gaussian_noise_freq,
    rescale_channels,
    random_mean,
    shift_samples,
    flip_sign,
    resample,
    temporal_dropout,
    channel_dropout,
]

# TODO
# - random low-pass, high-pass, notch-filtering
# - randomize phase in frequency domain (same for all channels)