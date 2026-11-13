"""
Log-spectral distance (LSD) metric.

Used as a validation metric for the enhancement-model GAN training pipeline
(src/training/lightningmodule/mixture_restoration_gan.py), mirroring the
`val/lsd` metric Hachimi reports (https://github.com/linfeng-feng/Hachimi).
Hachimi's own metric implementation (`utils/func.py`) isn't published in their
repository, so this is a standard-formula reimplementation rather than a port.
"""

import torch


def lsd_metric(pred_stft: torch.Tensor, target_stft: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Log-spectral distance between two complex STFT tensors of matching shape
    [..., F, T]: the mean, over time frames, of the RMS-over-frequency dB-magnitude
    difference.
    """
    pred_db = 20.0 * torch.log10(pred_stft.abs().clamp(min=eps))
    target_db = 20.0 * torch.log10(target_stft.abs().clamp(min=eps))
    squared_diff = (pred_db - target_db) ** 2
    per_frame = torch.sqrt(squared_diff.mean(dim=-2))  # average over frequency axis
    return per_frame.mean()
