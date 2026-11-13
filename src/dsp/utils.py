"""Generic DSP helpers."""

from __future__ import annotations

from typing import Tuple, Union, Dict
import random
import numpy as np
import torch


def mix_stems_to_snr_range(
    stems: Union[Dict[str, np.ndarray], Dict[str, torch.Tensor]],
    snr_range: Tuple[float, float],
    reference_stem_key: str,
    normalize_output: bool = True,
) -> Tuple[Union[np.ndarray, torch.Tensor], Dict[str, float]]:
    """
    Mix multiple stems with SNR-based level control.
    
    This function mixes multiple audio stems together, controlling their relative loudness
    based on a signal-to-noise ratio (SNR) range. Each stem (except the reference) is
    scaled relative to the reference stem to achieve a random SNR within the specified range.
    
    Parameters
    ----------
    stems:
        Dict of audio stems to mix. Each stem should be the same shape (channels, samples)
        or (samples,) for mono. Can be numpy arrays or torch tensors.
    snr_range:
        Tuple of (min_snr_db, max_snr_db) for the SNR range in dB.
    reference_stem_key:
        Key name of the reference stem.
        All other stems will be scaled relative to this one. Default: None
    normalize_output:
        If True, normalize the output mixture to prevent clipping (scale to max 0.95).
        Default: True
    
    Returns
    -------
    mixture:
        Mixed audio signal with the same shape and type as input stems
    scales:
        Dictionary mapping stem index/key to the scale factor applied to each stem
    
    Examples
    --------
    >>> import numpy as np
    >>> stems = [np.random.randn(2, 48000), np.random.randn(2, 48000)]
    >>> mixture, scales = mix_stems_to_snr_range(stems, snr_range=(-3.0, 3.0))
    >>> print(f"Applied scales: {scales}")
    """
    if isinstance(stems, dict):
        stem_keys = list(stems.keys())
        stem_list = [stems[k] for k in stem_keys]
    else:
        raise NotImplementedError()

    
    if len(stem_list) < 2:
        raise ValueError("Need at least 2 stems to mix")
    
    # Determine if we're using numpy or torch
    first_stem = stem_list[0]
    is_torch = isinstance(first_stem, torch.Tensor)
    
    # Convert to numpy for processing if needed (we'll convert back at the end)
    if is_torch:
        device = first_stem.device
        dtype = first_stem.dtype
        stem_list_np = [s.detach().cpu().numpy() for s in stem_list]
    else:
        stem_list_np = [np.asarray(s, dtype=np.float32) for s in stem_list]
    
    # Ensure all stems have the same shape
    ref_shape = stem_list_np[0].shape
    for i, stem in enumerate(stem_list_np):
        if stem.shape != ref_shape:
            raise ValueError(f"Stem {i} has shape {stem.shape}, expected {ref_shape}")
    
    # Get reference stem
    ref_idx = stem_keys.index(reference_stem_key)
    
    reference_stem = stem_list_np[ref_idx]
    
    # Calculate power of reference stem
    ref_power = np.mean(reference_stem ** 2)
    if ref_power < 1e-8:
        # Reference is silent, just sum all stems
        mixture = np.sum(stem_list_np, axis=0)
        scales = {k: 1.0 for k in stem_keys}
    else:
        # Initialize mixture with reference stem
        mixture = reference_stem.copy()
        scales = {stem_keys[ref_idx]: 1.0}
        
        # Mix other stems with random SNR
        for i, stem in enumerate(stem_list_np):
            if i == ref_idx:
                continue
            
            stem_power = np.mean(stem ** 2)
            if stem_power < 1e-8:
                # Silent stem, skip it
                scales[stem_keys[i]] = 0.0
                continue
            
            # Random SNR in the specified range
            target_snr_db = random.uniform(snr_range[0], snr_range[1])
            target_snr_linear = 10 ** (target_snr_db / 10)
            
            # Calculate scale factor: SNR = ref_power / (stem_power * scale^2)
            # So: scale^2 = ref_power / (stem_power * SNR)
            # scale = sqrt(ref_power / (stem_power * SNR))
            scale = np.sqrt(ref_power / (stem_power * target_snr_linear))
            
            scaled_stem = stem * scale
            mixture = mixture + scaled_stem
            scales[stem_keys[i]] = float(scale)
    
    # Normalize to prevent clipping
    if normalize_output:
        max_amplitude = np.max(np.abs(mixture))
        if max_amplitude > 1.0:
            norm_factor = 0.95 / max_amplitude
            mixture = mixture * norm_factor
            # Adjust scales accordingly
            scales = {k: v * norm_factor for k, v in scales.items()}
    
    # Convert back to torch if needed
    if is_torch:
        mixture = torch.from_numpy(mixture).to(device=device, dtype=dtype)
    
    return mixture, scales


def fix_length_to_duration_torch(target: torch.Tensor, duration: int) -> torch.Tensor:
    """Trim or zero-pad a tensor to exactly `duration` samples along the last dimension."""
    target_duration = target.shape[-1]

    if target_duration < duration:
        target = torch.nn.functional.pad(target, (0, duration - target_duration))
    elif target_duration > duration:
        target = target[..., :duration]

    return target


__all__ = ["mix_stems_to_snr_range", "fix_length_to_duration_torch"]
