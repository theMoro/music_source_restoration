"""
Combined loss with automatic L1 scaling: L1 + MR-STFT + SI-SNR + SNR + Multi-Mel-SNR.

This module provides a loss function that combines L1, MR-STFT, SI-SNR, SNR, Multi-Mel-SNR losses
with automatic dynamic scaling of the L1 loss based on the mean amplitude of predictions.
The L1 weight is computed as: (1e5 / mean_amplitude).clamp(max=1e6)

Formula: l1_weight * l1_loss + mr_stft_loss + si_snr_weight * si_snr_loss + snr_weight * snr_loss + multi_mel_snr_weight * multi_mel_snr_loss
"""

import torch
from typing import Dict, Callable, List, Optional, Tuple
from torch.cuda.amp import autocast
from torchmetrics.functional.audio import (
    scale_invariant_signal_noise_ratio as si_snr,
    signal_noise_ratio as snr
)
from einops import rearrange

from src.training.loss.multi_mel_snr import get_loss_func as get_multi_mel_snr_loss_func


def get_loss_func(
    l1_base_weight: float = 1e5,
    l1_max_weight: float = 1e6,
    si_snr_weight: float = 0.1,
    snr_weight: float = 0.0,
    multi_mel_snr_weight: float = 0.0,
    multi_mel_snr_sample_rate: int = 48000,
    low_amplitude_penalty_weight: float = 0.0,
    low_amplitude_ratio: float = 0.4,
    low_amplitude_target_floor: float = 1e-4,
    multi_stft_resolution_loss_weight: float = 1.0,
    multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
    multi_stft_hop_size: int = 147,
    multi_stft_normalized: bool = False,
    multi_stft_window_fn: Callable = torch.hann_window,
    multi_stft_n_fft: int = 2048,
    stem_weights: Optional[List[float]] = None,
):
    """
    Factory function for combined L1 + MR-STFT + SI-SNR + SNR + Multi-Mel-SNR loss with automatic L1 scaling.
    
    Args:
        l1_base_weight: Base weight for L1 scaling (default: 1e5)
        l1_max_weight: Maximum allowed L1 weight (default: 1e6)
        si_snr_weight: Weight for SI-SNR loss component (default: 0.1)
        snr_weight: Weight for SNR (non-scale-invariant) loss component (default: 0.0)
        multi_mel_snr_weight: Weight for Multi-Mel-SNR loss component (default: 0.0)
        multi_mel_snr_sample_rate: Sample rate for Multi-Mel-SNR computation (default: 48000)
        low_amplitude_penalty_weight: Weight for the low-amplitude hinge penalty (default: 0.0 disables it)
        low_amplitude_ratio: Minimum fraction of target amplitude predictions should match when target is active
        low_amplitude_target_floor: Minimum target amplitude considered "active" for the penalty
        multi_stft_resolution_loss_weight: Weight for the multi-resolution loss
        multi_stft_resolutions_window_sizes: Tuple of window sizes for different resolutions
        multi_stft_hop_size: Hop size for STFT
        multi_stft_normalized: Whether to normalize the STFT
        multi_stft_window_fn: Window function to use
        multi_stft_n_fft: Base FFT size
        stem_weights: Per-source scaling factors for L1 and MR-STFT losses, in column order.
            None means all weights are 1.0 (no per-source weighting). The aggregate loss used
            for optimisation is the active-fraction-weighted mean of (stem_weight * per_source_loss).
            When all weights equal 1.0, this is identical to the unweighted behaviour.

    Returns:
        Loss function that takes (output_dict, target_dict) and returns loss_dict
    """
    
    # Initialize Multi-Mel-SNR loss function if weight > 0
    if multi_mel_snr_weight > 0.0:
        multi_mel_snr_loss_fn = get_multi_mel_snr_loss_func(sample_rate=multi_mel_snr_sample_rate)
    else:
        multi_mel_snr_loss_fn = None

    
    def loss_func(output: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate combined loss with automatic L1 scaling.
        
        Args:
            output: Dictionary containing 'waveform' key with model predictions
                   [batch_size, n_sources, n_channels, n_samples]
            target: Dictionary containing 'waveform' key with target sources
                   [batch_size, n_sources, n_channels, n_samples]
        
        Returns:
            Dictionary containing loss components:
                - 'loss': Total combined loss
                - 'l1_loss': Unweighted aggregate L1 loss (mean over active sources)
                - 'l1_loss_weighted': Stem-weighted L1 * dynamic l1_weight (used in total_loss)
                - 'l1_weight': Dynamic L1 weight used (for logging)
                - 'mean_amp': Mean absolute amplitude of predictions (for wandb logging)
                - 'mr_stft_loss': Unweighted aggregate MR-STFT loss
                - 'mr_stft_loss_weighted': Stem-weighted MR-STFT (used in total_loss)
                - 'si_snr_loss': Unweighted SI-SNR loss (negated SI-SNR)
                - 'si_snr_loss_weighted': Weighted SI-SNR loss
                - 'snr_loss': Unweighted SNR loss (negated SNR)
                - 'snr_loss_weighted': Weighted SNR loss
                - 'multi_mel_snr_loss': Unweighted Multi-Mel-SNR loss
                - 'multi_mel_snr_loss_weighted': Weighted Multi-Mel-SNR loss
                Per-source tensors (Tensor[S], not scalars — pop before .item() iteration):
                - 'l1_loss_per_source': Per-source unweighted L1 [S]
                - 'l1_loss_weighted_per_source': Per-source stem_weight * L1 [S]
                - 'mr_stft_loss_per_source': Per-source unweighted MR-STFT [S]
                - 'mr_stft_loss_weighted_per_source': Per-source stem_weight * MR-STFT [S]
        """
        pred = output['waveform']  # [B, S, C, T]
        target_wav = target['waveform']  # [B, S, C, T]

        # Retrieve optional source mask indicating active sources
        source_mask = target.get('source_mask', None)  # binary value for each source
        if source_mask is None:
            source_mask = torch.ones(
                pred.shape[0], pred.shape[1], device=pred.device, dtype=pred.dtype
            )
        else:
            source_mask = source_mask.to(device=pred.device, dtype=pred.dtype)
            if source_mask.dim() > 2:
                source_mask = source_mask.view(source_mask.shape[0], source_mask.shape[1])

        # Expand mask for broadcasting over channels/time
        expand_dims = pred.dim() - source_mask.dim()
        source_mask_expanded = source_mask.view(*source_mask.shape, *([1] * expand_dims))
        mask_sum = source_mask.sum()
        active_sources = mask_sum.clamp_min(1.0)

        # Per-source active counts and stem-weight tensor
        is_4d = pred.dim() == 4
        if is_4d:
            B, S, C, T = pred.shape
            active_per_source = source_mask.sum(dim=0)  # [S]
            active_frac = active_per_source / active_sources  # [S], sums to <=1
            if stem_weights is not None:
                sw = torch.tensor(
                    stem_weights[:S], device=pred.device, dtype=pred.dtype
                )
            else:
                sw = torch.ones(S, device=pred.device, dtype=pred.dtype)

        # Compute unweighted L1 loss (masked to active sources)
        abs_diff = torch.abs(pred - target_wav) * source_mask_expanded
        l1_denominator = (active_sources * pred.shape[2] * pred.shape[3]).clamp_min(1.0)
        l1_loss = abs_diff.sum() / l1_denominator

        # Per-source L1 and stem-weighted aggregate
        if is_4d:
            per_source_l1_denom = (active_per_source * C * T).clamp_min(1.0)  # [S]
            per_source_l1 = abs_diff.sum(dim=(0, 2, 3)) / per_source_l1_denom  # [S]
            per_source_l1_weighted = per_source_l1 * sw  # [S]
            # Aggregate used in total_loss: active-fraction-weighted mean of (sw * per_source)
            # When sw=all-1: equals l1_loss exactly
            l1_loss_for_total = (per_source_l1 * sw * active_frac).sum()
        else:
            per_source_l1 = None
            per_source_l1_weighted = None
            l1_loss_for_total = l1_loss

        # Compute unweighted MR-STFT loss
        device = pred.device
        if is_4d:
            pred_flat = pred.view(B * S, C, T)
            target_flat = target_wav.view(B * S, C, T)
            mask_flat = source_mask.view(B * S)
            mask_channels = mask_flat.repeat_interleave(C)
            mask_channels = mask_channels[:, None, None]
            per_source_mr_stft = torch.zeros(S, device=device, dtype=pred.dtype)
        else:
            pred_flat = pred
            target_flat = target_wav
            mask_flat = source_mask.view(-1)
            mask_channels = mask_flat[:, None, None]
            per_source_mr_stft = None

        mr_stft_loss_unweighted = 0.0
        for window_size in multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, multi_stft_n_fft),
                win_length=window_size,
                return_complex=True,
                window=multi_stft_window_fn(window_size, device=device),
                hop_length=multi_stft_hop_size,
                normalized=multi_stft_normalized
            )

            with autocast(enabled=False):
                recon_Y = torch.stft(rearrange(pred_flat, '... s t -> (... s) t'), **res_stft_kwargs)
                target_Y = torch.stft(rearrange(target_flat, '... s t -> (... s) t'), **res_stft_kwargs)

            diff_raw = torch.abs(recon_Y - target_Y)  # [B*S*C, F, T_stft]
            F_sz, T_stft_sz = diff_raw.shape[1], diff_raw.shape[2]

            # Aggregate unweighted resolution loss (backward-compat formula)
            diff_masked = diff_raw * mask_channels
            resolution_denominator = (
                mask_channels.sum() * F_sz * T_stft_sz
            ).clamp_min(1.0)
            resolution_loss = diff_masked.sum() / resolution_denominator
            mr_stft_loss_unweighted += resolution_loss

            # Per-source resolution loss accumulation
            if is_4d:
                diff_per_src = diff_raw.view(B, S, C, F_sz, T_stft_sz) * source_mask.view(B, S, 1, 1, 1)
                per_src_denom = (active_per_source * C * F_sz * T_stft_sz).clamp_min(1.0)  # [S]
                per_source_mr_stft = per_source_mr_stft + diff_per_src.sum(dim=(0, 2, 3, 4)) / per_src_denom

        if is_4d:
            per_source_mr_stft_weighted = per_source_mr_stft * sw  # [S]
            # Aggregate used in total_loss: active-fraction-weighted mean of (sw * per_source)
            mr_stft_loss_for_total = (per_source_mr_stft * sw * active_frac).sum()
        else:
            per_source_mr_stft_weighted = None
            mr_stft_loss_for_total = mr_stft_loss_unweighted

        # Compute weighted MR-STFT loss (stem-weighted, then resolution weight)
        mr_stft_loss_weighted = mr_stft_loss_for_total * multi_stft_resolution_loss_weight
        
        # Compute SI-SNR loss (reuse flattened tensors from MR-STFT computation)
        # si_snr returns [B*S, C] (reduced over time); denominator must include C
        batch_si_snrs = si_snr(pred_flat, target_flat)  # [B*S, C]
        n_active_channels = mask_flat.sum().clamp_min(1.0) * pred_flat.shape[1]
        masked_si_snr = (batch_si_snrs * mask_flat.unsqueeze(-1)).sum() / n_active_channels
        si_snr_loss_unweighted = -masked_si_snr  # Negated SI-SNR as loss
        si_snr_loss_weighted = si_snr_weight * si_snr_loss_unweighted

        # Compute SNR loss (non-scale-invariant, reuse flattened tensors)
        batch_snrs = snr(pred_flat, target_flat)  # [B*S, C]
        masked_snr = (batch_snrs * mask_flat.unsqueeze(-1)).sum() / n_active_channels
        snr_loss_unweighted = -masked_snr  # Negated SNR as loss
        snr_loss_weighted = snr_weight * snr_loss_unweighted
        
        # Compute Multi-Mel-SNR loss if enabled
        if multi_mel_snr_weight > 0.0 and multi_mel_snr_loss_fn is not None:
            multi_mel_snr_loss_dict = multi_mel_snr_loss_fn(output, target)
            multi_mel_snr_loss_unweighted = multi_mel_snr_loss_dict['loss']
            multi_mel_snr_loss_weighted = multi_mel_snr_weight * multi_mel_snr_loss_unweighted
        else:
            multi_mel_snr_loss_unweighted = torch.tensor(0.0, device=pred.device)
            multi_mel_snr_loss_weighted = torch.tensor(0.0, device=pred.device)
        
        # Dynamic L1 weight based on mean amplitude of prediction
        with torch.no_grad():
            mean_amp_numerator = (pred.abs() * source_mask_expanded).sum()
            mean_amp = (mean_amp_numerator / l1_denominator).clamp(min=1e-8)
            l1_weight = (float(l1_base_weight) / mean_amp).clamp(max=float(l1_max_weight))

        # Stem-weighted aggregate L1 scaled by dynamic weight
        l1_loss_weighted = l1_weight * l1_loss_for_total
        
        # Low amplitude penalty: hinge on mean abs amplitude per source when target is active
        
        if low_amplitude_penalty_weight > 0.0:
            pred_amp = pred.abs().mean(dim=(2, 3))
            target_amp = target_wav.abs().mean(dim=(2, 3))
            target_active_mask = (target_amp >= low_amplitude_target_floor).to(pred.dtype)
            penalty_mask = target_active_mask * source_mask
            penalty_denominator = penalty_mask.sum().clamp_min(1.0)
            required_amp = target_amp * float(low_amplitude_ratio)
            amp_deficit = (required_amp - pred_amp).clamp_min(0.0)
            low_amp_penalty = (amp_deficit * penalty_mask).sum() / penalty_denominator
            low_amp_penalty_weighted = float(low_amplitude_penalty_weight) * low_amp_penalty
        else: 
            low_amp_penalty_weighted = torch.tensor([0.0], device=pred.device)
            low_amp_penalty = torch.tensor([0.0], device=pred.device)

        # Combine losses
        total_loss = (
            l1_loss_weighted
            + mr_stft_loss_weighted
            + si_snr_loss_weighted
            + snr_loss_weighted
            + multi_mel_snr_loss_weighted
            + low_amp_penalty_weighted
        )
        
        loss_dict = {
            'loss': total_loss,
            'l1_loss': l1_loss,
            'l1_loss_weighted': l1_loss_weighted,
            'l1_weight': l1_weight,  # Extra logging key for the dynamic weight
            'mean_amp': mean_amp,  # Mean absolute amplitude of predictions (for wandb logging)
            'mr_stft_loss': mr_stft_loss_unweighted,
            'mr_stft_loss_weighted': mr_stft_loss_weighted,
            'si_snr_loss': si_snr_loss_unweighted,
            'si_snr_loss_weighted': si_snr_loss_weighted,
            'snr_loss': snr_loss_unweighted,
            'snr_loss_weighted': snr_loss_weighted,
            'multi_mel_snr_loss': multi_mel_snr_loss_unweighted,
            'multi_mel_snr_loss_weighted': multi_mel_snr_loss_weighted,
            'low_amp_penalty': low_amp_penalty,
            'low_amp_penalty_weighted': low_amp_penalty_weighted,
            # Per-source tensors [S] — not scalars; pop these before .item() iteration
            'l1_loss_per_source': per_source_l1,
            'l1_loss_weighted_per_source': per_source_l1_weighted,
            'mr_stft_loss_per_source': per_source_mr_stft,
            'mr_stft_loss_weighted_per_source': per_source_mr_stft_weighted,
        }

        return loss_dict
    
    return loss_func


if __name__ == '__main__':
    # Test the combined loss function
    import torch

    batch_size, n_sources, n_channels, n_samples = 2, 4, 2, 48000
    pred = torch.randn(batch_size, n_sources, n_channels, n_samples) * 0.1
    target = torch.randn(batch_size, n_sources, n_channels, n_samples) * 0.1

    output_dict = {'waveform': pred}
    target_dict = {'waveform': target}

    loss_fn = get_loss_func(stem_weights=[1.0, 1.5, 0.5, 2.0])
    loss_dict = loss_fn(output_dict, target_dict)

    print("Combined Loss Results:")
    for key, value in loss_dict.items():
        if value is None:
            print(f"{key}: None")
        elif isinstance(value, torch.Tensor) and value.dim() > 0:
            print(f"{key}: {value.tolist()}")
        elif isinstance(value, torch.Tensor):
            print(f"{key}: {value.item():.6f}")
        else:
            print(f"{key}: {value:.6f}")

