"""
Multi-resolution log-mel L1 loss.

Puts the reconstruction objective directly in the domain Multi-Mel-SNR (this project's top-
priority restoration metric, src.training.metrics.multi_mel_snr) is evaluated in, instead of
relying on MultiResolutionSTFTLoss's linear-frequency STFT loss to correlate with it indirectly.
Uses the same three (n_fft, hop_length, n_mels) resolutions Multi-Mel-SNR averages over, so loss
and metric see the same spectral detail at each scale.
"""

from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from src.training.metrics.multi_mel_snr import multi_mel_snr_mel_f_max


class MultiResolutionMelLoss(nn.Module):
    """L1 loss between log1p-compressed mel power spectrograms, averaged over three resolutions
    (matching src.training.metrics.multi_mel_snr's (n_fft, hop, n_mels) configs)."""

    _DEFAULT_CONFIGS: Tuple[Tuple[int, int, int], ...] = (
        (512, 256, 80),
        (1024, 512, 128),
        (2048, 1024, 192),
    )

    def __init__(self, sample_rate: int, configs: Optional[Sequence[Tuple[int, int, int]]] = None):
        super().__init__()
        configs = configs if configs is not None else self._DEFAULT_CONFIGS
        f_max = multi_mel_snr_mel_f_max(sample_rate)
        self.mels = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft, hop_length=hop,
                n_mels=n_mels, f_min=0, f_max=f_max, power=2.0,
            )
            for n_fft, hop, n_mels in configs
        ])

    def forward(self, output_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0
        for mel in self.mels:
            mel_output = torch.log1p(mel(output_audio))
            mel_target = torch.log1p(mel(target_audio))
            total_loss = total_loss + F.l1_loss(mel_output, mel_target)
        return total_loss / len(self.mels)


class ScaleInvariantMultiMelSNRLoss(nn.Module):
    """Negated scale-invariant Multi-Mel-SNR, computed exactly the way
    src.training.metrics.multi_mel_snr.multi_mel_snr_stereo_ct (val/mel_snr, the
    checkpoint-selection metric) is: per flattened [batch*channel] row, project the
    output onto the target (alpha = <ref,pred>/<pred,pred>) before measuring log-mel error, then
    average dB SNR over the same three (n_fft, hop, n_mels) resolutions.

    MultiResolutionMelLoss (above) puts the objective in the same *domain* as val/mel_snr but
    still minimizes an L1 magnitude error, not SNR itself -- validation mel loss can improve
    epoch-over-epoch while val/mel_snr falls (concentrated in easy/high-SNR degradation
    families), i.e. lower mel L1 does not imply higher Mel-SNR.
    This loss instead optimizes the validation metric's own formula directly, including its
    scale-invariant projection step, so a low-error/high-relative-SNR case (an easy family) can no
    longer be improved on the metric's own terms by trading a large relative Mel-SNR loss there for
    a smaller absolute L1 gain on a harder case.
    """

    _DEFAULT_CONFIGS = MultiResolutionMelLoss._DEFAULT_CONFIGS

    def __init__(
        self,
        sample_rate: int,
        configs: Optional[Sequence[Tuple[int, int, int]]] = None,
        eps: float = 1e-8,
    ):
        super().__init__()
        configs = configs if configs is not None else self._DEFAULT_CONFIGS
        f_max = multi_mel_snr_mel_f_max(sample_rate)
        self.mels = nn.ModuleList([
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate, n_fft=n_fft, hop_length=hop,
                n_mels=n_mels, f_min=0, f_max=f_max, power=2.0,
            )
            for n_fft, hop, n_mels in configs
        ])
        self.eps = float(eps)

    def forward(self, output_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        """output_audio/target_audio: any shape ``[..., T]`` (e.g. ``[B, C, T]``) -- flattened to
        ``[N, T]`` so the scale-invariant projection and SNR are computed per batch*channel row,
        matching multi_mel_snr_stereo_ct's per-channel treatment exactly."""
        time_len = output_audio.shape[-1]
        output_flat = output_audio.reshape(-1, time_len)
        target_flat = target_audio.reshape(-1, time_len)

        dot_ref_pred = (target_flat * output_flat).sum(dim=-1)
        dot_pred_pred = (output_flat * output_flat).sum(dim=-1) + self.eps
        alpha = dot_ref_pred / dot_pred_pred
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
        output_scaled = alpha.unsqueeze(-1) * output_flat

        snrs = []
        for mel in self.mels:
            mel_ref = mel(target_flat)
            mel_pred = mel(output_scaled)
            ref_power = mel_ref.pow(2).sum(dim=(-2, -1))
            diff_power = (mel_ref - mel_pred).pow(2).sum(dim=(-2, -1)) + self.eps
            snr = 10 * torch.log10((ref_power / diff_power).clamp_min(self.eps))
            snr = torch.where(torch.isfinite(snr), snr, torch.full_like(snr, -80.0))
            snrs.append(snr)

        avg_snr = torch.stack(snrs, dim=0).mean(dim=0)  # [N]
        return -avg_snr.mean()
