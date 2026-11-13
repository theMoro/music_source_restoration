"""
GAN training losses, ported from Hachimi's model.py
(https://github.com/linfeng-feng/Hachimi/blob/main/model.py).

Used by the GAN fine-tuning pipelines (src/training/lightningmodule/mixture_restoration_gan.py and
stem_restoration_gan.py):
  - MultiResolutionSTFTLoss: the reconstruction loss (Hachimi's `multi_resolution_loss`)
  - discriminator_loss_with_stats / generator_gan_loss: LSGAN losses (Apollo paper Eq. 3 / Eq. 4)
  - feature_matching_loss: Apollo paper Eq. 5

Formulas are unchanged from the original; only turned into standalone
functions/module (Hachimi has them as MSSModel methods operating on
self.specs / self.discriminator).
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram


class MultiResolutionSTFTLoss(nn.Module):
    """Multi-resolution STFT L1 loss over stacked [real, imag, abs], each
    resolution normalized by the target's mean abs magnitude."""

    def __init__(self, window_sizes: Sequence[int] = (512, 1024, 2048)):
        super().__init__()
        self.specs = nn.ModuleList([
            Spectrogram(n_fft=w, win_length=w, hop_length=w // 4, power=None)
            for w in window_sizes
        ])

    def forward(self, output_audio: torch.Tensor, target_audio: torch.Tensor) -> torch.Tensor:
        eps = torch.finfo(output_audio.dtype).eps
        total_loss = 0.0
        for spec_transform in self.specs:
            stft_output = spec_transform(output_audio)
            stft_target = spec_transform(target_audio)
            stft_output = torch.stack([stft_output.real, stft_output.imag, stft_output.abs()], dim=-1)
            stft_target = torch.stack([stft_target.real, stft_target.imag, stft_target.abs()], dim=-1)
            total_loss = total_loss + F.l1_loss(stft_output, stft_target) / stft_target.abs().mean().clip(min=eps)
        return total_loss / len(self.specs)


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """1-indexed ranks of x, with tied values assigned their group's mean rank (standard
    mid-rank tie-breaking for rank-sum statistics). Needed because a collapsed discriminator --
    a common GAN failure mode -- outputs near-identical logits for
    every input, i.e. exactly the many-ties case: plain argsort-based ranking (no tie handling)
    silently reports whatever order torch's sort happens to break ties in as "perfect separation".
    """
    sorted_vals, sorted_idx = torch.sort(x)
    n = x.numel()
    positions = torch.arange(1, n + 1, dtype=torch.float64, device=x.device)
    unique_vals, inverse, counts = torch.unique(sorted_vals, return_inverse=True, return_counts=True)
    group_sum = torch.zeros(unique_vals.numel(), dtype=torch.float64, device=x.device)
    group_sum.scatter_add_(0, inverse, positions)
    avg_rank_per_group = group_sum / counts.to(torch.float64)
    sorted_ranks = avg_rank_per_group[inverse]
    ranks = torch.empty(n, dtype=torch.float64, device=x.device)
    ranks[sorted_idx] = sorted_ranks
    return ranks


def _pooled_auc(real_logit: torch.Tensor, fake_logit: torch.Tensor) -> float:
    """Mann-Whitney-U / rank-sum AUC estimator: P(a random real logit > a random fake logit),
    treating every element of real_logit/fake_logit as an independent pooled sample (not paired
    by spatial position) with label 1/0 respectively. Equivalent to sklearn's roc_auc_score on
    the flattened, pooled scores, without adding a sklearn dependency. Ties (see _average_ranks)
    correctly resolve to auc=0.5 for a fully collapsed (constant-output) discriminator.
    """
    real_flat = real_logit.detach().float().reshape(-1)
    fake_flat = fake_logit.detach().float().reshape(-1)
    n_real = real_flat.numel()
    n_fake = fake_flat.numel()
    if n_real == 0 or n_fake == 0:
        return float('nan')
    pooled = torch.cat([real_flat, fake_flat])
    ranks = _average_ranks(pooled)
    real_rank_sum = ranks[:n_real].sum()
    auc = (real_rank_sum - n_real * (n_real + 1) / 2) / (n_real * n_fake)
    return float(auc)


def _binary_discrimination_stats(real_logit: torch.Tensor, fake_logit: torch.Tensor) -> dict:
    """Diagnostics answering "is D discriminating?" directly, rather than via aggregate losses
    that can hide overlapping real/fake score distributions (e.g. d_loss~=0.5 is consistent both
    with a healthy discriminator seeing well-fooling fakes and with a collapsed one outputting a
    constant ~0.5 regardless of input). real_logit/fake_logit must have identical shape (true for
    every call site here: both come from the same discriminator, on real/fake audio already
    trimmed to matching length).

    Returns a dict of Python floats/None:
        gap_mean/gap_median/gap_p10/gap_p90: elementwise (real_logit - fake_logit) mean/quantiles
            -- real_logit and fake_logit share spatial shape [B, 1, F, T] (same STFT config, same
            audio length), so this is a true paired elementwise difference, not a difference of
            independent distributions.
        frac_real_gt_fake: elementwise fraction of positions where real_logit > fake_logit.
        frac_tied: elementwise fraction of positions where real_logit == fake_logit exactly --
            near 1.0 is a strong collapsed-discriminator (or identical-input) signal that
            gap_median alone can't distinguish from "D learned a genuinely tiny but nonzero gap".
        auc: pooled (unpaired) rank-sum AUC on the raw (pre-activation) logits -- see _pooled_auc.
        balanced_accuracy: 0.5 * (frac(real_logit > 0.5) + frac(fake_logit < 0.5)), using the
            LSGAN target midpoint (real target 1, fake target 0) as the decision threshold.
    """
    if real_logit.shape != fake_logit.shape:
        return {
            'gap_mean': None, 'gap_median': None, 'gap_p10': None, 'gap_p90': None,
            'frac_real_gt_fake': None, 'frac_tied': None,
            'auc': _pooled_auc(real_logit, fake_logit),
            'balanced_accuracy': None,
        }
    diff = (real_logit.detach().float() - fake_logit.detach().float()).reshape(-1)
    quantiles = torch.quantile(diff, torch.tensor([0.1, 0.5, 0.9], device=diff.device, dtype=diff.dtype))
    real_flat = real_logit.detach().float().reshape(-1)
    fake_flat = fake_logit.detach().float().reshape(-1)
    # Half credit for exact ties at the 0.5 decision boundary (the collapsed-discriminator
    # failure mode: constant output pinned near 0.5 for both real and fake), rather than the
    # (0 or 1)-only accuracy an unconditional > / < comparison gives at that single point --
    # matches how ties are handled in the AUC estimator above.
    real_correct = (real_flat > 0.5).float() + 0.5 * (real_flat == 0.5).float()
    fake_correct = (fake_flat < 0.5).float() + 0.5 * (fake_flat == 0.5).float()
    balanced_accuracy = 0.5 * (real_correct.mean() + fake_correct.mean())
    return {
        'gap_mean': float(diff.mean()),
        'gap_p10': float(quantiles[0]),
        'gap_median': float(quantiles[1]),
        'gap_p90': float(quantiles[2]),
        'frac_real_gt_fake': float((diff > 0).float().mean()),
        'frac_tied': float((diff == 0).float().mean()),
        'auc': _pooled_auc(real_logit, fake_logit),
        'balanced_accuracy': float(balanced_accuracy),
    }


def discriminator_loss_with_stats(
    discriminator: nn.Module, real_audio: torch.Tensor, fake_audio: torch.Tensor,
):
    """LSGAN discriminator loss: sum_i E[(D_i(real)-1)^2] + sum_i E[D_i(fake)^2] (averaged over the
    sub-discriminators). Additionally returns the real/fake loss terms separately and, per sub-discriminator, diagnostics on the raw (pre-loss) D(real) and
    D(fake) logits -- for tracking discriminator health during GAN training (e.g. a collapsing
    D(real)/D(fake) gap, or one resolution's sub-discriminator saturating while others don't,
    both invisible in the single aggregated d_loss scalar). Aggregate means alone can hide
    overlapping score distributions (see _binary_discrimination_stats), so AUC/balanced accuracy
    are included alongside them.

    Returns:
        loss: the discriminator loss.
        loss_real: mean_i E[(D_i(real)-1)^2] term only (loss == loss_real + loss_fake).
        loss_fake: mean_i E[D_i(fake)^2] term only.
        stats: list of dicts (one per sub-discriminator, same order as e.g.
            MultiFrequencyDiscriminator.discriminators/.window), each with real_mean/real_std/
            fake_mean/fake_std plus the gap_median/gap_p10/gap_p90/frac_real_gt_fake/auc/
            balanced_accuracy fields from _binary_discrimination_stats (Python floats, or None
            where not computable).
    """
    if real_audio.dim() == 2:
        real_audio = real_audio.unsqueeze(1)
    if fake_audio.dim() == 2:
        fake_audio = fake_audio.unsqueeze(1)

    real_logits, _ = discriminator(real_audio)
    fake_logits, _ = discriminator(fake_audio.detach())

    loss_real = 0.0
    loss_fake = 0.0
    stats = []
    for real_logit, fake_logit in zip(real_logits, fake_logits):
        loss_real = loss_real + torch.mean((real_logit - 1) ** 2)
        loss_fake = loss_fake + torch.mean(fake_logit ** 2)
        s = {
            'real_mean': real_logit.mean().item(),
            'real_std': real_logit.std().item(),
            'fake_mean': fake_logit.mean().item(),
            'fake_std': fake_logit.std().item(),
        }
        s.update(_binary_discrimination_stats(real_logit, fake_logit))
        stats.append(s)

    n = len(real_logits)
    loss_real = loss_real / n
    loss_fake = loss_fake / n
    return loss_real + loss_fake, loss_real, loss_fake, stats


def generator_gan_loss(
    discriminator: nn.Module, fake_audio: torch.Tensor, real_audio: torch.Tensor,
):
    """LSGAN generator loss: sum_i E[(D_i(fake)-1)^2]. Also returns the feature
    maps needed for feature_matching_loss."""
    if fake_audio.dim() == 2:
        fake_audio = fake_audio.unsqueeze(1)
    if real_audio.dim() == 2:
        real_audio = real_audio.unsqueeze(1)

    fake_logits, fake_features = discriminator(fake_audio)
    real_logits, real_features = discriminator(real_audio)

    loss = 0.0
    for fake_logit in fake_logits:
        loss = loss + torch.mean((fake_logit - 1) ** 2)
    loss = loss / len(fake_logits)

    return loss, fake_features, real_features


def feature_matching_loss(fake_features, real_features) -> torch.Tensor:
    """Apollo paper Eq. 5: normalized L1 distance between discriminator feature maps."""
    loss = 0.0
    for fake_fmaps, real_fmaps in zip(fake_features, real_features):
        for fake_fmap, real_fmap in zip(fake_fmaps, real_fmaps):
            real_fmap = real_fmap.detach()
            diff = torch.mean(torch.abs(fake_fmap - real_fmap))
            norm = torch.mean(torch.abs(real_fmap)) + 1e-8
            loss = loss + diff / norm

    total_layers = sum(len(fmaps) for fmaps in fake_features)
    return loss / total_layers if total_layers > 0 else loss
