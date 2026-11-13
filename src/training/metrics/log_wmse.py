"""
LogWMSE metric utility — wraps the torch_log_wmse package.

Computes Log-Weighted Mean Square Error per source using fixed-size audio chunks
(torch_log_wmse < 1.0.0 needs a constant ``audio_length`` constructor argument; 1.x
derives the length per call).  10-second chunks are used by default (matching the
chunked validation protocol).

LogWMSE measures reconstruction error in log-frequency scale, using the mixture
as a noise-floor reference so that errors in louder regions are weighted more.
Uses ``return_as_loss=False`` so values are positive and higher is better
(higher means less error relative to the mixture noise floor).

Reference implementation: https://github.com/crlandsc/torch-log-wmse
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

try:
    from torch_log_wmse import LogWMSE as _LogWMSE

    _LOG_WMSE_AVAILABLE = True
except ImportError:
    _LOG_WMSE_AVAILABLE = False
    _LogWMSE = None  # type: ignore[assignment,misc]


def is_available() -> bool:
    """Return True when torch_log_wmse is installed."""
    return _LOG_WMSE_AVAILABLE


@lru_cache(maxsize=8)
def _get_log_wmse_fn(audio_length_seconds: float, sample_rate: int) -> "_LogWMSE":
    """Return a cached LogWMSE instance (positive metric, higher is better).

    torch_log_wmse >= 1.0.0 removed ``audio_length`` (the length is derived per call) and
    ``return_as_loss`` (``LogWMSE`` is always the metric); older versions require both.
    """
    if not _LOG_WMSE_AVAILABLE:
        raise ImportError(
            "torch_log_wmse is required. Install with: pip install torch-log-wmse"
        )
    try:
        return _LogWMSE(sample_rate=sample_rate, bypass_filter=False)
    except TypeError:
        # torch_log_wmse < 1.0.0: audio_length is a required argument
        return _LogWMSE(
            audio_length=audio_length_seconds,
            sample_rate=sample_rate,
            return_as_loss=False,
            bypass_filter=False,
        )


def compute_log_wmse_chunked(
    mixture_ct: torch.Tensor,
    estimated_ct: torch.Tensor,
    target_ct: torch.Tensor,
    sample_rate: int,
    chunk_size_seconds: float = 10.0,
) -> float:
    """Compute LogWMSE for a single source by averaging over fixed-size chunks.

    Higher is better. Silent target chunks (RMS <= 1e-8) are skipped.

    Args:
        mixture_ct:   Mixture audio [C, T] (the unprocessed input).
        estimated_ct: Separated estimate [C, T] for one source.
        target_ct:    Ground-truth [C, T] for one source.
        sample_rate:  Audio sample rate in Hz.
        chunk_size_seconds: Length of each chunk in seconds (default 10 s).

    Returns:
        Mean LogWMSE across all non-silent chunks, or ``float('nan')`` when
        no valid chunks are available.
    """
    if not _LOG_WMSE_AVAILABLE:
        raise ImportError(
            "torch_log_wmse is required. Install with: pip install torch-log-wmse"
        )

    chunk_size = int(chunk_size_seconds * sample_rate)
    T = estimated_ct.shape[-1]
    n_chunks = T // chunk_size
    if n_chunks == 0:
        return float("nan")

    log_wmse_fn = _get_log_wmse_fn(chunk_size_seconds, sample_rate)

    scores: list[float] = []
    for i in range(n_chunks):
        s = i * chunk_size
        e = s + chunk_size

        mix_chunk = mixture_ct[:, s:e].float().cpu()   # [C, chunk]
        est_chunk = estimated_ct[:, s:e].float().cpu() # [C, chunk]
        tgt_chunk = target_ct[:, s:e].float().cpu()    # [C, chunk]

        # Skip silent target chunks (log_wmse is undefined / meaningless there).
        tgt_rms = float(torch.sqrt(torch.mean(tgt_chunk ** 2)).item())
        if tgt_rms <= 1e-8:
            continue

        # LogWMSE expects shapes:
        #   unprocessed_audio: [B, C, T]
        #   processed_audio:   [B, C, S, T]
        #   target_audio:      [B, C, S, T]
        mix_in = mix_chunk.unsqueeze(0)              # [1, C, chunk]
        est_in = est_chunk.unsqueeze(0).unsqueeze(2) # [1, C, 1, chunk]
        tgt_in = tgt_chunk.unsqueeze(0).unsqueeze(2) # [1, C, 1, chunk]

        try:
            val = float(log_wmse_fn(mix_in, est_in, tgt_in).item())
            if math.isfinite(val):
                scores.append(val)
        except Exception:
            pass

    if not scores:
        return float("nan")
    return float(sum(scores) / len(scores))