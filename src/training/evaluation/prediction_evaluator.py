"""
PyTorch Lightning module for evaluating predictions against ground truths.

This evaluator is shared by:
- Offline evaluation of saved predictions
- Online in-memory MSRBench inference+evaluation

Protocol highlights:
- Chunked evaluation (10s and 1s) with per-sample median aggregation per stem
- Configurable metric registry with optional heavy metrics only on 10s chunks
- Distributed-safe state gather/merge support
"""

from __future__ import annotations

import copy
import math

from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from lightning_fabric.utilities.exceptions import MisconfigurationException
from pytorch_lightning import LightningModule
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr

import museval
import torchaudio
import zimtohrli

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

try:
    from pypesq import pesq as pypesq_pesq
except Exception:
    pypesq_pesq = None

from src.training.metrics.multi_mel_snr import multi_mel_snr, multi_mel_snr_mel_f_max
from src.training.metrics.fad_clap import FADCLAPMetric
from src.training.metrics import log_wmse as _log_wmse_module
from src.training.metrics.additional_metrics import (
    L1Freq_metric,
    AuraSTFT_metric,
    AuraMRSTFT_metric,
    bleed_full,
)


class PredictionEvaluator(LightningModule):
    """Lightning module for evaluating saved/in-memory predictions."""

    _MULTI_MEL_CONFIGS = [(512, 256, 80), (1024, 512, 128), (2048, 1024, 192)]

    _DEFAULT_METRICS = [
        "sdr",
        "snr",
        "si_snr",
        "multi_mel_snr",
        "zimtohrli",
        "mel_ssim",
        "mel_kl_div",
        "kl",
        "pesq",
        "production_quality",
    ]

    _SUPPORTED_METRICS = set(_DEFAULT_METRICS) | {"log_wmse", "l1_freq", "aura_stft", "aura_mrstft", "bleed", "fullness"}
    _HEAVY_METRICS = {"pesq", "production_quality", "log_wmse"}

    _METRIC_DISPLAY_NAMES = {
        "sdr": "SDR",
        "snr": "SNR",
        "si_snr": "SI-SNR",
        "multi_mel_snr": "Multi-Mel-SNR",
        "zimtohrli": "Zimtohrli",
        "mel_ssim": "MEL_SSIM",
        "mel_kl_div": "mel_kl_div",
        "kl": "KL",
        "pesq": "PESQ",
        "production_quality": "production_quality",
        "log_wmse": "LogWMSE",
        "l1_freq": "L1Freq",
        "aura_stft": "AuraSTFT",
        "aura_mrstft": "AuraMRSTFT",
        "bleed": "Bleed",
        "fullness": "Fullness",
    }

    _EPS = 1e-8

    # The 5 "core" stems used for the dedicated 5-stem aggregate in the per-stem summary CSV.

    def __init__(
        self,
        columns: list[str],
        sampling_rate: int = 48_000,
        prediction_metric_stats: str = "both",
        prediction_metrics: list[str] | None = None,
        prediction_pesq_sample_rate: int = 16_000,
        prediction_mel_config: dict[str, int] | None = None,
        per_sample_scores_output_path: str | Path | None = None,
        per_stem_metrics_output_path: str | Path | None = None,
    ):
        super().__init__()
        self.columns = columns
        self.n_sources = len(columns)
        self.sampling_rate = int(sampling_rate)
        self._multi_mel_transforms_cache: list[torch.nn.Module] | None = None
        self._zimt_calculator = None
        self.prediction_pesq_sample_rate = int(prediction_pesq_sample_rate)

        mode = str(prediction_metric_stats).lower().strip()
        if mode not in ("both", "median", "mean"):
            raise ValueError(
                f"prediction_metric_stats must be one of ['both','median','mean'] (got {prediction_metric_stats!r})"
            )
        self.prediction_metric_stats_mode = mode
        self.prediction_metric_include_median = mode in ("both", "median")
        self.prediction_metric_include_mean = mode in ("both", "mean")

        self._warned_backends: set[str] = set()
        # Mel config for MEL_SSIM / MEL-KL metrics
        mel_cfg = prediction_mel_config or {}
        self.prediction_mel_config = {
            "n_fft": int(mel_cfg.get("n_fft", 1024)),
            "hop_length": int(mel_cfg.get("hop_length", 512)),
            "n_mels": int(mel_cfg.get("n_mels", 128)),
        }

        self._mel_transform = None
        self._db_transform = None
        self._mel_metric_backend_available = False
        self._setup_mel_backend_if_possible()

        self._production_predictor = None
        self._production_predictor_checked = False

        self.metric_keys = self._resolve_metric_keys(prediction_metrics)
        self.metric_registry = self._build_metric_registry()

        if "mel_ssim" in self.metric_keys and skimage_ssim is None:
            self._warn_once(
                "skimage_missing_mel_ssim",
                "Warning: scikit-image unavailable; MEL_SSIM metric will be skipped (NaN).",
            )

        # FAD-CLAP metric for 10-second segments only (enabled by including "fad_clap" in prediction_metrics).
        # It is intentionally excluded from metric_keys (chunked metrics), so we check
        # directly against the requested metrics list.
        fad_clap_requested = self._is_metric_requested(prediction_metrics, "fad_clap")
        if fad_clap_requested:
            try:
                self.fad_clap_metric: FADCLAPMetric | None = FADCLAPMetric(
                    max_samples_per_epoch=0,  # no limit — evaluate all samples
                    max_audio_seconds=10.0,
                )
                print("FAD-CLAP enabled in PredictionEvaluator (10s only)")
            except Exception as e:
                print(f"Warning: Failed to initialize FAD-CLAP in PredictionEvaluator: {e}")
                self.fad_clap_metric = None
        else:
            self.fad_clap_metric = None

        self.per_sample_scores_output_path: Path | None = (
            Path(per_sample_scores_output_path) if per_sample_scores_output_path is not None else None
        )
        self.per_stem_metrics_output_path: Path | None = (
            Path(per_stem_metrics_output_path) if per_stem_metrics_output_path is not None else None
        )
        self.per_sample_scores: list[dict] = []

        self.reset_metrics()
        self.total_samples_processed = 0
        self._metrics_initialized = False

    # ---------------------------------------------------------------------
    # Initialization helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _normalize_metric_key(metric_name: str) -> str:
        return str(metric_name).strip().lower().replace("-", "_")

    @classmethod
    def _is_metric_requested(cls, prediction_metrics: list[str] | None, metric_name: str) -> bool:
        if prediction_metrics is None:
            return False
        wanted = cls._normalize_metric_key(metric_name)
        return any(cls._normalize_metric_key(m) == wanted for m in prediction_metrics)

    def _warn_once(self, key: str, message: str) -> None:
        if key not in self._warned_backends:
            print(message)
            self._warned_backends.add(key)

    def _setup_mel_backend_if_possible(self) -> None:
        if torchaudio is None:
            self._warn_once(
                "torchaudio_missing_mel_metrics",
                "Warning: torchaudio unavailable, MEL_SSIM/mel_kl_div/KL metrics will be skipped.",
            )
            return

        try:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sampling_rate,
                n_fft=self.prediction_mel_config["n_fft"],
                hop_length=self.prediction_mel_config["hop_length"],
                n_mels=self.prediction_mel_config["n_mels"],
                f_min=0,
                f_max=self.sampling_rate // 2,
                power=2.0,
            )
            self._db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")
            self._mel_metric_backend_available = True
        except Exception as e:
            self._mel_metric_backend_available = False
            self._warn_once(
                "mel_backend_init_failed",
                f"Warning: Failed to initialize mel backend ({e}); MEL_SSIM/mel_kl_div/KL metrics will be skipped.",
            )

    def _setup_zimtohrli_backend_if_needed(self, metric_keys: list[str]) -> None:
        if "zimtohrli" not in metric_keys:
            return

        if zimtohrli is None:
            self._warn_once(
                "zimtohrli_missing",
                "Warning: zimtohrli package unavailable; Zimtohrli metric will be skipped.",
            )
            return

        try:
            self._zimt_calculator = zimtohrli.Pyohrli()
        except Exception as e:
            self._warn_once(
                "zimtohrli_init_failed",
                f"Warning: Failed to initialize zimtohrli backend ({e}); Zimtohrli metric will be skipped.",
            )
            self._zimt_calculator = None

    def _resolve_metric_keys(self, prediction_metrics: list[str] | None) -> list[str]:
        if prediction_metrics is None:
            metrics = list(self._DEFAULT_METRICS)
        else:
            metrics = [self._normalize_metric_key(m) for m in prediction_metrics]

        deduped: list[str] = []
        for m in metrics:
            if m not in deduped:
                deduped.append(m)

        # fad_clap is handled separately (not in metric_keys for chunked computation)
        deduped = [m for m in deduped if m != "fad_clap"]

        unknown = [m for m in deduped if m not in self._SUPPORTED_METRICS]
        if unknown:
            raise ValueError(f"Unsupported prediction metrics: {unknown}. Supported: {sorted(self._SUPPORTED_METRICS)}")

        # Initialize zimtohrli backend if needed; remove from list if backend unavailable.
        self._setup_zimtohrli_backend_if_needed(deduped)
        if "zimtohrli" in deduped and self._zimt_calculator is None:
            deduped = [m for m in deduped if m != "zimtohrli"]

        # Validate log_wmse availability if requested.
        if "log_wmse" in deduped and not _log_wmse_module.is_available():
            self._warn_once(
                "log_wmse_missing",
                "Warning: torch_log_wmse not installed; LogWMSE metric will be skipped. "
                "Install with: pip install torch-log-wmse",
            )
            deduped = [m for m in deduped if m != "log_wmse"]

        return deduped

    def _build_metric_registry(self) -> dict[str, Any]:
        return {
            "sdr": self._metric_sdr,
            "snr": self._metric_snr,
            "si_snr": self._metric_si_snr,
            "multi_mel_snr": self._metric_multi_mel_snr,
            "zimtohrli": self._metric_zimtohrli,
            "mel_ssim": self._metric_mel_ssim,
            "mel_kl_div": self._metric_mel_kl_div,
            "kl": self._metric_kl_alias,
            "pesq": self._metric_pesq,
            "production_quality": self._metric_production_quality,
            "log_wmse": self._metric_log_wmse,
            "l1_freq": self._metric_l1_freq,
            "aura_stft": self._metric_aura_stft,
            "aura_mrstft": self._metric_aura_mrstft,
            "bleed": self._metric_bleed,
            "fullness": self._metric_fullness,
        }

    # ---------------------------------------------------------------------
    # Accumulator helpers
    # ---------------------------------------------------------------------
    def _empty_per_source_map(self, metrics: list[str]) -> dict[str, dict[int, list[float]]]:
        return {metric: {source_idx: [] for source_idx in range(self.n_sources)} for metric in metrics}

    def reset_metrics(self) -> None:
        """Reset all metric accumulators."""
        self.per_sample_scores = []
        self.per_source_metric_values = self._empty_per_source_map(self.metric_keys)

        self.per_family_metric_values: dict[str, dict[str, list[float]]] = {
            metric: {} for metric in self.metric_keys
        }
        self.per_family_and_source_metric_values: dict[str, dict[str, dict[int, list[float]]]] = {
            metric: {} for metric in self.metric_keys
        }


        if self.fad_clap_metric is not None:
            self.fad_clap_metric.reset()

    @staticmethod
    def _nanmean(values: list[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return float("nan")
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean(finite))

    @staticmethod
    def _nanmedian(values: list[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return float("nan")
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return float("nan")
        return float(np.median(finite))

    @staticmethod
    def _nanstd(values: list[float]) -> float:
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return float("nan")
        return float(np.nanstd(arr))

    def _compute_source_summaries(
        self, per_source_values: dict[int, list[float]]
    ) -> dict[int, dict[str, float]]:
        """
        Compute per-source median/mean/std (over accumulated chunks).
        Returns: {source_idx: {"median": x, "mean": y, "std": z, "n": n}}
        """
        out: dict[int, dict[str, float]] = {}
        for source_idx in range(self.n_sources):
            values = per_source_values.get(source_idx, [])
            if len(values) == 0:
                out[source_idx] = {
                    "median": float("nan"),
                    "mean": float("nan"),
                    "std": float("nan"),
                    "n": 0.0,
                }
                continue
            out[source_idx] = {
                "median": self._nanmedian(values) if self.prediction_metric_include_median else float("nan"),
                "mean": self._nanmean(values) if self.prediction_metric_include_mean else float("nan"),
                "std": self._nanstd(values) if self.prediction_metric_include_mean else float("nan"),
                "n": float(len(values)),
            }
        return out

    def _compute_avg_over_source_subset(
        self, per_source_summaries: dict[int, dict[str, float]], source_indices: list[int]
    ) -> dict[str, float]:
        """
        Average summary across a subset of sources (ignores NaNs).
        Returns: {"median": avg_median, "mean": avg_mean, "std": avg_std, "n_sources": k}
        """
        medians: list[float] = []
        means: list[float] = []
        stds: list[float] = []
        for src_idx in source_indices:
            s = per_source_summaries.get(src_idx, {})
            m1 = float(s.get("median", float("nan")))
            m2 = float(s.get("mean", float("nan")))
            m3 = float(s.get("std", float("nan")))
            if math.isfinite(m1):
                medians.append(m1)
            if math.isfinite(m2):
                means.append(m2)
            if math.isfinite(m3):
                stds.append(m3)
        return {
            "median": self._nanmean(medians) if self.prediction_metric_include_median else float("nan"),
            "mean": self._nanmean(means) if self.prediction_metric_include_mean else float("nan"),
            "std": self._nanmean(stds) if self.prediction_metric_include_mean else float("nan"),
            "n_sources": float(max(len(medians), len(means), len(stds))),
        }

    def _compute_avg_over_sources(
        self, per_source_summaries: dict[int, dict[str, float]]
    ) -> dict[str, float]:
        """Average summary across all sources except "other" (ignores NaNs). See `_compute_avg_over_source_subset`."""
        indices = [i for i in range(self.n_sources) if self.columns[i] != "other"]
        return self._compute_avg_over_source_subset(per_source_summaries, indices)

    @staticmethod
    def _format_table_number(x: float | None) -> str:
        if x is None:
            return ""
        if not isinstance(x, (float, int)) or not math.isfinite(float(x)):
            return ""
        return f"{float(x):.3f}".replace(".", ",")

    def _format_metric_stats(self, median_val: float, mean_val: float, n: int, n_label: str = "n") -> str:
        parts: list[str] = []
        if self.prediction_metric_include_median:
            parts.append(f"median={median_val:.3f}")
        if self.prediction_metric_include_mean:
            parts.append(f"mean={mean_val:.3f}")
        joined = ", ".join(parts) if parts else ""
        return f"{joined} ({n_label}={n})"

    def get_state_for_gather(self) -> dict:
        return {
            "metric_keys": copy.deepcopy(self.metric_keys),
            "per_sample_scores": copy.deepcopy(self.per_sample_scores),
            "per_source_metric_values": copy.deepcopy(self.per_source_metric_values),
            "per_family_metric_values": copy.deepcopy(self.per_family_metric_values),
            "per_family_and_source_metric_values": copy.deepcopy(self.per_family_and_source_metric_values),
            "fad_clap_embeddings_state": (
                self.fad_clap_metric.get_concatenated_embeddings_state()
                if self.fad_clap_metric is not None
                else None
            ),
            "fad_clap_sample_count": (
                int(self.fad_clap_metric.sample_count) if self.fad_clap_metric is not None else 0
            ),
            "total_samples_processed": self.total_samples_processed,
        }

    def _ensure_metric_in_maps(self, metric: str) -> None:
        if metric not in self.per_source_metric_values:
            self.per_source_metric_values[metric] = {i: [] for i in range(self.n_sources)}
        if metric not in self.per_family_metric_values:
            self.per_family_metric_values[metric] = {}
        if metric not in self.per_family_and_source_metric_values:
            self.per_family_and_source_metric_values[metric] = {}

    def merge_state_from_other(self, other: dict) -> None:
        self.total_samples_processed += int(other.get("total_samples_processed", 0))
        self.per_sample_scores.extend(other.get("per_sample_scores", []) or [])

        other_metrics = other.get("metric_keys", list(other.get("per_source_metric_values", {}).keys()))
        for metric in other_metrics:
            if metric not in self.metric_keys:
                self.metric_keys.append(metric)
            self._ensure_metric_in_maps(metric)

        for metric, by_source in (other.get("per_source_metric_values", {}) or {}).items():
            self._ensure_metric_in_maps(metric)
            for source_idx, values in by_source.items():
                self.per_source_metric_values[metric].setdefault(int(source_idx), []).extend(values)

        for metric, by_family in (other.get("per_family_metric_values", {}) or {}).items():
            self._ensure_metric_in_maps(metric)
            for family, values in by_family.items():
                self.per_family_metric_values[metric].setdefault(family, []).extend(values)

        for metric, by_family in (other.get("per_family_and_source_metric_values", {}) or {}).items():
            self._ensure_metric_in_maps(metric)
            for family, by_source in by_family.items():
                dst = self.per_family_and_source_metric_values[metric].setdefault(
                    family, {i: [] for i in range(self.n_sources)}
                )
                for source_idx, values in by_source.items():
                    dst.setdefault(int(source_idx), []).extend(values)

        if self.fad_clap_metric is not None:
            other_state = other.get("fad_clap_embeddings_state", None)
            if other_state is not None:
                other_sample_count = int(other.get("fad_clap_sample_count", 0))
                self.fad_clap_metric.sample_count += other_sample_count

                target_state = other_state.get("target", {}) or {}
                output_state = other_state.get("output", {}) or {}
                for s, arr in target_state.items():
                    si = int(s)
                    if arr is None or getattr(arr, "size", 0) == 0:
                        continue
                    self.fad_clap_metric.target_embeddings_by_source.setdefault(si, []).append(arr)
                for s, arr in output_state.items():
                    si = int(s)
                    if arr is None or getattr(arr, "size", 0) == 0:
                        continue
                    self.fad_clap_metric.output_embeddings_by_source.setdefault(si, []).append(arr)

    # ---------------------------------------------------------------------
    # Metric primitives
    # ---------------------------------------------------------------------
    @staticmethod
    def _compute_snr_value(estimated: torch.Tensor, target: torch.Tensor) -> float:
        """Compute SNR in dB between estimate and target [C, T]. Higher is better.

        Formula: 10 * log10(sum(target^2) / (sum((estimate-target)^2) + eps))
        """
        # estimated/target: [C, T]
        tgt = target.detach().cpu().float().reshape(-1)
        est = estimated.detach().cpu().float().reshape(-1)

        signal_power = torch.sum(tgt ** 2)
        if signal_power <= PredictionEvaluator._EPS:
            return float("nan")
        noise_power = torch.sum((est - tgt) ** 2)
        ratio = signal_power / (noise_power + PredictionEvaluator._EPS)
        value = 10.0 * torch.log10(ratio + PredictionEvaluator._EPS)
        return float(value.item()) if torch.isfinite(value) else float("nan")

    def _zimtohrli_distance(self, estimated: torch.Tensor, target: torch.Tensor) -> float:
        """Compute perceptual Zimtohrli distance per channel, averaged. Lower is better."""
        if self._zimt_calculator is None:
            return float("nan")
        est = estimated.detach().cpu().float()
        tgt = target.detach().cpu().float()
        n_channels = est.shape[0]
        scores: list[float] = []
        for ch in range(n_channels):
            ref_np = tgt[ch].numpy()
            pred_np = est[ch].numpy()
            try:
                d = float(self._zimt_calculator.distance(ref_np, pred_np))
            except Exception:
                continue
            if math.isfinite(d):
                scores.append(d)
        if not scores:
            return float("nan")
        return float(sum(scores) / len(scores))

    def _get_multi_mel_transforms(self) -> list:
        """Lazy-init cached MelSpectrogram transforms for Multi-Mel-SNR (avoids creating 3 per chunk)."""
        if self._multi_mel_transforms_cache is None and torchaudio is not None:
            f_max = multi_mel_snr_mel_f_max(float(self.sampling_rate))
            self._multi_mel_transforms_cache = [
                torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sampling_rate,
                    n_fft=n_fft,
                    hop_length=hop,
                    n_mels=n_mels,
                    f_min=0,
                    f_max=f_max,
                    power=2.0,
                )
                for n_fft, hop, n_mels in self._MULTI_MEL_CONFIGS
            ]
        return self._multi_mel_transforms_cache or []

    def _multi_mel_snr_cached(self, estimated: torch.Tensor, target: torch.Tensor) -> float:
        mel_transforms = self._get_multi_mel_transforms()
        if not mel_transforms:
            return float(multi_mel_snr(target.cpu().numpy(), estimated.cpu().numpy(), sr=self.sampling_rate))

        est_cpu = estimated.cpu().float()
        tgt_cpu = target.cpu().float()
        n_channels = est_cpu.shape[0]
        channel_snrs: list[float] = []
        for c in range(n_channels):
            ref = torch.where(torch.isfinite(tgt_cpu[c]), tgt_cpu[c], torch.zeros_like(tgt_cpu[c]))
            pred = torch.where(torch.isfinite(est_cpu[c]), est_cpu[c], torch.zeros_like(est_cpu[c]))

            dot_rp = torch.dot(ref, pred)
            dot_pp = torch.dot(pred, pred) + self._EPS
            alpha = dot_rp / dot_pp if torch.isfinite(dot_pp) else torch.tensor(0.0)
            if not torch.isfinite(alpha):
                alpha = torch.tensor(0.0)
            pred = alpha * pred

            snrs = []
            for mel in mel_transforms:
                M_ref = mel(ref.unsqueeze(0)).squeeze(0)
                M_pred = mel(pred.unsqueeze(0)).squeeze(0)
                M_ref = torch.where(torch.isfinite(M_ref), M_ref, torch.zeros_like(M_ref))
                M_pred = torch.where(torch.isfinite(M_pred), M_pred, torch.zeros_like(M_pred))
                ref_power = M_ref.pow(2).sum()
                diff_power = (M_ref - M_pred).pow(2).sum() + self._EPS
                if not torch.isfinite(ref_power):
                    ref_power = torch.tensor(0.0)
                if not torch.isfinite(diff_power):
                    diff_power = torch.tensor(self._EPS)
                snr = (10.0 * torch.log10((ref_power / diff_power).clamp_min(self._EPS))).item()
                if math.isfinite(snr):
                    snrs.append(float(snr))
            if snrs:
                channel_snrs.append(float(sum(snrs) / len(snrs)))

        if channel_snrs:
            return float(sum(channel_snrs) / len(channel_snrs))
        return float("nan")

    def compute_mel_spectrogram(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute mel spectrogram in dB scale from audio [C,T] or [T]."""
        if not self._mel_metric_backend_available or self._mel_transform is None or self._db_transform is None:
            raise RuntimeError("Mel metric backend unavailable")

        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        mel_spec = self._mel_transform(audio)
        mel_spec_db = self._db_transform(mel_spec)
        return mel_spec_db

    def _log_mel_from_audio(self, audio: torch.Tensor) -> np.ndarray:
        mel = self.compute_mel_spectrogram(audio.detach().cpu().float())
        mel_np = mel.detach().cpu().numpy()
        return mel_np

    @staticmethod
    def _ssim_from_log_mel(pred_log_mel: np.ndarray, target_log_mel: np.ndarray) -> float:
        """Compute MEL-SSIM between log-mel spectrograms, averaged over channels. Higher is better.

        Returns values in [-1, 1]; practically [0, 1] for audio. 1.0 = identical spectra.
        """
        if skimage_ssim is None:
            return float("nan")

        def _single_ssim(a: np.ndarray, b: np.ndarray) -> float:
            if a.shape != b.shape:
                return float("nan")
            if a.ndim != 2:
                return float("nan")

            min_side = min(a.shape[0], a.shape[1])
            if min_side < 3:
                return float("nan")
            win_size = 7 if min_side >= 7 else min_side
            if win_size % 2 == 0:
                win_size -= 1
            if win_size < 3:
                return float("nan")

            data_min = float(min(a.min(), b.min()))
            data_max = float(max(a.max(), b.max()))
            data_range = data_max - data_min
            if data_range <= 0:
                return 1.0

            try:
                return float(skimage_ssim(a, b, data_range=data_range, win_size=win_size))
            except Exception:
                return float("nan")

        if pred_log_mel.ndim == 3:
            scores = []
            for i in range(pred_log_mel.shape[0]):
                s = _single_ssim(target_log_mel[i], pred_log_mel[i])
                if math.isfinite(s):
                    scores.append(s)
            return float(np.mean(scores)) if scores else float("nan")

        return _single_ssim(target_log_mel, pred_log_mel)

    @staticmethod
    def _softmax_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        x = x.astype(np.float64)
        x = x - np.max(x)
        x = np.clip(x, -60.0, 60.0)
        ex = np.exp(x)
        denom = float(np.sum(ex)) + eps
        return ex / denom

    @classmethod
    def _kl_from_log_mel(cls, pred_log_mel: np.ndarray, target_log_mel: np.ndarray) -> float:
        """Compute KL divergence KL(target || pred) on flattened softmax log-mel spectrograms.

        Lower is better. 0 means the predicted and reference spectral distributions match.
        Used by both mel_kl_div and kl metrics.
        """
        def _single_kl(p: np.ndarray, t: np.ndarray) -> float:
            if p.shape != t.shape:
                return float("nan")
            q = cls._softmax_np(p.reshape(-1))
            r = cls._softmax_np(t.reshape(-1))
            eps = 1e-12
            value = float(np.sum(r * np.log((r + eps) / (q + eps))))
            return value if math.isfinite(value) else float("nan")

        if pred_log_mel.ndim == 3:
            scores = []
            for i in range(pred_log_mel.shape[0]):
                s = _single_kl(pred_log_mel[i], target_log_mel[i])
                if math.isfinite(s):
                    scores.append(s)
            return float(np.mean(scores)) if scores else float("nan")

        return _single_kl(pred_log_mel, target_log_mel)

    def _compute_pesq(self, estimated: torch.Tensor, target: torch.Tensor) -> float:
        if pypesq_pesq is None:
            self._warn_once(
                "pypesq_missing",
                "Warning: pypesq unavailable; PESQ metric will be skipped (NaN).",
            )
            return float("nan")

        if torchaudio is None:
            self._warn_once(
                "torchaudio_missing_pesq",
                "Warning: torchaudio unavailable; PESQ metric will be skipped (NaN).",
            )
            return float("nan")

        if self.prediction_pesq_sample_rate not in (8_000, 16_000):
            self._warn_once(
                "invalid_pesq_sr",
                f"Warning: prediction_pesq_sample_rate={self.prediction_pesq_sample_rate} is invalid for pypesq. Use 8000 or 16000.",
            )
            return float("nan")

        try:
            # pypesq expects mono waveform arrays.
            ref_mono = target.detach().cpu().float().mean(dim=0, keepdim=True)
            deg_mono = estimated.detach().cpu().float().mean(dim=0, keepdim=True)

            if self.sampling_rate != self.prediction_pesq_sample_rate:
                ref_mono = torchaudio.functional.resample(
                    ref_mono,
                    orig_freq=self.sampling_rate,
                    new_freq=self.prediction_pesq_sample_rate,
                )
                deg_mono = torchaudio.functional.resample(
                    deg_mono,
                    orig_freq=self.sampling_rate,
                    new_freq=self.prediction_pesq_sample_rate,
                )

            ref = ref_mono.squeeze(0).numpy()
            deg = deg_mono.squeeze(0).numpy()
            min_len = min(ref.shape[0], deg.shape[0])
            if min_len <= 0:
                return float("nan")
            ref = ref[:min_len]
            deg = deg[:min_len]

            score = float(pypesq_pesq(ref, deg, self.prediction_pesq_sample_rate))
            return score if math.isfinite(score) else float("nan")
        except Exception:
            return float("nan")

    def _get_production_predictor(self):
        if self._production_predictor_checked:
            return self._production_predictor

        self._production_predictor_checked = True
        try:
            from audiobox_aesthetics.infer import AesPredictor

            self._production_predictor = AesPredictor(checkpoint_pth=None, data_col="waveform")
        except Exception as e:
            self._production_predictor = None
            self._warn_once(
                "audiobox_init_failed",
                f"Warning: failed to initialize audiobox_aesthetics ({e}); production_quality metric will be skipped (NaN).",
            )
        return self._production_predictor

    def _compute_production_quality(self, estimated: torch.Tensor) -> float:
        predictor = self._get_production_predictor()
        if predictor is None:
            return float("nan")

        try:
            # Audiobox model is no-reference; use prediction only.
            mono = estimated.detach().cpu().float().mean(dim=0, keepdim=True)
            outputs = predictor.forward([{"waveform": mono, "sample_rate": self.sampling_rate}])
            if not outputs:
                return float("nan")
            pq = float(outputs[0].get("PQ", float("nan")))
            return pq if math.isfinite(pq) else float("nan")
        except Exception:
            return float("nan")

    def _get_or_compute_log_mel_pair(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if "log_mel_pair" in source_context:
            return source_context["log_mel_pair"]

        if not self._mel_metric_backend_available:
            self._warn_once(
                "mel_metrics_backend_unavailable",
                "Warning: MEL_SSIM/mel_kl_div/KL backend unavailable; returning NaN for mel-domain metrics.",
            )
            source_context["log_mel_pair"] = (None, None)
            return source_context["log_mel_pair"]

        try:
            pred_log_mel = self._log_mel_from_audio(estimated)
            target_log_mel = self._log_mel_from_audio(target)
            source_context["log_mel_pair"] = (pred_log_mel, target_log_mel)
        except Exception:
            source_context["log_mel_pair"] = (None, None)
        return source_context["log_mel_pair"]

    def _get_or_compute_mel_kl(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        if "mel_kl_value" in source_context:
            return float(source_context["mel_kl_value"])
        pred_log_mel, target_log_mel = self._get_or_compute_log_mel_pair(estimated, target, source_context)
        if pred_log_mel is None or target_log_mel is None:
            source_context["mel_kl_value"] = float("nan")
            return float("nan")
        source_context["mel_kl_value"] = self._kl_from_log_mel(pred_log_mel, target_log_mel)
        return float(source_context["mel_kl_value"])

    def _metric_sdr(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        """SDR (Signal-to-Distortion Ratio) via museval BSS Eval. Higher is better.

        Standard benchmark metric for MUSDB18. Returns dB; typical good values 8–20+ dB.
        """
        if museval is None:
            return float("nan")
        estimated_frame_np = source_context.get("estimated_frame_np")
        target_frame_np = source_context.get("target_frame_np")
        if estimated_frame_np is None or target_frame_np is None:
            return float("nan")
        try:
            est_np = estimated_frame_np[source_idx]
            tgt_np = target_frame_np[source_idx]
            t_len = est_np.shape[-1]
            est_np = est_np.T[np.newaxis, :, :]
            tgt_np = tgt_np.T[np.newaxis, :, :]
            sdr, _, _, _, _ = museval.metrics.bss_eval(
                tgt_np,
                est_np,
                window=t_len,
                hop=t_len,
                compute_permutation=False,
            )
            return float(np.mean(sdr))
        except Exception:
            return float("nan")

    def _metric_snr(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        del source_idx, source_context
        return self._compute_snr_value(estimated, target)

    def _metric_si_snr(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        del source_idx, source_context
        try:
            v = si_snr(estimated, target)
            return float(v.mean().item() if v.dim() > 0 else v.item())
        except Exception:
            return float("nan")

    def _metric_multi_mel_snr(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        del source_idx, source_context
        return self._multi_mel_snr_cached(estimated, target)

    def _metric_zimtohrli(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        del source_idx, source_context
        return self._zimtohrli_distance(estimated, target)

    def _metric_mel_ssim(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        del source_idx
        pred_log_mel, target_log_mel = self._get_or_compute_log_mel_pair(estimated, target, source_context)
        if pred_log_mel is None or target_log_mel is None:
            return float("nan")
        return self._ssim_from_log_mel(pred_log_mel, target_log_mel)

    def _metric_mel_kl_div(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        del source_idx
        return self._get_or_compute_mel_kl(estimated, target, source_context)

    def _metric_kl_alias(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        del source_idx
        return self._get_or_compute_mel_kl(estimated, target, source_context)

    def _metric_pesq(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        """PESQ perceptual speech quality score (ITU-T P.862). Higher is better.

        Range roughly -0.5 to 4.5 (MOS-LQO scale). Audio is resampled to mono at
        prediction_pesq_sample_rate (default 16 kHz) before scoring.
        """
        del source_idx, source_context
        return self._compute_pesq(estimated, target)

    def _metric_production_quality(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        """No-reference production quality score from AudioBox Aesthetics (PQ key). Higher is better."""
        del source_idx, target, source_context
        return self._compute_production_quality(estimated)

    def _metric_log_wmse(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        """LogWMSE for a single 10-second chunk. Requires 'mixture' in source_context."""
        del source_idx
        mixture = source_context.get("mixture")
        if mixture is None:
            return float("nan")
        audio_secs = estimated.shape[-1] / self.sampling_rate
        try:
            log_wmse_fn = _log_wmse_module._get_log_wmse_fn(audio_secs, self.sampling_rate)
            mix_in = mixture.float().cpu().unsqueeze(0)              # [1, C, T]
            est_in = estimated.float().cpu().unsqueeze(0).unsqueeze(2)  # [1, C, 1, T]
            tgt_in = target.float().cpu().unsqueeze(0).unsqueeze(2)    # [1, C, 1, T]
            val = float(log_wmse_fn(mix_in, est_in, tgt_in).item())
            return val if math.isfinite(val) else float("nan")
        except Exception:
            return float("nan")

    def _metric_l1_freq(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        del source_idx, source_context
        try:
            est_np = estimated.float().cpu().numpy()
            tgt_np = target.float().cpu().numpy()
            return float(L1Freq_metric(tgt_np, est_np))
        except Exception:
            return float("nan")

    def _metric_aura_stft(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        del source_idx, source_context
        try:
            est_np = estimated.float().cpu().numpy()
            tgt_np = target.float().cpu().numpy()
            return float(AuraSTFT_metric(tgt_np, est_np))
        except Exception:
            return float("nan")

    def _metric_aura_mrstft(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        del source_idx, source_context
        try:
            est_np = estimated.float().cpu().numpy()
            tgt_np = target.float().cpu().numpy()
            return float(AuraMRSTFT_metric(tgt_np, est_np, sample_rate=self.sampling_rate))
        except Exception:
            return float("nan")

    def _metric_bleed(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        del source_idx, source_context
        try:
            est_np = estimated.float().cpu().numpy()
            tgt_np = target.float().cpu().numpy()
            bleedless, _ = bleed_full(tgt_np, est_np, sr=self.sampling_rate)
            return float(bleedless)
        except Exception:
            return float("nan")

    def _metric_fullness(
        self,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        del source_idx, source_context
        try:
            est_np = estimated.float().cpu().numpy()
            tgt_np = target.float().cpu().numpy()
            _, fullness = bleed_full(tgt_np, est_np, sr=self.sampling_rate)
            return float(fullness)
        except Exception:
            return float("nan")

    def _compute_metric_value(
        self,
        metric: str,
        source_idx: int,
        estimated: torch.Tensor,
        target: torch.Tensor,
        source_context: dict[str, Any],
    ) -> float:
        metric_fn = self.metric_registry.get(metric)
        if metric_fn is None:
            return float("nan")
        try:
            return float(metric_fn(source_idx, estimated, target, source_context))
        except Exception:
            return float("nan")

    # ---------------------------------------------------------------------
    # Compute path
    # ---------------------------------------------------------------------
    def _safe_log(self, name: str, value: float, **kwargs) -> None:
        try:
            self.log(name, value, **kwargs)
        except MisconfigurationException:
            pass

    def _append_metric_value(
        self,
        values_per_metric: dict[str, dict[int, list[float]]],
        metric: str,
        source_idx: int,
        value: float,
    ) -> None:
        values_per_metric[metric][source_idx].append(float(value))

    def _compute_metrics(
        self,
        estimated: torch.Tensor,
        target: torch.Tensor,
        mixture: Optional[torch.Tensor] = None,
    ) -> dict[str, dict[int, float]]:
        """
        Compute all configured metrics on the full audio per source.

        Args:
            estimated: [n_sources, C, T]
            target:    [n_sources, C, T]
            mixture:   [C, T] — required for LogWMSE

        Returns:
            {metric_key: {source_idx: value}}
        """
        if not self.metric_keys:
            return {}

        if "sdr" in self.metric_keys and museval is None:
            raise ImportError("museval package is required for SDR. Install with: pip install museval")

        n_sources, n_channels, total_length = estimated.shape
        signal_threshold = 1e-6

        estimated_np = estimated.detach().cpu().numpy() if "sdr" in self.metric_keys else None
        target_np = target.detach().cpu().numpy() if "sdr" in self.metric_keys else None

        metric_dict: dict[str, dict[int, float]] = {metric: {} for metric in self.metric_keys}

        for source_idx in range(n_sources):
            est_source = estimated[source_idx]  # [C, T]
            tgt_source = target[source_idx]     # [C, T]

            if float(torch.sqrt((tgt_source ** 2).mean()).item()) <= signal_threshold:
                for metric in self.metric_keys:
                    metric_dict[metric][source_idx] = float("nan")
                continue

            source_context: dict[str, Any] = {"mixture": mixture}
            if estimated_np is not None:
                source_context["estimated_frame_np"] = estimated_np
                source_context["target_frame_np"] = target_np

            if "si_snr" in self.metric_keys:
                try:
                    v = si_snr(est_source, tgt_source)
                    metric_dict["si_snr"][source_idx] = float(v.mean().item() if v.dim() > 0 else v.item())
                except Exception:
                    metric_dict["si_snr"][source_idx] = float("nan")

            for metric in self.metric_keys:
                if metric == "si_snr":
                    continue
                metric_dict[metric][source_idx] = self._compute_metric_value(
                    metric=metric,
                    source_idx=source_idx,
                    estimated=est_source,
                    target=tgt_source,
                    source_context=source_context,
                )

        return metric_dict

    # ---------------------------------------------------------------------
    # Accumulation/reporting helpers
    # ---------------------------------------------------------------------
    @staticmethod
    def _is_valid_metric_value(x: float) -> bool:
        return isinstance(x, (float, int))

    def _accumulate_per_source_metrics(
        self,
        metrics: dict[str, dict[int, float]],
        per_source_store: dict[str, dict[int, list[float]]],
    ) -> None:
        for metric, per_source_vals in metrics.items():
            if metric not in per_source_store:
                per_source_store[metric] = {i: [] for i in range(self.n_sources)}
            for source_idx in range(self.n_sources):
                if source_idx not in per_source_vals:
                    continue
                val = float(per_source_vals[source_idx])
                if self._is_valid_metric_value(val):
                    per_source_store[metric].setdefault(source_idx, []).append(val)

    def _accumulate_family_metrics(
        self,
        family: str,
        metrics: dict[str, dict[int, float]],
        family_store: dict[str, dict[str, list[float]]],
        family_source_store: dict[str, dict[str, dict[int, list[float]]]],
    ) -> None:
        for metric, per_source_vals in metrics.items():
            family_store.setdefault(metric, {})
            family_source_store.setdefault(metric, {})

            family_store[metric].setdefault(family, [])
            family_source_store[metric].setdefault(family, {i: [] for i in range(self.n_sources)})

            for source_idx in range(self.n_sources):
                if source_idx not in per_source_vals:
                    continue
                val = float(per_source_vals[source_idx])
                if not self._is_valid_metric_value(val):
                    continue
                family_store[metric][family].append(val)
                family_source_store[metric][family].setdefault(source_idx, []).append(val)

    def _summaries_for_metrics(
        self,
        per_source_store: dict[str, dict[int, list[float]]],
        metrics: list[str],
    ) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, dict[str, float]]]:
        summaries = {}
        averages = {}
        for metric in metrics:
            per_source_values = per_source_store.get(metric, {i: [] for i in range(self.n_sources)})
            metric_summary = self._compute_source_summaries(per_source_values)
            summaries[metric] = metric_summary
            averages[metric] = self._compute_avg_over_sources(metric_summary)
        return summaries, averages

    def _print_final_results_table(
        self,
        title: str,
        source_names: list[str],
        summaries_by_metric: dict[str, dict[int, dict[str, float]]],
        avg_by_metric: dict[str, dict[str, float]],
        metrics: list[str],
        system_name: str = "model 1",
    ) -> None:
        include_mean = self.prediction_metric_include_mean
        include_median = self.prediction_metric_include_median

        headers_with_system = ["System", "Stem"]
        for metric in metrics:
            display = self._METRIC_DISPLAY_NAMES.get(metric, metric)
            if include_mean:
                headers_with_system.append(f"{display}_Mean")
                headers_with_system.append(f"{display}_STD")
            if include_median:
                headers_with_system.append(f"{display}_Median")

        rows: list[list[str]] = []
        for source_idx, source_name in enumerate(source_names):
            if source_name == "other":
                continue
            row = [system_name, source_name]
            for metric in metrics:
                summary = summaries_by_metric[metric][source_idx]
                if include_mean:
                    row.append(self._format_table_number(summary["mean"]))
                    row.append(self._format_table_number(summary.get("std")))
                if include_median:
                    row.append(self._format_table_number(summary["median"]))
            rows.append(row)

        avg_row = [system_name, "avg"]
        for metric in metrics:
            avg = avg_by_metric[metric]
            if include_mean:
                avg_row.append(self._format_table_number(avg["mean"]))
                avg_row.append(self._format_table_number(avg.get("std")))
            if include_median:
                avg_row.append(self._format_table_number(avg["median"]))
        rows.append(avg_row)

        print("\n" + title)
        print(";".join(headers_with_system))
        for row in rows:
            print(";".join(row))

        print()
        headers_no_system = headers_with_system[1:]
        print(";".join(headers_no_system))
        for row in rows:
            print(";".join(row[1:]))

    def _log_metric_stats(self, metric: str, values: list[float], log_prefix: str, label: str) -> None:
        if not values:
            print(f"  WARNING: No {self._METRIC_DISPLAY_NAMES.get(metric, metric)} values for {label}")
            return

        median_value = self._nanmedian(values)
        mean_value = self._nanmean(values)

        print(
            f"  {self._METRIC_DISPLAY_NAMES.get(metric, metric)}: "
            f"{self._format_metric_stats(median_value, mean_value, n=len(values))}"
        )

        if self.prediction_metric_include_median:
            self._safe_log(
                f"{log_prefix}_median/{metric}",
                median_value,
                sync_dist=False,
                on_epoch=True,
            )
        if self.prediction_metric_include_mean:
            self._safe_log(
                f"{log_prefix}_mean/{metric}",
                mean_value,
                sync_dist=False,
                on_epoch=True,
            )

    # ---------------------------------------------------------------------
    # Lightning hooks
    # ---------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        batch_size = batch["ground_truth"]["waveform"].shape[0]

        families = batch.get("family", [None] * batch_size)
        if not isinstance(families, list):
            families = [families] if families is not None else [None] * batch_size

        ground_truth_lengths = batch["ground_truth"]["lengths"]
        prediction_lengths = batch["prediction"]["lengths"]

        # Optional mixture (unprocessed audio) for LogWMSE; shape [B, C, T]
        unprocessed_batch = batch.get("unprocessed")

        for batch_idx_sample in range(batch_size):
            family = families[batch_idx_sample] if batch_idx_sample < len(families) else None

            gt_length = int(ground_truth_lengths[batch_idx_sample].item())
            pred_length = int(prediction_lengths[batch_idx_sample].item())

            gt_waveform = batch["ground_truth"]["waveform"][batch_idx_sample, :, :gt_length]
            pred_waveform = batch["prediction"]["waveform"][batch_idx_sample, :, :pred_length]

            if gt_length == 0 or pred_length == 0:
                continue

            min_length = min(gt_length, pred_length)
            gt_waveform = gt_waveform[:, :min_length]
            pred_waveform = pred_waveform[:, :min_length]

            # Per-sample mixture for LogWMSE [C, T], trimmed to min_length
            mixture_sample: Optional[torch.Tensor] = None
            if unprocessed_batch is not None:
                unp_lengths = unprocessed_batch.get("lengths")
                if unp_lengths is not None:
                    unp_len = int(unp_lengths[batch_idx_sample].item())
                else:
                    unp_len = unprocessed_batch["waveform"].shape[-1]
                unp_wav = unprocessed_batch["waveform"][batch_idx_sample, :, :unp_len]
                mixture_sample = unp_wav[:, :min_length]

            self.total_samples_processed += 1

            source_label = batch["source_label"][batch_idx_sample]
            if source_label not in self.columns:
                continue
            source_idx = self.columns.index(source_label)

            n_sources = self.n_sources
            n_channels = gt_waveform.shape[0]
            T = gt_waveform.shape[-1]

            target_sources = torch.zeros((n_sources, n_channels, T), device=gt_waveform.device)
            separated_sources = torch.zeros((n_sources, n_channels, T), device=pred_waveform.device)
            target_sources[source_idx] = gt_waveform
            separated_sources[source_idx] = pred_waveform

            if self.fad_clap_metric is not None:
                source_mask = torch.zeros((n_sources,), device=target_sources.device, dtype=torch.float32)
                source_mask[source_idx] = 1.0
                self.fad_clap_metric.update(
                    {"waveform": separated_sources.unsqueeze(0)},
                    {"waveform": target_sources.unsqueeze(0), "source_mask": source_mask.unsqueeze(0)},
                )

            metrics = self._compute_metrics(separated_sources, target_sources, mixture_sample)

            self._accumulate_per_source_metrics(metrics, self.per_source_metric_values)

            # ── Per-sample score recording ────────────────────────────────────
            indices = batch.get("index", [None] * batch_size)
            dataset_idx = indices[batch_idx_sample] if batch_idx_sample < len(indices) else None
            clip_id_sample = batch["clip_id"][batch_idx_sample]

            row: dict = {
                "dataset_idx": dataset_idx,
                "clip_id": clip_id_sample,
                "source_label": source_label,
            }
            for metric, per_source_vals in metrics.items():
                val = per_source_vals.get(source_idx, float("nan"))
                row[metric] = round(float(val), 3) if math.isfinite(float(val)) else None
            self.per_sample_scores.append(row)
            # ─────────────────────────────────────────────────────────────────

            if family is not None:
                self._accumulate_family_metrics(
                    family,
                    metrics,
                    self.per_family_metric_values,
                    self.per_family_and_source_metric_values,
                )

            if self.total_samples_processed % 500 == 0:
                self._print_intermediate_results(self.total_samples_processed)

    def _print_intermediate_results(self, sample_count: int) -> None:
        source_names = [self.columns[i] for i in range(self.n_sources)]

        print("\n" + "=" * 80)
        print(f"INTERMEDIATE RESULTS (after {sample_count} samples)")
        print("=" * 80)

        if not self.metric_keys:
            return
        print("\n--- Per-Source Metrics ---")
        summaries, averages = self._summaries_for_metrics(self.per_source_metric_values, self.metric_keys)

        for source_idx in range(self.n_sources):
            source_name = source_names[source_idx]
            print(f"\n{source_name}:")
            for metric in self.metric_keys:
                vals = self.per_source_metric_values.get(metric, {}).get(source_idx, [])
                if vals:
                    med = self._nanmedian(vals)
                    mean = self._nanmean(vals)
                    print(
                        f"  {self._METRIC_DISPLAY_NAMES.get(metric, metric)}: "
                        f"{self._format_metric_stats(med, mean, n=len(vals))}"
                    )
                else:
                    print(f"  {self._METRIC_DISPLAY_NAMES.get(metric, metric)}: WARNING - No values")

        print("\n--- Avg over Sources ---")
        for metric in self.metric_keys:
            avg = averages[metric]
            print(
                f"  {self._METRIC_DISPLAY_NAMES.get(metric, metric)}: "
                f"{self._format_metric_stats(avg['median'], avg['mean'], n=int(avg['n_sources']), n_label='n_sources')}"
            )

        print("=" * 80 + "\n")

    def on_test_epoch_start(self):
        if not self._metrics_initialized:
            self.reset_metrics()
            self._metrics_initialized = True
            self.total_samples_processed = 0

    def on_test_epoch_end(self):
        source_names = [self.columns[i] for i in range(self.n_sources)]

        print("\n" + "=" * 80)
        print(f"FINAL PREDICTION EVALUATION METRICS (after {self.total_samples_processed} samples)")
        print("=" * 80)

        has_data = any(
            len(self.per_source_metric_values.get(metric, {}).get(source_idx, [])) > 0
            for metric in self.metric_keys
            for source_idx in range(self.n_sources)
        )
        if not has_data:
            print("  No samples found")
            print("=" * 80 + "\n")
            return

        fad_clap_per_source: dict[int, float] = {}
        if self.fad_clap_metric is not None:
            fad_clap_score = self.fad_clap_metric.compute()
            overall_fad = float(fad_clap_score.get("fad_clap", 0.0))
            print(f"\nFAD-CLAP (10s): overall={overall_fad:.3f}")
            self._safe_log("eval_prediction_overall/fad_clap", overall_fad, sync_dist=False, on_epoch=True)
            per_source_scores = fad_clap_score.get("fad_clap_per_source", {}) or {}
            fad_clap_per_source = per_source_scores
            for source_idx in range(self.n_sources):
                source_name = source_names[source_idx]
                if source_idx in per_source_scores:
                    v = float(per_source_scores[source_idx])
                    self._safe_log(
                        f"eval_prediction_sources_10s_median/{source_name}/fad_clap",
                        v,
                        sync_dist=False,
                        on_epoch=True,
                    )

        # Per-source reporting
        print("\n--- Per-Source Metrics ---")
        for source_idx in range(self.n_sources):
            source_name = source_names[source_idx]
            print(f"\n{source_name}:")
            for metric in self.metric_keys:
                values = self.per_source_metric_values.get(metric, {}).get(source_idx, [])
                if not values:
                    print(f"  WARNING: No {self._METRIC_DISPLAY_NAMES.get(metric, metric)} values for {source_name}")
                    continue
                median_val = self._nanmedian(values)
                mean_val = self._nanmean(values)
                print(
                    f"  {self._METRIC_DISPLAY_NAMES.get(metric, metric)}: "
                    f"{self._format_metric_stats(median_val, mean_val, n=len(values))}"
                )
                if self.prediction_metric_include_median:
                    self._safe_log(
                        f"eval_prediction_sources_median/{source_name}/{metric}",
                        median_val, sync_dist=False, on_epoch=True,
                    )
                if self.prediction_metric_include_mean:
                    self._safe_log(
                        f"eval_prediction_sources_mean/{source_name}/{metric}",
                        mean_val, sync_dist=False, on_epoch=True,
                    )

        summaries, averages = self._summaries_for_metrics(self.per_source_metric_values, self.metric_keys)
        print("\n--- Avg over Sources ---")
        for metric in self.metric_keys:
            avg = averages[metric]
            print(
                f"  {self._METRIC_DISPLAY_NAMES.get(metric, metric)}: "
                f"{self._format_metric_stats(avg['median'], avg['mean'], n=int(avg['n_sources']), n_label='n_sources')}"
            )

        table_metrics = [m for m in self.metric_keys if m != "snr"]
        self._print_final_results_table(
            title="--- FINAL RESULTS TABLE ---",
            source_names=source_names,
            summaries_by_metric=summaries,
            avg_by_metric=averages,
            metrics=table_metrics,
        )

        # ── Save per-stem (+ 5-stem / overall) metrics summary to CSV ──────────
        if self.per_stem_metrics_output_path is not None:
            per_stem_rows = self._build_per_stem_summary_rows(source_names, summaries, fad_clap_per_source)
            self._save_per_stem_metrics_csv(self.per_stem_metrics_output_path, per_stem_rows)
            self._print_per_stem_metrics_csv_table(self.per_stem_metrics_output_path)

        # Per-family + per-family-source reporting
        all_families: set[str] = set()
        for metric in self.metric_keys:
            all_families.update(self.per_family_metric_values.get(metric, {}).keys())

        if all_families:
            print("\n--- Per-Family Metrics ---")
            for family in sorted(all_families):
                print(f"\n{family}:")
                for metric in self.metric_keys:
                    values = self.per_family_metric_values.get(metric, {}).get(family, [])
                    if not values:
                        print(f"  WARNING: No {self._METRIC_DISPLAY_NAMES.get(metric, metric)} values for family {family}")
                        continue
                    median_val = self._nanmedian(values)
                    mean_val = self._nanmean(values)
                    print(
                        f"  {self._METRIC_DISPLAY_NAMES.get(metric, metric)}: "
                        f"{self._format_metric_stats(median_val, mean_val, n=len(values))}"
                    )
                    if self.prediction_metric_include_median:
                        self._safe_log(
                            f"eval_prediction_families_median/{family}/{metric}",
                            median_val, sync_dist=False, on_epoch=True,
                        )
                    if self.prediction_metric_include_mean:
                        self._safe_log(
                            f"eval_prediction_families_mean/{family}/{metric}",
                            mean_val, sync_dist=False, on_epoch=True,
                        )

                print(f"\n  --- Per-Family+Per-Source ---")
                for source_idx in range(self.n_sources):
                    source_name = source_names[source_idx]
                    print(f"\n  {source_name}:")
                    for metric in self.metric_keys:
                        values = (
                            self.per_family_and_source_metric_values.get(metric, {})
                            .get(family, {})
                            .get(source_idx, [])
                        )
                        if not values:
                            print(f"    WARNING: No {self._METRIC_DISPLAY_NAMES.get(metric, metric)} values for {family}/{source_name}")
                            continue
                        median_val = self._nanmedian(values)
                        mean_val = self._nanmean(values)
                        print(
                            f"    {self._METRIC_DISPLAY_NAMES.get(metric, metric)}: "
                            f"{self._format_metric_stats(median_val, mean_val, n=len(values))}"
                        )
                        if self.prediction_metric_include_median:
                            self._safe_log(
                                f"eval_prediction_families_median/{family}/{source_name}/{metric}",
                                median_val, sync_dist=False, on_epoch=True,
                            )
                        if self.prediction_metric_include_mean:
                            self._safe_log(
                                f"eval_prediction_families_mean/{family}/{source_name}/{metric}",
                                mean_val, sync_dist=False, on_epoch=True,
                            )

        # ── Save per-sample scores to CSV ─────────────────────────────────────
        if self.per_sample_scores and self.per_sample_scores_output_path is not None:
            self._save_per_sample_scores(self.per_sample_scores_output_path)

        print("\n" + "=" * 80)

    def _save_per_sample_scores(self, output_path: Path) -> None:
        """Write per-sample metric scores to a CSV file (3 decimal places)."""
        import csv

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.per_sample_scores:
            return

        fixed_cols = ["dataset_idx", "clip_id", "source_label"]
        all_cols = fixed_cols + list(self.metric_keys)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
            writer.writeheader()
            for row in self.per_sample_scores:
                writer.writerow({col: row.get(col, "") for col in all_cols})

        print(f"\nPer-sample scores saved to: {output_path}")

    def _build_per_stem_summary_rows(
        self,
        source_names: list[str],
        summaries: dict[str, dict[int, dict[str, float]]],
        fad_clap_per_source: dict[int, float],
    ) -> list[dict]:
        """
        Build one row per stem plus an "overall_mean" row (all stems except "other")
        with median/mean/std for every metric in `self.metric_keys` and, if FAD-CLAP
        was enabled, a "fad_clap" column computed for that same row's stem(s).
        """
        has_fad_clap = self.fad_clap_metric is not None
        n_count_metric = self.metric_keys[0] if self.metric_keys else None

        def _metric_value_cols(metric: str, summary: dict[str, float]) -> dict:
            cols: dict = {}
            if self.prediction_metric_include_mean:
                cols[f"{metric}_mean"] = summary.get("mean")
                cols[f"{metric}_std"] = summary.get("std")
            if self.prediction_metric_include_median:
                cols[f"{metric}_median"] = summary.get("median")
            return cols

        def _row_for_indices(label: str, indices: list[int]) -> dict:
            row: dict = {"stem": label}
            if n_count_metric is not None:
                row["n_samples"] = sum(
                    int(summaries[n_count_metric].get(i, {}).get("n", 0)) for i in indices
                )
            for metric in self.metric_keys:
                summary = (
                    summaries[metric].get(indices[0], {})
                    if len(indices) == 1
                    else self._compute_avg_over_source_subset(summaries[metric], indices)
                )
                row.update(_metric_value_cols(metric, summary))
            if has_fad_clap:
                if len(indices) == 1:
                    row["fad_clap"] = fad_clap_per_source.get(indices[0])
                else:
                    row["fad_clap"] = self.fad_clap_metric.compute_for_sources(indices)
            return row

        rows: list[dict] = [
            _row_for_indices(source_name, [source_idx])
            for source_idx, source_name in enumerate(source_names)
        ]

        overall_indices = [i for i, name in enumerate(source_names) if name != "other"]
        if overall_indices:
            rows.append(_row_for_indices("overall_mean", overall_indices))

        return rows

    def _save_per_stem_metrics_csv(self, output_path: Path, rows: list[dict]) -> None:
        """Write the per-stem (+overall) aggregated metrics to a CSV (3 decimal places)."""
        import csv

        if not rows:
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_cols: list[str] = []
        for row in rows:
            for col in row:
                if col not in all_cols:
                    all_cols.append(col)

        def _fmt(value):
            if value is None:
                return ""
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                return value
            if not math.isfinite(float(value)):
                return ""
            return int(value) if float(value).is_integer() else round(float(value), 3)

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_cols, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({col: _fmt(row.get(col)) for col in all_cols})

        print(f"\nPer-stem metrics summary saved to: {output_path}")

    def _print_per_stem_metrics_csv_table(self, csv_path: Path) -> None:
        """Load the per-stem metrics CSV again and print the main metrics as a table (one row per
        stem except "other", then the overall mean). Means are shown when available,
        otherwise medians; FAD-CLAP is a single value per row."""
        import csv

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return

        suffix = "mean" if self.prediction_metric_include_mean else "median"
        columns = [
            (metric, f"{metric}_{suffix}")
            for metric in ("sdr", "si_snr", "multi_mel_snr", "zimtohrli")
            if f"{metric}_{suffix}" in rows[0]
        ]
        if "fad_clap" in rows[0]:
            columns.append(("fad_clap", "fad_clap"))
        if not columns:
            return

        col_width = 16

        def _fmt_row(label: str, values: list[str]) -> str:
            return "    " + "".join(f"{v:<{col_width}}" for v in [label] + values)

        def _fmt_value(value: str | None) -> str:
            return f"{float(value):.3f}" if value else "n/a"

        separator = "    " + "-" * (col_width * (len(columns) + 1))
        stem_rows = [r for r in rows if r["stem"] not in ("other", "overall_mean")]
        summary_rows = [r for r in rows if r["stem"] == "overall_mean"]

        print(f"\n  PER-STEM RESULTS ({suffix} over clips, from {csv_path}):")
        print(_fmt_row("stem", [name for name, _ in columns]))
        print(separator)
        for row in stem_rows:
            print(_fmt_row(row["stem"], [_fmt_value(row.get(col)) for _, col in columns]))
        if summary_rows:
            print(separator)
            for row in summary_rows:
                print(_fmt_row(row["stem"], [_fmt_value(row.get(col)) for _, col in columns]))

    def configure_optimizers(self):
        return None
