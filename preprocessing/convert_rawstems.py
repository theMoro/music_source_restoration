#!/usr/bin/env python3

"""
Utilities to convert the Raw Stems dataset into a Hugging Face dataset.

Each track directory under the dataset root has subdirectories for instrument
families that contain one or more FLAC stems, sometimes organised into nested
subfolders (e.g. `Rhy/DK/01_Kick.flac`). The instrument taxonomy roughly maps to
the same high-level families we already use elsewhere:

    vocals, guitars, bass, keyboards, synthesizers, drums, percussions, orchestral

This script walks the dataset tree, maps every available stem to the canonical
categories above, renders per-category summed WAV files, and applies quality
checks (optional, `--quality-filter`) to reject problematic tracks (sample-rate
mismatch, duration mismatch, low volume, misalignment, and high correlation/leakage).
It writes quality manifests under `<save_dir>/quality_reports`. Stems in folders that cannot be
mapped to any category are summed into the residual `other` file (empty string if
there are none).

Usage:
    python preprocessing/convert_rawstems.py --dataset-root <RawStems> --save-dir <hf_root>/raw_stems
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
from pathlib import Path
from shutil import move, rmtree
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
from datasets import Dataset, DatasetDict, Features, Sequence, Value

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    tqdm = None


ADDITIONAL_WAVS_DIRNAME = "additional_wavs"
QUALITY_REPORTS_DIRNAME = "quality_reports"

DEFAULT_MIN_RMS_DB = -60.0
DEFAULT_MAX_ALIGNMENT_OFFSET_MS = 10.0
DEFAULT_MAX_STEM_CORRELATION = 0.95
DEFAULT_MIN_DURATION_SEC = 5.0
DEFAULT_MAX_ANALYSIS_SECONDS = 30.0
DEFAULT_MAX_STEMS_FOR_PAIRWISE = 12
DEFAULT_STRICT_ALIGNMENT_MODE = False
DEFAULT_STRICT_ALIGNMENT_MIN_CONFIDENCE = 0.30
DEFAULT_STRICT_ALIGNMENT_WINDOW_SECONDS = 5.0
DEFAULT_STRICT_ALIGNMENT_WINDOW_HOP_SECONDS = 2.5

TARGET_CATEGORIES: Tuple[str, ...] = (
    "vocals",
    "guitars",
    "bass",
    "keyboards",
    "synthesizers",
    "drums",
    "percussions",
    "orchestral",
)

# Main directory names (normalised to lowercase) mapped to canonical categories.
RAW_MAIN_CATEGORY_MAP: Dict[str, str] = {
    "voc": "vocals",
    "voclv": "vocals",
    "oc": "vocals",  # stray directory missing the leading 'v'
    "gtr": "guitars",
    "bass": "bass",
    "bss": "bass",  # duplicated spelling
    "kbs": "keyboards",
    "synth": "synthesizers",
    "orch": "orchestral",
    "misc": "synthesizers",  # miscellaneous effects and alternate captures
    "room-ar70.wav": "synthesizers",  # room microphones; keep in "other" mix
}

# Rhythm section subdirectory names mapped to canonical categories.
RHY_SUBCATEGORY_MAP: Dict[str, str] = {
    "dk": "drums",
    "perc": "percussions",
    "oerc": "percussions",  # occasional misspelling
}


def _validate_directory(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")
    return path


def _initialise_category_record() -> Dict[str, object]:
    return {"paths": [], "is_present": False, "num_stems": 0}


def _initialise_category_records() -> Dict[str, Dict[str, object]]:
    return {category: _initialise_category_record() for category in TARGET_CATEGORIES}


def _load_stem_audio(path: Path) -> Tuple[np.ndarray, int]:
    frames, sample_rate = sf.read(str(path), always_2d=True)
    return frames.astype(np.float32, copy=False), sample_rate


def _prepare_stem_cache(paths: Iterable[str]) -> Tuple[Dict[str, np.ndarray], int, int, int]:
    loaded: Dict[str, np.ndarray] = {}
    ref_len = 0
    ref_channels = 0
    ref_sr: int | None = None

    for path_str in paths:
        path = Path(path_str)
        audio, sr = _load_stem_audio(path)
        loaded[path_str] = audio
        ref_len = max(ref_len, audio.shape[0])
        ref_channels = max(ref_channels, audio.shape[1])
        if ref_sr is None:
            ref_sr = sr
        elif sr != ref_sr:
            raise ValueError(f"Sample rate mismatch for stems: {sr} vs expected {ref_sr} ({path})")

    if ref_sr is None:
        raise ValueError("No stems available to determine reference sample rate.")

    return loaded, ref_len, ref_channels, ref_sr


def _align_cached(audio: np.ndarray, ref_len: int, ref_channels: int) -> np.ndarray:
    frames = audio
    if frames.shape[1] == 1 and ref_channels == 2:
        frames = np.repeat(frames, 2, axis=1)
    elif frames.shape[1] != ref_channels:
        raise ValueError(f"Unexpected channel count {frames.shape[1]} (expected {ref_channels}).")

    if frames.shape[0] < ref_len:
        padded = np.zeros((ref_len, ref_channels), dtype=np.float32)
        padded[: frames.shape[0], : frames.shape[1]] = frames
        return padded
    return frames[:ref_len, :ref_channels]


def _sum_cached(paths: Iterable[str], cache: Dict[str, np.ndarray], ref_len: int, ref_channels: int) -> np.ndarray:
    paths = list(paths)
    if not paths:
        return np.zeros((ref_len, ref_channels), dtype=np.float32)
    mixdown = np.zeros((ref_len, ref_channels), dtype=np.float32)
    for path_str in paths:
        mixdown += _align_cached(cache[path_str], ref_len, ref_channels)
    return mixdown


def _render_category_bounces(
    song_id: str,
    split_name: str,
    category_records: Dict[str, Dict[str, object]],
    residual_paths: List[str],
    temp_dir: Path,
    final_dir: Path,
) -> Tuple[Dict[str, str], str, int]:
    all_paths = [path for path in chain.from_iterable(record["paths"] for record in category_records.values())]
    if not all_paths:
        raise ValueError(f"No stems found for song '{song_id}'")

    cache, ref_len, ref_channels, sample_rate = _prepare_stem_cache(all_paths + residual_paths)
    category_audio_paths: Dict[str, str] = {}

    for category in TARGET_CATEGORIES:
        stems = category_records[category]["paths"]
        if stems:
            summed = _sum_cached(stems, cache, ref_len, ref_channels)
            filename = f"{split_name}_{song_id}_{category}.wav"
            temp_path = temp_dir / filename
            final_path = final_dir / filename
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(str(temp_path), summed, sample_rate)
            category_audio_paths[category] = str(final_path.resolve())
        else:
            category_audio_paths[category] = ""

    # Residual "other": sum of all stems in folders that could not be mapped to a category.
    other_audio_path = ""
    if residual_paths:
        other_audio = _sum_cached(residual_paths, cache, ref_len, ref_channels)
        filename = f"{split_name}_{song_id}_other.wav"
        temp_other = temp_dir / filename
        final_other = final_dir / filename
        temp_other.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(temp_other), other_audio, sample_rate)
        other_audio_path = str(final_other.resolve())

    return category_audio_paths, other_audio_path, sample_rate


def _iter_song_dirs(dataset_root: Path) -> Iterable[Path]:
    for song_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        yield song_dir


def _normalise_name(name: str) -> str:
    return name.strip().lower()


def _collect_category_records(song_dir: Path) -> Tuple[Dict[str, Dict[str, object]], List[str], List[Path], List[str]]:
    category_records = _initialise_category_records()
    unmapped: List[str] = []
    residual_paths: List[str] = []
    fallback_candidates: List[Path] = []

    for category_dir in sorted(path for path in song_dir.iterdir() if path.is_dir()):
        normalised = _normalise_name(category_dir.name)
        if normalised == "rhy":
            for sub_dir in sorted(path for path in category_dir.iterdir() if path.is_dir()):
                sub_name = _normalise_name(sub_dir.name)
                target_category = RHY_SUBCATEGORY_MAP.get(sub_name)
                if target_category is None:
                    unmapped.append(f"{song_dir.name}:{category_dir.name}/{sub_dir.name}")
                    residual_paths.extend(str(stem.resolve()) for stem in sorted(sub_dir.rglob("*.flac")))
                    continue
                stems = sorted(sub_dir.rglob("*.flac"))
                if not stems:
                    continue
                record = category_records[target_category]
                for stem in stems:
                    resolved = stem.resolve()
                    record["paths"].append(str(resolved))
                    record["num_stems"] += 1
                    record["is_present"] = True
                    fallback_candidates.append(resolved)
            direct_stems = sorted(category_dir.glob("*.flac"))
            if direct_stems:
                record = category_records["drums"]
                for stem in direct_stems:
                    resolved = stem.resolve()
                    record["paths"].append(str(resolved))
                    record["num_stems"] += 1
                    record["is_present"] = True
                    fallback_candidates.append(resolved)
            continue

        target_category = RAW_MAIN_CATEGORY_MAP.get(normalised)
        if target_category is None:
            unmapped.append(f"{song_dir.name}:{category_dir.name}")
            residual_paths.extend(str(stem.resolve()) for stem in sorted(category_dir.rglob("*.flac")))
            continue

        stems = sorted(category_dir.rglob("*.flac"))
        if not stems:
            continue
        record = category_records[target_category]
        for stem in stems:
            resolved = stem.resolve()
            record["paths"].append(str(resolved))
            record["num_stems"] += 1
            record["is_present"] = True
            fallback_candidates.append(resolved)

    return category_records, unmapped, fallback_candidates, residual_paths


def _compute_duration_seconds(fallbacks: Iterable[Path]) -> float:
    for path in fallbacks:
        if not path.exists():
            continue
        try:
            with sf.SoundFile(str(path)) as handle:
                samplerate = float(handle.samplerate)
                if samplerate <= 0:
                    continue
                return float(handle.frames) / samplerate
        except (RuntimeError, OSError, ValueError):
            continue
    candidate_list = ", ".join(str(path) for path in fallbacks)
    raise ValueError(f"Unable to compute duration; no readable audio among: {candidate_list}")


def _build_features() -> Features:
    feature_dict = {
        "song_id": Value("string"),
        "artist": Value("string"),
        "song_name": Value("string"),
        "split": Value("string"),
        "duration_seconds": Value("float32"),
        "sample_rate": Value("int32"),
        "unmapped_items": Sequence(Value("string")),
        "categories_present": Sequence(Value("string")),
    }

    for category in TARGET_CATEGORIES:
        feature_dict[f"{category}_stems"] = Sequence(Value("string"))
        feature_dict[f"{category}_is_present"] = Value("bool")
        feature_dict[f"{category}_num_stems"] = Value("int32")
        feature_dict[f"{category}_audio"] = Value("string")

    feature_dict["other_audio"] = Value("string")

    return Features(feature_dict)


def _split_artist_song(song_id: str) -> Tuple[str, str]:
    if "|" not in song_id:
        return "", song_id
    artist, title = song_id.split("|", 1)
    return artist, title


def _path_label(path: Path) -> str:
    parts = path.parts
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    return path.name


def _compute_rms_db(path: Path) -> float:
    total_power = 0.0
    total_samples = 0
    with sf.SoundFile(str(path)) as handle:
        for block in handle.blocks(blocksize=262_144, dtype="float32", always_2d=True):
            if block.size == 0:
                continue
            mono = block.mean(axis=1)
            total_power += float(np.dot(mono, mono))
            total_samples += int(mono.shape[0])
    if total_samples <= 0:
        return -120.0
    rms = np.sqrt(total_power / float(total_samples))
    return float(20.0 * np.log10(rms + 1e-8))


def _load_mono_excerpt(path: Path, max_seconds: float) -> Tuple[np.ndarray, int]:
    with sf.SoundFile(str(path)) as handle:
        sample_rate = int(handle.samplerate)
        if sample_rate <= 0:
            raise ValueError(f"Invalid samplerate in {path}: {sample_rate}")
        max_frames = int(round(max_seconds * sample_rate)) if max_seconds > 0 else -1
        audio = handle.read(frames=max_frames, dtype="float32", always_2d=True)
    if audio.size == 0:
        return np.zeros((0,), dtype=np.float32), sample_rate
    mono = audio.mean(axis=1).astype(np.float32, copy=False)
    return mono, sample_rate


def _estimate_offset_ms(reference: np.ndarray, other: np.ndarray, sample_rate: int) -> float:
    offset_ms, _, _ = _estimate_offset_and_peak_ratio(reference, other, sample_rate, use_phat=False)
    return offset_ms


def _estimate_offset_and_peak_ratio(
    reference: np.ndarray,
    other: np.ndarray,
    sample_rate: int,
    *,
    use_phat: bool,
) -> Tuple[float, float, float]:
    min_len = min(reference.shape[0], other.shape[0])
    if min_len <= 1:
        return 0.0, 0.0, float(sample_rate)

    max_points = 20_000
    stride = max(1, min_len // max_points)
    ref = reference[:min_len:stride].astype(np.float64, copy=False)
    oth = other[:min_len:stride].astype(np.float64, copy=False)
    if ref.size <= 1 or oth.size <= 1:
        return 0.0, 0.0, float(sample_rate) / float(stride)

    ref -= ref.mean()
    oth -= oth.mean()
    ref_energy = float(np.dot(ref, ref))
    oth_energy = float(np.dot(oth, oth))
    if ref_energy <= 1e-12 or oth_energy <= 1e-12:
        return 0.0, 0.0, float(sample_rate) / float(stride)

    n = ref.size + oth.size - 1
    nfft = 1 << (n - 1).bit_length()
    fft_ref = np.fft.rfft(ref, nfft)
    fft_oth = np.fft.rfft(oth, nfft)
    spectrum = fft_ref * np.conj(fft_oth)
    if use_phat:
        denom = np.abs(spectrum)
        spectrum = spectrum / np.maximum(denom, 1e-12)
    corr = np.fft.irfft(spectrum, nfft)
    corr = np.concatenate((corr[-(oth.size - 1) :], corr[: ref.size]))
    corr_abs = np.abs(corr)
    peak_idx = int(np.argmax(corr_abs))
    lag = int(peak_idx - (oth.size - 1))
    exclusion = 2
    left = max(0, peak_idx - exclusion)
    right = min(corr_abs.size, peak_idx + exclusion + 1)
    peak = float(corr_abs[peak_idx])
    if left < right:
        masked = np.concatenate((corr_abs[:left], corr_abs[right:]))
    else:
        masked = corr_abs
    second = float(np.max(masked)) if masked.size else 0.0
    peak_ratio = peak / (second + 1e-8)
    effective_sr = float(sample_rate) / float(stride)
    return float((lag / effective_sr) * 1000.0), float(peak_ratio), effective_sr


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size <= 1:
        return values
    kernel = np.ones((window,), dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _compute_onset_envelope(signal_in: np.ndarray, sample_rate: float) -> np.ndarray:
    if signal_in.size <= 1:
        return signal_in.astype(np.float64, copy=False)
    diff = np.abs(np.diff(signal_in.astype(np.float64, copy=False), prepend=signal_in[0]))
    smooth_window = max(1, int(round(0.02 * sample_rate)))
    return _moving_average(diff, smooth_window)


def _compute_active_ratio(a: np.ndarray, b: np.ndarray) -> float:
    min_len = min(a.shape[0], b.shape[0])
    if min_len <= 1:
        return 0.0
    x = a[:min_len]
    y = b[:min_len]
    x_rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) + 1e-12
    y_rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2))) + 1e-12
    x_thr = max(1e-5, 0.1 * x_rms)
    y_thr = max(1e-5, 0.1 * y_rms)
    mask = (np.abs(x) > x_thr) & (np.abs(y) > y_thr)
    return float(np.mean(mask))


def _estimate_offset_with_confidence(reference: np.ndarray, other: np.ndarray, sample_rate: int) -> Tuple[float, float, Dict[str, float]]:
    base_offset, base_peak, effective_sr = _estimate_offset_and_peak_ratio(
        reference, other, sample_rate, use_phat=False
    )
    gcc_offset, gcc_peak, _ = _estimate_offset_and_peak_ratio(reference, other, sample_rate, use_phat=True)

    ref_env = _compute_onset_envelope(reference, effective_sr)
    oth_env = _compute_onset_envelope(other, effective_sr)
    env_offset, env_peak, _ = _estimate_offset_and_peak_ratio(ref_env, oth_env, sample_rate, use_phat=False)

    lags = np.array([base_offset, gcc_offset, env_offset], dtype=np.float64)
    lag_med = float(np.median(lags))
    lag_spread = float(np.median(np.abs(lags - lag_med)))
    activity_ratio = _compute_active_ratio(reference, other)

    peak_norm = np.clip(np.array([base_peak, gcc_peak, env_peak], dtype=np.float64) - 1.0, 0.0, None)
    peak_score = float(np.clip(np.mean(peak_norm) / 4.0, 0.0, 1.0))
    agreement_score = float(np.exp(-lag_spread / 8.0))
    activity_score = float(np.clip(activity_ratio / 0.25, 0.0, 1.0))
    confidence = float(np.clip(0.5 * peak_score + 0.3 * agreement_score + 0.2 * activity_score, 0.0, 1.0))

    details = {
        "lag_waveform_ms": float(base_offset),
        "lag_gcc_phat_ms": float(gcc_offset),
        "lag_onset_ms": float(env_offset),
        "peak_ratio_waveform": float(base_peak),
        "peak_ratio_gcc_phat": float(gcc_peak),
        "peak_ratio_onset": float(env_peak),
        "lag_spread_ms": float(lag_spread),
        "activity_ratio": float(activity_ratio),
    }
    return lag_med, confidence, details


def _windowed_alignment_stats(
    reference: np.ndarray,
    other: np.ndarray,
    sample_rate: int,
    *,
    window_seconds: float,
    hop_seconds: float,
    confidence_threshold: float,
    offset_threshold_ms: float,
) -> Dict[str, float]:
    if sample_rate <= 0 or reference.size <= 1 or other.size <= 1:
        return {
            "num_windows_total": 0.0,
            "num_windows_high_conf": 0.0,
            "median_abs_lag_ms": 0.0,
            "iqr_lag_ms": 0.0,
            "exceed_ratio": 0.0,
        }

    win = max(1, int(round(window_seconds * sample_rate)))
    hop = max(1, int(round(hop_seconds * sample_rate)))
    total_len = min(reference.shape[0], other.shape[0])
    if total_len <= win:
        lag, conf, _ = _estimate_offset_with_confidence(reference[:total_len], other[:total_len], sample_rate)
        abs_lag = abs(lag)
        high = 1.0 if conf >= confidence_threshold else 0.0
        exceed = 1.0 if abs_lag > offset_threshold_ms and high > 0 else 0.0
        return {
            "num_windows_total": 1.0,
            "num_windows_high_conf": high,
            "median_abs_lag_ms": float(abs_lag if high else 0.0),
            "iqr_lag_ms": 0.0,
            "exceed_ratio": float(exceed),
        }

    abs_lags: List[float] = []
    high_conf_lags: List[float] = []
    windows_total = 0
    windows_high = 0
    windows_exceed = 0

    for start in range(0, total_len - win + 1, hop):
        end = start + win
        lag_ms, conf, _ = _estimate_offset_with_confidence(reference[start:end], other[start:end], sample_rate)
        windows_total += 1
        abs_lag = abs(float(lag_ms))
        abs_lags.append(abs_lag)
        if conf >= confidence_threshold:
            windows_high += 1
            high_conf_lags.append(abs_lag)
            if abs_lag > offset_threshold_ms:
                windows_exceed += 1

    if not high_conf_lags:
        return {
            "num_windows_total": float(windows_total),
            "num_windows_high_conf": float(windows_high),
            "median_abs_lag_ms": 0.0,
            "iqr_lag_ms": 0.0,
            "exceed_ratio": 0.0,
        }

    high_arr = np.array(high_conf_lags, dtype=np.float64)
    q25 = float(np.percentile(high_arr, 25))
    q75 = float(np.percentile(high_arr, 75))
    return {
        "num_windows_total": float(windows_total),
        "num_windows_high_conf": float(windows_high),
        "median_abs_lag_ms": float(np.median(high_arr)),
        "iqr_lag_ms": float(q75 - q25),
        "exceed_ratio": float(windows_exceed / max(1, windows_high)),
    }


def _pair_alignment_threshold_ms(
    category_a: str | None,
    category_b: str | None,
    base_threshold_ms: float,
    strict_alignment_mode: bool,
) -> float:
    if not strict_alignment_mode:
        return float(base_threshold_ms)
    rhythm = {"drums", "percussions"}
    cat_a = category_a or ""
    cat_b = category_b or ""
    if cat_a in rhythm and cat_b in rhythm:
        return float(min(base_threshold_ms, 8.0))
    if cat_a in rhythm or cat_b in rhythm:
        return float(min(max(base_threshold_ms, 10.0), 12.0))
    return float(max(base_threshold_ms * 2.0, 20.0))


def _choose_alignment_reference(
    labels: List[str],
    stem_categories: Dict[str, str],
    rms_db_by_label: Dict[str, float],
) -> str:
    if not labels:
        return ""
    rhythm = {"drums", "percussions"}
    rhythm_labels = [label for label in labels if stem_categories.get(label, "") in rhythm]
    if rhythm_labels:
        return max(rhythm_labels, key=lambda label: rms_db_by_label.get(label, -120.0))
    return max(labels, key=lambda label: rms_db_by_label.get(label, -120.0))


def _compute_abs_correlation(a: np.ndarray, b: np.ndarray) -> float:
    min_len = min(a.shape[0], b.shape[0])
    if min_len <= 1:
        return 0.0

    max_points = 20_000
    stride = max(1, min_len // max_points)
    x = a[:min_len:stride].astype(np.float64, copy=False)
    y = b[:min_len:stride].astype(np.float64, copy=False)
    if x.size <= 1 or y.size <= 1:
        return 0.0

    x -= x.mean()
    y -= y.mean()
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 1e-12:
        return 0.0
    corr = float(np.dot(x, y) / denom)
    return float(abs(np.clip(corr, -1.0, 1.0)))


def _check_track_quality(
    stem_paths: Iterable[Path],
    *,
    min_rms_db: float,
    max_alignment_offset_ms: float,
    max_stem_correlation: float,
    min_duration_sec: float,
    max_analysis_seconds: float,
    max_stems_for_pairwise: int,
    stem_categories: Dict[str, str] | None = None,
    strict_alignment_mode: bool = DEFAULT_STRICT_ALIGNMENT_MODE,
    strict_alignment_min_confidence: float = DEFAULT_STRICT_ALIGNMENT_MIN_CONFIDENCE,
    strict_alignment_window_seconds: float = DEFAULT_STRICT_ALIGNMENT_WINDOW_SECONDS,
    strict_alignment_window_hop_seconds: float = DEFAULT_STRICT_ALIGNMENT_WINDOW_HOP_SECONDS,
) -> Tuple[bool, List[str], Dict[str, object]]:
    paths = list(stem_paths)
    categories_lookup = dict(stem_categories or {})
    reasons: List[str] = []
    metrics: Dict[str, object] = {}
    if not paths:
        return False, ["no_stems_discovered"], metrics

    sample_rates: List[int] = []
    durations: List[float] = []
    num_samples: List[int] = []
    mono_cache: Dict[str, np.ndarray] = {}
    rms_db: Dict[str, float] = {}
    sr_lookup: Dict[str, int] = {}
    label_categories: Dict[str, str] = {}

    for path in paths:
        label = _path_label(path)
        try:
            info = sf.info(str(path))
            sr = int(info.samplerate)
            frames = int(info.frames)
            if sr <= 0 or frames <= 0:
                reasons.append(f"invalid_audio:{label}")
                continue
            sample_rates.append(sr)
            num_samples.append(frames)
            durations.append(float(frames) / float(sr))
            sr_lookup[label] = sr
            label_categories[label] = categories_lookup.get(str(path.resolve()), categories_lookup.get(label, ""))

            rms_value = _compute_rms_db(path)
            rms_db[label] = rms_value
            if rms_value < min_rms_db:
                reasons.append(f"low_volume:{label}:{rms_value:.1f}dB")

            excerpt, excerpt_sr = _load_mono_excerpt(path, max_analysis_seconds)
            mono_cache[label] = excerpt
            sr_lookup[label] = excerpt_sr
        except Exception as exc:  # pragma: no cover - defensive, data dependent
            reasons.append(f"load_error:{label}:{exc}")

    metrics["sample_rates"] = sample_rates
    metrics["durations"] = durations
    metrics["num_samples"] = num_samples
    metrics["rms_db"] = rms_db
    metrics["stem_categories"] = label_categories
    metrics["strict_alignment_mode"] = bool(strict_alignment_mode)

    if not sample_rates:
        return False, reasons or ["no_readable_stems"], metrics

    if len(set(sample_rates)) > 1:
        reasons.append(f"sample_rate_mismatch:{sample_rates}")

    max_samples = max(num_samples)
    min_samples = min(num_samples)
    if (max_samples - min_samples) > 1:
        reasons.append(f"duration_mismatch:{min_samples}..{max_samples}samples")

    min_duration = min(durations)
    if min_duration < min_duration_sec:
        reasons.append(f"too_short:{min_duration:.2f}s<{min_duration_sec:.2f}s")

    alignment_offsets: Dict[str, float] = {}
    alignment_confidences: Dict[str, float] = {}
    alignment_windows: Dict[str, Dict[str, float]] = {}
    alignment_details: Dict[str, Dict[str, float]] = {}
    alignment_flags: List[str] = []
    stem_correlations: Dict[str, float] = {}
    sorted_labels = sorted(mono_cache.keys())
    if max_stems_for_pairwise > 0:
        sorted_labels = sorted_labels[:max_stems_for_pairwise]
    metrics["stems_analyzed_for_pairwise"] = sorted_labels

    if len(sorted_labels) >= 2 and len(set(sample_rates)) == 1:
        reference_label = _choose_alignment_reference(sorted_labels, label_categories, rms_db)
        if not reference_label:
            reference_label = sorted_labels[0]
        reference = mono_cache[reference_label]
        reference_sr = sr_lookup.get(reference_label, sample_rates[0])
        metrics["alignment_reference"] = reference_label

        for label in sorted_labels:
            if label == reference_label:
                continue
            pair_threshold = _pair_alignment_threshold_ms(
                label_categories.get(reference_label, ""),
                label_categories.get(label, ""),
                max_alignment_offset_ms,
                strict_alignment_mode,
            )
            pair_key = f"{reference_label}_vs_{label}"
            if strict_alignment_mode:
                offset_ms, confidence, details = _estimate_offset_with_confidence(reference, mono_cache[label], reference_sr)
                window_stats = _windowed_alignment_stats(
                    reference,
                    mono_cache[label],
                    reference_sr,
                    window_seconds=strict_alignment_window_seconds,
                    hop_seconds=strict_alignment_window_hop_seconds,
                    confidence_threshold=strict_alignment_min_confidence,
                    offset_threshold_ms=pair_threshold,
                )
                alignment_confidences[pair_key] = float(confidence)
                alignment_windows[pair_key] = window_stats
                alignment_details[pair_key] = details
                high_conf_windows = int(round(window_stats["num_windows_high_conf"]))
                consistent_large_offset = (
                    high_conf_windows >= 3
                    and window_stats["median_abs_lag_ms"] > pair_threshold
                    and window_stats["exceed_ratio"] >= 0.6
                    and window_stats["iqr_lag_ms"] <= max(pair_threshold, 1.0)
                )
                if abs(offset_ms) > pair_threshold and confidence >= strict_alignment_min_confidence:
                    reasons.append(
                        f"misalignment:{pair_key}:{offset_ms:.1f}ms(thr={pair_threshold:.1f},conf={confidence:.2f})"
                    )
                elif consistent_large_offset:
                    reasons.append(
                        f"misalignment_windowed:{pair_key}:median={window_stats['median_abs_lag_ms']:.1f}ms"
                    )
                elif abs(offset_ms) > pair_threshold:
                    alignment_flags.append(
                        f"alignment_flag_low_conf:{pair_key}:{offset_ms:.1f}ms(conf={confidence:.2f})"
                    )
            else:
                offset_ms = _estimate_offset_ms(reference, mono_cache[label], reference_sr)
                if abs(offset_ms) > pair_threshold:
                    reasons.append(f"misalignment:{pair_key}:{offset_ms:.1f}ms(thr={pair_threshold:.1f})")
            alignment_offsets[pair_key] = offset_ms

        for idx, stem_a in enumerate(sorted_labels):
            for stem_b in sorted_labels[idx + 1 :]:
                corr = _compute_abs_correlation(mono_cache[stem_a], mono_cache[stem_b])
                pair_key = f"{stem_a}_vs_{stem_b}"
                stem_correlations[pair_key] = corr
                if corr > max_stem_correlation:
                    reasons.append(f"high_correlation:{pair_key}:{corr:.3f}")

    metrics["alignment_offsets_ms"] = alignment_offsets
    metrics["alignment_confidence"] = alignment_confidences
    metrics["alignment_window_stats"] = alignment_windows
    metrics["alignment_detail"] = alignment_details
    metrics["alignment_flags"] = alignment_flags
    metrics["stem_correlations"] = stem_correlations
    metrics["num_stems_checked"] = len(paths)
    metrics["analysis_seconds"] = float(max_analysis_seconds)

    passed = len(reasons) == 0
    return passed, reasons, metrics


def _build_song_entry(
    song_dir: Path,
    category_records: Dict[str, Dict[str, object]],
    fallback_candidates: List[Path],
    unmapped: List[str],
    residual_paths: List[str],
    temp_dir: Path,
    final_dir: Path,
) -> Dict[str, object]:
    category_audio_paths, other_audio_path, sample_rate = _render_category_bounces(
        song_dir.name,
        "train",
        category_records,
        residual_paths,
        temp_dir,
        final_dir,
    )

    duration_seconds = _compute_duration_seconds(fallback_candidates)
    artist, song_name = _split_artist_song(song_dir.name)

    entry: Dict[str, object] = {
        "song_id": song_dir.name,
        "artist": artist,
        "song_name": song_name,
        "split": "train",
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "unmapped_items": unmapped,
    }

    for category, record in category_records.items():
        entry[f"{category}_stems"] = record["paths"]
        entry[f"{category}_is_present"] = record["is_present"]
        entry[f"{category}_num_stems"] = record["num_stems"]
        entry[f"{category}_audio"] = category_audio_paths[category]

    present_categories = sorted(cat for cat, record in category_records.items() if record["is_present"])
    entry["categories_present"] = present_categories
    entry["other_audio"] = other_audio_path
    return entry


def _process_song(
    song_dir: Path,
    dataset_root: Path,
    temp_dir: Path,
    final_dir: Path,
    *,
    apply_quality_filter: bool,
    min_rms_db: float,
    max_alignment_offset_ms: float,
    max_stem_correlation: float,
    min_duration_sec: float,
    max_analysis_seconds: float,
    max_stems_for_pairwise: int,
    strict_alignment_mode: bool,
    strict_alignment_min_confidence: float,
    strict_alignment_window_seconds: float,
    strict_alignment_window_hop_seconds: float,
) -> Dict[str, object]:
    category_records, unmapped, fallback_candidates, residual_paths = _collect_category_records(song_dir)
    stems_by_category = {
        category: list(record["paths"])
        for category, record in category_records.items()
        if record["paths"]
    }
    stem_categories: Dict[str, str] = {}
    for category, record in category_records.items():
        for path_str in record["paths"]:
            resolved = str(Path(path_str).resolve())
            stem_categories[resolved] = category
            stem_categories[_path_label(Path(path_str))] = category

    report: Dict[str, object] = {
        "track_id": song_dir.relative_to(dataset_root).as_posix(),
        "track_dir": str(song_dir.resolve()),
        "stems": stems_by_category,
        "passed": False,
        "reasons": [],
        "metrics": {},
    }

    if not fallback_candidates:
        report["reasons"] = ["no_stems_discovered"]
        return {"entry": None, "report": report}

    passed = True
    reasons: List[str] = []
    metrics: Dict[str, object] = {}

    if apply_quality_filter:
        passed, reasons, metrics = _check_track_quality(
            fallback_candidates,
            min_rms_db=min_rms_db,
            max_alignment_offset_ms=max_alignment_offset_ms,
            max_stem_correlation=max_stem_correlation,
            min_duration_sec=min_duration_sec,
            max_analysis_seconds=max_analysis_seconds,
            max_stems_for_pairwise=max_stems_for_pairwise,
            stem_categories=stem_categories,
            strict_alignment_mode=strict_alignment_mode,
            strict_alignment_min_confidence=strict_alignment_min_confidence,
            strict_alignment_window_seconds=strict_alignment_window_seconds,
            strict_alignment_window_hop_seconds=strict_alignment_window_hop_seconds,
        )

    report["passed"] = passed
    report["reasons"] = reasons
    report["metrics"] = metrics

    if not passed:
        return {"entry": None, "report": report}

    try:
        entry = _build_song_entry(
            song_dir=song_dir,
            category_records=category_records,
            fallback_candidates=fallback_candidates,
            unmapped=unmapped,
            residual_paths=residual_paths,
            temp_dir=temp_dir,
            final_dir=final_dir,
        )
    except Exception as exc:  # pragma: no cover - defensive, data dependent
        report["passed"] = False
        report["reasons"] = [f"entry_build_error:{exc}"]
        return {"entry": None, "report": report}

    return {"entry": entry, "report": report}


def _process_song_worker(
    args: Tuple[str, str, str, str, bool, float, float, float, float, float, int, bool, float, float, float]
) -> Dict[str, object]:
    (
        song_dir_str,
        dataset_root_str,
        temp_dir_str,
        final_dir_str,
        apply_quality_filter,
        min_rms_db,
        max_alignment_offset_ms,
        max_stem_correlation,
        min_duration_sec,
        max_analysis_seconds,
        max_stems_for_pairwise,
        strict_alignment_mode,
        strict_alignment_min_confidence,
        strict_alignment_window_seconds,
        strict_alignment_window_hop_seconds,
    ) = args
    return _process_song(
        song_dir=Path(song_dir_str),
        dataset_root=Path(dataset_root_str),
        temp_dir=Path(temp_dir_str),
        final_dir=Path(final_dir_str),
        apply_quality_filter=apply_quality_filter,
        min_rms_db=min_rms_db,
        max_alignment_offset_ms=max_alignment_offset_ms,
        max_stem_correlation=max_stem_correlation,
        min_duration_sec=min_duration_sec,
        max_analysis_seconds=max_analysis_seconds,
        max_stems_for_pairwise=max_stems_for_pairwise,
        strict_alignment_mode=strict_alignment_mode,
        strict_alignment_min_confidence=strict_alignment_min_confidence,
        strict_alignment_window_seconds=strict_alignment_window_seconds,
        strict_alignment_window_hop_seconds=strict_alignment_window_hop_seconds,
    )


def _write_quality_reports(reports: List[Dict[str, object]], report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_all = report_dir / "manifest_all.jsonl"
    manifest_clean = report_dir / "manifest_clean.jsonl"
    manifest_reject = report_dir / "manifest_reject.jsonl"
    summary_csv = report_dir / "summary.csv"

    clean_reports = [report for report in reports if bool(report.get("passed", False))]
    reject_reports = [report for report in reports if not bool(report.get("passed", False))]

    with manifest_all.open("w", encoding="utf-8") as handle:
        for report in reports:
            handle.write(json.dumps(report, ensure_ascii=True) + "\n")

    with manifest_clean.open("w", encoding="utf-8") as handle:
        for report in clean_reports:
            handle.write(json.dumps(report, ensure_ascii=True) + "\n")

    with manifest_reject.open("w", encoding="utf-8") as handle:
        for report in reject_reports:
            handle.write(json.dumps(report, ensure_ascii=True) + "\n")

    with summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["track_id", "passed", "num_reasons", "reasons", "track_dir"])
        for report in reports:
            reasons = report.get("reasons", [])
            writer.writerow(
                [
                    report.get("track_id", ""),
                    bool(report.get("passed", False)),
                    len(reasons),
                    "; ".join(str(reason) for reason in reasons),
                    report.get("track_dir", ""),
                ]
            )


def _collect_entries(
    dataset_root: Path,
    temp_dir: Path,
    final_dir: Path,
    num_workers: int,
    apply_quality_filter: bool,
    min_rms_db: float,
    max_alignment_offset_ms: float,
    max_stem_correlation: float,
    min_duration_sec: float,
    max_analysis_seconds: float,
    max_stems_for_pairwise: int,
    strict_alignment_mode: bool,
    strict_alignment_min_confidence: float,
    strict_alignment_window_seconds: float,
    strict_alignment_window_hop_seconds: float,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    entries: List[Dict[str, object]] = []
    reports: List[Dict[str, object]] = []
    song_dirs = list(_iter_song_dirs(dataset_root))
    progress = None
    worker_count = max(1, num_workers)

    if worker_count == 1:
        if tqdm is not None:
            progress = tqdm(song_dirs, desc="Processing Raw Stems songs", unit="song")
            iterator = progress
        else:
            iterator = song_dirs

        for song_dir in iterator:
            result = _process_song(
                song_dir=song_dir,
                dataset_root=dataset_root,
                temp_dir=temp_dir,
                final_dir=final_dir,
                apply_quality_filter=apply_quality_filter,
                min_rms_db=min_rms_db,
                max_alignment_offset_ms=max_alignment_offset_ms,
                max_stem_correlation=max_stem_correlation,
                min_duration_sec=min_duration_sec,
                max_analysis_seconds=max_analysis_seconds,
                max_stems_for_pairwise=max_stems_for_pairwise,
                strict_alignment_mode=strict_alignment_mode,
                strict_alignment_min_confidence=strict_alignment_min_confidence,
                strict_alignment_window_seconds=strict_alignment_window_seconds,
                strict_alignment_window_hop_seconds=strict_alignment_window_hop_seconds,
            )
            reports.append(result["report"])
            if result["entry"] is not None:
                entries.append(result["entry"])

        if progress is not None:
            progress.close()
    else:
        total = len(song_dirs)
        if tqdm is not None:
            progress = tqdm(total=total, desc="Processing Raw Stems songs", unit="song")

        args_iter = (
            (
                str(song_dir),
                str(dataset_root),
                str(temp_dir),
                str(final_dir),
                apply_quality_filter,
                min_rms_db,
                max_alignment_offset_ms,
                max_stem_correlation,
                min_duration_sec,
                max_analysis_seconds,
                max_stems_for_pairwise,
                strict_alignment_mode,
                strict_alignment_min_confidence,
                strict_alignment_window_seconds,
                strict_alignment_window_hop_seconds,
            )
            for song_dir in song_dirs
        )
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for result in executor.map(_process_song_worker, args_iter):
                reports.append(result["report"])
                if result["entry"] is not None:
                    entries.append(result["entry"])
                if progress is not None:
                    progress.update(1)

        if progress is not None:
            progress.close()

    if not entries:
        raise ValueError(f"No valid songs discovered under {dataset_root}")
    return entries, reports


def _compute_safe_save_num_proc(dataset_dict: DatasetDict, requested_num_proc: int) -> int:
    requested = max(1, int(requested_num_proc))
    split_sizes = [len(split_dataset) for split_dataset in dataset_dict.values()]
    non_empty_split_sizes = [size for size in split_sizes if size > 0]
    if not non_empty_split_sizes:
        return 1
    return max(1, min(requested, min(non_empty_split_sizes)))


def create_dataset(
    dataset_root: Path,
    save_dir: Path,
    overwrite: bool = False,
    num_workers: int = 0,
    num_proc: int = 1,
    apply_quality_filter: bool = False,
    min_rms_db: float = DEFAULT_MIN_RMS_DB,
    max_alignment_offset_ms: float = DEFAULT_MAX_ALIGNMENT_OFFSET_MS,
    max_stem_correlation: float = DEFAULT_MAX_STEM_CORRELATION,
    min_duration_sec: float = DEFAULT_MIN_DURATION_SEC,
    max_analysis_seconds: float = DEFAULT_MAX_ANALYSIS_SECONDS,
    max_stems_for_pairwise: int = DEFAULT_MAX_STEMS_FOR_PAIRWISE,
    strict_alignment_mode: bool = DEFAULT_STRICT_ALIGNMENT_MODE,
    strict_alignment_min_confidence: float = DEFAULT_STRICT_ALIGNMENT_MIN_CONFIDENCE,
    strict_alignment_window_seconds: float = DEFAULT_STRICT_ALIGNMENT_WINDOW_SECONDS,
    strict_alignment_window_hop_seconds: float = DEFAULT_STRICT_ALIGNMENT_WINDOW_HOP_SECONDS,
) -> DatasetDict:
    dataset_root = _validate_directory(dataset_root.resolve(), "Raw Stems dataset root")
    save_dir = save_dir.resolve()

    if save_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target directory already exists: {save_dir}. Use --overwrite to replace it."
            )
        rmtree(save_dir)

    features = _build_features()
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_wav_root = Path(tempfile.mkdtemp(prefix="rawstems_wavs_", dir=save_dir.parent))
    final_wav_root = save_dir / ADDITIONAL_WAVS_DIRNAME

    dataset_dict: DatasetDict | None = None
    worker_count = max(1, num_workers or (os.cpu_count() or 1))
    max_stems_for_pairwise = max(1, int(max_stems_for_pairwise))
    strict_alignment_min_confidence = float(np.clip(strict_alignment_min_confidence, 0.0, 1.0))
    strict_alignment_window_seconds = max(0.5, float(strict_alignment_window_seconds))
    strict_alignment_window_hop_seconds = max(0.1, float(strict_alignment_window_hop_seconds))

    try:
        entries, reports = _collect_entries(
            dataset_root=dataset_root,
            temp_dir=temp_wav_root,
            final_dir=final_wav_root,
            num_workers=worker_count,
            apply_quality_filter=apply_quality_filter,
            min_rms_db=min_rms_db,
            max_alignment_offset_ms=max_alignment_offset_ms,
            max_stem_correlation=max_stem_correlation,
            min_duration_sec=min_duration_sec,
            max_analysis_seconds=max_analysis_seconds,
            max_stems_for_pairwise=max_stems_for_pairwise,
            strict_alignment_mode=strict_alignment_mode,
            strict_alignment_min_confidence=strict_alignment_min_confidence,
            strict_alignment_window_seconds=strict_alignment_window_seconds,
            strict_alignment_window_hop_seconds=strict_alignment_window_hop_seconds,
        )

        dataset = Dataset.from_list(entries, features=features)
        dataset_dict = DatasetDict({"train": dataset})
        save_num_proc = _compute_safe_save_num_proc(dataset_dict, num_proc)
        if save_num_proc < max(1, int(num_proc)):
            print(
                "Capping --num-proc from "
                f"{int(num_proc)} to {save_num_proc} for save_to_disk "
                f"(smallest non-empty split has {save_num_proc} examples)."
            )

        try:
            dataset_dict.save_to_disk(str(save_dir), num_proc=save_num_proc)
        except (PermissionError, IndexError):
            if save_num_proc == 1:
                raise
            if save_dir.exists():
                rmtree(save_dir)
            dataset_dict.save_to_disk(str(save_dir), num_proc=1)

        final_wav_root.parent.mkdir(parents=True, exist_ok=True)
        if final_wav_root.exists():
            rmtree(final_wav_root)
        move(str(temp_wav_root), str(final_wav_root))

        _write_quality_reports(reports, save_dir / QUALITY_REPORTS_DIRNAME)
    finally:
        if temp_wav_root.exists():
            rmtree(temp_wav_root)

    if dataset_dict is None:
        raise RuntimeError("Failed to assemble Raw Stems dataset.")
    return dataset_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face dataset for Raw Stems.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the extracted Raw Stems dataset root (one folder per song)",
    )
    parser.add_argument(
        "--save-dir",
        required=True,
        help="Directory where the Hugging Face dataset should be saved",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the target save directory if it already exists.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Worker processes for per-song processing (0 uses all available cores).",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=1,
        help="Number of processes to use when saving the dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--quality-filter",
        action="store_true",
        help="Enable quality filtering (reject tracks with sample-rate/duration mismatch, low volume, "
        "misalignment or leakage, see the --min-*/--max-* options). Disabled by default: the mss8s "
        "dataset used in our experiments was built without it.",
    )
    parser.add_argument(
        "--min-rms-db",
        type=float,
        default=DEFAULT_MIN_RMS_DB,
        help="Minimum stem RMS level in dB (default: %(default)s).",
    )
    parser.add_argument(
        "--max-alignment-offset-ms",
        type=float,
        default=DEFAULT_MAX_ALIGNMENT_OFFSET_MS,
        help="Maximum alignment offset in milliseconds (default: %(default)s).",
    )
    parser.add_argument(
        "--max-stem-correlation",
        type=float,
        default=DEFAULT_MAX_STEM_CORRELATION,
        help="Maximum absolute correlation between stems (default: %(default)s).",
    )
    parser.add_argument(
        "--min-duration-sec",
        type=float,
        default=DEFAULT_MIN_DURATION_SEC,
        help="Minimum track duration in seconds (default: %(default)s).",
    )
    parser.add_argument(
        "--max-analysis-seconds",
        type=float,
        default=DEFAULT_MAX_ANALYSIS_SECONDS,
        help="Max seconds per stem for alignment/correlation checks (default: %(default)s).",
    )
    parser.add_argument(
        "--max-stems-for-pairwise",
        type=int,
        default=DEFAULT_MAX_STEMS_FOR_PAIRWISE,
        help="Max stems per track used for pairwise checks (default: %(default)s).",
    )
    parser.add_argument(
        "--strict-alignment-mode",
        action="store_true",
        help="Enable robust alignment mode (confidence gating + windowed checks + category-aware thresholds).",
    )
    parser.add_argument(
        "--strict-alignment-min-confidence",
        type=float,
        default=DEFAULT_STRICT_ALIGNMENT_MIN_CONFIDENCE,
        help="Minimum confidence for hard misalignment rejection in strict mode (default: %(default)s).",
    )
    parser.add_argument(
        "--strict-alignment-window-seconds",
        type=float,
        default=DEFAULT_STRICT_ALIGNMENT_WINDOW_SECONDS,
        help="Window length in seconds for strict alignment stability checks (default: %(default)s).",
    )
    parser.add_argument(
        "--strict-alignment-window-hop-seconds",
        type=float,
        default=DEFAULT_STRICT_ALIGNMENT_WINDOW_HOP_SECONDS,
        help="Hop length in seconds for strict alignment stability checks (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    save_dir = Path(args.save_dir)

    dataset_dict = create_dataset(
        dataset_root=dataset_root,
        save_dir=save_dir,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        num_proc=args.num_proc,
        apply_quality_filter=args.quality_filter,
        min_rms_db=args.min_rms_db,
        max_alignment_offset_ms=args.max_alignment_offset_ms,
        max_stem_correlation=args.max_stem_correlation,
        min_duration_sec=args.min_duration_sec,
        max_analysis_seconds=args.max_analysis_seconds,
        max_stems_for_pairwise=args.max_stems_for_pairwise,
        strict_alignment_mode=args.strict_alignment_mode,
        strict_alignment_min_confidence=args.strict_alignment_min_confidence,
        strict_alignment_window_seconds=args.strict_alignment_window_seconds,
        strict_alignment_window_hop_seconds=args.strict_alignment_window_hop_seconds,
    )
    print(f"Raw Stems dataset prepared with split: {', '.join(dataset_dict)}")
    print(f"Saved to: {save_dir}")
    print(f"Quality reports: {save_dir / QUALITY_REPORTS_DIRNAME}")


if __name__ == "__main__":
    main()
