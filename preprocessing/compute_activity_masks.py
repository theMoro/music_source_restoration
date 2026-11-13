#!/usr/bin/env python3

"""
Compute the activity-mask metadata of the mss8s dataset (see build_mss8s.py).

For every song and stem, the RMS (in dB) is computed per second (max over the stereo
channels); a second is marked active if it is above RMS_THRESHOLD. Stems without any active
second get no mask. The "other" stem is deliberately skipped: it is never used to build
training mixtures (see src/data/dataset_8s.py).

Two files are written to `<dataset_root>_metadata/` (the folder expected by
src/data/dataset_8s.py):
  * activity_masks_v1.jsonl: one line per song with {"original_idx", "masks": {stem: {...}}}
  * stem_mapping_v1.jsonl: {stem: [original_idx of every song in which the stem is active]}
    (used to draw stems from other songs for random mixing)

Usage:
    python preprocessing/compute_activity_masks.py --dataset-root <hf_root>/mss8s --num-workers 16
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import numpy as np
from tqdm import tqdm

# Allow `python preprocessing/compute_activity_masks.py` from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset_8s import load_mss8s_split  # noqa: E402
from src.data.helpers import get_absolute_path  # noqa: E402

COLUMNS = ["vocals", "bass", "drums", "guitars", "keyboards", "synthesizers", "percussions", "orchestral"]

SAMPLING_RATE = 48000
RMS_THRESHOLD = -40.0  # RMS threshold in dB
WINDOW_SIZE = 1.0  # Window size in seconds for activity detection
STEREO_RMS_METHOD = 'max'  # How to combine the RMS of the stereo channels: 'mean', 'max' or 'sum'


def compute_rms_per_second(audio_path: str, sample_rate: int = 48000, stereo_method: str = 'max') -> np.ndarray:
    """Compute the RMS in dB per second of an audio file (mono or stereo)."""
    try:
        audio, sr = librosa.load(audio_path, sr=sample_rate, mono=False)

        if len(audio) == 0:
            return np.array([])

        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # (1, samples)

        num_channels, num_samples = audio.shape
        samples_per_second = sample_rate
        num_seconds = int(np.ceil(num_samples / samples_per_second))

        rms_per_second = []
        for i in range(num_seconds):
            start_idx = i * samples_per_second
            end_idx = min((i + 1) * samples_per_second, num_samples)

            segment = audio[:, start_idx:end_idx]

            if segment.shape[1] == 0:
                rms_per_second.append(-np.inf)
                continue

            rms_per_channel = np.sqrt(np.mean(segment ** 2, axis=1))

            if stereo_method == 'mean':
                rms = np.mean(rms_per_channel)
            elif stereo_method == 'max':
                rms = np.max(rms_per_channel)
            elif stereo_method == 'sum':
                rms = np.sum(rms_per_channel)
            else:
                raise ValueError(f"Unknown stereo_method: {stereo_method}. Use 'mean', 'max', or 'sum'")

            rms_db = 20 * np.log10(rms + 1e-10)
            rms_per_second.append(rms_db)

        return np.array(rms_per_second)
    except Exception as e:
        print(f"Error loading {audio_path}: {e}")
        return np.array([])


def compute_activity_mask(rms_values: np.ndarray, window_size: float, rms_threshold: float) -> np.ndarray:
    """Boolean activity mask per second (at least 80% of the window must be above rms_threshold)."""
    if len(rms_values) == 0:
        return np.array([], dtype=bool)

    window_size_int = int(np.ceil(window_size))

    if len(rms_values) < window_size_int:
        return np.zeros(len(rms_values), dtype=bool)

    is_loud = rms_values > rms_threshold

    sum_loud = np.convolve(is_loud.astype(float), np.ones(window_size_int), 'valid')
    avg_loud_enough = sum_loud / window_size_int > 0.8

    mask = np.zeros(len(rms_values), dtype=bool)
    mask[:len(avg_loud_enough)] = avg_loud_enough

    return mask


def process_row(args) -> Dict[str, Any]:
    row, columns, prefix_path = args
    original_idx = int(row['original_idx'])
    row_activity_masks: Dict[str, Any] = {
        'dataset_index': original_idx,
        'original_idx': original_idx,
        'masks': {},
    }

    for column in columns:
        audio_path = row.get(column)
        if isinstance(audio_path, dict):
            audio_path = audio_path.get('path', '')
        if audio_path is None or not str(audio_path).strip():
            row_activity_masks['masks'][column] = None
            continue

        relative_path = str(audio_path)
        absolute_path = get_absolute_path(relative_path, prefix_path)

        if not Path(absolute_path).exists():
            print(f"Warning: File not found: {absolute_path}")
            row_activity_masks['masks'][column] = None
            continue

        try:
            rms_per_second = compute_rms_per_second(str(absolute_path), SAMPLING_RATE, stereo_method=STEREO_RMS_METHOD)
            if len(rms_per_second) == 0:
                row_activity_masks['masks'][column] = None
                continue

            activity_mask = compute_activity_mask(rms_per_second, WINDOW_SIZE, RMS_THRESHOLD)

            # No active seconds: store None so downstream treats the stem as absent
            if activity_mask.sum() == 0:
                row_activity_masks['masks'][column] = None
                continue

            row_activity_masks['masks'][column] = {
                'rms_db_per_second': rms_per_second.tolist(),
                'activity_mask': activity_mask.tolist(),
                'filepath': relative_path,
                'stereo_method': STEREO_RMS_METHOD,
            }
        except Exception as e:
            print(f"Error processing {column} for original_idx {original_idx}: {e}")
            row_activity_masks['masks'][column] = None

    return row_activity_masks


def build_stem_to_indices(activity_masks: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """stem -> sorted list of original_idx of the songs in which the stem is active."""
    stem_to_indices = defaultdict(list)
    for entry in activity_masks:
        for stem_name, stem_data in entry['masks'].items():
            if stem_data is not None and stem_data.get('activity_mask') is not None and sum(stem_data['activity_mask']) > 0:
                stem_to_indices[stem_name].append(entry['original_idx'])
    return {stem: sorted(indices) for stem, indices in stem_to_indices.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute activity masks and the stem mapping of the mss8s dataset.")
    parser.add_argument("--dataset-root", required=True, help="Path to the mss8s dataset (output of build_mss8s.py).")
    parser.add_argument("--prefix-path", default=None,
                        help="Prefix for relative audio paths in the dataset (not needed for absolute paths).")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel worker processes.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    prefix_path: Optional[Path] = Path(args.prefix_path) if args.prefix_path else None
    output_dir = Path(str(dataset_root.expanduser().resolve()) + "_metadata")
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_mss8s_split(args.split, root=dataset_root)
    columns = [col for col in COLUMNS if col in dataset.column_names]
    print(f"Loaded {len(dataset)} songs from {dataset_root}, processing columns: {columns}")
    print(f"Parameters: RMS threshold={RMS_THRESHOLD} dB, window size={WINDOW_SIZE} s, stereo method={STEREO_RMS_METHOD}")

    jobs = ((dataset[idx], columns, prefix_path) for idx in range(len(dataset)))
    if args.num_workers > 1:
        with Pool(args.num_workers) as pool:
            activity_masks = list(tqdm(pool.imap(process_row, jobs, chunksize=4), total=len(dataset)))
    else:
        activity_masks = [process_row(job) for job in tqdm(jobs, total=len(dataset))]

    masks_path = output_dir / "activity_masks_v1.jsonl"
    with open(masks_path, 'w', encoding='utf-8') as f:
        for entry in activity_masks:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    mapping = build_stem_to_indices(activity_masks)
    mapping_path = output_dir / "stem_mapping_v1.jsonl"
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"Saved activity masks of {len(activity_masks)} songs to {masks_path}")
    print(f"Saved stem mapping to {mapping_path}")
    for column in columns:
        print(f"  {column:15s} active in {len(mapping.get(column, [])):6d} songs")


if __name__ == "__main__":
    main()
