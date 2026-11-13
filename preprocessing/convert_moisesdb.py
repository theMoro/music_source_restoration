#!/usr/bin/env python3

"""
Utilities to convert the MoisesDB dataset into a Hugging Face dataset.

Each song in the MoisesDB v0.1 release lives under a UUID-named directory. The
directory contains a `data.json` metadata file plus one subdirectory per stem,
where each stem directory stores one or more mono WAV tracks. Stem directory
names (e.g. `vocals`, `drums`, `other_keys`) capture high-level instrument
families that we collapse into the same canonical categories used for Slakh2100:

    vocals, guitars, bass, keyboards, synthesizers, drums, percussions, orchestral

This script walks the dataset tree, maps every stem directory to a canonical
category, renders one summed WAV file per category (`<save_dir>/additional_wavs/`)
and stores the paths plus simple metadata (counts and presence flags) in a Hugging
Face dataset. Stems that cannot be mapped to any category are summed into the
residual `other` file (empty string if there are none). Each row also stores a
`duration_seconds` value derived from the first readable stem.

Usage:
    python preprocessing/convert_moisesdb.py --dataset-root <moisesdb>/moisesdb_v0.1 --save-dir <hf_root>/moisesdb
"""

from __future__ import annotations

import argparse
import json
import tempfile
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

# Mapping from MoisesDB stem directory names to canonical categories.
STEM_CATEGORY_MAP: Dict[str, str] = {
    "vocals": "vocals",
    "bass": "bass",
    "guitar": "guitars",
    "other_plucked": "guitars",
    "piano": "keyboards",
    "other_keys": "synthesizers",
    "drums": "drums",
    "percussion": "percussions",
    "bowed_strings": "orchestral",
    "wind": "orchestral",
    "other": "synthesizers",
}


def _validate_directory(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")
    return path


def _load_metadata(metadata_path: Path) -> Dict[str, object]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


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
            sf.write(temp_path, summed, sample_rate)
            category_audio_paths[category] = str(final_path.resolve())
        else:
            category_audio_paths[category] = ""

    # Residual "other": sum of all stems that could not be mapped to one of the categories.
    other_audio_path = ""
    if residual_paths:
        other_audio = _sum_cached(residual_paths, cache, ref_len, ref_channels)
        filename = f"{split_name}_{song_id}_other.wav"
        temp_other = temp_dir / filename
        final_other = final_dir / filename
        temp_other.parent.mkdir(parents=True, exist_ok=True)
        sf.write(temp_other, other_audio, sample_rate)
        other_audio_path = str(final_other.resolve())

    return category_audio_paths, other_audio_path, sample_rate


def _iter_song_dirs(dataset_root: Path) -> Iterable[Path]:
    for song_dir in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        yield song_dir


def _collect_song_entry(song_dir: Path, temp_dir: Path, final_dir: Path) -> Dict[str, object]:
    metadata_path = song_dir / "data.json"
    metadata = _load_metadata(metadata_path)

    category_records = _initialise_category_records()
    unmapped: List[str] = []
    residual_paths: List[str] = []
    fallback_candidates: List[Path] = []

    for stem in metadata.get("stems", []):
        stem_name = (stem.get("stemName") or "").strip().lower()
        if not stem_name:
            unmapped.append(f"{song_dir.name}:<missing stemName>")
            continue

        category = STEM_CATEGORY_MAP.get(stem_name)
        if category is None:
            unmapped.append(f"{song_dir.name}:{stem_name}")
            # Keep the audio of unmapped stems for the residual "other" file.
            for track in stem.get("tracks", []):
                track_path = song_dir / stem_name / f"{track.get('id')}.{track.get('extension', 'wav')}"
                if track.get("id") and track_path.exists():
                    residual_paths.append(str(track_path.resolve()))
            continue

        stem_dir = song_dir / stem_name
        if not stem_dir.exists():
            unmapped.append(f"{song_dir.name}:{stem_name} (missing directory)")
            continue

        record = category_records[category]
        tracks = stem.get("tracks", [])
        if not tracks:
            unmapped.append(f"{song_dir.name}:{stem_name} (no tracks)")
            continue

        for track in tracks:
            track_id = track.get("id")
            extension = track.get("extension", "wav")
            if not track_id:
                unmapped.append(f"{song_dir.name}:{stem_name} (missing track id)")
                continue

            track_path = stem_dir / f"{track_id}.{extension}"
            if not track_path.exists():
                unmapped.append(f"{song_dir.name}:{stem_name}:{track_id} (missing file)")
                continue

            resolved_path = track_path.resolve()
            record["paths"].append(str(resolved_path))
            record["num_stems"] += 1
            record["is_present"] = True
            fallback_candidates.append(resolved_path)

    duration_seconds = _compute_duration_seconds(None, fallback_candidates)
    category_audio_paths, other_audio_path, sample_rate = _render_category_bounces(
        song_dir.name,
        "train",
        category_records,
        residual_paths,
        temp_dir,
        final_dir,
    )

    entry: Dict[str, object] = {
        "song_id": song_dir.name,
        "song_name": str(metadata.get("song") or ""),
        "artist": str(metadata.get("artist") or ""),
        "genre": str(metadata.get("genre") or ""),
        "metadata_path": str(metadata_path.resolve()),
        "split": "train",
        "duration_seconds": duration_seconds,
        "unmapped_stems": unmapped,
        "sample_rate": sample_rate,
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


def _collect_entries(dataset_root: Path, temp_dir: Path, final_dir: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    song_dirs = list(_iter_song_dirs(dataset_root))
    progress = None
    if tqdm is not None:
        progress = tqdm(song_dirs, desc="Processing MoisesDB songs", unit="song")
        iterator = progress
    else:
        iterator = song_dirs

    for song_dir in iterator:
        entry = _collect_song_entry(song_dir, temp_dir, final_dir)
        entries.append(entry)

    if progress is not None:
        progress.close()
    if not entries:
        raise ValueError(f"No songs discovered under {dataset_root}")
    return entries


def _build_features() -> Features:
    feature_dict = {
        "song_id": Value("string"),
        "song_name": Value("string"),
        "artist": Value("string"),
        "genre": Value("string"),
        "metadata_path": Value("string"),
        "split": Value("string"),
        "duration_seconds": Value("float32"),
        "sample_rate": Value("int32"),
        "unmapped_stems": Sequence(Value("string")),
        "categories_present": Sequence(Value("string")),
    }

    for category in TARGET_CATEGORIES:
        feature_dict[f"{category}_stems"] = Sequence(Value("string"))
        feature_dict[f"{category}_is_present"] = Value("bool")
        feature_dict[f"{category}_num_stems"] = Value("int32")
        feature_dict[f"{category}_audio"] = Value("string")

    feature_dict["other_audio"] = Value("string")

    return Features(feature_dict)


def _probe_duration_seconds(path: Path) -> float:
    """Return the duration of an audio file in seconds."""
    with sf.SoundFile(str(path)) as audio_file:
        samplerate = float(audio_file.samplerate)
        if samplerate <= 0:
            raise ValueError(f"Invalid samplerate for {path}")
        return float(audio_file.frames) / samplerate


def _compute_duration_seconds(primary: Path | None, fallbacks: Iterable[Path]) -> float:
    """Use the mixture when provided; otherwise fall back to stem audio."""
    candidates: List[Path] = []
    if primary is not None:
        candidates.append(primary)
    candidates.extend(fallbacks)
    for candidate in candidates:
        if candidate is None or not candidate.exists():
            continue
        try:
            return _probe_duration_seconds(candidate)
        except (RuntimeError, OSError, ValueError):
            continue
    candidate_list = ", ".join(str(path) for path in candidates if path is not None)
    raise ValueError(f"Unable to compute duration; no readable audio among: {candidate_list}")


def create_dataset(
    dataset_root: Path,
    save_dir: Path,
    overwrite: bool = False,
    num_proc: int = 1,
) -> DatasetDict:
    dataset_root = _validate_directory(dataset_root.resolve(), "MoisesDB dataset root")
    save_dir = save_dir.resolve()

    if save_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target directory already exists: {save_dir}. Use --overwrite to replace it."
            )
        rmtree(save_dir)

    features = _build_features()
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_wav_root = Path(tempfile.mkdtemp(prefix="moises_wavs_", dir=save_dir.parent))
    final_wav_root = save_dir / ADDITIONAL_WAVS_DIRNAME

    dataset_dict: DatasetDict | None = None

    try:
        entries = _collect_entries(dataset_root, temp_wav_root, final_wav_root)
        dataset = Dataset.from_list(entries, features=features)
        dataset_dict = DatasetDict({"train": dataset})

        try:
            dataset_dict.save_to_disk(str(save_dir), num_proc=num_proc)
        except PermissionError:
            if num_proc == 1:
                raise
            if save_dir.exists():
                rmtree(save_dir)
            dataset_dict.save_to_disk(str(save_dir), num_proc=1)

        final_wav_root.parent.mkdir(parents=True, exist_ok=True)
        if final_wav_root.exists():
            rmtree(final_wav_root)
        move(str(temp_wav_root), str(final_wav_root))
    finally:
        if temp_wav_root.exists():
            rmtree(temp_wav_root)

    if dataset_dict is None:
        raise RuntimeError("Failed to assemble MoisesDB dataset.")
    return dataset_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face dataset for MoisesDB.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the MoisesDB dataset root (the folder containing one sub-folder per song, e.g. moisesdb_v0.1)",
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
        "--num-proc",
        type=int,
        default=1,
        help="Number of processes to use when saving the dataset (default: %(default)s).",
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
        num_proc=args.num_proc,
    )
    print(f"MoisesDB dataset prepared with split: {', '.join(dataset_dict)}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
