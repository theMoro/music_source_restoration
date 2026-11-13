#!/usr/bin/env python3

"""
Utilities to convert the Slakh2100 (FLAC redux) dataset into a Hugging Face dataset.

The original Slakh2100 dataset stores one directory per track, containing a
`mix.flac` file along with individual stem files (`stems/SXX.flac`) described in
`metadata.yaml`. Each stem has an `inst_class` field that we map to a reduced set
of eight high-level instrument families:

    vocals, guitars, bass, keyboards, synthesizers, drums, percussions, orchestral

This script organises the stems under those categories, renders one summed WAV
file per category at 48 kHz (`<save_dir>/additional_wavs/<split>/`) and stores the
paths plus simple metadata (counts and presence flags) in a Hugging Face dataset.
Stems whose `inst_class` cannot be mapped are summed into the residual `other` file
(empty string if there are none). Each entry also carries a `duration_seconds` value
derived from the mixture.

Note: the dataset does not contain vocal stems; the `vocals_*` columns are empty.

The dataset root must contain the `train`, `validation` and `test` folders of the
Slakh2100 FLAC redux release. A root without these folders (e.g. BabySlakh) is
treated as a single `train` split. Stems/mixtures can be FLAC or WAV.

Usage:
    python preprocessing/convert_slakh2100.py --dataset-root <slakh2100_flac_redux> --save-dir <hf_root>/slakh2100
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, cast

import numpy as np
import soundfile as sf
import yaml
from tqdm import tqdm
from datasets import Dataset, DatasetDict, Features, Sequence, Value
from shutil import move, rmtree
import tempfile

try:  # pragma: no cover - optional dependency
    import librosa
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    librosa = None


ADDITIONAL_WAVS_DIRNAME = "additional_wavs"
SAMPLE_RATE = 48_000
AUDIO_EXTENSIONS = (".flac", ".wav")

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

# Mapping from `inst_class` (lowercase) to one of the TARGET_CATEGORIES.
INSTRUMENT_CATEGORY_MAP: Dict[str, str] = {
    "bass": "bass",
    "guitar": "guitars",
    "drums": "drums",
    "percussive": "percussions",
    "chromatic percussion": "percussions",
    "piano": "keyboards",
    "organ": "keyboards",
    "synth lead": "synthesizers",
    "synth pad": "synthesizers",
    "sound effects": "synthesizers",
    "brass": "orchestral",
    "strings": "orchestral",
    "strings (continued)": "orchestral",
    "reed": "orchestral",
    "pipe": "orchestral",
    "ethnic": "orchestral",
    # The dataset currently lacks vocal stems, but the mapping can be extended if needed.
}


def _find_audio(base: Path) -> Path | None:
    """Return `base` with the first existing extension of AUDIO_EXTENSIONS (e.g. mix.flac / mix.wav)."""
    for ext in AUDIO_EXTENSIONS:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _validate_directory(path: Path, description: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")
    return path


def _load_metadata(metadata_path: Path) -> Dict[str, object]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _normalise_inst_class(inst_class: str | None) -> str | None:
    if not inst_class:
        return None
    return inst_class.strip().lower()


def _initialise_category_record() -> Dict[str, object]:
    return {"paths": [], "is_present": False, "num_stems": 0}


def _initialise_category_records() -> Dict[str, Dict[str, object]]:
    return {category: _initialise_category_record() for category in TARGET_CATEGORIES}


def _load_reference_audio(path: Path) -> np.ndarray:
    if librosa is None:  # pragma: no cover - runtime guard
        raise RuntimeError("librosa is required to render Slakh2100 stem bounces but is not installed")
    audio, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=False)
    if audio.ndim == 1:
        audio = audio[np.newaxis, :]
    return audio.T.astype(np.float32, copy=False)


def _load_and_align(path: Path, ref_len: int, ref_channels: int) -> np.ndarray:
    frames = _load_reference_audio(path)
    if frames.shape[1] == 1 and ref_channels == 2:
        frames = np.repeat(frames, 2, axis=1)
    elif frames.shape[1] != ref_channels:
        raise ValueError(f"Unexpected channel count in {path}: {frames.shape[1]} (wanted {ref_channels})")

    if frames.shape[0] < ref_len:
        padded = np.zeros((ref_len, ref_channels), dtype=np.float32)
        padded[: frames.shape[0]] = frames
        return padded
    return frames[:ref_len]


def _sum_stem_paths(paths: Iterable[str], ref_len: int, ref_channels: int) -> np.ndarray:
    paths = list(paths)
    if not paths:
        return np.zeros((ref_len, ref_channels), dtype=np.float32)
    mixdown = np.zeros((ref_len, ref_channels), dtype=np.float32)
    for path_str in paths:
        mixdown += _load_and_align(Path(path_str), ref_len, ref_channels)
    return mixdown


def _is_existing_audio_valid(path: Path, expected_sr: int, expected_channels: int) -> bool:
    if not path.exists():
        return False
    try:
        info = sf.info(str(path))
    except RuntimeError:
        return False
    return info.samplerate == expected_sr and info.channels == expected_channels


def _render_category_bounces(
    piece_name: str,
    split_name: str,
    mix_path: Path,
    category_records: Dict[str, Dict[str, object]],
    residual_paths: List[str],
    final_dir: Path,
) -> Tuple[Dict[str, str], str]:
    ref_audio = _load_reference_audio(mix_path)
    ref_len, ref_channels = ref_audio.shape

    category_audio_paths: Dict[str, str] = {}

    for category in TARGET_CATEGORIES:
        stems = category_records[category]["paths"]
        if stems:
            summed = _sum_stem_paths(stems, ref_len, ref_channels)
            filename = f"{split_name}_{piece_name}_{category}.wav"
            final_path = final_dir / filename
            final_path.parent.mkdir(parents=True, exist_ok=True)
            if not _is_existing_audio_valid(final_path, SAMPLE_RATE, ref_channels):
                sf.write(str(final_path), summed, SAMPLE_RATE)
            category_audio_paths[category] = str(final_path.resolve())
        else:
            category_audio_paths[category] = ""

    # Residual "other": sum of all stems that could not be mapped to one of the categories.
    other_audio_path = ""
    if residual_paths:
        other_audio = _sum_stem_paths(residual_paths, ref_len, ref_channels)
        filename = f"{split_name}_{piece_name}_other.wav"
        final_other = final_dir / filename
        final_other.parent.mkdir(parents=True, exist_ok=True)
        if not _is_existing_audio_valid(final_other, SAMPLE_RATE, ref_channels):
            sf.write(str(final_other), other_audio, SAMPLE_RATE)
        other_audio_path = str(final_other.resolve())

    return category_audio_paths, other_audio_path


def _build_entry(
    track_dir: Path,
    split_name: str,
    final_wav_dir: Path,
    mix_override: Path | None = None,
) -> Dict[str, object]:
    metadata_path = track_dir / "metadata.yaml"
    metadata = _load_metadata(metadata_path)

    mix_path = mix_override or _find_audio(track_dir / "mix")
    if mix_path is None or not mix_path.exists():
        raise FileNotFoundError(f"Mix file not found for track '{track_dir.name}'")
    mix_path = mix_path.resolve()

    stems_dir = track_dir / "stems"
    category_records = _initialise_category_records()
    unmapped: List[str] = []
    residual_paths: List[str] = []

    stems = metadata.get("stems", {})
    for stem_id, stem_meta in stems.items():
        if not stem_meta.get("audio_rendered", False):
            continue

        inst_class = _normalise_inst_class(stem_meta.get("inst_class"))
        if inst_class is None:
            unmapped.append(f"{stem_id}:<missing>")
            continue

        stem_path = _find_audio(stems_dir / stem_id) or (stems_dir / f"{stem_id}.flac")

        category = INSTRUMENT_CATEGORY_MAP.get(inst_class)
        if category is None:
            unmapped.append(f"{stem_id}:{stem_meta.get('inst_class')}")
            # Keep the audio of unmapped stems for the residual "other" file.
            if stem_path.exists():
                residual_paths.append(str(stem_path.resolve()))
            continue

        if not stem_path.exists():
            unmapped.append(f"{stem_id}:{stem_meta.get('inst_class')} (missing audio)")
            continue

        record = category_records[category]
        record["paths"].append(str(stem_path.resolve()))
        record["num_stems"] += 1
        record["is_present"] = True

    fallback_paths: List[Path] = []
    for record in category_records.values():
        for path_str in record["paths"]:
            fallback_paths.append(Path(path_str))

    duration_seconds = _compute_duration_seconds(mix_path, fallback_paths)

    entry: Dict[str, object] = {
        "piece_name": track_dir.name,
        "split": split_name,
        "metadata_path": str(metadata_path.resolve()),
        "mixture": str(mix_path),
        "duration_seconds": duration_seconds,
        "unmapped_stems": unmapped,
    }

    for category, record in category_records.items():
        entry[f"{category}_stems"] = record["paths"]
        entry[f"{category}_is_present"] = record["is_present"]
        entry[f"{category}_num_stems"] = record["num_stems"]

    present_categories = sorted(cat for cat, record in category_records.items() if record["is_present"])
    entry["categories_present"] = present_categories

    category_audio_paths, other_audio_path = _render_category_bounces(
        track_dir.name,
        split_name,
        mix_path,
        category_records,
        residual_paths,
        final_wav_dir,
    )

    for category, path_str in category_audio_paths.items():
        entry[f"{category}_audio"] = path_str
    entry["other_audio"] = other_audio_path

    return entry


def _collect_split_entries(
    split_root: Path,
    split_name: str,
    final_wav_dir: Path,
    num_workers: int,
) -> List[Dict[str, object]]:
    split_root = _validate_directory(split_root, f"{split_name} split directory")
    track_dirs = sorted(path for path in split_root.iterdir() if path.is_dir())
    if not track_dirs:
        raise ValueError(f"No tracks discovered under {split_root}")

    final_wav_dir.mkdir(parents=True, exist_ok=True)

    if num_workers <= 1:
        iterator = track_dirs
        if tqdm is not None:
            iterator = tqdm(track_dirs, desc=f"{split_name} tracks", unit="track")
        return [_build_entry(track_dir, split_name, final_wav_dir) for track_dir in iterator]

    args = [(str(track_dir), split_name, str(final_wav_dir)) for track_dir in track_dirs]
    results: List[Optional[Dict[str, object]]] = [None] * len(args)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_index = {
            executor.submit(_build_entry_worker, arg): idx for idx, arg in enumerate(args)
        }
        iterator = as_completed(future_to_index)
        if tqdm is not None:
            iterator = tqdm(iterator, total=len(future_to_index), desc=f"{split_name} tracks", unit="track")
        for future in iterator:
            idx = future_to_index[future]
            results[idx] = future.result()

    return cast(List[Dict[str, object]], results)


def _build_entry_worker(args: Tuple[str, str, str]) -> Dict[str, object]:
    track_dir_str, split_name, final_wav_dir_str = args
    track_dir = Path(track_dir_str)
    final_wav_dir = Path(final_wav_dir_str)
    return _build_entry(track_dir, split_name, final_wav_dir)


def _build_features() -> Features:
    feature_dict = {
        "piece_name": Value("string"),
        "split": Value("string"),
        "metadata_path": Value("string"),
        "mixture": Value("string"),
        "duration_seconds": Value("float32"),
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


def _compute_duration_seconds(primary: Path, fallbacks: Iterable[Path]) -> float:
    """Prefer the mixture path, falling back to the first readable stem."""
    candidates = [primary, *fallbacks]
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
    num_workers: int | None = None,
    save_num_proc: int = 1,
) -> DatasetDict:
    print(f"Preparing Hugging Face dataset from Slakh2100 at: {dataset_root}")
    dataset_root = _validate_directory(dataset_root.resolve(), "Slakh2100 dataset root")
    save_dir = save_dir.resolve()

    if num_workers is None or num_workers <= 0:
        num_workers = max(1, os.cpu_count() or 1)

    reuse_tmp_parent: Path | None = None
    existing_additional_tmp: Path | None = None
    if save_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target directory already exists: {save_dir}. Use --overwrite to replace it."
            )
        additional_dir = save_dir / ADDITIONAL_WAVS_DIRNAME
        if additional_dir.exists():
            reuse_tmp_parent = Path(tempfile.mkdtemp(prefix="slakh_existing_", dir=save_dir.parent))
            existing_additional_tmp = reuse_tmp_parent / ADDITIONAL_WAVS_DIRNAME
            move(str(additional_dir), str(existing_additional_tmp))
        rmtree(save_dir)

    split_names = ("train", "validation", "test")
    # Without split folders (e.g. BabySlakh), the dataset root itself is the (only) train split.
    split_roots = {name: dataset_root / name for name in split_names if (dataset_root / name).is_dir()}
    if not split_roots:
        split_roots = {"train": dataset_root}
    features = _build_features()
    dataset_splits = {}

    save_dir.parent.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    final_wav_root = save_dir / ADDITIONAL_WAVS_DIRNAME
    if existing_additional_tmp and existing_additional_tmp.exists():
        move(str(existing_additional_tmp), str(final_wav_root))
    else:
        final_wav_root.mkdir(parents=True, exist_ok=True)

    dataset_dict: DatasetDict | None = None
    try:
        print("Processing dataset splits...")
        for split_name, split_root in split_roots.items():
            split_wav_dir = final_wav_root / split_name
            entries = _collect_split_entries(split_root, split_name, split_wav_dir, num_workers)
            split_dataset = Dataset.from_list(entries, features=features)
            dataset_splits[split_name] = split_dataset

        dataset_dict = DatasetDict(dataset_splits)
        dataset_dict.save_to_disk(str(save_dir), num_proc=save_num_proc)
    finally:
        if reuse_tmp_parent and reuse_tmp_parent.exists():
            rmtree(reuse_tmp_parent, ignore_errors=True)

    if dataset_dict is None:
        raise RuntimeError("Failed to create dataset splits.")
    return dataset_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face dataset for Slakh2100 (FLAC redux).")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the Slakh2100 FLAC redux root (containing train/validation/test)",
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
        help="Number of worker processes for stem rendering (0 uses CPU count).",
    )
    parser.add_argument(
        "--save-num-proc",
        type=int,
        default=1,
        help="Number of processes to use when saving the dataset to disk.",
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
        save_num_proc=args.save_num_proc,
    )
    print(f"Dataset prepared with splits: {', '.join(dataset_dict)}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
