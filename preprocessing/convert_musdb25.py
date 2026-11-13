#!/usr/bin/env python3

"""
Convert the MUSDB25 alpha dataset (https://huggingface.co/datasets/kwatcharasupat/musdb25, a
fully multitrack recreation of MUSDB18) into a Hugging Face dataset.

Each song directory contains a MedleyDB-style metadata YAML file plus a
`raws.npz` archive holding the raw microphone tracks grouped by stem ID. The
archive stores one NumPy array per microphone capture using keys such as
`vocals/S11R01_leadvocals_00`. We collapse all captures belonging to the same
stem, map the reported instrument to our canonical source families, and render
per-category bounces mirroring the structure produced for MoisesDB.

Outputs:
  * A `DatasetDict` saved via `datasets` with one split (`train`).
  * Per-stem WAV files stored under `<save_dir>/stem_wavs`.
  * Per-category summed WAV files plus a residual "other" track (sum of all stems that
    cannot be mapped to a category; empty string if there are none) under
    `<save_dir>/additional_wavs`.

Usage:
    python preprocessing/convert_musdb25.py --dataset-root <musdb25> --save-dir <hf_root>/kwatcharasupat_musdb25
"""

from __future__ import annotations

import argparse
import tempfile
from collections import defaultdict
from pathlib import Path
from shutil import move, rmtree
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import soundfile as sf
import yaml
from datasets import Dataset, DatasetDict, Features, Sequence, Value

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    tqdm = None


ADDITIONAL_WAVS_DIRNAME = "additional_wavs"
STEM_WAVS_DIRNAME = "stem_wavs"
DEFAULT_SAMPLE_RATE = 44_100  # sample rate of the raws.npz arrays (MUSDB data is 44.1 kHz)
SPLIT_NAME = "train"

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

# Exact instrument name overrides (lowercase) mapped to canonical categories.
INSTRUMENT_CATEGORY_MAP: Dict[str, str] = {
    # Vocals
    "vocals": "vocals",
    "vocalists": "vocals",
    "female singer": "vocals",
    "male singer": "vocals",
    "male rapper": "vocals",
    "vocals.synthesizer": "vocals",
    # Guitars and plucked strings
    "acoustic guitar": "guitars",
    "banjo": "guitars",
    "clean electric guitar": "guitars",
    "distorted electric guitar": "guitars",
    "lap steel guitar": "guitars",
    "string.plucked": "guitars",
    "string.plucked.lute": "guitars",
    "string.plucked.lute.guitar": "guitars",
    # Bass
    "bass": "bass",
    "double bass": "bass",
    "electric bass": "bass",
    "string.plucked.bass": "bass",
    "string.plucked.lute.guitar.bass": "bass",
    "synthesizer.bass": "bass",
    # Keyboards / piano
    "keys": "keyboards",
    "keys.piano": "keyboards",
    "keys.piano.electronic": "keyboards",
    "keys.organ": "keyboards",
    "keys.free_reeds": "keyboards",
    "piano": "keyboards",
    "tack piano": "keyboards",
    # Synthesizers / electronic
    "string.plucked.lute.guitar.synthesizer": "synthesizers",
    "string.synthesizer": "synthesizers",
    "synthesizer": "synthesizers",
    "sampler": "synthesizers",
    "fx_processed": "synthesizers",
    "fx_processed.sample": "synthesizers",
    "fx_processed.sfx": "synthesizers",
    "fx/processed sound": "synthesizers",
    "misc": "synthesizers",
    "other": "synthesizers",
    "ensemble": "orchestral",  # default to orchestral blend
    # Drums
    "drum machine": "drums",
    "drum set": "drums",
    "drumset": "drums",
    "percussion.drumset": "drums",
    # Percussions (non-drumset)
    "auxiliary percussion": "percussions",
    "percussion": "percussions",
    "percussion.pitched": "percussions",
    "percussion.timpani": "percussions",
    "percussion.unpitched.idiophones": "percussions",
    "percussion.unpitched.idiophones.body": "percussions",
    "percussion.unpitched.idiophones.metal": "percussions",
    "percussion.unpitched.idiophones.rattles": "percussions",
    "percussion.unpitched.idiophones.wood": "percussions",
    "percussion.unpitched.membraphones": "percussions",
    "tambourine": "percussions",
    "shaker": "percussions",
    "claps": "percussions",
    "cymbal": "percussions",
    "scratches": "percussions",
    "timpani": "percussions",
    "glockenspiel": "percussions",
    "vibraphone": "percussions",
    # Orchestral / acoustic sections
    "string": "orchestral",
    "string section": "orchestral",
    "string.bowed": "orchestral",
    "violin section": "orchestral",
    "viola section": "orchestral",
    "horn section": "orchestral",
    "brass section": "orchestral",
    "trumpet section": "orchestral",
    "clarinet": "orchestral",
    "wind.brass": "orchestral",
    "wind.brass.bass": "orchestral",
    "wind.woodwind.flutes": "orchestral",
    "wind.woodwind.free_reeds": "orchestral",
    "wind.woodwind.single_reeds": "orchestral",
    "harmonica": "orchestral",
}

# Fallback mapping based on component labels when available.
COMPONENT_CATEGORY_MAP: Dict[str, str] = {
    "bass": "bass",
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


def _load_metadata(metadata_path: Path) -> Dict[str, object]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _iter_song_dirs(dataset_root: Path) -> Iterator[Path]:
    for path in sorted(dataset_root.iterdir()):
        if path.name.startswith("."):
            continue
        if path.is_dir():
            yield path


def _load_npz_stems(npz_path: Path) -> Dict[str, List[np.ndarray]]:
    """Return a mapping from stem ID (e.g. 'S11') to raw arrays."""
    stem_audio: Dict[str, List[np.ndarray]] = defaultdict(list)
    with np.load(str(npz_path), allow_pickle=False) as archive:
        for key in archive.files:
            parts = key.split("/", 1)
            if len(parts) != 2:
                continue
            second = parts[1]
            if not second.startswith("S"):
                continue
            stem_id = second.split("R", 1)[0]
            audio = archive[key].astype(np.float32, copy=False)
            stem_audio[stem_id].append(audio)
    return stem_audio


def _map_instrument_to_category(instrument: str | None, component: str | None) -> str | None:
    inst = (instrument or "").strip().lower()
    if inst:
        direct = INSTRUMENT_CATEGORY_MAP.get(inst)
        if direct is not None:
            return direct

    comp = (component or "").strip().lower()
    if comp:
        direct = COMPONENT_CATEGORY_MAP.get(comp)
        if direct is not None:
            return direct

    token = inst or comp
    if not token:
        return None

    if "vocal" in token or "singer" in token or "rapper" in token:
        return "vocals"
    if "guitar" in token or "banjo" in token or "lute" in token:
        return "guitars"
    if "bass" in token and "synth" not in token:
        return "bass"
    if "piano" in token or "keys" in token or "organ" in token:
        return "keyboards"
    if "synth" in token or "fx" in token or "sampler" in token:
        return "synthesizers"
    if "drum" in token:
        return "drums"
    if "percussion" in token or "tambour" in token or "shaker" in token or "clap" in token:
        return "percussions"
    if "horn" in token or "brass" in token or "string" in token or "wind" in token or "ensemble" in token:
        return "orchestral"
    if "harmonica" in token or "clarinet" in token:
        return "orchestral"
    return None


def _compute_reference_dims(arrays: Iterable[np.ndarray]) -> Tuple[int, int]:
    ref_channels = 0
    ref_len = 0
    for audio in arrays:
        if audio.ndim != 2:
            raise ValueError(f"Expected (channels, samples) audio array; got shape {audio.shape}")
        ref_channels = max(ref_channels, audio.shape[0])
        ref_len = max(ref_len, audio.shape[1])
    if ref_channels == 0 or ref_len == 0:
        raise ValueError("Unable to determine reference dimensions; no audio available.")
    return ref_channels, ref_len


def _align_audio(audio: np.ndarray, ref_channels: int, ref_len: int) -> np.ndarray:
    frames = audio
    if frames.shape[0] == 1 and ref_channels == 2:
        frames = np.repeat(frames, 2, axis=0)
    elif frames.shape[0] < ref_channels:
        padded = np.zeros((ref_channels, frames.shape[1]), dtype=np.float32)
        padded[: frames.shape[0], : frames.shape[1]] = frames
        frames = padded
    elif frames.shape[0] > ref_channels:
        frames = frames[:ref_channels, :]

    if frames.shape[1] < ref_len:
        padded = np.zeros((frames.shape[0], ref_len), dtype=np.float32)
        padded[:, : frames.shape[1]] = frames
        frames = padded
    elif frames.shape[1] > ref_len:
        frames = frames[:, :ref_len]
    return frames


def _sum_audio(arrays: Iterable[np.ndarray], ref_channels: int, ref_len: int) -> np.ndarray:
    arrays = list(arrays)
    if not arrays:
        return np.zeros((ref_channels, ref_len), dtype=np.float32)
    mixdown = np.zeros((ref_channels, ref_len), dtype=np.float32)
    for audio in arrays:
        mixdown += _align_audio(audio, ref_channels, ref_len)
    return mixdown


def _write_audio_file(audio: np.ndarray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio.T, sample_rate)


def _render_category_bounces(
    song_id: str,
    category_audio_arrays: Dict[str, List[np.ndarray]],
    residual_arrays: List[np.ndarray],
    sample_rate: int,
    temp_dir: Path,
    final_dir: Path,
) -> Tuple[Dict[str, str], str, int, int]:
    all_arrays = [audio for arrays in category_audio_arrays.values() for audio in arrays]
    if not all_arrays:
        raise ValueError(f"No audio arrays available for song '{song_id}'")

    ref_channels, ref_len = _compute_reference_dims(all_arrays)

    category_audio_paths: Dict[str, str] = {}

    for category in TARGET_CATEGORIES:
        arrays = category_audio_arrays.get(category, [])
        if arrays:
            summed = _sum_audio(arrays, ref_channels, ref_len)
            filename = f"{SPLIT_NAME}_{song_id}_{category}.wav"
            temp_path = temp_dir / filename
            final_path = final_dir / filename
            _write_audio_file(summed, sample_rate, temp_path)
            category_audio_paths[category] = str(final_path.resolve())
        else:
            category_audio_paths[category] = ""

    # Residual "other": sum of all stems that could not be mapped to one of the categories.
    other_audio_path = ""
    if residual_arrays:
        other_audio = _sum_audio(residual_arrays, ref_channels, ref_len)
        other_filename = f"{SPLIT_NAME}_{song_id}_other.wav"
        temp_other = temp_dir / other_filename
        final_other = final_dir / other_filename
        _write_audio_file(other_audio, sample_rate, temp_other)
        other_audio_path = str(final_other.resolve())

    return category_audio_paths, other_audio_path, ref_channels, ref_len


def _collect_song_entry(
    song_dir: Path,
    temp_stem_dir: Path,
    final_stem_dir: Path,
    temp_category_dir: Path,
    final_category_dir: Path,
    sample_rate: int,
) -> Dict[str, object]:
    stem_arrays_path = song_dir / "raws.npz"
    if not stem_arrays_path.exists():
        raise FileNotFoundError(f"Missing raws.npz for song '{song_dir.name}'")
    metadata_path_candidates = list(song_dir.glob("*_METADATA.yaml"))
    if not metadata_path_candidates:
        raise FileNotFoundError(f"No metadata YAML found for song '{song_dir.name}'")
    metadata_path = metadata_path_candidates[0]
    metadata = _load_metadata(metadata_path)

    stem_audio_map = _load_npz_stems(stem_arrays_path)
    category_records = _initialise_category_records()
    category_audio_arrays: Dict[str, List[np.ndarray]] = {category: [] for category in TARGET_CATEGORIES}
    unmapped: List[str] = []
    residual_arrays: List[np.ndarray] = []

    stems_meta = metadata.get("stems", {})
    present_audio: List[np.ndarray] = []

    for stem_id in sorted(stems_meta):
        stem_info = stems_meta[stem_id] or {}
        instrument = stem_info.get("instrument")
        component = stem_info.get("component")
        category = _map_instrument_to_category(instrument, component)
        if category is None:
            unmapped.append(f"{stem_id}:{instrument or component or '<unknown>'}")
            # Keep the audio of unmapped stems for the residual "other" file.
            unmapped_arrays = stem_audio_map.get(stem_id)
            if unmapped_arrays:
                residual_arrays.append(_sum_audio(unmapped_arrays, *_compute_reference_dims(unmapped_arrays)))
            continue

        raw_arrays = stem_audio_map.get(stem_id)
        if not raw_arrays:
            unmapped.append(f"{stem_id}:{instrument or component or '<unknown>'} (no audio)")
            continue

        ref_channels, ref_len = _compute_reference_dims(raw_arrays)
        combined = _sum_audio(raw_arrays, ref_channels, ref_len)
        present_audio.append(combined)

        stem_filename = f"{SPLIT_NAME}_{song_dir.name}_{stem_id}.wav"
        temp_stem_path = temp_stem_dir / stem_filename
        final_stem_path = final_stem_dir / stem_filename
        _write_audio_file(combined, sample_rate, temp_stem_path)

        record = category_records[category]
        record["paths"].append(str(final_stem_path.resolve()))
        record["num_stems"] += 1
        record["is_present"] = True

        category_audio_arrays[category].append(combined)

    if not present_audio:
        raise ValueError(f"No mapped stems found for song '{song_dir.name}'")

    category_audio_paths, other_audio_path, ref_channels, ref_len = _render_category_bounces(
        song_dir.name,
        category_audio_arrays,
        residual_arrays,
        sample_rate,
        temp_category_dir,
        final_category_dir,
    )

    duration_seconds = float(ref_len) / float(sample_rate)

    entry: Dict[str, object] = {
        "song_id": song_dir.name,
        "song_name": str(metadata.get("title") or ""),
        "artist": str(metadata.get("artist") or ""),
        "genre": str(metadata.get("genre") or ""),
        "metadata_path": str(metadata_path.resolve()),
        "split": SPLIT_NAME,
        "duration_seconds": duration_seconds,
        "sample_rate": sample_rate,
        "unmapped_stems": unmapped,
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


def _collect_entries(
    dataset_root: Path,
    temp_stem_dir: Path,
    final_stem_dir: Path,
    temp_category_dir: Path,
    final_category_dir: Path,
    sample_rate: int,
) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    song_dirs = list(_iter_song_dirs(dataset_root))
    iterator: Iterable[Path]
    progress = None

    if tqdm is not None:
        progress = tqdm(song_dirs, desc="Processing MUSDB25 songs", unit="song")
        iterator = progress
    else:
        iterator = song_dirs

    for song_dir in iterator:
        entry = _collect_song_entry(
            song_dir,
            temp_stem_dir,
            final_stem_dir,
            temp_category_dir,
            final_category_dir,
            sample_rate,
        )
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


def create_dataset(
    dataset_root: Path,
    save_dir: Path,
    overwrite: bool = False,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    num_proc: int = 1,
) -> DatasetDict:
    dataset_root = _validate_directory(dataset_root.resolve(), "kwatcharasupat MUSDB25 dataset root")
    save_dir = save_dir.resolve()

    if save_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target directory already exists: {save_dir}. Use --overwrite to replace it."
            )
        rmtree(save_dir)

    features = _build_features()
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    temp_root = Path(tempfile.mkdtemp(prefix="kwatcharasupat_tmp_", dir=save_dir.parent))
    temp_stem_dir = temp_root / STEM_WAVS_DIRNAME
    temp_category_dir = temp_root / ADDITIONAL_WAVS_DIRNAME
    final_stem_dir = save_dir / STEM_WAVS_DIRNAME
    final_category_dir = save_dir / ADDITIONAL_WAVS_DIRNAME

    dataset_dict: DatasetDict | None = None

    try:
        entries = _collect_entries(
            dataset_root,
            temp_stem_dir,
            final_stem_dir,
            temp_category_dir,
            final_category_dir,
            sample_rate,
        )
        dataset = Dataset.from_list(entries, features=features)
        dataset_dict = DatasetDict({SPLIT_NAME: dataset})

        try:
            dataset_dict.save_to_disk(str(save_dir), num_proc=num_proc)
        except PermissionError:
            if num_proc == 1:
                raise
            if save_dir.exists():
                rmtree(save_dir)
            dataset_dict.save_to_disk(str(save_dir), num_proc=1)

        if final_stem_dir.exists():
            rmtree(final_stem_dir)
        if final_category_dir.exists():
            rmtree(final_category_dir)
        move(str(temp_stem_dir), str(final_stem_dir))
        move(str(temp_category_dir), str(final_category_dir))
    finally:
        if temp_root.exists():
            rmtree(temp_root, ignore_errors=True)

    if dataset_dict is None:
        raise RuntimeError("Failed to assemble kwatcharasupat MUSDB25 dataset.")
    return dataset_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face dataset for kwatcharasupat MUSDB25.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the MUSDB25 dataset root (one folder per song with raws.npz + *_METADATA.yaml)",
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
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Sample rate of the raws.npz arrays (default: %(default)s).",
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
        sample_rate=args.sample_rate,
        num_proc=args.num_proc,
    )
    print(f"kwatcharasupat MUSDB25 dataset prepared with split: {', '.join(dataset_dict)}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
