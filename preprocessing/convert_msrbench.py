#!/usr/bin/env python3

"""
Utilities to convert the MSR-Bench dataset (https://huggingface.co/datasets/yongyizang/MSRBench)
into a Hugging Face dataset.

The dataset is organised by source family (Bass, Vocals, etc.). Each family
directory contains two subdirectories:

    mixture/   --- mixture audio files (multiple decorrelated variants per clip)
    target/    --- the isolated source corresponding to each mixture clip

Mixture filenames follow the pattern `<clip_id>_DT<variant>.flac`, while target
files are stored as `<clip_id>.flac`. Each mixture therefore has exactly one
matching source stem. We map every family directory to the canonical target
categories used throughout this project:

    vocals, guitars, bass, keyboards, synthesizers, drums, percussions, orchestral

For every mixture variant we create one dataset row that stores the canonical
category metadata together with direct pointers to the mixture and isolated
source audio.

Usage (dataset root = folder containing the unzipped Bass/, Drums/, ... folders):
    python preprocessing/convert_msrbench.py --dataset-root <msrbench> --save-dir <hf_root>/msrbench
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree
from typing import Dict, Iterable, Iterator, List, Tuple

import soundfile as sf
from datasets import Dataset, DatasetDict, Features, Sequence as HFSequence, Value

try:  # pragma: no cover - optional dependency
    from tqdm import tqdm
except ModuleNotFoundError:  # pragma: no cover - runtime guard
    tqdm = None


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

# Mapping from the MSR-Bench family directory name (case-insensitive) to the
# canonical category label used for the Hugging Face dataset.
FAMILY_CATEGORY_MAP: Dict[str, str] = {
    "vocals": "vocals",
    "guitars": "guitars",
    "bass": "bass",
    "keyboards": "keyboards",
    "synthesizers": "synthesizers",
    "drums": "drums",
    "percussions": "percussions",
    "orchestral elements": "orchestral",
}

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".flac", ".wav", ".ogg")


@dataclass(frozen=True)
class AudioProperties:
    """Simple container for duration and sample rate metadata."""

    duration_seconds: float
    sample_rate: int


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


def _normalise_family_name(name: str) -> str:
    return name.strip().lower()


def _iter_audio_files(directory: Path) -> Iterator[Path]:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            yield path


def _match_source_for_mixture(mix_path: Path, source_dir: Path) -> Path:
    stem = mix_path.stem
    clip_id = stem.split("_", 1)[0]
    expected = source_dir / f"{clip_id}{mix_path.suffix}"
    if expected.exists():
        return expected

    # Fall back to any file with matching clip id regardless of extension.
    candidates = list(source_dir.glob(f"{clip_id}.*"))
    if not candidates:
        raise FileNotFoundError(f"No matching target found for mixture: {mix_path}")
    if len(candidates) > 1:
        raise ValueError(f"Multiple matching targets found for mixture {mix_path}: {candidates}")
    return candidates[0]


def _probe_audio_properties(path: Path) -> AudioProperties:
    with sf.SoundFile(str(path)) as audio_file:
        samplerate = int(audio_file.samplerate)
        if samplerate <= 0:
            raise ValueError(f"Invalid samplerate in {path}: {samplerate}")
        duration = float(audio_file.frames) / float(samplerate)
    return AudioProperties(duration_seconds=duration, sample_rate=samplerate)


def _collect_family_entries(family_dir: Path, canonical_category: str) -> List[Dict[str, object]]:
    mixture_dir = _validate_directory(family_dir / "mixture", f"mixture directory for {family_dir.name}")
    target_dir = _validate_directory(family_dir / "target", f"target directory for {family_dir.name}")

    entries: List[Dict[str, object]] = []
    mixture_files = list(_iter_audio_files(mixture_dir))
    progress = None
    iterator: Iterable[Path]
    if tqdm is not None and len(mixture_files) > 200:
        progress = tqdm(mixture_files, desc=f"Processing {family_dir.name}", unit="clip")
        iterator = progress
    else:
        iterator = mixture_files

    for mix_path in iterator:
        source_path = _match_source_for_mixture(mix_path, target_dir)
        props = _probe_audio_properties(mix_path)

        category_records = _initialise_category_records()
        record = category_records[canonical_category]
        resolved_source = str(source_path.resolve())
        record["paths"].append(resolved_source)
        record["num_stems"] = 1
        record["is_present"] = True

        entry: Dict[str, object] = {
            "clip_id": mix_path.stem,
            "family": family_dir.name,
            "source_label": canonical_category,
            "split": "train",
            "duration_seconds": props.duration_seconds,
            "sample_rate": props.sample_rate,
            "unmapped_items": [],
            "mixture_audio": str(mix_path.resolve()),
            "source_audio": resolved_source,
        }

        for category, cat_record in category_records.items():
            entry[f"{category}_stems"] = cat_record["paths"]
            entry[f"{category}_is_present"] = cat_record["is_present"]
            entry[f"{category}_num_stems"] = cat_record["num_stems"]
            entry[f"{category}_audio"] = resolved_source if category == canonical_category else ""

        entry["categories_present"] = [canonical_category]
        entry["other_audio"] = ""

        entries.append(entry)

    if progress is not None:
        progress.close()
    return entries


def _collect_entries(dataset_root: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    family_dirs = [path for path in sorted(dataset_root.iterdir()) if path.is_dir()]
    progress = None
    iterator: Iterable[Path]

    if tqdm is not None:
        progress = tqdm(family_dirs, desc="Processing MSR-Bench families", unit="family")
        iterator = progress
    else:
        iterator = family_dirs

    for family_dir in iterator:
        family_name = _normalise_family_name(family_dir.name)
        canonical = FAMILY_CATEGORY_MAP.get(family_name)
        if canonical is None:
            raise KeyError(f"No canonical mapping defined for MSR family '{family_dir.name}'")
        entries.extend(_collect_family_entries(family_dir, canonical))

    if progress is not None:
        progress.close()

    if not entries:
        raise ValueError(f"No entries discovered under {dataset_root}")
    return entries


def _build_features() -> Features:
    feature_dict = {
        "clip_id": Value("string"),
        "family": Value("string"),
        "source_label": Value("string"),
        "split": Value("string"),
        "duration_seconds": Value("float32"),
        "sample_rate": Value("int32"),
        "unmapped_items": HFSequence(Value("string")),
        "categories_present": HFSequence(Value("string")),
        "mixture_audio": Value("string"),
        "source_audio": Value("string"),
    }

    for category in TARGET_CATEGORIES:
        feature_dict[f"{category}_stems"] = HFSequence(Value("string"))
        feature_dict[f"{category}_is_present"] = Value("bool")
        feature_dict[f"{category}_num_stems"] = Value("int32")
        feature_dict[f"{category}_audio"] = Value("string")

    feature_dict["other_audio"] = Value("string")

    return Features(feature_dict)


def create_dataset(
    dataset_root: Path,
    save_dir: Path,
    overwrite: bool = False,
    num_proc: int = 1,
) -> DatasetDict:
    dataset_root = _validate_directory(dataset_root.resolve(), "MSR-Bench dataset root")
    save_dir = save_dir.resolve()

    if save_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Target directory already exists: {save_dir}. Use --overwrite to replace it."
            )
        rmtree(save_dir)

    entries = _collect_entries(dataset_root)
    features = _build_features()

    dataset = Dataset.from_list(entries, features=features)
    dataset_dict = DatasetDict({"train": dataset})
    save_dir.parent.mkdir(parents=True, exist_ok=True)

    dataset_dict.save_to_disk(str(save_dir), num_proc=num_proc)

    return dataset_dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare a Hugging Face dataset for MSR-Bench.")
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Path to the MSR-Bench root containing the unzipped family folders (Bass/, Drums/, ...)",
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
    print(f"MSR-Bench dataset prepared with split: {', '.join(dataset_dict)}")
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
