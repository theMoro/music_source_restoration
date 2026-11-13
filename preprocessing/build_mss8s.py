#!/usr/bin/env python3

"""
Assemble the mss8s Hugging Face dataset by merging the preprocessed datasets of MoisesDB,
RawStems, Slakh2100 and MUSDB25 (created with the convert_*.py scripts in this folder).

Each source already provides one audio file per instrument category (vocals, guitars, bass,
keyboards, synthesizers, drums, percussions, orchestral) plus the residual "other" file. This
script aligns their column schemas, annotates every example with the originating dataset name
(`dataset_name`), puts all examples (incl. the Slakh2100 validation/test splits) into a single
`train` split and adds the `original_idx` column (the row index, used as key for the activity
masks, see compute_activity_masks.py).

Usage:
    python preprocessing/build_mss8s.py \
        --moisesdb-path <hf_root>/moisesdb \
        --raw-stems-path <hf_root>/raw_stems \
        --slakh2100-path <hf_root>/slakh2100 \
        --musdb25-path <hf_root>/kwatcharasupat_musdb25 \
        --output-path <hf_root>/mss8s
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict, List

import pyarrow as pa
from datasets import Dataset, DatasetDict, concatenate_datasets, load_from_disk

TARGET_INSTRUMENTS: List[str] = [
    "guitars",
    "bass",
    "drums",
    "vocals",
    "keyboards",
    "synthesizers",
    "percussions",
    "orchestral",
]

SLAKH_SAMPLE_RATE = 48_000

BASE_COLUMNS: List[str] = [
    "song_id",
    "song_name",
    "artist",
    "genre",
    "dataset_name",
    "source_split",
    "split",
    "duration_seconds",
    "sample_rate",
    "metadata_path",
    "mixture_audio",
    "unmapped_sources",
    "categories_present",
]

INSTRUMENT_COLUMNS: List[str] = []
for instrument in TARGET_INSTRUMENTS:
    INSTRUMENT_COLUMNS.extend(
        [
            f"{instrument}_stems",
            f"{instrument}_is_present",
            f"{instrument}_num_stems",
            instrument,
        ]
    )

FINAL_COLUMNS: List[str] = BASE_COLUMNS + INSTRUMENT_COLUMNS + ["other"]


def _rename_if_exists(dataset: Dataset, old: str, new: str) -> Dataset:
    if old in dataset.column_names and old != new:
        dataset = dataset.rename_column(old, new)
    return dataset


def _ensure_column_default(dataset: Dataset, column: str, value) -> Dataset:
    if column in dataset.column_names:
        return dataset
    values = [value] * dataset.num_rows
    return dataset.add_column(column, values)


def _copy_column(dataset: Dataset, source: str, target: str) -> Dataset:
    if target in dataset.column_names:
        return dataset
    if source not in dataset.column_names:
        raise KeyError(f"Cannot derive '{target}' – source column '{source}' is missing.")
    data = dataset[source]
    return dataset.add_column(target, data)


def _validate_instrument_columns(dataset: Dataset) -> None:
    missing: List[str] = []
    for instrument in TARGET_INSTRUMENTS:
        for suffix in ("_stems", "_is_present", "_num_stems"):
            column = f"{instrument}{suffix}"
            if column not in dataset.column_names:
                missing.append(column)
        if instrument not in dataset.column_names:
            missing.append(instrument)
    if "other" not in dataset.column_names:
        missing.append("other")
    if missing:
        raise ValueError(f"Dataset split missing expected instrument columns: {', '.join(sorted(missing))}")


def _ensure_final_columns(dataset: Dataset) -> Dataset:
    missing = [column for column in FINAL_COLUMNS if column not in dataset.column_names]
    if missing:
        raise ValueError(f"Dataset split is missing required columns: {', '.join(sorted(missing))}")
    extra = [column for column in dataset.column_names if column not in FINAL_COLUMNS]
    if extra:
        dataset = dataset.remove_columns(extra)
    return dataset.select_columns(FINAL_COLUMNS)


def _finalise_split(dataset: Dataset, final_split: str) -> Dataset:
    if "split" in dataset.column_names:
        dataset = dataset.remove_columns("split")
    dataset = _ensure_column_default(dataset, "split", final_split)
    dataset = _ensure_final_columns(dataset)
    return dataset


def _rename_audio_columns(dataset: Dataset) -> Dataset:
    for instrument in TARGET_INSTRUMENTS:
        dataset = _rename_if_exists(dataset, f"{instrument}_audio", instrument)
    dataset = _rename_if_exists(dataset, "other_audio", "other")
    return dataset


def _prepare_common(dataset: Dataset, dataset_name: str) -> Dataset:
    dataset = _rename_if_exists(dataset, "split", "source_split")
    dataset = _rename_if_exists(dataset, "unmapped_stems", "unmapped_sources")
    dataset = _rename_if_exists(dataset, "unmapped_items", "unmapped_sources")
    dataset = _ensure_column_default(dataset, "dataset_name", dataset_name)
    dataset = _ensure_column_default(dataset, "mixture_audio", "")
    dataset = _ensure_column_default(dataset, "genre", "")
    dataset = _ensure_column_default(dataset, "metadata_path", "")
    dataset = _ensure_column_default(dataset, "song_name", "")
    dataset = _ensure_column_default(dataset, "artist", "")
    dataset = _rename_audio_columns(dataset)
    _validate_instrument_columns(dataset)
    return dataset


def _prepare(path: Path, dataset_name: str) -> DatasetDict:
    dataset_dict = load_from_disk(str(path))
    return DatasetDict({split: _prepare_common(ds, dataset_name) for split, ds in dataset_dict.items()})


def _prepare_slakh(path: Path) -> DatasetDict:
    dataset_dict = load_from_disk(str(path))
    prepared: Dict[str, Dataset] = {}

    for split_name, dataset in dataset_dict.items():
        dataset = _rename_if_exists(dataset, "piece_name", "song_id")
        dataset = _rename_if_exists(dataset, "mixture", "mixture_audio")
        dataset = _copy_column(dataset, "song_id", "song_name")
        if "sample_rate" not in dataset.column_names:
            sample_rates = pa.array([SLAKH_SAMPLE_RATE] * dataset.num_rows, type=pa.int32())
            dataset = dataset.add_column("sample_rate", sample_rates)
        prepared[split_name] = _prepare_common(dataset, "slakh2100")

    return DatasetDict(prepared)


def build_mss8s_dataset(
    moisesdb_path: Path,
    raw_stems_path: Path,
    slakh_path: Path,
    musdb25_path: Path,
) -> DatasetDict:
    sources = {
        "moisesdb": _prepare(moisesdb_path, "moisesdb"),
        "raw_stems": _prepare(raw_stems_path, "raw_stems"),
        "slakh2100": _prepare_slakh(slakh_path),
        "kwatcharasupat_musdb25": _prepare(musdb25_path, "kwatcharasupat_musdb25"),
    }

    # All examples are used for training (validation is done on MSRBench): concatenate the
    # train splits of all sources, followed by the validation and test splits (Slakh2100 only).
    parts: List[Dataset] = []
    for split_name in ("train", "validation", "test"):
        for dataset_dict in sources.values():
            if split_name in dataset_dict:
                parts.append(_finalise_split(dataset_dict[split_name], "train"))
    if not parts:
        raise RuntimeError("No examples found; check the source datasets.")

    train = concatenate_datasets(parts)
    train = train.add_column("original_idx", list(range(train.num_rows)))
    return DatasetDict({"train": train})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge the MoisesDB, RawStems, Slakh2100 and MUSDB25 HF datasets into mss8s."
    )
    parser.add_argument("--moisesdb-path", required=True, help="Path to the converted MoisesDB dataset.")
    parser.add_argument("--raw-stems-path", required=True, help="Path to the converted RawStems dataset.")
    parser.add_argument("--slakh2100-path", required=True, help="Path to the converted Slakh2100 dataset.")
    parser.add_argument("--musdb25-path", required=True, help="Path to the converted MUSDB25 dataset.")
    parser.add_argument("--output-path", required=True, help="Destination directory for the combined dataset.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output directory if it already exists.",
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

    output_path = Path(args.output_path)
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)

    dataset = build_mss8s_dataset(
        moisesdb_path=Path(args.moisesdb_path),
        raw_stems_path=Path(args.raw_stems_path),
        slakh_path=Path(args.slakh2100_path),
        musdb25_path=Path(args.musdb25_path),
    )

    dataset.save_to_disk(str(output_path), num_proc=args.num_proc)
    counts: Dict[str, int] = {}
    for name in dataset["train"]["dataset_name"]:
        counts[name] = counts.get(name, 0) + 1
    print(f"Saved mss8s dataset with {dataset['train'].num_rows} songs: {counts}")
    print(f"Destination: {output_path}")


if __name__ == "__main__":
    main()
