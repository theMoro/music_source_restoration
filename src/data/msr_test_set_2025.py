"""
MSRChallenge/2025TestSet dataset loader.

Reads a local snapshot of the MSRChallenge/2025TestSet Hugging Face dataset (downloaded via
`hf download MSRChallenge/2025TestSet --repo-type dataset --local-dir <path>`)
directly from `metadata.jsonl` + the `mixtures/`/`targets/` folders, instead of going through
`datasets.load_dataset()` — the generic folder-based builder crashes on rows whose
`target_file_name` is null (blind subsets have no ground truth), since it calls
`os.path.normpath()` on every path-like column unconditionally.

Subclasses MultiAudioFullSongDataset (msr_bench.py) so it reuses its __getitem__/collate_fn/
audio-loading logic unchanged, and so it is recognized by the "pre-mixed, single-source-per-
sample, full-clip" test_step pathway in MultiSourceSeparationLightning (which isinstance-checks
against MultiAudioFullSongDataset).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from src.data.msr_bench import MultiAudioFullSongDataset

# Maps the (capitalized, space-separated) instrument names used in metadata.jsonl / the
# mixtures//targets/ folder names to the lowercase column names used throughout the project.
INSTRUMENT_TO_COLUMN = {
    "Bass": "bass",
    "Drums": "drums",
    "Guitars": "guitars",
    "Keyboards": "keyboards",
    "Orchestral Elements": "orchestral",
    "Percussions": "percussions",
    "Synthesizers": "synthesizers",
    "Vocals": "vocals",
}


class MSRTestSet2025Dataset(MultiAudioFullSongDataset):
    """
    Local snapshot of the MSRChallenge/2025TestSet evaluation set.

    Each row in metadata.jsonl describes one mixture clip and (for non-blind subsets) the single
    ground-truth stem it was derived from. Only rows with `has_target=True` (i.e. subsets like
    "non-blind"/"streaming") are usable for evaluation; blind subsets ("live", "radio",
    "cylinder") have `target_file_name=None` and are skipped.

    Parameters
    ----------
    root:
        Directory containing metadata.jsonl and the mixtures/ and targets/ folders.
    subsets:
        Optional list of `subset` values to include (e.g. ["non-blind", "streaming"]). If None,
        all rows with a ground-truth target are included regardless of subset.
    columns:
        Stem/column names (lowercase) to expose per sample, matching the lightning module's
        `self.columns`. Defaults to the known instrument columns above.
    sr:
        Target sampling rate for loaded audio.
    mono:
        Whether to load audio as mono.
    """

    def __init__(
        self,
        root: Union[str, Path],
        subsets: Optional[Sequence[str]] = None,
        columns: Optional[Sequence[str]] = None,
        sr: int = 48_000,
        mono: bool = False,
    ) -> None:
        root = Path(root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"MSRTestSet2025 root not found: {root}")

        self.root = root
        self.prefix_path = root
        self.sr = int(sr)
        self.mono = bool(mono)
        self.columns = list(columns) if columns is not None else list(INSTRUMENT_TO_COLUMN.values())
        self.presence_columns: Dict[str, str] = {}
        self.presence_suffix = "_is_present"

        subsets_set = set(subsets) if subsets is not None else None

        metadata_path = root / "metadata.jsonl"
        if not metadata_path.exists():
            raise FileNotFoundError(f"metadata.jsonl not found under {root}")

        self.ds: List[Dict[str, Any]] = []
        with open(metadata_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                if not row.get("has_target") or row.get("target_file_name") is None:
                    continue
                if subsets_set is not None and row.get("subset") not in subsets_set:
                    continue

                instrument = row.get("instrument", "")
                column = INSTRUMENT_TO_COLUMN.get(instrument)
                if column is None:
                    print(f"Warning: unknown instrument '{instrument}' in metadata row {row.get('file_id')}, skipping")
                    continue

                sample: Dict[str, Any] = {
                    "clip_id": row.get("file_id", ""),
                    "source_label": column,
                    "split": row.get("subset", ""),
                    "duration_seconds": 0.0,
                    "sample_rate": self.sr,
                    "mixture_audio": row["mixture_file_name"],
                    "source_audio": row["target_file_name"],
                    # Explicit, since MultiAudioFullSongDataset.collate_fn (msr_bench.py) defaults
                    # missing dataset_name to "msr_bench" — without this, per-dataset breakdowns
                    # mislabel MSRTestSet2025 samples as MSRBench.
                    "dataset_name": "msr_test_set_2025",
                }

                for col in self.columns:
                    is_target_column = col == column
                    sample[f"{col}_audio"] = row["target_file_name"] if is_target_column else ""
                    sample[f"{col}_is_present"] = is_target_column

                self.ds.append(sample)

        if len(self.ds) == 0:
            raise ValueError(
                f"No usable samples found in {root} for subsets={subsets} "
                f"(only subsets with ground-truth targets can be used for evaluation)"
            )

        self.ds.sort(key=lambda s: s["clip_id"])

        print(
            f"MSRTestSet2025Dataset: loaded {len(self.ds)} samples "
            f"from subsets={sorted(subsets_set) if subsets_set else 'all with targets'}"
        )


__all__ = ["MSRTestSet2025Dataset"]
