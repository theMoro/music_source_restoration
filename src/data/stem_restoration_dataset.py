"""
Paired stem-restoration dataset for fine-tuning per-stem restoration models on the output of
`generate_inference_dataset_with_trainer` (src/train.py, `gen_inf_dataset`), e.g. the
`final_dataset` produced by scripts/stem_restoration/generate_dataset.sh.

Each row of that HuggingFace dataset holds, for one stem of one song: `separated_stem` (the
post-separator prediction — augmented/degraded) and `clean_stem` (the raw, unaugmented ground
truth), plus a `label` column (stem name) and a `dataset_name` column. This dataset filters to a
single stem and exposes the filtered HF dataset as `self.ds` — the attribute name
src.datamodules.datamodule.DataModule's `data_scheduler` looks for (via `hasattr(..., 'ds')`) to
build a per-epoch `WeightedRandomSampler` over `dataset_name`. That's what drives the "fine-tune
on everything, then decay down to raw_stems only" curriculum (see
src.datamodules.data_scheduler_utils.decay_to_single_dataset and
configs/stem_restoration/apollo_stem_restoration.yaml) — this class itself has no epoch-dependent
sampling logic, it just returns whatever row the sampler/DataLoader asks for.
"""

import hashlib
import warnings
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
from datasets import load_from_disk


class StemRestorationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root: str,
        stem: str,
        split: str = "train",
        val_fraction: float = 0.05,
        val_dataset_name: Optional[str] = "raw_stems",
        input_column: str = "separated_stem",
        target_column: str = "clean_stem",
        sr: int = 48000,
        seed: int = 42,
        columns: Optional[List[str]] = None,
        training_chunk_size: Optional[int] = None,
        chunk_seed: Optional[int] = None,
    ):
        # `columns` is accepted and discarded, not used for filtering: src/train.py::main()
        # unconditionally force-injects a `columns` list (derived from
        # lightning_module.args.n_sources) into this dataset's args dict, before any
        # pipeline-specific `.enabled` gate — every dataset class in this repo must accept it
        # even when unused. See StemRestorationGANLightning's docstring for the full explanation.
        super().__init__()
        if not dataset_root:
            raise ValueError("dataset_root must be set")
        if not stem:
            raise ValueError("stem must be set")

        self.stem = stem
        self.split = split
        self.input_column = input_column
        self.target_column = target_column
        self.sr = sr
        # Default None = whole-file loading (this class's original behavior). Set this (e.g.
        # 192000 = 4s @ 48kHz, matching the mixture restoration training_chunk_size) for
        # generators whose activation memory scales with input duration -- e.g. Apollo's
        # per-frame band Roformer (src.models.apollo.model.BSNet) OOMs on whole songs, since
        # every layer's band-attention batches over every STFT frame of the full track.
        self.training_chunk_size = int(training_chunk_size) if training_chunk_size else None
        # `seed` only selects the train/val split (fixed across training seeds); chunk offsets are
        # drawn per (chunk_seed, epoch, idx) -- independent of DataLoader workers / ranks and of
        # resumes. chunk_seed=None falls back to `seed` (e.g. the validation split: fixed offsets).
        self.chunk_seed = int(chunk_seed) if chunk_seed is not None else int(seed)
        self._epoch = 0

        full_ds = load_from_disk(dataset_root)
        full_ds = full_ds.filter(
            lambda x: x["label"] == stem, desc=f"Filtering dataset for stem={stem}"
        )
        if len(full_ds) == 0:
            raise ValueError(f"No samples found for stem='{stem}' in {dataset_root}")

        # Deterministic train/val split based on a hash of each sample's own identity (its
        # original source path, falling back to dataset_index) rather than its row position, so
        # the split stays stable across dataset regenerations/reorderings.
        def _val_bucket(example) -> int:
            key = f"{seed}_{example.get('original_path') or example.get('dataset_index')}"
            digest = hashlib.md5(key.encode("utf-8")).hexdigest()
            return int(digest, 16) % 10_000

        val_threshold = int(val_fraction * 10_000)
        val_mask = np.array([_val_bucket(row) < val_threshold for row in full_ds])

        if split == "train":
            self.ds = full_ds.select(np.where(~val_mask)[0].tolist())
        elif split == "val":
            self.ds = full_ds.select(np.where(val_mask)[0].tolist())
            if val_dataset_name is not None and "dataset_name" in self.ds.column_names:
                self.ds = self.ds.filter(
                    lambda x: x["dataset_name"] == val_dataset_name,
                    desc=f"Filtering val split to dataset_name={val_dataset_name}",
                )
        else:
            raise ValueError(f"Unknown split: {split}")

        dataset_names = sorted(set(self.ds["dataset_name"])) if len(self.ds) else []
        print(
            f"[StemRestorationDataset] stem={stem} split={split}: {len(self.ds)} samples "
            f"(dataset_names={dataset_names})"
        )

    def set_epoch(self, epoch: int) -> None:
        """New chunk offsets every epoch (called by DataModule.train_dataloader)."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int, _attempts: int = 0):
        row = self.ds[idx]
        try:
            input_path = row[self.input_column]
            target_path = row[self.target_column]
            start, frames = 0, -1
            if self.training_chunk_size is not None:
                # sf.info reads only the header, not the samples, so this stays cheap even for
                # long files.
                length = min(sf.info(input_path).frames, sf.info(target_path).frames)
                if length > self.training_chunk_size:
                    rng = np.random.default_rng(np.random.SeedSequence([self.chunk_seed, self._epoch, idx]))
                    start = int(rng.integers(0, length - self.training_chunk_size + 1))
                    frames = self.training_chunk_size
            input_wave, _ = sf.read(
                input_path, start=start, frames=frames, dtype="float32", always_2d=True
            )
            target_wave, _ = sf.read(
                target_path, start=start, frames=frames, dtype="float32", always_2d=True
            )
        except Exception as exc:  # pragma: no cover - guard corrupted files
            warnings.warn(
                f"Failed to load audio for idx={idx} "
                f"(input={row[self.input_column]!r}, target={row[self.target_column]!r}): "
                f"{type(exc).__name__}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            if _attempts >= len(self.ds):
                raise
            fallback_idx = (idx + 1 + _attempts) % len(self.ds)
            return self.__getitem__(fallback_idx, _attempts=_attempts + 1)

        input_wave = torch.from_numpy(input_wave.T)  # [C, T]
        target_wave = torch.from_numpy(target_wave.T)  # [C, T]
        length = min(input_wave.shape[-1], target_wave.shape[-1])
        return {
            "input": input_wave[..., :length],
            "target": target_wave[..., :length],
            "dataset_name": row.get("dataset_name", "unknown"),
            "label": row["label"],
        }

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        length = min(item["input"].shape[-1] for item in batch)
        return {
            "input": torch.stack([item["input"][..., :length] for item in batch]),
            "target": torch.stack([item["target"][..., :length] for item in batch]),
            "dataset_name": [item["dataset_name"] for item in batch],
            "label": [item["label"] for item in batch],
        }