"""
MSRBench dataset utilities for multi-source audio training.

This module provides utilities to load the MSRBench dataset which stores audio
as bytes in the HuggingFace dataset format.
"""

from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Audio, Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset


try:  # pragma: no cover - optional dependency
    import librosa
    import soundfile as sf
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise ImportError("librosa and soundfile are required for audio loading.") from exc


def load_msr_bench(split: str, root: Path = None) -> Dataset:
    """
    Load the specified MSRBench split and return a Hugging Face dataset.

    The dataset is stored on disk (via ``save_to_disk``). Audio columns are kept
    with decode=False to access the raw bytes directly.
    """
    if root is None:
        raise Exception()
    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"MSRBench root not found: {root}")
    dataset_dict = DatasetDict.load_from_disk(str(root))
    if split not in dataset_dict:
        raise KeyError(f"Split '{split}' not present in MSRBench dataset (available: {list(dataset_dict.keys())})")

    dataset = dataset_dict[split]
    # Ensure audio columns are not decoded so we can access bytes directly
    for column, feature in dataset.features.items():
        if isinstance(feature, Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    return dataset


def _resolve_audio_path(relative_path: str, prefix_path: Optional[Path]) -> str:
    """Convert a relative path from the dataset to an absolute path (see
    MultiAudioFullSongDataset._get_absolute_path, which this backs)."""
    if not relative_path:
        return relative_path
    if os.path.isabs(relative_path):
        return relative_path
    if prefix_path is not None:
        return str(prefix_path / relative_path)
    return relative_path


def _load_audio(
    audio_data: Union[Dict[str, Any], str],
    sr: int,
    mono: bool,
    prefix_path: Optional[Path] = None,
) -> torch.Tensor:
    """Load one audio file from bytes (if available) or a file path, resampling/channel-
    normalizing to (sr, mono) (see MultiAudioFullSongDataset._load_full_audio, which this backs).
    `audio_data` may be a raw path string or an HF Audio dict; dict paths are resolved through
    `prefix_path` via `_resolve_audio_path` first.
    """
    if isinstance(audio_data, dict):
        if 'bytes' in audio_data and audio_data['bytes'] is not None:
            audio_bytes = audio_data['bytes']
            wav, wav_sr = sf.read(io.BytesIO(audio_bytes))
            if wav.ndim == 1:
                wav = wav.reshape(1, -1)
            else:
                wav = wav.T

            if wav_sr != sr:
                if wav.shape[0] == 1:
                    wav = librosa.resample(wav[0], orig_sr=wav_sr, target_sr=sr)
                    wav = wav.reshape(1, -1)
                else:
                    wav_resampled = []
                    for ch in range(wav.shape[0]):
                        wav_resampled.append(librosa.resample(wav[ch], orig_sr=wav_sr, target_sr=sr))
                    wav = np.stack(wav_resampled, axis=0)

            if mono and wav.shape[0] > 1:
                wav = np.mean(wav, axis=0, keepdims=True)
        elif 'path' in audio_data and audio_data['path']:
            path = _resolve_audio_path(audio_data['path'], prefix_path)
            wav, _ = librosa.load(path, sr=sr, mono=mono)
            if wav.ndim == 1:
                wav = np.expand_dims(wav, axis=0)
        else:
            raise ValueError("Audio data dict must contain either 'bytes' or 'path'")
    else:
        path = _resolve_audio_path(audio_data, prefix_path)
        wav, _ = librosa.load(path, sr=sr, mono=mono)
        if wav.ndim == 1:
            wav = np.expand_dims(wav, axis=0)

    if not mono and wav.shape[0] == 1:
        wav = np.repeat(wav, 2, axis=0)

    return torch.from_numpy(wav).to(torch.float32)


class MultiAudioFullSongDataset(TorchDataset):
    """
    Load full songs (no segmenting) for evaluation purposes.
    
    This dataset loads complete audio files without any segmenting, suitable for
    standard evaluation procedures like MUSDB18-HQ evaluation.
    
    MSRBench dataset structure:
    - Each sample has a mixture_audio (always present)
    - Each sample has a source_audio (the single source that's present)
    - Each sample has a source_label (label indicating which source it is)
    - Multiple stem columns (vocals_audio, bass_audio, etc.) with *_is_present flags
    - Only one source is present per sample
    
    Parameters
    ----------
    split:
        The key of the dataset split to load.
    columns:
        Optional sequence of dataset columns that contain audio file paths.
        If None, uses mixture_audio and source_audio.
    filter_dt_families:
        Optional sequence of DT families (DT0, DT1, ...) to filter the dataset by.
    presence_columns:
        Optional dict mapping column names to presence indicator column names.
    presence_suffix:
        Optional suffix to append to column names to find presence indicators.
        Defaults to '_is_present' for MSRBench.
    sr:
        Target sampling rate for loaded audio.
    mono:
        Whether to load audio as mono. When ``False`` the original channel layout
        is preserved.
    """
    
    def __init__(
        self,
        split: str,
        columns: Optional[Sequence[str]] = None,
        filter_dt_families: Optional[Sequence[str]] = None,
        presence_columns: Optional[Dict[str, str]] = None,
        presence_suffix: Optional[str] = None,
        sr: int = 48_000,
        mono: bool = False,
        root: Optional[Union[str, Path]] = None,
        prefix_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.split = split
        # Use provided root, or fall back to default from config_updates
        if root is None:
            raise Exception()
        else:
            root = Path(root)

        # Store prefix path for converting relative paths to absolute
        self.prefix_path = Path(prefix_path) if prefix_path is not None else None

        self.ds = load_msr_bench(split, root=root)

        # Filter by DT families if provided
        if filter_dt_families is not None: 
            ## first create a column for the DT family in the dataset
            self.ds = self.ds.map(lambda x: {
                'dt_family': x['clip_id'].split('_')[1] if '_' in x['clip_id'] and len(x['clip_id'].split('_')) > 1 else ''
            })

            # Filter the dataset to only include samples with matching DT families
            self.ds = self.ds.filter(lambda x: x['dt_family'] in filter_dt_families)

        # If columns not specified, use mixture_audio and source_audio
        if columns is None:
            self.columns = ['mixture_audio', 'source_audio']
        else:
            self.columns = list(columns)
        
        self.sr = int(sr)
        self.mono = bool(mono)
        self.presence_columns = presence_columns or {}
        # Default presence suffix for MSRBench is '_is_present'
        self.presence_suffix = presence_suffix if presence_suffix is not None else '_is_present'
    
    def _get_absolute_path(self, relative_path: str) -> str:
        """Convert a relative path from the dataset to an absolute path (via prefix_path)."""
        return _resolve_audio_path(relative_path, self.prefix_path)

    def __len__(self) -> int:
        return len(self.ds)
    
    def _is_present(self, row: Dict[str, Any], column: str) -> bool:
        if column in self.presence_columns:
            key = self.presence_columns[column]
            return bool(row.get(key, False))
        if self.presence_suffix:
            # For MSRBench, presence flags are like vocals_is_present (not vocals_audio_is_present)
            # So we need to remove _audio suffix if present before adding the presence suffix
            base_column = column.replace('_audio', '') if column.endswith('_audio') else column
            key = f"{base_column}{self.presence_suffix}"
            if key in row:
                return bool(row.get(key, False))
        return bool(row.get(column))
    
    def _load_full_audio(self, audio_data: Union[Dict[str, Any], str]) -> torch.Tensor:
        """
        Load the full audio file from bytes (if available) or file path.

        Parameters
        ----------
        audio_data:
            Either a dict with 'bytes' key containing audio bytes, or a string path
            to an audio file.
        """
        return _load_audio(audio_data, self.sr, self.mono, self.prefix_path)

    def _extract_stem(self, path: str) -> str:
        """
        Extract stem name from path.
        
        Parameters
        ----------
        path:
            File path that may contain extension like .zip
            
        Returns
        -------
        Stem name without extension
        """
        if not path:
            raise Exception()
        # Remove any extension (e.g., .zip, .wav, etc.)
        stem = path.split("/")[0]
        return stem
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[int(idx)]
        clip_id = row.get("clip_id", "")
        # Family = DT type from clip_id (e.g. "0_DT0" -> "DT0")
        family = clip_id.split("_")[1] if "_" in clip_id and len(clip_id.split("_")) > 1 else ""

        item: Dict[str, Any] = {
            "index": idx,
            "clip_id": clip_id,
            "family": family,
            "source_label": row.get("source_label", ""),
            "split": row.get("split", ""),
            "duration_seconds": row.get("duration_seconds", 0.0),
            "sample_rate": row.get("sample_rate", self.sr),
            "dataset_name": row.get("dataset_name", "msr_bench"),
        }

        # Load mixture_audio (always present)
        if 'mixture_audio' in row and row['mixture_audio']:
            mixture_data = row['mixture_audio']
            # Extract relative path for metadata
            if isinstance(mixture_data, dict):
                mixture_relative_path = mixture_data.get('path', '')
            else:
                mixture_relative_path = mixture_data if mixture_data else ''
            
            # Convert relative path to absolute using prefix_path
            mixture_absolute_path = self._get_absolute_path(mixture_relative_path) if mixture_relative_path else None
            mixture_waveform = self._load_full_audio(mixture_absolute_path) if mixture_absolute_path else None
            
            item['mixture'] = {
                "waveform": mixture_waveform,
                "path": mixture_relative_path,  # Store original relative path for metadata
                "is_present": True,
            }
        
        # Load source_audio (the single source that's present)
        if 'source_audio' in row and row['source_audio']:
            source_data = row['source_audio']
            # Extract relative path for metadata
            if isinstance(source_data, dict):
                source_relative_path = source_data.get('path', '')
            else:
                source_relative_path = source_data if source_data else ''
            
            # Convert relative path to absolute using prefix_path
            source_absolute_path = self._get_absolute_path(source_relative_path) if source_relative_path else None
            source_waveform = self._load_full_audio(source_absolute_path) if source_absolute_path else None

            source_label = row.get("source_label", "")
            
            item['source'] = {
                "waveform": source_waveform,
                "path": source_relative_path,  # Store original relative path for metadata
                "label": source_label,
                "is_present": True,
            }
        
        # Optionally load individual stem columns if requested
        for col_ in self.columns:
            if col_ in ['mixture_audio', 'source_audio']:
                # Already handled above
                continue

            stem = col_
            path = ''
            waveform = torch.zeros((2, 480000), dtype=torch.float32)

            col = f"{col_}_audio"
            present = self._is_present(row, col)
            audio_data = row.get(col, '')

            if audio_data and audio_data == '':
                audio_data = None

            if present and audio_data:
                # Extract path for metadata (could be in dict or direct string)
                if isinstance(audio_data, dict):
                    path = audio_data.get('path', '')
                else:
                    path = audio_data

                # Extract stem from path (e.g., "Bass.zip" -> "Bass")
                # CRITICAL: For MSRBench, we should use the column name as the stem identifier,
                # not the path, because the path might not match the column name.
                # The column name (e.g., 'vocals_audio') is the source of truth for which stem this is.
                # Only extract from path if path is available and we want to validate it matches.
                if path:
                    # path_stem = self._extract_stem(path)
                    # # Use column name as the stem identifier (remove _audio suffix if present)
                    col_key_for_stem = col.replace('_audio', '') if col.endswith('_audio') else col
                    # # Normalize both for comparison (case-insensitive)
                    # path_stem_lower = path_stem.lower()
                    # col_key_lower = col_key_for_stem.lower()
                    # # If they don't match, warn but use column name as source of truth
                    # if path_stem_lower != col_key_lower and path_stem_lower not in ['', 'none', 'null']:
                    #     raise ValueError(f"Stem mismatch in MSRBench dataset: column '{col}' (expected stem: '{col_key_for_stem}') has path with stem '{path_stem}'")
                    stem = col_key_for_stem  # Use column name as stem identifier
                
                    # Convert relative path to absolute using prefix_path
                    audio_absolute_path = self._get_absolute_path(path)
                    waveform = self._load_full_audio(audio_absolute_path) if audio_absolute_path else None
            else:
                # Create silent waveform with correct channel structure
                # Use mixture length as reference if available
                if 'mixture' in item:
                    ref_length = item['mixture']['waveform'].shape[-1]
                else:
                    ref_length = 1
                channels = 1 if self.mono else 2
                waveform = torch.zeros((channels, ref_length), dtype=torch.float32)
            
            # Use column name without _audio suffix for cleaner keys
            col_key = col.replace('_audio', '') if col.endswith('_audio') else col
            item[col_key] = {
                "waveform": waveform,
                "stem": stem,
                "path": path,
                "is_present": bool(present),
            }
        
        return item
    
    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Collate function that pads waveforms to the same length."""
        # Find the maximum length in the batch
        max_length = 0
        for b in batch:
            # Check mixture and source first
            if 'mixture' in b:
                waveform = b['mixture']['waveform']
                if waveform.shape[-1] > max_length:
                    max_length = waveform.shape[-1]
            if 'source' in b:
                waveform = b['source']['waveform']
                if waveform.shape[-1] > max_length:
                    max_length = waveform.shape[-1]
            # Check other columns
            for col in self.columns:
                if col in ['mixture_audio', 'source_audio']:
                    continue
                col_key = col.replace('_audio', '') if col.endswith('_audio') else col
                if col_key in b:
                    waveform = b[col_key]["waveform"]
                    if waveform.shape[-1] > max_length:
                        max_length = waveform.shape[-1]
        
        # Pad all waveforms to the same length
        out: Dict[str, Any] = {
            "index": [b["index"] for b in batch],
            "clip_id": [b.get("clip_id", "") for b in batch],
            "family": [b.get("family", "") for b in batch],
            "source_label": [b.get("source_label", "") for b in batch],
            "split": [b.get("split", "") for b in batch],
            "duration_seconds": [b.get("duration_seconds", 0.0) for b in batch],
            "dataset_name": [b.get("dataset_name", "msr_bench") for b in batch],
        }
        
        # Collate mixture
        if 'mixture' in batch[0]:
            waveforms = []
            paths = []
            lengths = []
            for b in batch:
                waveform = b['mixture']['waveform']
                lengths.append(waveform.shape[-1])
                if waveform.shape[-1] < max_length:
                    pad_length = max_length - waveform.shape[-1]
                    waveform = F.pad(waveform, (0, pad_length), mode='constant', value=0.0)
                waveforms.append(waveform)
                paths.append(b['mixture'].get('path', ''))
            out['mixture'] = {
                "waveform": torch.stack(waveforms, dim=0),
                "lengths": torch.tensor(lengths, dtype=torch.long),
                "paths": paths,
            }
        
        # Collate source
        if 'source' in batch[0]:
            waveforms = []
            paths = []
            labels = []
            lengths = []
            for b in batch:
                waveform = b['source']['waveform']
                lengths.append(waveform.shape[-1])
                if waveform.shape[-1] < max_length:
                    pad_length = max_length - waveform.shape[-1]
                    waveform = F.pad(waveform, (0, pad_length), mode='constant', value=0.0)
                waveforms.append(waveform)
                paths.append(b['source'].get('path', ''))
                labels.append(b['source'].get('label', ''))
            out['source'] = {
                "waveform": torch.stack(waveforms, dim=0),
                "lengths": torch.tensor(lengths, dtype=torch.long),
                "paths": paths,
                "labels": labels,
            }
        
        # Collate other columns
        for col in self.columns:
            if col in ['mixture_audio', 'source_audio']:
                continue
            col_key = col.replace('_audio', '') if col.endswith('_audio') else col
            if col_key in batch[0]:
                waveforms = []
                stems = []
                paths = []
                present = []
                lengths = []
                for b in batch:
                    waveform = b[col_key]["waveform"]
                    lengths.append(waveform.shape[-1])
                    if waveform.shape[-1] < max_length:
                        pad_length = max_length - waveform.shape[-1]
                        waveform = F.pad(waveform, (0, pad_length), mode='constant', value=0.0)
                    waveforms.append(waveform)
                    stems.append(b[col_key].get("stem", ""))
                    paths.append(b[col_key].get("path", ""))
                    present.append(b[col_key].get("is_present", False))
                out[col_key] = {
                    "waveform": torch.stack(waveforms, dim=0),
                    "lengths": torch.tensor(lengths, dtype=torch.long),
                    "stems": stems,
                    "paths": paths,
                    "is_present": present,
                }
        
        return out


class MSRBenchRestorationValDataset(TorchDataset):
    """
    Real (not synthetically degraded) mixture-restoration validation pairs, built directly from
    MSRBench, for the mixture-restoration GAN's validation
    (configs/mixture_restoration/apollo_mixture_restoration.yaml).

    Rationale: the previous unofficial-val approach loaded only DT0 (clean) MSRBench mixtures and
    applied the *same synthetic* degradation_pipeline used at training time (see
    MixtureRestorationGANLightning.build_pair) -- so validation quality was gated on how well the
    synthetic degradation probabilities happened to match real degradations, not by measuring
    against real ones. MSRBench is a crossed design: the same origin content is degraded once per
    DT family (clip_id "<origin_id>_DT<n>", e.g. "0_DT0" .. "0_DT12" all share origin file "0";
    see MultiAudioFullSongDataset's docstring), so for each origin file, its DT0 mixture *is* the
    clean ground truth for every other DT-family mixture of that same file. This class pairs them
    up directly -- one (degraded_mixture, clean_mixture) sample per (origin_id, family) -- so
    unofficial validation exercises real MSRBench degradations across every family, comparable in
    kind (just cheaper, via subset_fraction) to the official full-MSRBench evaluation, with no
    degradation-probability tuning needed to make it representative. DT0 is included by default:
    its pair is (clean, clean) -- an identity-mapping check that the model doesn't degrade
    already-clean input -- see exclude_dt_families to drop it.

    Parameters
    ----------
    split:
        The key of the dataset split to load.
    subset_fraction:
        Fraction in (0, 1] of *origin* files to keep. Selection is stratified by construction:
        since every kept origin file contributes one sample per DT family, every family ends up
        with exactly the same sample count -- unlike sampling clips independently per family,
        which would scramble the origin_id x DT_family crossed design and confound content
        difficulty with per-family differences.
    subset_seed:
        Seed for the subset_fraction origin-file selection. Deterministic across reloads.
    exclude_dt_families:
        DT families to exclude from the *degraded* side of each pair. Empty by default (DT0
        included, paired against itself as an identity-mapping check). Pass ["DT0"] to drop the
        identity case and validate only on genuinely degraded input.
    sr, mono, root, prefix_path:
        Same meaning as MultiAudioFullSongDataset.
    columns:
        Accepted and discarded, not used (this dataset only ever loads `mixture_audio`):
        src/train.py::main() unconditionally force-injects a `columns` list (derived from
        lightning_module.args.n_sources) into every train/val/test dataset's args dict, before
        any pipeline-specific `.enabled` gate -- every dataset class in this repo must accept it
        even when unused. See StemRestorationDataset for the same convention.
    """

    def __init__(
        self,
        split: str,
        subset_fraction: float = 1.0,
        subset_seed: int = 42,
        exclude_dt_families: Optional[Sequence[str]] = None,
        sr: int = 48_000,
        mono: bool = False,
        root: Optional[Union[str, Path]] = None,
        prefix_path: Optional[Union[str, Path]] = None,
        columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.split = split
        if root is None:
            raise Exception()
        root = Path(root)

        self.prefix_path = Path(prefix_path) if prefix_path is not None else None
        self.sr = int(sr)
        self.mono = bool(mono)

        if not (0.0 < subset_fraction <= 1.0):
            raise ValueError(f"subset_fraction must be in (0, 1], got {subset_fraction}")

        ds = load_msr_bench(split, root=root)

        origin_to_family_mixture: Dict[str, Dict[str, Any]] = {}
        for row in ds:
            clip_id = row["clip_id"]
            origin_id, _, family = clip_id.partition("_DT")
            family = f"DT{family}"
            origin_to_family_mixture.setdefault(origin_id, {})[family] = row["mixture_audio"]

        origin_ids = sorted(o for o, families in origin_to_family_mixture.items() if "DT0" in families)
        if len(origin_ids) == 0:
            raise ValueError(f"No origin files with a DT0 (clean) mixture found in {root} (split={split})")

        if subset_fraction < 1.0:
            subset_threshold = int(subset_fraction * 10_000)

            def _keep_origin(origin_id: str) -> bool:
                digest = hashlib.md5(f"{subset_seed}_{origin_id}".encode("utf-8")).hexdigest()
                return (int(digest, 16) % 10_000) < subset_threshold

            origin_ids = [o for o in origin_ids if _keep_origin(o)]

        exclude = set(exclude_dt_families or [])
        self.pairs: List[Tuple[str, str, Any, Any]] = [
            (origin_id, family, mixture_audio, origin_to_family_mixture[origin_id]["DT0"])
            for origin_id in origin_ids
            for family, mixture_audio in origin_to_family_mixture[origin_id].items()
            if family not in exclude
        ]
        if len(self.pairs) == 0:
            raise ValueError("No (origin_id, DT family) pairs left after subset/exclude filtering")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        origin_id, family, degraded_audio, clean_audio = self.pairs[idx]
        return {
            "origin_id": origin_id,
            "family": family,
            "clip_id": f"{origin_id}_{family}",
            "degraded_waveform": _load_audio(degraded_audio, self.sr, self.mono, self.prefix_path),
            "clean_waveform": _load_audio(clean_audio, self.sr, self.mono, self.prefix_path),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        max_length = max(
            max(b["degraded_waveform"].shape[-1], b["clean_waveform"].shape[-1]) for b in batch
        )

        def _pad_stack(key: str) -> torch.Tensor:
            waveforms = []
            for b in batch:
                waveform = b[key]
                if waveform.shape[-1] < max_length:
                    waveform = F.pad(waveform, (0, max_length - waveform.shape[-1]))
                waveforms.append(waveform)
            return torch.stack(waveforms, dim=0)

        return {
            "origin_id": [b["origin_id"] for b in batch],
            "family": [b["family"] for b in batch],
            "clip_id": [b["clip_id"] for b in batch],
            "mixture": {"waveform": _pad_stack("clean_waveform")},
            "degraded_mixture": {"waveform": _pad_stack("degraded_waveform")},
        }


__all__ = [
    "MultiAudioFullSongDataset",
    "MSRBenchRestorationValDataset",
    "load_msr_bench",
]
