"""
Random segment dataset utilities for multi-source audio training.

The helpers here make it easy to sample aligned fixed-length excerpts from
datasets that store multiple audio file paths per example—ideal for source
separation or remix tasks.  A convenience loader for the saved MSS8S dataset is
included so training code can directly iterate over random fixed-length chunks.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from datasets import Audio, Dataset, DatasetDict
from torch.utils.data import Dataset as TorchDataset
from .helpers import get_absolute_path
from ..config_updates import EIGHT_STEMS

# The "other" stem (residual of sources that cannot be mapped to one of the 8 instrument
# categories) is never loaded and never added to a training mixture: it is ignored in the
# activity masks and the stem mapping, and excluded from random mixing.
EXCLUDED_STEMS = {"other"}
MIX_STEMS = [stem for stem in EIGHT_STEMS if stem not in EXCLUDED_STEMS]

try:  # pragma: no cover - optional dependency
    import librosa
except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
    raise ImportError("librosa is required for random segment loading.") from exc

first_run = True
first_run_get_item = True


def _active_start_offsets(mask: np.ndarray, chunk_seconds: int = 10, min_active_fraction: float = 0.8) -> np.ndarray:
    """Return start indices where >= min_active_fraction of the next chunk_seconds are active.
    Falls back to the index/indices with the highest window activity when none meet the threshold
    (e.g. for audio shorter than chunk_seconds)."""
    if len(mask) == 0:
        return np.array([], dtype=np.intp)
    cumsum = np.concatenate([[0.0], np.cumsum(mask.astype(float))])
    ends = np.minimum(np.arange(len(mask)) + chunk_seconds, len(mask))
    window_sum = cumsum[ends] - cumsum[np.arange(len(mask))]
    valid = np.where(window_sum >= min_active_fraction * chunk_seconds)[0]
    if len(valid) > 0:
        return valid
    return np.where(window_sum == window_sum.max())[0]


def _count_active_seconds(waveform: torch.Tensor, sr: int, threshold_db: float = -40.0) -> int:
    """Count seconds in a waveform whose peak-channel RMS exceeds threshold_db."""
    n_seconds = waveform.shape[1] // sr
    if n_seconds == 0:
        return 0
    chunk = waveform[:, :n_seconds * sr].reshape(waveform.shape[0], n_seconds, sr)
    rms = torch.sqrt(torch.mean(chunk.float() ** 2, dim=2))  # (channels, n_seconds)
    rms_db = 20.0 * torch.log10(rms.max(dim=0).values + 1e-10)  # (n_seconds,)
    return int((rms_db > threshold_db).sum().item())


def _make_fill_gate(
    waveform: torch.Tensor,
    sr: int,
    rng: np.random.Generator,
    threshold_db: float = -40.0,
) -> torch.Tensor:
    """Return a (1, T) binary gate: 1 where filler should be applied, 0 elsewhere.

    For each contiguous inactive region the gate is 1 between a uniformly random
    onset and a uniformly random offset within that region.  No fades are applied —
    abrupt onsets are natural in music and using them prevents the model from
    learning any systematic transition pattern introduced by the fill process.
    """
    T = waveform.shape[1]
    n_seconds = T // sr
    gate = torch.zeros(T, dtype=torch.float32)
    if n_seconds == 0:
        return gate.unsqueeze(0)

    chunk = waveform[:, :n_seconds * sr].reshape(waveform.shape[0], n_seconds, sr)
    rms = torch.sqrt(torch.mean(chunk.float() ** 2, dim=2))
    rms_db = 20.0 * torch.log10(rms.max(dim=0).values + 1e-10)
    active = rms_db > threshold_db  # (n_seconds,) bool tensor

    # Collect contiguous inactive regions as sample ranges [r_start, r_end)
    regions: List[Tuple[int, int]] = []
    s = 0
    while s < n_seconds:
        if not active[s].item():  # if second s is not active
            start_s = s
            while s < n_seconds and not active[s].item():  # as long as we're in an inactive region
                s += 1
            regions.append((start_s * sr, min(s * sr, T)))  # save inactive region
        else:
            s += 1

    for r_start, r_end in regions:  # loop over inactive regions
        r_len = r_end - r_start  # get length of inactive region r in samples
        if r_len < 1:
            continue
        onset = r_start + int(rng.integers(0, min(sr, r_len)))
        remaining = r_end - onset
        offset = onset + int(rng.integers(max(1, remaining - min(sr, remaining)), remaining + 1))
        gate[onset:offset] = 1.0

    return gate.unsqueeze(0)  # (1, T) — broadcasts over channels


class MultiAudioRandomSegmentDataset(TorchDataset):
    """
    Sample fixed-length segments from multiple audio paths stored in a HF split.

    Parameters
    ----------
    split:
        The key of the dataset split to load.
    columns:
        Sequence of dataset columns that contain audio file paths.
    duration_column:
        Either a single column name holding a shared duration (seconds) or a list
        matching ``columns`` containing per-file durations.  ``None`` disables
        duration hints.
    sr:
        Target sampling rate for loaded audio.
    mono:
        Whether to load audio as mono.  When ``False`` the original channel layout
        is preserved.
    training_chunk_size:
        Length of the sampled segment (samples).
    seed:
        Base seed for random offset generation.
    resample_each_epoch:
        If ``True``, calling :meth:`set_epoch` will change the sampled offsets each
        epoch (useful when paired with a distributed sampler).
    dataset_names:
        Optional sequence of dataset names to filter by. If provided, only samples
        with ``dataset_name`` matching one of the provided names will be included.
        If ``None``, all samples in the split are used.
    """

    def __init__(
        self,
        split: str,
        columns: Sequence[str],
        duration_column: Optional[Union[str, Sequence[str]]] = None,
        sr: int = 48_000,
        mono: bool = False,
        training_chunk_size: int = 480000,
        seed: int = 17,
        resample_each_epoch: bool = True,
        root: Optional[Union[str, Path]] = None,
        dataset_names: Optional[Sequence[str]] = None,
        all_dataset_names_except: Optional[Sequence[str]] = None,
        prefix_path: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> None:
        if len(columns) == 0:
            raise ValueError("columns must contain at least one audio column")
        self.split = split
        # Use provided root, or fall back to default from config_updates
        if root is None:
            raise Exception()
        else:
            root = Path(root)
        self.ds = load_mss8s_split(split, root=root)

        self.prefix_path = Path(prefix_path) if prefix_path is not None else None

        # Filter by dataset_names / all_dataset_names_except if provided
        if dataset_names is not None and all_dataset_names_except is not None:
            raise ValueError("dataset_names and all_dataset_names_except are mutually exclusive")
        if dataset_names is not None:
            dataset_names_set = set(dataset_names)
            if len(dataset_names_set) == 0:
                raise ValueError("dataset_names cannot be an empty list")
            # Get available dataset names before filtering for error message
            if 'dataset_name' in self.ds.column_names:
                available_names = set(self.ds['dataset_name'])
            else:
                available_names = None
            # Filter the dataset to only include samples with matching dataset_name
            original_len = len(self.ds)
            self.ds = self.ds.filter(lambda example: example.get("dataset_name") in dataset_names_set)
            if len(self.ds) == 0:
                available_msg = f"Available dataset names in split '{split}': {available_names}" if available_names is not None else "dataset_name column not found"
                raise ValueError(
                    f"No samples found with dataset_name in {dataset_names}. "
                    f"{available_msg}"
                )
            print(f"Filtered dataset from {original_len} to {len(self.ds)} samples using datasets: {dataset_names}")
        elif all_dataset_names_except is not None:
            exclude_set = set(all_dataset_names_except)
            if len(exclude_set) == 0:
                raise ValueError("all_dataset_names_except cannot be an empty list")
            original_len = len(self.ds)
            self.ds = self.ds.filter(lambda example: example.get("dataset_name") not in exclude_set)
            if len(self.ds) == 0:
                raise ValueError(
                    f"all_dataset_names_except={list(exclude_set)} excluded all samples in split '{split}'"
                )
            print(f"Filtered dataset from {original_len} to {len(self.ds)} samples excluding datasets: {list(exclude_set)}")

        self.columns = list(columns)
        self.duration_column = duration_column
        self.sr = int(sr)
        self.mono = bool(mono)
        self.training_chunk_size = int(training_chunk_size)
        self.training_chunk_seconds = self.training_chunk_size / self.sr
        self.resample_each_epoch = bool(resample_each_epoch)

        self._base_seed = int(seed)
        self._epoch = 0
        self._target_frames = self.training_chunk_size
        self._required_sampling_warning_emitted = False

        self.activity_masks_path = str(root) + f"_metadata/activity_masks_v1.jsonl"
        
        # Load activity masks data
        activity_masks_path = Path(self.activity_masks_path)
        if not activity_masks_path.exists():
            raise Exception(f"Activity masks file not found: {activity_masks_path}")

        print(f"Loading activity masks / metadata from JSON file {self.activity_masks_path} ... ")
        self.metadata_dict: Dict[int, Dict[str, Any]] = {}
        with open(activity_masks_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                metadata_index = int(data.get("original_idx"))
                self.metadata_dict[metadata_index] = {
                    stem: mask for stem, mask in data['masks'].items() if stem not in EXCLUDED_STEMS
                }
        print("Loading done.")
        
        # Initialize buffer of valid samples (samples with max_n_of_stems_present >= 2)
        print("Building buffer of valid samples...")
        self._valid_samples_buffer: List[int] = []
        self._buffer_size = 100
        self._populate_valid_buffer()
        print(f"Buffer populated with {len(self._valid_samples_buffer)} valid samples.")
        # Counter for samples with max_n_of_stems_present < 2 (buffer fallbacks per epoch).
        # Printed and reset in set_epoch(). Note: with DataLoader num_workers > 0 only the
        # main process's dataset is updated; worker copies have separate counters.
        self._buffer_fallback_count = 0

    def _metadata_index(self, idx: int, sample: Optional[Dict[str, Any]] = None) -> int:
        """Return stable metadata key for a sample (original_idx if available)."""
        if sample is None:
            sample = self.ds[int(idx)]
        return int(sample.get("original_idx"))


    def _is_valid_sample(self, idx: int) -> bool:
        """Check if a sample has max_n_of_stems_present >= 2. For multi-stem training, there should be at least 2 stems (includes other) to separate."""
        sample = self.ds[int(idx)]
        metadata_idx = self._metadata_index(idx, sample)
        metadata_of_sample = self.metadata_dict.get(metadata_idx)
        if metadata_of_sample is None:
            return False
        
        mask_lengths = [
            len(v['activity_mask'])
            for _, v in metadata_of_sample.items()
            if v is not None and 'activity_mask' in v
        ]
        n_seconds = int(min(mask_lengths)) if len(mask_lengths) > 0 else 0
        activity_masks_of_all_stems_dict = {
            stem: v['activity_mask'][:n_seconds]
            if v is not None and 'activity_mask' in v
            else [False] * n_seconds
            for stem, v in metadata_of_sample.items()
        }
        
        if len(activity_masks_of_all_stems_dict) == 0:
            return False
        
        activity_masks_of_all_stems_list = list(activity_masks_of_all_stems_dict.values())
        activity_masks_tensor = torch.stack([torch.tensor(mask) for mask in activity_masks_of_all_stems_list])
        n_of_stems_present_over_time = activity_masks_tensor.sum(dim=0)
        max_n_of_stems_present = n_of_stems_present_over_time.max().item() if n_of_stems_present_over_time.numel() > 0 else 0
        
        return max_n_of_stems_present >= 2
    
    def _populate_valid_buffer(self) -> None:
        """Populate the buffer with valid sample indices."""
        # Try to find valid samples by checking random indices
        rng = np.random.default_rng(self._base_seed)
        checked_indices = set()
        max_attempts = min(1000, len(self.ds) * 2)  # Check up to 1000 indices or 2x dataset size
        
        for _ in range(max_attempts):
            if len(self._valid_samples_buffer) >= self._buffer_size:
                break
            
            # Try random indices first, then sequential if needed
            if len(checked_indices) < len(self.ds):
                idx = rng.integers(0, len(self.ds))
                while idx in checked_indices and len(checked_indices) < len(self.ds):
                    idx = rng.integers(0, len(self.ds))
                checked_indices.add(idx)
            else:
                # If we've checked all indices, break
                break
            
            if self._is_valid_sample(idx):
                self._valid_samples_buffer.append(idx)
        
        # If buffer is still not full, try sequential search
        if len(self._valid_samples_buffer) < self._buffer_size:
            for idx in range(len(self.ds)):
                if len(self._valid_samples_buffer) >= self._buffer_size:
                    break
                if idx not in checked_indices and self._is_valid_sample(idx):
                    self._valid_samples_buffer.append(idx)
        
        # If we still don't have enough, duplicate entries to fill buffer
        if len(self._valid_samples_buffer) == 0:
            raise ValueError("No valid samples found in dataset (all samples have max_n_of_stems_present < 2)")
        
        # Ensure buffer has at least buffer_size entries by duplicating if necessary
        while len(self._valid_samples_buffer) < self._buffer_size:
            needed = self._buffer_size - len(self._valid_samples_buffer)
            self._valid_samples_buffer.extend(self._valid_samples_buffer[:needed])

    def _load_window(self, path: str, offset_s: float) -> torch.Tensor:
        """Load an aligned segment and pad/trim to the target length."""
        target_frames = self._target_frames
        wav = None
        try:
            wav, _ = librosa.load(
                path,
                sr=self.sr,
                mono=self.mono,
                offset=float(offset_s),
                duration=self.training_chunk_seconds,
            )
        except Exception as exc:  # pragma: no cover - guard corrupted files
            import warnings
            warnings.warn(f"Failed to load audio at {path!r}: {exc}", RuntimeWarning, stacklevel=2)
            channels = 1 if self.mono else 2
            wav = np.zeros((channels, self._target_frames), dtype=np.float32)

        if wav.ndim == 1:
            wav = np.expand_dims(wav, axis=0)

        # If mono=False and we got mono audio, duplicate to stereo
        if not self.mono and wav.shape[0] == 1:
            wav = np.repeat(wav, 2, axis=0)

        channels, frames = wav.shape
        if frames < target_frames:
            pad = np.zeros((channels, target_frames - frames), dtype=wav.dtype)
            wav = np.concatenate([wav, pad], axis=1)
        elif frames > target_frames:
            wav = wav[:, :target_frames]
        return torch.from_numpy(wav).to(torch.float32)


    # --------------------------------------------------------------------- access
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self._load_8stem_item(int(idx) % len(self.ds))

    def _load_8stem_item(self, actual_idx: int) -> Dict[str, Any]:
        """Load one training item from a known 8_stem dataset index."""
        rng = self._rng(actual_idx)
        sample = self.ds[actual_idx]

        metadata_idx = self._metadata_index(actual_idx, sample)

        # handle durations of the stems in this sample
        duration = sample[self.duration_column]

        # get activity masks for all stems in this sample
        metadata_of_sample = self.metadata_dict.get(metadata_idx)

        # Guard: sample not present in activity masks (e.g. newly added samples not yet processed).
        # Fall back to a pre-validated buffer entry instead of crashing.
        if metadata_of_sample is None:
            self._buffer_fallback_count += 1
            buffer_idx = rng.integers(0, len(self._valid_samples_buffer))
            valid_idx = int(self._valid_samples_buffer[buffer_idx])
            return self._load_8stem_item(valid_idx)

        mask_lengths = [
            len(v['activity_mask'])
            for _, v in metadata_of_sample.items()
            if v is not None and 'activity_mask' in v
        ]
        n_seconds = int(min(mask_lengths)) if len(mask_lengths) > 0 else 0
        activity_masks_of_all_stems_dict = {
            stem: v['activity_mask'][:n_seconds]
            if v is not None and 'activity_mask' in v
            else [False] * n_seconds
            for stem, v in metadata_of_sample.items()
        }

        all_stems = list(activity_masks_of_all_stems_dict.keys())

        # create a tensor with all the activity_masks for this sample, shape: (num_stems, num_seconds)
        activity_masks_of_all_stems_list = list(activity_masks_of_all_stems_dict.values())
        activity_masks_tensor = torch.stack([torch.tensor(mask) for mask in activity_masks_of_all_stems_list])

        # sum across stems to get number of stems present at each second, shape: (num_seconds,)
        n_of_stems_present_over_time = activity_masks_tensor.sum(dim=0)

        # find the maximum number of stems present at any second in this sample
        max_n_of_stems_present = n_of_stems_present_over_time.max().item() if n_of_stems_present_over_time.numel() > 0 else 0

        if max_n_of_stems_present < 2:
            # Sample from the buffer of valid samples instead of recursing
            self._buffer_fallback_count += 1
            buffer_idx = rng.integers(0, len(self._valid_samples_buffer))
            valid_idx = int(self._valid_samples_buffer[buffer_idx])
            return self._load_8stem_item(valid_idx)
        elif max_n_of_stems_present == 2:
            n_of_stems_to_use = 2
        else:
            # randomly choose how many stems to use for this sample (between 2 and the maximum number of stems present, inclusive)
            n_of_stems_to_use = rng.integers(2, max_n_of_stems_present + 1)

        # see which offsets you could use (at least 80% of the 10-second window must be active)
        valid_mask = (n_of_stems_present_over_time >= n_of_stems_to_use).numpy()
        possible_start_seconds = _active_start_offsets(valid_mask, chunk_seconds=int(self.training_chunk_seconds))
        assert len(possible_start_seconds) != 0, "len(possible_start_seconds) should never be 0!"

        # randomly choose an offset from the possible offsets
        offset_s = rng.choice(possible_start_seconds)

        # offset_s may fall in the inactive 20% of the window, so find a reference second within
        # the window that is guaranteed to have >= n_of_stems_to_use stems active simultaneously
        window_end = min(int(offset_s) + int(self.training_chunk_seconds), activity_masks_tensor.shape[1])
        valid_secs_in_window = (n_of_stems_present_over_time[int(offset_s):window_end] >= n_of_stems_to_use).nonzero().squeeze(1)
        ref_second = int(offset_s) + valid_secs_in_window[rng.integers(0, len(valid_secs_in_window)).item()].item()
        stems_present_at_offset = [stem for stem in all_stems if activity_masks_tensor[all_stems.index(stem), ref_second] == 1]

        # choose n_stems_to_use stems from the stems that are present at the chosen offset
        chosen_stems = rng.choice(stems_present_at_offset, size=n_of_stems_to_use, replace=False)

        # randomly shift offset_s by up to +/- 0.5 seconds for extra variation
        offset_s_shifted = offset_s + rng.uniform(-0.5, 0.5)
        offset_s_shifted = max(0.0, offset_s_shifted)
        offset_s_shifted = min(float(offset_s_shifted), duration - self.training_chunk_seconds - 1e-6)

        # create the item dict
        item: Dict[str, Any] = {
            "index": actual_idx,
            "offset_seconds": offset_s_shifted,
            "training_chunk_size": self.training_chunk_size,
            "dataset_name": sample["dataset_name"]
        }

        for col in self.columns:
            relative_path = sample[col]
            path = self._get_absolute_path(relative_path) if relative_path else None

            if col in chosen_stems:
                waveform = self._load_window(path, offset_s_shifted)
            else:
                channels = 1 if self.mono else 2
                waveform = torch.zeros((channels, self._target_frames), dtype=torch.float32)

            item[col] = {
                "waveform": waveform,
                "path": relative_path,
                "is_present": col in chosen_stems,
            }

        return item

    def get_item_with_offset(self, idx: int, offset_seconds: float) -> Dict[str, Any]:
        """
        Return a sample for the given validation index at the exact offset (same
        structure as __getitem__). Stems present at int(offset_seconds) are loaded;
        others get zero waveforms. Use with validation_samples_path for fixed wandb uploads.
        """
        idx = int(idx)
        sample = self.ds[idx]
        metadata_idx = self._metadata_index(idx, sample)

        # handle durations of the stems in this sample
        duration = sample[self.duration_column]

        # clamp offset so we don't go past the end of the track
        offset_s = int(offset_seconds)
        offset_s_shifted = max(0.0, min(float(offset_seconds), duration - self.training_chunk_seconds - 1e-6))

        # get activity masks for all stems in this sample
        metadata_of_sample = self.metadata_dict.get(metadata_idx)

        mask_lengths = [
            len(v['activity_mask'])
            for _, v in metadata_of_sample.items()
            if v is not None and 'activity_mask' in v
        ]
        n_seconds = int(min(mask_lengths)) if len(mask_lengths) > 0 else 0
        activity_masks_of_all_stems_dict = {
            stem: v['activity_mask'][:n_seconds]
            if v is not None and 'activity_mask' in v
            else [False] * n_seconds
            for stem, v in metadata_of_sample.items()
        }
        all_stems = list(activity_masks_of_all_stems_dict.keys())

        # create a tensor with all the activity_masks for this sample, shape: (num_stems, num_seconds)
        activity_masks_of_all_stems_list = list(activity_masks_of_all_stems_dict.values())
        activity_masks_tensor = torch.stack([torch.tensor(mask) for mask in activity_masks_of_all_stems_list])

        # get the stems that are present at the chosen offset
        stems_present_at_offset = [stem for stem in all_stems if offset_s < activity_masks_tensor.shape[1] and activity_masks_tensor[all_stems.index(stem), offset_s] == 1]
        chosen_stems = set(stems_present_at_offset) if stems_present_at_offset else set()

        # create the item dict
        item: Dict[str, Any] = {
            "index": idx,
            "offset_seconds": offset_s_shifted,
            "training_chunk_size": self.training_chunk_size,
            "dataset_name": sample["dataset_name"]
        }

        for col in self.columns:
            relative_path = sample[col]
            path = self._get_absolute_path(relative_path) if relative_path else None

            if col in chosen_stems:
                waveform = self._load_window(path, offset_s_shifted)
            else:
                channels = 1 if self.mono else 2
                waveform = torch.zeros((channels, self._target_frames), dtype=torch.float32)

            item[col] = {
                "waveform": waveform,
                "path": relative_path,
                "is_present": col in chosen_stems,
            }

        return item

    # ------------------------------------------------------------------ metadata
    def set_epoch(self, epoch: int) -> None:
        """Set the logical epoch for deterministic resampling."""
        if epoch > 0:
            print(f"[Dataset] Epoch {epoch - 1}: samples with max_n_of_stems_present < 2 (buffer fallback): {self._buffer_fallback_count}")
            self._buffer_fallback_count = 0
        else:
            self._buffer_fallback_count = 0
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.ds)

    # ------------------------------------------------------------------ internals
    def _rng(self, idx: int) -> np.random.Generator:
        """Deterministic RNG for (epoch, index). Do not mix in DataLoader worker id:
        which worker serves an index can vary across runs and would change segments."""
        epoch = self._epoch if self.resample_each_epoch else 0
        return np.random.default_rng(
            np.random.SeedSequence([self._base_seed, epoch, idx])
        )

    def _get_absolute_path(self, relative_path: str) -> str:
        """Convert a relative path from the dataset to an absolute path."""
        return get_absolute_path(relative_path, self.prefix_path)

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "index": [b["index"] for b in batch],
            "offset_seconds": [b["offset_seconds"] for b in batch],
            "training_chunk_size": [b["training_chunk_size"] for b in batch],
            "dataset_name": [b["dataset_name"] for b in batch],
        }
        for col in self.columns:
            if col not in batch[0].keys():
                continue
            waveforms = [b[col]["waveform"] for b in batch]
            paths = [b[col]["path"] for b in batch]
            present = [b[col].get("is_present", True) for b in batch]
            padded, lengths = _pad_stack(waveforms)
            out[col] = {
                "waveform": padded,
                "lengths": lengths,
                "paths": paths,
                "is_present": present,
            }
        return out


class MultiAudioRandomSegmentMixDataset(MultiAudioRandomSegmentDataset):
    """
    Sample fixed-length segments from multiple audio paths stored in a HF split.

    Parameters
    ----------
    split:
        The key of the dataset split to load.
    columns:
        Sequence of dataset columns that contain audio file paths.
    duration_column:
        Either a single column name holding a shared duration (seconds) or a list
        matching ``columns`` containing per-file durations.  ``None`` disables
        duration hints.
    sr:
        Target sampling rate for loaded audio.
    mono:
        Whether to load audio as mono.  When ``False`` the original channel layout
        is preserved.
    training_chunk_size:
        Length of the sampled segment (samples).
    seed:
        Base seed for random offset generation.
    resample_each_epoch:
        If ``True``, calling :meth:`set_epoch` will change the sampled offsets each
        epoch (useful when paired with a distributed sampler).
    dataset_names:
        Optional sequence of dataset names to filter by. If provided, only samples
        with ``dataset_name`` matching one of the provided names will be included.
        If ``None``, all samples in the split are used.
    """

    def __init__(
        self,
        split: str,
        columns: Sequence[str],
        duration_column: Optional[Union[str, Sequence[str]]] = None,
        sr: int = 48_000,
        mono: bool = False,
        training_chunk_size: int = 480000,
        seed: int = 17,
        resample_each_epoch: bool = True,
        root: Optional[Union[str, Path]] = None,
        dataset_names: Optional[Sequence[str]] = None,
        all_dataset_names_except: Optional[Sequence[str]] = None,
        prefix_path: Optional[Union[str, Path]] = None,
        mix_probability_start: Optional[float] = None,
        mix_probability_end: Optional[float] = None,
        mix_probability_total_epochs: Optional[int] = None,
        num_stems_to_mix: int = 8,
        check_chunk_activity: bool = True,
        boost_rare_stems: bool = False,
        boost_stems: Optional[Sequence[str]] = None,
        boost_factor: float = 3.0,
        **kwargs,
    ) -> None:
        """
        mix_probability_start:
            Mixing probability at epoch 0. Required — must be set in the config.
        mix_probability_end:
            Mixing probability at epoch ``mix_probability_total_epochs``. Required.
        mix_probability_total_epochs:
            Number of epochs over which the probability is linearly interpolated
            from start to end. After this epoch the probability stays at end. Required.
        boost_rare_stems:
            When True, the stems listed in ``boost_stems`` are sampled more often
            than others, both when picking the anchor stem from an existing sample
            and when selecting additional stems to mix in. This increases the
            frequency of rare stems (e.g. orchestral, percussions, synthesizers)
            in training batches without changing the dataset itself.
        boost_stems:
            Stems to over-sample. Defaults to ["orchestral", "percussions", "synthesizers"].
        boost_factor:
            Relative weight multiplier for boosted stems (e.g. 3.0 means 3× more
            likely to be chosen vs. a non-boosted stem of equal natural frequency).
        """

        super().__init__(
            split=split,
            columns=columns,
            duration_column=duration_column,
            sr=sr,
            mono=mono,
            training_chunk_size=training_chunk_size,
            seed=seed,
            resample_each_epoch=resample_each_epoch,
            root=root,
            dataset_names=dataset_names,
            all_dataset_names_except=all_dataset_names_except,
            prefix_path=prefix_path,
            **kwargs
        )

        if mix_probability_start is None:
            raise ValueError("mix_probability_start must be provided in the config (no default)")
        if mix_probability_end is None:
            raise ValueError("mix_probability_end must be provided in the config (no default)")
        if mix_probability_total_epochs is None:
            raise ValueError("mix_probability_total_epochs must be provided in the config (no default)")

        self._mix_probability_start = float(np.clip(mix_probability_start, 0.0, 1.0))
        self._mix_probability_end = float(np.clip(mix_probability_end, 0.0, 1.0))
        self._mix_probability_total_epochs = int(mix_probability_total_epochs)
        self._mix_probability = self._mix_probability_start
        self.num_stems_to_mix = num_stems_to_mix

        self.boost_rare_stems = bool(boost_rare_stems)
        self._boost_stems_set: set = set(boost_stems) if boost_stems is not None else {"orchestral", "percussions", "synthesizers"}
        self._boost_factor = float(boost_factor)
        if self.boost_rare_stems:
            print(f"Rare-stem boosting enabled: stems={sorted(self._boost_stems_set)}, factor={self._boost_factor}")

        print(
            f"[MixDataset] mix_probability schedule: {self._mix_probability_start:.4f} -> {self._mix_probability_end:.4f} "
            f"over {self._mix_probability_total_epochs} epochs (linear)"
        )

        file_path = str(root) + f"_metadata/stem_mapping_v1.jsonl"
        with open(file_path) as f:
            loaded_mapping = json.load(f)

        # Build mapping between stable original_idx and the current (possibly filtered)
        # dataset index space used by self.ds.
        self._original_idx_to_filtered_idx: Dict[int, int] = {}
        for filtered_idx in range(len(self.ds)):
            sample = self.ds[filtered_idx]
            original_idx = self._metadata_index(filtered_idx, sample)
            self._original_idx_to_filtered_idx[original_idx] = filtered_idx

        # Convert mapping indices into current self.ds index space.
        self.stem_to_indices_mapping = {}
        for stem_name, original_indices in loaded_mapping.items():
            if stem_name in EXCLUDED_STEMS:
                continue
            filtered_indices = [
                self._original_idx_to_filtered_idx[int(original_idx)]
                for original_idx in original_indices
                if int(original_idx) in self._original_idx_to_filtered_idx
            ]
            self.stem_to_indices_mapping[stem_name] = filtered_indices

        self.check_chunk_activity = bool(check_chunk_activity)

        # Optional per-sample weights from WeightedRandomSampler (data scheduler).
        # When set by the DataModule, internal sampling (get_stem and __getitem__ retry) uses these weights.
        self._sample_weights: Optional[np.ndarray] = None

        self._fill_initial_below: int = 0
        self._fill_still_below_after: List[int] = [0, 0, 0]  # after fill attempt 1, 2, 3

        print(f"Info: Using MultiAudioRandomSegmentMixDataset: split: {split}")

    @property
    def mix_probability(self) -> float:
        """The current mixing probability (updated each epoch by the linear schedule)."""
        return self._mix_probability

    def get_mix_probability_bounds(self) -> Tuple[float, float]:
        """Return (mix_probability_start, mix_probability_end)."""
        return self._mix_probability_start, self._mix_probability_end

    def set_mix_probability(self, probability: float) -> None:
        """Override the scheduled probability for this epoch."""
        self._mix_probability = float(np.clip(probability, 0.0, 1.0))

    def reset_mix_probability(self) -> None:
        """Reset probability to the start value."""
        self._mix_probability = self._mix_probability_start

    def set_epoch(self, epoch: int) -> None:
        if epoch > 0:
            print(
                f"[MixDataset] Epoch {epoch - 1}: initial waveform below 80%: {self._fill_initial_below} | "
                f"still below after fill 1: {self._fill_still_below_after[0]} | "
                f"still below after fill 2: {self._fill_still_below_after[1]} | "
                f"still below after fill 3 (final): {self._fill_still_below_after[2]}"
            )
        self._fill_initial_below = 0
        self._fill_still_below_after = [0, 0, 0]

        # Linear schedule: interpolate from start to end over mix_probability_total_epochs
        t = min(float(epoch) / self._mix_probability_total_epochs, 1.0) if self._mix_probability_total_epochs > 0 else 1.0
        self._mix_probability = float(np.clip(
            self._mix_probability_start + t * (self._mix_probability_end - self._mix_probability_start),
            0.0, 1.0,
        ))
        print(
            f"[MixDataset] Epoch {epoch}: mix_probability={self._mix_probability:.4f} "
            f"(schedule {self._mix_probability_start:.4f} -> {self._mix_probability_end:.4f} "
            f"over {self._mix_probability_total_epochs} epochs)"
        )

        super().set_epoch(epoch)

    def _possible_offsets(self, activity_mask: list) -> np.ndarray:
        """Return valid start-second candidates for a stem.

        check_chunk_activity=True  (option 1): offsets where >=80% of the
            training chunk is active — same as the standalone helper.
        check_chunk_activity=False (option 2): any second that is active in the
            mask, regardless of what the surrounding chunk looks like.
        """
        mask_arr = np.array(activity_mask)
        if self.check_chunk_activity:
            offsets = _active_start_offsets(mask_arr, chunk_seconds=int(self.training_chunk_seconds))
            assert len(offsets) != 0, "len(possible_start_seconds) should never be 0!"
            return offsets
        active = np.where(mask_arr)[0]
        return active if len(active) > 0 else np.arange(len(mask_arr))

    def _stem_weights(self, stems: Sequence[str]) -> Optional[np.ndarray]:
        """Return normalised sampling weights for a list of stem names.
        Returns None when boosting is disabled (caller should use uniform sampling).
        """
        if not self.boost_rare_stems:
            return None
        w = np.array([self._boost_factor if s in self._boost_stems_set else 1.0 for s in stems], dtype=np.float64)
        w /= w.sum()
        return w

    def set_sampler_weights(self, weights: Optional[np.ndarray]) -> None:
        """
        Set per-sample weights used for internal sampling (get_stem and __getitem__ retry).
        Should be called by the DataModule when using the data scheduler so that
        stem/retry sampling respects the same dataset-wise distribution as the
        WeightedRandomSampler. Pass None to clear (uniform sampling).
        """
        if weights is not None and len(weights) != len(self.ds):
            raise ValueError(f"weights length {len(weights)} must match dataset size {len(self.ds)}")
        self._sample_weights = weights

    def _load_window_with_fill(
        self,
        stem_name: str,
        path: Optional[str],
        offset_s: float,
        rng: np.random.Generator,
        max_fill: int = 3,
    ) -> torch.Tensor:
        """Load a stem waveform, mixing in extra samples until >= 80% of the chunk is active.
        Bounded by max_fill iterations so it always terminates."""
        waveform = self._load_window(path, offset_s)
        n_chunk = int(self.training_chunk_seconds)
        min_active = int(0.8 * n_chunk)  # 8 out of 10 seconds

        if _count_active_seconds(waveform, self.sr) < min_active:
            self._fill_initial_below += 1
            for fill_idx in range(max_fill):
                fill_result = self.get_stem(stem_name, rng)
                filler_path = self._get_absolute_path(fill_result[0]) if fill_result[0] else None
                if filler_path is None:
                    continue

                filler_waveform = self._load_window(filler_path, fill_result[1])
                gate = _make_fill_gate(waveform, self.sr, rng)
                waveform = waveform + filler_waveform * gate
                if _count_active_seconds(waveform, self.sr) >= min_active:
                    break
                self._fill_still_below_after[fill_idx] += 1

        return waveform

    def get_stem(self, stem_name: str, rng: np.random.Generator):
        global first_run
        
        indices = self.stem_to_indices_mapping[stem_name]
        if len(indices) == 0:
            raise ValueError(f"No available samples for stem '{stem_name}' after dataset filtering.")
        if self._sample_weights is not None:
            w = np.asarray(self._sample_weights)[indices]
            w_sum = w.sum()
            if w_sum > 0:
                p = w / w_sum
                idx = int(rng.choice(indices, p=p))

                if first_run: 
                    print("first_run, set(w): ", set(w), " w_sum: ", w_sum, " set(p): ", set(p))
                    first_run = False
            else:
                idx = int(rng.choice(indices))
        else:
            idx = int(rng.choice(indices))

        if first_run: 
            print("first_run: get_stem: no weights set!")
            first_run = False

        sample = self.ds[idx]
        metadata_idx = self._metadata_index(idx, sample)

        # get activity masks for all stems in this sample
        metadata_of_sample = self.metadata_dict.get(metadata_idx)

        # Resolve activity mask, retrying with a different index if the stem is absent from
        # this sample's metadata (can happen when the stem mapping and activity_masks_v1.jsonl
        # are out of sync after a dataset update).
        stem_entry = metadata_of_sample.get(stem_name) if metadata_of_sample is not None else None
        activity_mask_of_stem = stem_entry.get('activity_mask') if stem_entry is not None else None

        if activity_mask_of_stem is None:
            for _retry in range(10):
                idx = int(rng.choice(indices))
                sample = self.ds[idx]
                metadata_idx = self._metadata_index(idx, sample)
                metadata_of_sample = self.metadata_dict.get(metadata_idx)
                stem_entry = metadata_of_sample.get(stem_name) if metadata_of_sample is not None else None
                activity_mask_of_stem = stem_entry.get('activity_mask') if stem_entry is not None else None
                if activity_mask_of_stem is not None:
                    break
            else:
                raise ValueError(
                    f"No valid activity mask found for stem '{stem_name}' after 10 retries. "
                    f"The stem mapping (stem_mapping_v1.jsonl) may be out of sync with the "
                    f"current activity_masks_v1.jsonl."
                )

        possible_start_seconds = self._possible_offsets(activity_mask_of_stem)

        # randomly choose an offset from the possible offsets
        offset_s = rng.choice(possible_start_seconds)

        # optional idea: randomly shift offset_s by a few milliseconds to add some more variation
        offset_s_shifted = offset_s + rng.uniform(-0.5, 0.5)  # shift by up to +/- 0.5 seconds
        offset_s_shifted = max(0.0, offset_s_shifted)  #
        offset_s_shifted = min(float(offset_s_shifted), sample[self.duration_column] - self.training_chunk_seconds - 1e-6)  # ensure we don't go past the end of the track (subtract a small epsilon to avoid edge case)

        return sample[stem_name], offset_s_shifted, sample['dataset_name']

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        global first_run_get_item

        for loop_idx in range(100):

            idx = int(idx)
            rng = self._rng(idx)

            if loop_idx != 0:
                if self._sample_weights is not None:
                    p = np.asarray(self._sample_weights, dtype=np.float64)
                    p = p / p.sum()
                    idx = int(rng.choice(len(self.ds), p=p))
                else:
                    idx = int(rng.choice(len(self.ds)))

                if first_run_get_item:
                    print("first_run_get_item, used sample weights: ", self._sample_weights is None)
                    first_run_get_item = False

                rng = self._rng(idx)

            if rng.uniform() >= self._mix_probability:
                result = super().__getitem__(idx)
                n_present = sum(1 for col in result if isinstance(result[col], dict) and result[col].get('is_present'))
                if n_present >= 2:
                    return result
                continue

            relative_paths = {}
            offsets = {}
            dataset_names = {}

            # -------------------- get first stem --------------------
            sample = self.ds[idx]
            metadata_idx = self._metadata_index(idx, sample)

            # get activity masks for all stems in this sample
            metadata_of_sample = self.metadata_dict.get(metadata_idx)

            # Guard: sample not present in activity masks — skip this attempt.
            if metadata_of_sample is None:
                continue

            # get activity masks of present stems
            activity_masks_of_present_stems_dict = {stem: v['activity_mask'] for stem, v in metadata_of_sample.items() if v is not None and 'activity_mask' in v and sum(v['activity_mask']) > 0}
            if len(activity_masks_of_present_stems_dict) == 0:
                continue

            # choose one stem randomly from the present stems (biased toward rare stems when boosting)
            present_stems = list(activity_masks_of_present_stems_dict.keys())
            chosen_stem = rng.choice(present_stems, p=self._stem_weights(present_stems))

            activity_mask_of_chosen_stem = activity_masks_of_present_stems_dict[chosen_stem]
            possible_start_seconds = self._possible_offsets(activity_mask_of_chosen_stem)

            # randomly choose an offset from the possible offsets
            offset_s = rng.choice(possible_start_seconds)

            # optional idea: randomly shift offset_s by a few milliseconds to add some more variation
            offset_s_shifted = offset_s + rng.uniform(-0.5, 0.5)  # shift by up to +/- 0.5 seconds
            offset_s_shifted = max(0.0, offset_s_shifted)  # ensure non-negative offset
            offset_s_shifted = min(float(offset_s_shifted), sample[self.duration_column] - self.training_chunk_seconds - 1e-6)  # ensure we don't go past the end of the track (subtract a small epsilon to avoid edge case)

            # update the dicts with the chosen stem
            relative_paths[chosen_stem] = sample[chosen_stem]
            offsets[chosen_stem] = offset_s_shifted
            dataset_names[chosen_stem] = sample['dataset_name']

            # stems that can be mixed in (copy so we don't mutate the global MIX_STEMS)
            rest_of_stems = list(MIX_STEMS)
            rest_of_stems.remove(chosen_stem)

            # -------------------- select other stems to mix in --------------------
            amount_of_stems = rng.integers(1, min(self.num_stems_to_mix - 1, len(rest_of_stems) + 1))
            other_stems = rng.choice(rest_of_stems, size=amount_of_stems, replace=False, p=self._stem_weights(rest_of_stems))
            for stem in other_stems:
                result = self.get_stem(stem, rng)
                relative_paths[stem] = result[0]
                offsets[stem] = result[1]
                dataset_names[stem] = result[2]

            # create the item dict
            item: Dict[str, Any] = {
                "index": idx,
                "training_chunk_size": self.training_chunk_size,
            }

            for col in self.columns:
                if col in relative_paths:
                    path = self._get_absolute_path(relative_paths[col]) if relative_paths[col] else None
                    if self.check_chunk_activity:
                        waveform = self._load_window_with_fill(col, path, offsets[col], rng)
                    else:
                        waveform = self._load_window(path, offsets[col])
                    is_present = True
                else:
                    channels = 1 if self.mono else 2
                    waveform = torch.zeros((channels, self._target_frames), dtype=torch.float32)
                    is_present = False

                item[col] = {
                    "waveform": waveform,
                    "path": relative_paths.get(col, None),
                    "is_present": is_present,
                    "dataset_name": dataset_names.get(col, None)
                }

            n_present = sum(1 for val in item.values() if isinstance(val, dict) and val.get('is_present'))
            if n_present < 2:
                continue

            return item

        raise ValueError("Didn't find valid sample in getitem function of MultiAudioRandomSegmentMixDataset!")

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "index": [b["index"] for b in batch],
            "training_chunk_size": [b["training_chunk_size"] for b in batch]
        }

        for col in self.columns:
            if col not in batch[0].keys():
                continue
            waveforms = [b[col]["waveform"] for b in batch]
            paths = [b[col]["path"] for b in batch]
            present = [b[col].get("is_present", False) for b in batch]
            dataset_names = [b[col].get("dataset_name", None) for b in batch]
            padded, lengths = _pad_stack(waveforms)
            out[col] = {
                "waveform": padded,
                "lengths": lengths,
                "paths": paths,
                "is_present": present,
                "dataset_name": dataset_names
            }
        return out


def _pad_stack(tensors: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad waveforms with varying channel counts to a common (B, C_max, T)."""
    batch_size = len(tensors)
    channel_max = max(t.shape[0] for t in tensors)
    time = tensors[0].shape[1]
    padded = tensors[0].new_zeros((batch_size, channel_max, time))
    lengths = torch.full((batch_size,), time, dtype=torch.long)
    for idx, tensor in enumerate(tensors):
        channels = tensor.shape[0]
        padded[idx, :channels] = tensor
    return padded, lengths


def load_mss8s_split(split: str, root: Path = None) -> Dataset:
    """
    Load the specified MSS8S split and return a Hugging Face dataset.

    The dataset is stored on disk (via ``save_to_disk``).  This helper resolves the
    path and also ensures that audio columns remain as plain string paths (which is
    the format expected by :class:`MultiAudioRandomSegmentDataset`).
    """

    root = root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"MSS8S root not found: {root}")
    dataset_dict = DatasetDict.load_from_disk(str(root))
    if split not in dataset_dict:
        raise KeyError(f"Split '{split}' not present in MSS8S dataset (available: {list(dataset_dict.keys())})")

    dataset = dataset_dict[split]
    # The saved dataset stores raw string paths.  If someone later saves it with
    # Audio features, we cast them back to plain strings to keep compatibility.
    for column, feature in dataset.features.items():
        if isinstance(feature, Audio):
            dataset = dataset.cast_column(column, Audio(decode=False))
    return dataset


__all__ = [
    "MultiAudioRandomSegmentDataset",
    "load_mss8s_split",
]
