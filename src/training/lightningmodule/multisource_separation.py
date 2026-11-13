import random
import math

import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import time
import logging
from torch.utils.data import Subset, Dataset
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from typing import List, Dict, Any, Optional, Sequence

from src.config_updates import EIGHT_STEMS
from src.dsp.torch_mastering import MasteringPipeline


# Set up logger for wandb capture
logger = logging.getLogger(__name__)

from src.training.lightningmodule.base_lightningmodule import BaseLightningModule
from src.training.metrics.fad_clap import FADCLAPMetric
from src.training.metrics.multi_mel_snr import multi_mel_snr
from src.training.metrics import log_wmse as _log_wmse_module
from src.training.metrics.additional_metrics import (
    L1Freq_metric,
    AuraSTFT_metric,
    AuraMRSTFT_metric,
    bleed_full,
)
from src.data.msr_bench import MultiAudioFullSongDataset as MultiAudioFullSongDatasetMSRBench


from torch.cuda.amp import autocast

import museval
import zimtohrli

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception as _skimage_import_error:
    skimage_ssim = None
    logger.warning(f"Warning: Failed to import scikit-image (mel_ssim will be skipped/NaN if requested): {_skimage_import_error}")

from src.dsp import MusicEffectPipeline, mix_stems_to_snr_range


class SubsetWithCollateFn(Subset):
    """
    A Subset wrapper that forwards the collate_fn attribute to the underlying dataset.
    
    This is needed because PyTorch's Subset class doesn't forward custom attributes
    like collate_fn, which are needed by the DataLoader.
    """
    def __init__(self, dataset: Dataset, indices: List[int]):
        super().__init__(dataset, indices)
        self._dataset = dataset
    
    def __getattr__(self, name):
        # Forward collate_fn and any other attributes to the underlying dataset
        if name == 'collate_fn' or hasattr(self._dataset, name):
            return getattr(self._dataset, name)
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


class MultiSourceSeparationLightning(BaseLightningModule):
    """
    Lightning module for multi-source separation.
    
    The model outputs all target sources simultaneously.
    For music source separation, this outputs: bass, drums, vocals, other
    
    Args:
        n_sources: Number of sources to separate (default: 4)
        progress_bar_metrics: List of full metric names to show in progress bar.
                             If None (default), all metrics appear in progress bar.
                             If list, only specified metrics appear in progress bar.
                             Example: ['step_train/loss', 'epoch_val_overall/si_snr'] to show only specific metrics.
    """
    def __init__(
            self,
            model: dict,
            loss: dict,
            optimizer: dict,
            lr_scheduler: dict = None,
            n_sources: int = 4,
            val_metrics: list = None,
            test_metrics: list = None,
            fad_clap_batch_size: int = 4,
            fad_clap_max_samples: int = 0,  # 0 = all clips
            fad_clap_force_cpu: bool = False,
            sampling_rate: int = 48000,
            progress_bar_metrics: list = None,
            use_ema: bool = False,
            ema_decay: float = 0.9999,
            apply_music_effect_pipeline: bool = False,
            apply_mastering_pipeline: bool = False,
            music_effect_pipeline_dataset_names: Optional[Sequence[str]] = None,
            target_key: str = "waveform",
            debug: bool = False,
            ir_paths: str = None,
            music_effect_pipeline_config: dict = None,
            mastering_pipeline_config: dict = None,
            min_lufs: float = -70.0,
            training_chunk_size: int = 480000,
            snr_range: tuple[float, float] = None,
            gain_range: tuple[float, float] = None,
            train_other_stem: bool = True,
    ):
        super().__init__(
            model=model,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            use_ema=use_ema,
            ema_decay=ema_decay,
        )


        self.debug = debug
        self.n_sources = n_sources
        self.sampling_rate = sampling_rate
        self.train_other_stem = train_other_stem

        self.columns = self.get_columns()
        self._zimt_calculator = zimtohrli.Pyohrli()

        # Mel backend for mel_ssim / mel_kl_div metrics (mirrors PredictionEvaluator's setup).
        self._mel_metric_backend_available = False
        self._mel_transform = None
        self._db_transform = None
        try:
            self._mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sampling_rate, n_fft=1024, hop_length=512, n_mels=128,
                f_min=0, f_max=sampling_rate // 2, power=2.0,
            )
            self._db_transform = torchaudio.transforms.AmplitudeToDB(stype="power")
            self._mel_metric_backend_available = True
        except Exception as e:
            logger.warning(f"Warning: Failed to initialize mel backend ({e}); mel_ssim/mel_kl_div metrics will be skipped.")
        if skimage_ssim is None:
            logger.warning("Warning: scikit-image unavailable; mel_ssim metric will be skipped (NaN) if requested.")

        # Configuration for which metrics appear in progress bar
        # If None, use default behavior (all metrics in progress bar)
        # If list, only metrics in the list will appear in progress bar
        self.progress_bar_metrics = progress_bar_metrics

        self.apply_music_effect_pipeline = apply_music_effect_pipeline
        self.apply_mastering_pipeline = apply_mastering_pipeline
        self.music_effect_pipeline_dataset_names = (
            set(music_effect_pipeline_dataset_names) if music_effect_pipeline_dataset_names is not None else None
        )
        self.target_key = target_key

        if self.target_key == "aug_waveform" and not self.apply_music_effect_pipeline:
            raise Exception("Warning: target_key is 'aug_waveform' but apply_music_effect_pipeline is False. No augmentation will be applied.")

        self.min_lufs = min_lufs

        # Initialize stem presence tracking counters
        # Will be reset each epoch in on_train_epoch_start
        self.stem_presence_counts = None

        # Metrics to compute during validation and test.
        # Available: sdr, si_snr, multi_mel_snr, zimtohrli, log_wmse, fad_clap, l1_freq, aura_stft,
        # aura_mrstft, bleed, fullness, mel_ssim, mel_kl_div
        _DEFAULT_METRICS = ["sdr", "si_snr", "multi_mel_snr", "zimtohrli"]
        self.val_metrics = list(val_metrics) if val_metrics is not None else list(_DEFAULT_METRICS)
        self.test_metrics = list(test_metrics) if test_metrics is not None else list(_DEFAULT_METRICS)

        # Initialize FAD-CLAP metrics if requested.
        # 'fad_clap' is always removed from val_metrics/test_metrics — it is tracked
        # separately via self.val_fad_clap_metric / self.test_fad_clap_metric.
        if 'fad_clap' in self.val_metrics:
            self.val_metrics = [m for m in self.val_metrics if m != 'fad_clap']
            try:
                self.val_fad_clap_metric = FADCLAPMetric(
                    batch_size=fad_clap_batch_size,
                    max_samples_per_epoch=fad_clap_max_samples,
                    force_cpu=fad_clap_force_cpu
                )
                print("FAD-CLAP metric enabled (validation)")
            except Exception as e:
                print(f"Warning: Failed to initialize FAD-CLAP metric: {e}")
                self.val_fad_clap_metric = None
        else:
            self.val_fad_clap_metric = None

        if 'fad_clap' in self.test_metrics:
            self.test_metrics = [m for m in self.test_metrics if m != 'fad_clap']
            try:
                self.test_fad_clap_metric = FADCLAPMetric(
                    batch_size=fad_clap_batch_size,
                    max_samples_per_epoch=fad_clap_max_samples,
                    force_cpu=fad_clap_force_cpu
                )
                print("FAD-CLAP metric enabled (test)")
            except Exception as e:
                print(f"Warning: Failed to initialize FAD-CLAP metric: {e}")
                self.test_fad_clap_metric = None
        else:
            self.test_fad_clap_metric = None

        self.training_chunk_size = int(training_chunk_size)

        self.snr_range = snr_range
        self.gain_range = gain_range

        ######################## OUR AUGMENTATION CONFIGURATION ########################

        ###### MUSIC EFFECT PIPELINE
        # Build augmentation config with defaults from MusicEffectPipeline, overridden by yaml config if provided
        # Convert YAML lists to tuples for parameters that expect tuples
        music_effect_pipeline_config = music_effect_pipeline_config or {}

        def to_tuple(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return tuple(value)
            return value

        self.music_effect_pipeline = MusicEffectPipeline(
            sample_rate=sampling_rate,
            ir_paths=ir_paths,  # ir_paths can be None (disabled)
            noise_snr_db=to_tuple(music_effect_pipeline_config.get('noise_snr_db')),
            high_pass_hz=to_tuple(music_effect_pipeline_config.get('high_pass_hz')),
            ir_p=music_effect_pipeline_config.get('ir_p'),
            downsample_target_rate=music_effect_pipeline_config.get('downsample_target_rate'),
            enable_low_pass=music_effect_pipeline_config.get('enable_low_pass'),
            low_pass_p=music_effect_pipeline_config.get('low_pass_p'),
            low_pass_hz=to_tuple(music_effect_pipeline_config.get('low_pass_hz')),
            enable_reverb=music_effect_pipeline_config.get('enable_reverb'),
            reverb_p=music_effect_pipeline_config.get('reverb_p'),
            run_on_gpu=music_effect_pipeline_config.get('run_on_gpu', False),
        )

        ###### MASTERING PIPELINE
        mastering_pipeline_config = mastering_pipeline_config or {}

        self.mastering_pipeline = MasteringPipeline(
            sample_rate=sampling_rate,
            apply_prob=mastering_pipeline_config.get('apply_prob'),
            tp_db=mastering_pipeline_config.get('tp_db'),
            width_db=mastering_pipeline_config.get('width_db'),
            oversample=mastering_pipeline_config.get('oversample'),
            lookahead_ms=mastering_pipeline_config.get('lookahead_ms'),
            release_ms=mastering_pipeline_config.get('release_ms'),
            run_on_gpu=mastering_pipeline_config.get('run_on_gpu', False),
            enable_mp3=mastering_pipeline_config.get('enable_mp3', True),
        )
        ###############################################################################


    @property
    def _other_stem_idx(self):
        """Index of 'other' in self.columns when train_other_stem=False, else None."""
        if not self.train_other_stem and self.columns is not None and 'other' in self.columns:
            return list(self.columns).index('other')
        return None

    def get_columns(self):
        if self.n_sources == 9:
            return EIGHT_STEMS
        raise Exception(f"Unsupported number of stems: {self.n_sources}. Expected 9.")

    def get_length(self, batch):
        # get lengths of all waveforms
        waveform_lengths = []
        for column in self.columns:
            waveform_lengths.append(batch[column]['waveform'].shape[2])

        waveform_lengths = list(set(waveform_lengths))
        if len(waveform_lengths) != 1:
            raise ValueError(f"waveform_lengths: {waveform_lengths} should be of length 1! ")

        return max(self.training_chunk_size, waveform_lengths[0])

    def pad_or_trim_to_length(self, w: torch.Tensor, length: int) -> torch.Tensor:
        if w.shape[1] < length:
            return torch.nn.functional.pad(w, (0, length - w.shape[1]), mode='constant')
        return w[:, :length]

    def _get_stem_aug_apply_mask(self, batch, batch_size: int):
        if not self.apply_music_effect_pipeline:
            return [False] * batch_size
        if self.music_effect_pipeline_dataset_names is None:
            return [True] * batch_size

        if batch.get('dataset_name', None) is not None:
            dataset_names = batch['dataset_name']  # list of dimension batch_size

            if not isinstance(dataset_names, list) or len(dataset_names) != batch_size:
                raise ValueError(f"dataset_names: {dataset_names} should be a list of length {batch_size}")

            return [
                (dataset_names[i] in self.music_effect_pipeline_dataset_names)
                for i in range(batch_size)
            ]
        else:
            dataset_names_dict = {}  # dictionary of lists of dimension batch_size
            for col in self.columns:
                if batch.get(col, None) is not None and batch[col].get('dataset_name', None) is not None:
                    dataset_names_for_stem = batch[col]['dataset_name']

                    if not isinstance(dataset_names_for_stem, list) or len(dataset_names_for_stem) != batch_size:
                        raise ValueError(f"dataset_names: {dataset_names_for_stem} for column {col} should be a list of length {batch_size}")

                    dataset_names_dict[col] = [dataset_names_for_stem[i] in self.music_effect_pipeline_dataset_names for i in range(batch_size)]

            return dataset_names_dict


    def create_mixtures(self, batch):

        mixtures = []
        waveform_dict = {}
        aug_waveform_dict = {}
        processing_dtype = torch.float32

        original_length = self.get_length(batch)

        B = len(batch[self.columns[0]]['waveform'])
        stem_aug_apply_mask = self._get_stem_aug_apply_mask(batch, B)

        for idx in range(B):  # iterate over batch dimension
            with autocast(enabled=False):
                present_stems_dict = {}
                original_stems_dict = {}

                for column in self.columns:
                    multi_channel_waveform = batch[column]["waveform"][idx].to(dtype=processing_dtype)

                    # now batch[column]['is_present'][idx] really says if the source is present or not
                    if batch[column]['is_present'][idx]:
                        original_stems_dict[column] = multi_channel_waveform

                        apply_stem_aug_for_this_stem_of_this_sample = stem_aug_apply_mask[idx] if isinstance(stem_aug_apply_mask, list) else stem_aug_apply_mask[column][idx]
                        if apply_stem_aug_for_this_stem_of_this_sample:
                            aug_multi_channel_waveform = self.music_effect_pipeline(multi_channel_waveform.unsqueeze(0), self.sampling_rate).squeeze(0)
                        else:
                            aug_multi_channel_waveform = multi_channel_waveform

                        present_stems_dict[column] = aug_multi_channel_waveform

                reference_stem = random.choice(list(present_stems_dict.keys()))
                mixture, scales = mix_stems_to_snr_range(present_stems_dict, self.snr_range, reference_stem_key=reference_stem)

                scaled_augmented_stems = {k: v*scales[k] for k, v in present_stems_dict.items()}
                scaled_original_stems = {k: v*scales[k] for k, v in original_stems_dict.items()}  # should be correct

                # mixture augmentation
                if self.apply_mastering_pipeline:
                    aug_mixture = self.mastering_pipeline(mixture.unsqueeze(0), self.sampling_rate).squeeze(0)
                else:
                    aug_mixture = mixture

                max_val = torch.max(torch.abs(aug_mixture))
                norm_mixture = aug_mixture / max_val
                norm_stems = {k: v / max_val for k, v in scaled_augmented_stems.items()}
                norm_original_stems = {k: v / max_val for k, v in scaled_original_stems.items()}  # not sure if correct

                # rescale with self.gain_range=(0.5, 1.0)
                rescale = np.random.uniform(*self.gain_range)
                final_mixture = torch.nan_to_num(norm_mixture * rescale)
                final_stems = {k: torch.nan_to_num(v * rescale) for k, v in norm_stems.items()}
                final_original_stems = {k: torch.nan_to_num(v * rescale) for k, v in norm_original_stems.items()}  # also not sure

                # padding / trimming
                target_length = original_length
                final_mixture = self.pad_or_trim_to_length(final_mixture, target_length)

                for k in final_stems.keys():
                    final_stems[k] = self.pad_or_trim_to_length(final_stems[k], target_length)
                    final_original_stems[k] = self.pad_or_trim_to_length(final_original_stems[k], target_length)

                for column in self.columns:
                    if column not in aug_waveform_dict:
                        aug_waveform_dict[column] = []

                    if column not in waveform_dict:
                        waveform_dict[column] = []

                    if column in final_stems:
                        aug_waveform_dict[column].append(final_stems[column])
                    else:
                        aug_waveform_dict[column].append(batch[column]['waveform'][idx])

                    if column in final_original_stems:
                        waveform_dict[column].append(final_original_stems[column])
                    else:
                        waveform_dict[column].append(batch[column]['waveform'][idx])

                mixtures.append(final_mixture)

        for column in self.columns:
            batch[column]['aug_waveform'] = torch.stack(aug_waveform_dict[column])
            batch[column]['waveform'] = torch.stack(waveform_dict[column])

        batch['mixture'] = torch.stack(mixtures)

        return batch

    def _gather_dataset_names_across_processes(self, local_dataset_names: list) -> set:
        """
        Gather dataset names from all processes and return union of all names.
        
        Args:
            local_dataset_names: List of dataset names from this process
            
        Returns:
            Set of all unique dataset names from all processes
        """
        try:
            # Get world size (use Lightning's trainer if available, otherwise check distributed)
            if hasattr(self, 'trainer') and self.trainer is not None:
                world_size = self.trainer.world_size if hasattr(self.trainer, 'world_size') else 1
            else:
                world_size = torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1
            
            if world_size == 1:
                # Not using distributed training, return local dataset names
                return set(local_dataset_names)
            
            # Convert to a format we can gather (use a delimiter to join names)
            # We'll gather as a single string and split it
            local_str = ','.join(sorted(local_dataset_names)) if local_dataset_names else ''
            
            # For string gathering, we need to use all_gather_object which Lightning doesn't have directly
            # So we'll use the distributed version, but Lightning's all_gather handles the distributed check
            # For strings, we still need all_gather_object, so check if distributed is available
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                # Gather all strings using all_gather_object (needed for strings)
                gathered_strings = ['' for _ in range(world_size)]
                torch.distributed.all_gather_object(gathered_strings, local_str)
                
                # Combine all dataset names
                all_dataset_names = set()
                for ds_str in gathered_strings:
                    if ds_str:
                        all_dataset_names.update(ds_str.split(','))
                
                return all_dataset_names
            else:
                return set(local_dataset_names)
        except Exception as e:
            # If gathering fails, return local dataset names as fallback
            print(f"Warning: Failed to gather dataset names across processes: {e}. Using local dataset names only.")
            return set(local_dataset_names)
    
    def _gather_metric_values_across_processes(self, local_values: list) -> list:
        """
        Gather metric values from all processes and return aggregated list.
        
        Args:
            local_values: List of metric values from this process
            
        Returns:
            List of all metric values from all processes (aggregated)
        """

        try:
            # Get device from trainer or use default
            device = next(self.parameters()).device if list(self.parameters()) else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Convert to tensor for gathering
            if len(local_values) == 0:
                local_tensor = torch.tensor([], dtype=torch.float32, device=device)
            else:
                local_tensor = torch.tensor(local_values, dtype=torch.float32, device=device)
            
            # Get world size (use Lightning's trainer if available, otherwise check distributed)
            if hasattr(self, 'trainer') and self.trainer is not None:
                world_size = self.trainer.world_size if hasattr(self.trainer, 'world_size') else 1
            else:
                world_size = torch.distributed.get_world_size() if torch.distributed.is_available() and torch.distributed.is_initialized() else 1
            
            if world_size == 1:
                # Not using distributed training, return local values
                return local_values
            
            # Gather sizes from all processes using Lightning's all_gather
            # Lightning's all_gather concatenates along a new dimension, so [1] becomes [world_size, 1] or [world_size]
            local_size = torch.tensor([len(local_values)], dtype=torch.long, device=device)
            gathered_sizes_tensor = self.all_gather(local_size)  # Returns tensor of shape [world_size, 1] or [world_size]
            # Flatten in case it's [world_size, 1] instead of [world_size]
            if gathered_sizes_tensor.dim() > 1:
                gathered_sizes_tensor = gathered_sizes_tensor.flatten()
            sizes = gathered_sizes_tensor.cpu().tolist()
            # Ensure sizes is a flat list of integers
            if sizes and isinstance(sizes[0], list):
                sizes = [s[0] if isinstance(s, list) else s for s in sizes]
            max_size = max(sizes) if sizes else 0
            
            if max_size == 0:
                # No data from any process
                return []
            
            # Pad local tensor to max size (all processes need same shape for all_gather)
            if len(local_tensor) < max_size:
                padding = torch.zeros(max_size - len(local_tensor), dtype=local_tensor.dtype, device=device)
                local_tensor = torch.cat([local_tensor, padding])
            elif len(local_tensor) > max_size:
                # This shouldn't happen, but handle it
                local_tensor = local_tensor[:max_size]
            
            # Gather all tensors using Lightning's all_gather
            # Lightning's all_gather concatenates along a new dimension, so [max_size] becomes [world_size, max_size]
            gathered_tensors_tensor = self.all_gather(local_tensor)  # Returns tensor of shape [world_size, max_size]
            
            # Extract values from each process and remove padding
            all_values = []
            for i in range(world_size):
                size = sizes[i]
                if size > 0:
                    # Extract this process's tensor: gathered_tensors_tensor[i, :size]
                    values = gathered_tensors_tensor[i, :size].cpu().tolist()
                    all_values.extend(values)

            return all_values
        except Exception as e:
            # If gathering fails, return local values as fallback
            print(f"Warning: Failed to gather metrics across processes: {e}. Using local values only.")
            return local_values

    def _compute_fad_clap_score_across_processes(self, fad_metric: FADCLAPMetric) -> Dict[str, Any]:
        """
        Compute FAD-CLAP globally across ranks by gathering CLAP embeddings from all processes.

        FAD-CLAP is not additive, so we must merge embeddings before computing the Fréchet distance.
        """
        try:
            if fad_metric is None:
                return {}

            # Single-process shortcut
            if (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()):
                return fad_metric.compute()

            world_size = (
                self.trainer.world_size
                if hasattr(self, "trainer") and self.trainer is not None and hasattr(self.trainer, "world_size")
                else torch.distributed.get_world_size()
            )
            if world_size == 1:
                return fad_metric.compute()

            local_state = fad_metric.get_concatenated_embeddings_state()

            gathered_states: list = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(gathered_states, local_state)

            # print(f"len of entries of gathered_states: {len(gathered_states[0])}, {len(gathered_states[1])}, {len(gathered_states[2])}, {len(gathered_states[3])}")

            # Only rank 0 computes the Fréchet distances
            if not self._is_rank_0():
                return {}

            merged_target_by_source: Dict[int, list] = {}
            merged_output_by_source: Dict[int, list] = {}

            for st in gathered_states:
                if st is None:
                    continue
                for s, arr in st.get("target", {}).items():
                    if arr is None or getattr(arr, "size", 0) == 0:
                        continue
                    merged_target_by_source.setdefault(int(s), []).append(arr)
                for s, arr in st.get("output", {}).items():
                    if arr is None or getattr(arr, "size", 0) == 0:
                        continue
                    merged_output_by_source.setdefault(int(s), []).append(arr)

            fad_per_source: Dict[int, float] = {}
            all_target_parts: List[np.ndarray] = []
            all_output_parts: List[np.ndarray] = []

            common_sources = sorted(set(merged_target_by_source.keys()) & set(merged_output_by_source.keys()))
            for s in common_sources:
                tgt_parts = merged_target_by_source.get(s, [])
                out_parts = merged_output_by_source.get(s, [])
                if not tgt_parts or not out_parts:
                    continue
                tgt = np.concatenate(tgt_parts, axis=0)
                out = np.concatenate(out_parts, axis=0)

                score = fad_metric._calculate_frechet_distance(tgt, out)
                fad_per_source[s] = float(score) if score is not None else 0.0

                all_target_parts.append(tgt)
                all_output_parts.append(out)

            if not all_target_parts or not all_output_parts:
                return {"fad_clap": 0.0, "fad_clap_overall": 0.0, "fad_clap_per_source": {}}

            all_target = np.concatenate(all_target_parts, axis=0)
            all_output = np.concatenate(all_output_parts, axis=0)
            overall = fad_metric._calculate_frechet_distance(all_target, all_output)
            overall_value = float(overall) if overall is not None else 0.0

            return {
                "fad_clap": overall_value,
                "fad_clap_overall": overall_value,
                "fad_clap_per_source": fad_per_source,
            }
        except Exception as e:
            # Fallback: compute locally (rank 0 will log)
            if self._is_rank_0():
                print(f"Warning: Failed to compute global FAD-CLAP across ranks: {e}. Falling back to local compute.")
            return fad_metric.compute() if fad_metric is not None else {}
    
    def _is_rank_0(self) -> bool:
        """Check if current process is rank 0."""
        return (not torch.distributed.is_available() or 
                not torch.distributed.is_initialized() or 
                torch.distributed.get_rank() == 0)
    
    def _gather_and_log_metric(self, local_values: list, metric_key: str, 
                               prefix: str = 'epoch_val_overall', 
                               prog_bar: bool = False) -> float:
        """
        Gather metric values across processes, compute median, and log if on rank 0.
        
        Args:
            local_values: List of metric values from this process
            metric_key: Key for the metric (e.g., 'sdr', 'si_snr')
            prefix: Logging prefix (e.g., 'epoch_val_overall', 'epoch_val_sources/vocals')
            prog_bar: Whether to show in progress bar
            
        Returns:
            Median value (0.0 if no values available)
        """
        all_values = self._gather_metric_values_across_processes(local_values)
        # print(f"Gathering across ranks for {metric_key} calculation: {len(local_values)} local values, {len(all_values)} total values")
        if len(all_values) > 0:
            median_value = torch.tensor(all_values).median().item()
            if self._is_rank_0():
                self.log(f'{prefix}/{metric_key}', median_value,
                        prog_bar=prog_bar, logger=True, on_epoch=True, sync_dist=False)
            return median_value
        return 0.0
    
    def _compute_overall_metrics_from_per_source(self, per_source_dict: dict, metric_name: str) -> tuple:
        """
        Compute overall metrics: concatenated median, mean of medians, and mean of means (per source).

        Per-sample values in per_source_dict are already medians over time chunks. For each stem,
        mean_of_medians uses median across samples then mean across stems; mean_of_means uses
        mean across samples then mean across stems.

        Args:
            per_source_dict: Dictionary mapping source_idx to list of values
            metric_name: Name of metric (e.g., 'sdr', 'si_snr', 'multi_mel_snr')

        Returns:
            Tuple of (concatenated_median, mean_of_medians, mean_of_means, source_mean_by_idx)
            where source_mean_by_idx maps source_idx -> mean-across-samples for that stem
            (only for stems that had data).
        """
        # Collect all values across sources for concatenated median
        all_values = []
        for source_idx in range(self.n_sources):
            if source_idx in per_source_dict:
                all_values.extend(per_source_dict[source_idx])

        # Gather all values across processes
        all_values_global = self._gather_metric_values_across_processes(all_values)

        # Compute concatenated median
        concatenated_median = 0.0
        if len(all_values_global) > 0:
            concatenated_median = torch.tensor(all_values_global).median().item()

        # Mean of medians / mean of means: per stem, aggregate across samples; then mean across stems
        source_medians = []
        source_means = []
        source_mean_by_idx: Dict[int, float] = {}
        for source_idx in range(self.n_sources):
            local_source_values = per_source_dict.get(source_idx, [])
            all_source_values = self._gather_metric_values_across_processes(local_source_values)
            if len(all_source_values) > 0:
                t = torch.tensor(all_source_values)
                source_medians.append(t.median().item())
                source_means.append(t.mean().item())
                source_mean_by_idx[source_idx] = t.mean().item()

        mean_of_medians = 0.0
        if len(source_medians) > 0:
            mean_of_medians = torch.tensor(source_medians).mean().item()

        mean_of_means = 0.0
        if len(source_means) > 0:
            mean_of_means = torch.tensor(source_means).mean().item()

        return concatenated_median, mean_of_medians, mean_of_means, source_mean_by_idx
    
    def should_show_in_progress_bar(self, metric_name: str) -> bool:
        """
        Determine if a metric should appear in the progress bar.
        
        Args:
            metric_name: Full name of the metric (e.g., 'step_train/mr_stft_loss', 'epoch_val_overall/si_snr')
            
        Returns:
            bool: True if metric should appear in progress bar, False otherwise
        """
        if self.progress_bar_metrics is None:
            # Default behavior: show all metrics in progress bar
            return True
        else:
            # Only show metrics that are in the specified list
            return metric_name in self.progress_bar_metrics
    
    def _log_wandb_heartbeat(self, phase: str = "validation"):
        """
        Log a heartbeat signal to wandb to prevent false crash detection during long-running operations.
        
        This is especially important during validation and test phases when processing full songs
        can take several minutes without any other logging activity.
        
        Args:
            phase: Phase name (e.g., "validation", "test") for the heartbeat metric name
        """
        if not self._is_rank_0():
            return
        
        try:
            # Log a simple heartbeat metric that wandb will recognize as activity
            # Using a timestamp-based value ensures it's always unique and wandb sees it as new data
            heartbeat_value = time.time() % 1000  # Use modulo to keep value reasonable
            self.log(
                f'heartbeat/{phase}',
                heartbeat_value,
                prog_bar=False,
                logger=True,
                on_epoch=False,
                on_step=True,
                sync_dist=False,
                batch_size=1
            )
        except Exception:
            # Silently fail if logging doesn't work (e.g., wandb not initialized)
            pass

    def encode_target_sources(self, batch, target_key=None):
        sources = []
        target_key_to_use = target_key if target_key is not None else self.target_key

        for column in self.columns:
            waveform = batch[column][target_key_to_use]  # self.target_key is either "waveform" or "aug_waveform"
            sources.append(waveform)
        
        target_sources = torch.stack(sources, dim=1)  # [B, n_columns, C, T]
        
        batch["target_sources"] = target_sources
        return batch

    def training_step(self, batch_data_dict, batch_idx):
        """
        Process a training step.
        
        Args:
            batch_data_dict: Dictionary containing:
                - mixture: [batch_size, 2, waveform_length] stereo audio
                - target_sources: [batch_size, n_sources, 2, waveform_length] stereo audio
        
        Returns:
            batchsize: int
            loss_dict: dict of losses
        """
        batch_data_dict = self.create_mixtures(batch_data_dict)
        batch_data_dict = self.encode_target_sources(batch_data_dict)

        batchsize = batch_data_dict['mixture'].shape[0]

        # Resample mixture if enabled
        mixture = batch_data_dict['mixture']  # [bs, nch, wlen] at 48kHz
        input_dict = {'mixture': mixture}

        if self.debug:
            output_dict = dict(
                waveform=batch_data_dict['target_sources'] - 0.1,
                source_mask=torch.ones(batch_data_dict['target_sources'].shape[0], 4,
                           device=batch_data_dict['target_sources'].device, dtype=batch_data_dict['target_sources'].dtype)
            )
        else:
            output_dict = self.model(input_dict) # output_dict['waveform']: [bs, n_sources, 2, wlen] (at 48kHz for dual-STFT, 44.1kHz if resampling enabled)

        # Build per-source presence mask [B, S]
        source_presence_tensors = []
        all_columns = list(self.columns)
        
        for column in all_columns:
            if column in batch_data_dict:
                # stored as list[bool]
                source_presence_flags = batch_data_dict[column]["is_present"]
                source_presence_tensors.append(torch.tensor(source_presence_flags, dtype=torch.float32, device=batch_data_dict['mixture'].device))
            else:
                # Column doesn't exist, treat as not present
                B = batch_data_dict['mixture'].shape[0]
                source_presence_tensors.append(torch.zeros(B, dtype=torch.float32, device=batch_data_dict['mixture'].device))
        
        # source_mask consists of just a binary value for each source
        source_mask = torch.stack(source_presence_tensors, dim=1)  # [B, n_columns or n_columns+1]
        
        # Pad source_mask if needed (shouldn't happen if we handled "other" correctly)
        if source_mask.shape[1] < self.n_sources:
            n_padding = self.n_sources - source_mask.shape[1]
            B = source_mask.shape[0]
            padding = torch.zeros(B, n_padding, dtype=source_mask.dtype, device=source_mask.device)
            source_mask = torch.cat([source_mask, padding], dim=1)  # [B, n_sources]

        # Exclude 'other' stem from training if flag is set
        if self._other_stem_idx is not None:
            source_mask[:, self._other_stem_idx] = 0.0

        target_dict = {
            'waveform': batch_data_dict['target_sources'],
            'source_mask': source_mask
        }
        
        # Track stem presence statistics (is_present=True AND has actual signal)
        if self.stem_presence_counts is not None and self.columns is not None:
            target_sources = batch_data_dict['target_sources']  # [B, S, C, T]
            B, S, C, T = target_sources.shape
            
            # target_sources is stacked in the order of self.columns
            # So target_sources[:, stem_idx, :, :] corresponds to self.columns[stem_idx]
            for stem_idx, column in enumerate(self.columns):
                if stem_idx < S:
                    # Get source presence flags for this stem
                    if column in batch_data_dict:
                        source_presence_flags = batch_data_dict[column]["is_present"]  # List[bool]
                    else:
                        source_presence_flags = [False] * B
                    
                    # Get waveforms for this stem [B, C, T]
                    stem_waveforms = target_sources[:, stem_idx, :, :]
                    
                    # Check each sample in the batch
                    for batch_idx in range(B):
                        if source_presence_flags[batch_idx]:
                            # Check if waveform has actual signal (not all zeros)
                            waveform = stem_waveforms[batch_idx]  # [C, T]
                            # Compute RMS to check if there's actual signal
                            # Use a small threshold to avoid numerical noise
                            rms = torch.sqrt(torch.mean(waveform ** 2))
                            signal_threshold = 1e-6  # Very small threshold to detect any meaningful signal
                            
                            if rms > signal_threshold:
                                self.stem_presence_counts[column] += 1
        
        loss_dict = self.loss_func(output_dict, target_dict)

        # Pop per-source tensors before scalar iteration; they are logged separately
        per_source_l1 = loss_dict.pop('l1_loss_per_source', None)
        per_source_l1_weighted = loss_dict.pop('l1_loss_weighted_per_source', None)
        per_source_mr_stft = loss_dict.pop('mr_stft_loss_per_source', None)
        per_source_mr_stft_weighted = loss_dict.pop('mr_stft_loss_weighted_per_source', None)

        loss = loss_dict['loss']

        # check if loss is nan
        if torch.isnan(loss_dict['loss']):
            print("loss is NaN!!!")

        # Calculate per-source amplitude statistics
        pred_waveform = output_dict['waveform']  # [B, S, C, T]
        target_waveform = target_dict['waveform']  # [B, S, C, T]
        
        # Calculate mean absolute amplitude per source for predictions and ground truth
        # Shape: [B, S] - mean over channels and time
        pred_amp_per_source = pred_waveform.abs().mean(dim=(2, 3))  # [B, S]
        target_amp_per_source = target_waveform.abs().mean(dim=(2, 3))  # [B, S]
        
        # Calculate absolute difference
        amp_diff_per_source = torch.abs(pred_amp_per_source - target_amp_per_source)  # [B, S]
        
        # Average over batch dimension to get per-source statistics
        # Only average over samples where the source is present (using source_mask)
        # Masked mean: sum over batch, divide by number of active sources
        pred_amp_masked = pred_amp_per_source * source_mask  # [B, S]
        target_amp_masked = target_amp_per_source * source_mask  # [B, S]
        amp_diff_masked = amp_diff_per_source * source_mask  # [B, S]
        
        # Sum over batch and divide by number of active samples per source
        active_samples_per_source = source_mask.sum(dim=0).clamp_min(1.0)  # [S]
        pred_amp_mean_per_source = pred_amp_masked.sum(dim=0) / active_samples_per_source  # [S]
        target_amp_mean_per_source = target_amp_masked.sum(dim=0) / active_samples_per_source  # [S]
        amp_diff_mean_per_source = amp_diff_masked.sum(dim=0) / active_samples_per_source  # [S]
        
        # Log per-source amplitude statistics (both step and epoch level)
        if self.columns is not None:
            for source_idx, source_name in enumerate(self.columns):
                if source_idx < len(pred_amp_mean_per_source):
                    # Log predicted amplitude (step and epoch)
                    self.log(
                        f'amplitudes/{source_name}/predicted',
                        pred_amp_mean_per_source[source_idx].item(),
                        prog_bar=False,
                        logger=True,
                        on_epoch=True,
                        on_step=True,
                        sync_dist=True,
                        batch_size=batchsize
                    )
                    # Log ground truth amplitude (step and epoch)
                    self.log(
                        f'amplitudes/{source_name}/ground_truth',
                        target_amp_mean_per_source[source_idx].item(),
                        prog_bar=False,
                        logger=True,
                        on_epoch=True,
                        on_step=True,
                        sync_dist=True,
                        batch_size=batchsize
                    )
                    # Log absolute difference (step and epoch)
                    self.log(
                        f'amplitudes/{source_name}/absolute_difference',
                        amp_diff_mean_per_source[source_idx].item(),
                        prog_bar=False,
                        logger=True,
                        on_epoch=True,
                        on_step=True,
                        sync_dist=True,
                        batch_size=batchsize
                    )
            
            # Log averages across all sources (step and epoch)
            # Average over all sources (weighted by active samples per source)
            total_active_samples = active_samples_per_source.sum().clamp_min(1.0)
            pred_amp_average = (pred_amp_mean_per_source * active_samples_per_source).sum() / total_active_samples
            target_amp_average = (target_amp_mean_per_source * active_samples_per_source).sum() / total_active_samples
            amp_diff_average = (amp_diff_mean_per_source * active_samples_per_source).sum() / total_active_samples
            
            # Log average predicted amplitude
            self.log(
                'amplitudes/average/predicted',
                pred_amp_average.item(),
                prog_bar=False,
                logger=True,
                on_epoch=True,
                on_step=True,
                sync_dist=True,
                batch_size=batchsize
            )
            # Log average ground truth amplitude
            self.log(
                'amplitudes/average/ground_truth',
                target_amp_average.item(),
                prog_bar=False,
                logger=True,
                on_epoch=True,
                on_step=True,
                sync_dist=True,
                batch_size=batchsize
            )
            # Log average absolute difference
            self.log(
                'amplitudes/average/absolute_difference',
                amp_diff_average.item(),
                prog_bar=False,
                logger=True,
                on_epoch=True,
                on_step=True,
                sync_dist=True,
                batch_size=batchsize
            )

        # log all items in loss_dict
        epoc_dict = {f'epoch_train/{name}': val.item() for name, val in loss_dict.items()}
        step_dict = {f'step_train/{name}': val.item() for name, val in loss_dict.items()}
        
        # Log epoch metrics with appropriate progress bar settings
        for key, value in epoc_dict.items():
            show_in_progress_bar = self.should_show_in_progress_bar(key)
            self.log(key, value, prog_bar=show_in_progress_bar, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)
        
        # Log step metrics with appropriate progress bar settings
        for key, value in step_dict.items():
            show_in_progress_bar = self.should_show_in_progress_bar(key)
            self.log(key, value, prog_bar=show_in_progress_bar, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)

        # Per-source L1 and MR-STFT logging (epoch-level only)
        if self.columns is not None:
            log_kwargs = dict(prog_bar=False, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)
            if per_source_l1 is not None:
                n_src = min(per_source_l1.shape[0], len(self.columns))
                for i, name in enumerate(self.columns[:n_src]):
                    self.log(f'epoch_train_l1_loss/{name}', per_source_l1[i].item(), **log_kwargs)
                    self.log(f'epoch_train_l1_loss_weighted/{name}', per_source_l1_weighted[i].item(), **log_kwargs)
                self.log('epoch_train_l1_loss/total', per_source_l1[:n_src].sum().item(), **log_kwargs)
                self.log('epoch_train_l1_loss_weighted/total', per_source_l1_weighted[:n_src].sum().item(), **log_kwargs)
            if per_source_mr_stft is not None:
                n_src = min(per_source_mr_stft.shape[0], len(self.columns))
                for i, name in enumerate(self.columns[:n_src]):
                    self.log(f'epoch_train_mr_stft_loss/{name}', per_source_mr_stft[i].item(), **log_kwargs)
                    self.log(f'epoch_train_mr_stft_loss_weighted/{name}', per_source_mr_stft_weighted[i].item(), **log_kwargs)
                self.log('epoch_train_mr_stft_loss/total', per_source_mr_stft[:n_src].sum().item(), **log_kwargs)
                self.log('epoch_train_mr_stft_loss_weighted/total', per_source_mr_stft_weighted[:n_src].sum().item(), **log_kwargs)

        self.log_dict({"epoch/lr": self.optimizer.param_groups[0]['lr']}, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch_data_dict, batch_idx, dataloader_idx=0):
        """
        Process a validation step.
        
        Args:
            batch_data_dict: Dictionary containing:
                - mixture: [batch_size, 2, waveform_length] stereo audio
                - target_sources: [batch_size, n_sources, 2, waveform_length] stereo audio
            batch_idx: Index of the batch
            dataloader_idx: Index of the dataloader (0 for MSR Bench if both exist)
        
        Returns:
            batchsize: int
            loss_dict: dict of losses and metrics
        """
        # -------------------------------------------------------------------------------------
        # ----------------------------- STANDARD VALIDATION -----------------------------------
        # -------------------------------------------------------------------------------------

        is_msrbench_dataset = False
        val_dataset = self.trainer.datamodule.val_dataset

        if val_dataset is not None:
            # Unwrap Subset if present
            if isinstance(val_dataset, Subset):
                unwrapped_dataset = val_dataset.dataset
            else:
                unwrapped_dataset = val_dataset

            is_msrbench_dataset = isinstance(unwrapped_dataset, MultiAudioFullSongDatasetMSRBench)

        if not is_msrbench_dataset:
            batch_data_dict = self.create_mixtures(batch_data_dict)
            target_key_to_use = None
            default_value = None
        else:
            batch_data_dict['mixture'] = batch_data_dict['mixture']['waveform']
            target_key_to_use = "waveform"
            default_value = "msr_bench"

        batchsize = batch_data_dict['mixture'].shape[0]
        device = batch_data_dict['mixture'].device

        batch_data_dict = self.encode_target_sources(batch_data_dict, target_key=target_key_to_use)
        
        # Extract dataset_name from batch (if available)
        dataset_names = batch_data_dict.get('dataset_name', [default_value] * batchsize)
        if not isinstance(dataset_names, list):
            dataset_names = [dataset_names] if dataset_names is not None else [None] * batchsize

        # Resample mixture if enabled
        mixture = batch_data_dict['mixture']  # [bs, nch, wlen] at 48kHz
        input_dict = {
            'mixture': mixture, # [bs, nch, wlen]
        }

        with torch.no_grad():
            output_dict = self.model(input_dict) # {'waveform': [bs, n_sources, 2, wlen]}, output is in 48kHz again
        
        # For validation, only use source presence mask, no frame-level masking
        source_presence_tensors = []
        
        for column in list(self.columns):
            if column in batch_data_dict:
                source_presence_flags = batch_data_dict[column]["is_present"]
                source_presence_tensors.append(torch.tensor(source_presence_flags, dtype=torch.float32, device=batch_data_dict['mixture'].device))
            else:
                # Column doesn't exist, treat as not present
                B = batch_data_dict['mixture'].shape[0]
                source_presence_tensors.append(torch.zeros(B, dtype=torch.float32, device=batch_data_dict['mixture'].device))
        
        source_mask = torch.stack(source_presence_tensors, dim=1)  # [B, n_columns or n_columns+1]
        
        # Pad source_mask if needed (shouldn't happen if we handled "other" correctly)
        if source_mask.shape[1] < self.n_sources:
            n_padding = self.n_sources - source_mask.shape[1]
            B = source_mask.shape[0]
            padding = torch.zeros(B, n_padding, dtype=source_mask.dtype, device=source_mask.device)
            source_mask = torch.cat([source_mask, padding], dim=1)  # [B, n_sources]

        # Exclude 'other' stem from validation if flag is set
        val_other_idx = self._other_stem_idx
        if val_other_idx is not None:
            source_mask[:, val_other_idx] = 0.0

        # No frame masking in validation - evaluate on full audio
        target_dict = {
            'waveform': batch_data_dict['target_sources'],
            'source_mask': source_mask,
        }
        loss_dict = self.loss_func(output_dict, target_dict)

        # Update FAD-CLAP metric for validation (only computed once for whole batch)
        if self.val_fad_clap_metric is not None:
            self.val_fad_clap_metric.update(output_dict, target_dict)

        # Pop per-source tensors before scalar conversion; log them separately
        val_per_source_l1 = loss_dict.pop('l1_loss_per_source', None)
        val_per_source_l1_weighted = loss_dict.pop('l1_loss_weighted_per_source', None)
        val_per_source_mr_stft = loss_dict.pop('mr_stft_loss_per_source', None)
        val_per_source_mr_stft_weighted = loss_dict.pop('mr_stft_loss_weighted_per_source', None)

        loss_dict = {k: v.item() for k,v in loss_dict.items()}

        # Compute metrics using 1-second chunking (consistent with full song validation)
        output_wav = output_dict['waveform']  # [B, S, C, T]
        target_wav = target_dict['waveform']  # [B, S, C, T]
        B, S, C, T = output_wav.shape

        # Process each sample in the batch
        for b in range(B):
            dataset_name = dataset_names[b] if b < len(dataset_names) else None
            
            # Extract this sample: [S, C, T]
            output_sample = output_wav[b]  # [S, C, T]
            target_sample = target_wav[b]  # [S, C, T]
            mixture_sample = mixture[b]    # [C, T] — needed for LogWMSE

            metrics = self._compute_metrics(output_sample, target_sample, metrics=self.val_metrics, mixture=mixture_sample)

            # Store per-source metrics
            for metric, per_stem in metrics.items():
                if metric not in self.val_per_source_metric_values:
                    continue
                for source_idx in range(min(S, self.n_sources)):
                    if val_other_idx is not None and source_idx == val_other_idx:
                        continue
                    if source_idx in per_stem and not np.isnan(per_stem[source_idx]):
                        self.val_per_source_metric_values[metric][source_idx].append(float(per_stem[source_idx]))

            # Store per-dataset and per-dataset+per-source metrics
            if dataset_name is not None:
                for metric, per_stem in metrics.items():
                    if metric not in self.val_per_dataset_metric_values:
                        continue
                    if dataset_name not in self.val_per_dataset_metric_values[metric]:
                        self.val_per_dataset_metric_values[metric][dataset_name] = []
                        self.val_per_dataset_and_source_metric_values[metric][dataset_name] = {i: [] for i in range(self.n_sources)}
                    for source_idx in range(min(S, self.n_sources)):
                        if val_other_idx is not None and source_idx == val_other_idx:
                            continue
                        if source_idx in per_stem and not np.isnan(per_stem[source_idx]):
                            v = float(per_stem[source_idx])
                            self.val_per_dataset_metric_values[metric][dataset_name].append(v)
                            self.val_per_dataset_and_source_metric_values[metric][dataset_name][source_idx].append(v)
        
        loss = loss_dict['loss']
        
        # Track overall loss
        loss_value = loss.item() if isinstance(loss, torch.Tensor) else loss
        if hasattr(self, 'val_overall_loss_values'):
            self.val_overall_loss_values.append(loss_value)

        # log all items in loss_dict with "step_val" prefix (step-level only, epoch-level logged separately)
        step_dict = {f'step_val/{name}': metric for name, metric in loss_dict.items()}

        # Log step metrics with appropriate progress bar settings
        for key, value in step_dict.items():
            show_in_progress_bar = self.should_show_in_progress_bar(key)
            self.log(key, value, prog_bar=show_in_progress_bar, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)

        # Per-source L1 and MR-STFT logging (epoch-level, auto-averaged by Lightning)
        if self.columns is not None:
            log_kwargs = dict(prog_bar=False, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)
            if val_per_source_l1 is not None:
                n_src = min(val_per_source_l1.shape[0], len(self.columns))
                for i, name in enumerate(self.columns[:n_src]):
                    self.log(f'epoch_val_l1_loss/{name}', val_per_source_l1[i].item(), **log_kwargs)
                    self.log(f'epoch_val_l1_loss_weighted/{name}', val_per_source_l1_weighted[i].item(), **log_kwargs)
                self.log('epoch_val_l1_loss/total', val_per_source_l1[:n_src].sum().item(), **log_kwargs)
                self.log('epoch_val_l1_loss_weighted/total', val_per_source_l1_weighted[:n_src].sum().item(), **log_kwargs)
            if val_per_source_mr_stft is not None:
                n_src = min(val_per_source_mr_stft.shape[0], len(self.columns))
                for i, name in enumerate(self.columns[:n_src]):
                    self.log(f'epoch_val_mr_stft_loss/{name}', val_per_source_mr_stft[i].item(), **log_kwargs)
                    self.log(f'epoch_val_mr_stft_loss_weighted/{name}', val_per_source_mr_stft_weighted[i].item(), **log_kwargs)
                self.log('epoch_val_mr_stft_loss/total', val_per_source_mr_stft[:n_src].sum().item(), **log_kwargs)
                self.log('epoch_val_mr_stft_loss_weighted/total', val_per_source_mr_stft_weighted[:n_src].sum().item(), **log_kwargs)

        return loss

    def on_fit_start(self):
        """Called at the very beginning of fit (before training starts)."""

        # Set dataset epoch to current_epoch when resuming (important for correct random sampling)
        if self.trainer.datamodule.train_dataset is not None:
            if hasattr(self.trainer.datamodule.train_dataset, 'set_epoch'):
                # When resuming, current_epoch will be > 0, so set it immediately
                # This ensures correct random sampling from the start
                initial_epoch = self.current_epoch
                print(f"\n[DataModule] on_fit_start called - setting initial dataset epoch to {initial_epoch}")
                self.trainer.datamodule.train_dataset.set_epoch(initial_epoch)
                print(f"[DataModule] Dataset epoch is now: {self.trainer.datamodule.train_dataset._epoch}")
        
        # ====================================================================
        # Handle validation datasets: Two cases
        # ====================================================================
        
        # Case 1: Multiple validation datasets (val_datasets is a list)
        val_datasets = getattr(self.trainer.datamodule, 'val_datasets', None)
        if val_datasets is not None:
            print(f"[Validation] Multiple validation datasets detected ({len(val_datasets)} datasets)")
            
            # Set epoch for each validation dataset
            for idx, val_dataset in enumerate(val_datasets):
                if val_dataset is not None:
                    # Unwrap if it's a Subset to get the original dataset
                    if isinstance(val_dataset, Subset):
                        original_dataset = val_dataset.dataset
                    else:
                        original_dataset = val_dataset
                    
                    # Set epoch if the dataset supports it
                    if hasattr(original_dataset, 'set_epoch'):
                        initial_epoch = self.current_epoch
                        original_dataset.set_epoch(initial_epoch)
                        print(f"[Validation] Set epoch for validation dataset {idx} to {initial_epoch}")
        
        # Case 2: Single validation dataset (val_dataset is not None, backward compatibility)
        elif self.trainer.datamodule.val_dataset is not None:
            print(f"[Validation] Single validation dataset detected")
            
            val_dataset = self.trainer.datamodule.val_dataset
            # Unwrap if it's a Subset to get the original dataset
            if isinstance(val_dataset, Subset):
                original_dataset = val_dataset.dataset
            else:
                original_dataset = val_dataset
            
            # Set epoch if the dataset supports it
            if hasattr(original_dataset, 'set_epoch'):
                initial_epoch = self.current_epoch
                original_dataset.set_epoch(initial_epoch)
                print(f"[Validation] Set epoch for validation dataset to {initial_epoch}")


    def on_train_epoch_start(self):
        """Called at the beginning of each training epoch."""
        # Reset stem presence tracking counters for this epoch
        self.stem_presence_counts = {column: 0 for column in self.columns}
        
        # Update epoch in dataset for random offset resampling
        print(f"\n[DataModule] on_train_epoch_start called - current_epoch={self.current_epoch}")
        if hasattr(self.trainer.datamodule.train_dataset, 'set_epoch'):
            print(f"[DataModule] Setting dataset epoch to {self.current_epoch}")
            self.trainer.datamodule.train_dataset.set_epoch(self.current_epoch)
            print(f"[DataModule] Dataset epoch is now: {self.trainer.datamodule.train_dataset._epoch}")
        else:
            print(f"[DataModule] Dataset does not have set_epoch method")
        if self.trainer.datamodule.val_dataset is not None and hasattr(self.trainer.datamodule.val_dataset, 'set_epoch'):
            self.trainer.datamodule.val_dataset.set_epoch(self.current_epoch)
        

    def on_train_epoch_end(self):
        """Compute and log FAD-CLAP and SI-SNR metrics at the end of training epoch."""
        
        # Stem presence statistics - log on all ranks with sync_dist=True
        # Note: This will sum counts across ranks, which is correct for total samples
        if self.stem_presence_counts is not None:
            for column in self.columns:
                count = self.stem_presence_counts.get(column, 0)
                self.log(
                    f'stem_stats/training_samples/{column}',
                    float(count),
                    prog_bar=False,
                    logger=True,
                    on_epoch=True,
                    on_step=False,
                    sync_dist=True,
                    reduce_fx='sum',  # sum across ranks to get total sample count
                )

        super().on_train_epoch_end()

    def on_validation_epoch_start(self):
        """Reset metric accumulators at the start of validation epoch."""
        if self.val_fad_clap_metric is not None:
            self.val_fad_clap_metric.reset()

        self.val_per_source_metric_values = {metric: {i: [] for i in range(self.n_sources)} for metric in self.val_metrics}
        self.val_per_dataset_metric_values = {metric: {} for metric in self.val_metrics}
        self.val_per_dataset_and_source_metric_values = {metric: {} for metric in self.val_metrics}
        self.val_overall_loss_values = []

        # Swap in EMA weights for the full validation epoch, then restore afterwards.
        if self.use_ema and self.ema is not None:
            self._ema_stored_params = self.ema.store()
            self.ema.apply()



    def on_validation_epoch_end(self):
        """Compute and log metrics at the end of validation epoch."""
        if len(self.val_overall_loss_values) == 0:
            return

        source_names = [self.columns[i] for i in range(self.n_sources)]

        # FAD-CLAP (computed separately, not part of per-source metric accumulators)
        value_fad_clap = None
        if self.val_fad_clap_metric is not None:
            fad_clap_score = self._compute_fad_clap_score_across_processes(self.val_fad_clap_metric)
            if self._is_rank_0():
                value_fad_clap = float(fad_clap_score.get("fad_clap", 0.0))
                self.log('epoch_val_overall/fad_clap', value_fad_clap,
                         prog_bar=self.should_show_in_progress_bar('fad_clap'),
                         logger=True, on_epoch=True, sync_dist=False)
                per_source_scores = fad_clap_score.get("fad_clap_per_source", {}) or {}
                for source_idx, source_name in enumerate(source_names):
                    if source_idx in per_source_scores:
                        self.log(f'epoch_val_sources/{source_name}/fad_clap',
                                 float(per_source_scores[source_idx]),
                                 prog_bar=False, logger=True, on_epoch=True, sync_dist=False)
            self.val_fad_clap_metric.reset()

        # Overall loss
        all_loss_values = self._gather_metric_values_across_processes(self.val_overall_loss_values)
        if all_loss_values and self._is_rank_0():
            self.log('epoch_val_overall/loss', torch.tensor(all_loss_values).mean().item(),
                     prog_bar=False, logger=True, on_epoch=True, sync_dist=False)

        # Overall mean-of-medians and mean-of-means for each metric
        overall_results = {}  # {metric: (mean_of_medians, mean_of_means)}
        for metric in self.val_metrics:
            _, mom, mome, _ = self._compute_overall_metrics_from_per_source(
                self.val_per_source_metric_values[metric], metric)
            overall_results[metric] = (mom, mome)
        if self._is_rank_0():
            for metric, (mom, mome) in overall_results.items():
                self.log(f'epoch_val_overall_mean_of_medians/{metric}', mom,
                         prog_bar=False, logger=True, on_epoch=True, sync_dist=False)
                self.log(f'epoch_val_overall_mean_of_means/{metric}', mome,
                         prog_bar=False, logger=True, on_epoch=True, sync_dist=False)

        # Per-source metrics
        for source_idx, source_name in enumerate(source_names):
            for metric in self.val_metrics:
                self._gather_and_log_metric(
                    self.val_per_source_metric_values[metric].get(source_idx, []),
                    metric, f'epoch_val_sources/{source_name}')

        # Gather dataset names across all processes
        local_dataset_names = set()
        for metric in self.val_metrics:
            local_dataset_names.update(self.val_per_dataset_metric_values[metric].keys())
            local_dataset_names.update(self.val_per_dataset_and_source_metric_values[metric].keys())
        all_dataset_names = self._gather_dataset_names_across_processes(list(local_dataset_names))

        if all_dataset_names:
            for dataset_name in sorted(all_dataset_names):
                # Per-dataset overall and per-dataset+per-source
                for metric in self.val_metrics:
                    self._gather_and_log_metric(
                        self.val_per_dataset_metric_values[metric].get(dataset_name, []),
                        metric, f'epoch_val_datasets/{dataset_name}')
                    dataset_source_dict = self.val_per_dataset_and_source_metric_values[metric].get(dataset_name, {})
                    for source_idx, source_name in enumerate(source_names):
                        local_vals = dataset_source_dict.get(source_idx, [])
                        all_vals = self._gather_metric_values_across_processes(local_vals)
                        if all_vals:
                            try:
                                median_val = torch.tensor(all_vals).median().item()
                                if self._is_rank_0():
                                    self.log(f'epoch_val_datasets_and_source/{dataset_name}/{source_name}/{metric}',
                                             median_val, prog_bar=False, logger=True, on_epoch=True, sync_dist=False)
                            except Exception as e:
                                print(f"Warning: Failed to log {metric} for {dataset_name}/{source_name}: {e}")

        # Print summary
        print("")
        print("")
        parts = ([f"{m} (mean of medians): {v[0]}" for m, v in overall_results.items()] +
                 [f"{m} (mean of means): {v[1]}" for m, v in overall_results.items()])
        if value_fad_clap is not None:
            parts.append(f"fad_clap: {value_fad_clap}")
        if parts:
            print(f"Epoch {self.current_epoch} validation finished: {', '.join(parts)}")
        print("")
        print("")

        # Restore actual training weights now that validation is done.
        if self.use_ema and self.ema is not None and hasattr(self, '_ema_stored_params'):
            self.ema.restore(self._ema_stored_params)
            del self._ema_stored_params

    def _compute_metrics(self, estimated: torch.Tensor, target: torch.Tensor, metrics=None, mixture: Optional[torch.Tensor] = None) -> Dict[str, Dict[int, float]]:
        """
        Compute metrics on the full audio sample per stem.

        Args:
            estimated: [n_sources, n_channels, T]
            target:    [n_sources, n_channels, T]
            metrics:   List of metric keys to compute.
            mixture:   [n_channels, T] — required for LogWMSE.

        Returns:
            {metric_key: {source_idx: value}}
        """
        if museval is None:
            raise ImportError("museval package is required. Install it with: pip install museval")

        if metrics is None:
            metrics = []

        n_sources, n_channels, total_length = estimated.shape
        compute_log_wmse = (
            'log_wmse' in metrics
            and mixture is not None
            and _log_wmse_module.is_available()
        )

        metric_dict: Dict[str, Dict[int, float]] = {
            'sdr': {},
            'si_snr': {},
            'multi_mel_snr': {},
        }
        if 'zimtohrli' in metrics:
            metric_dict['zimtohrli'] = {}
        if compute_log_wmse:
            metric_dict['log_wmse'] = {}
        for key in ('l1_freq', 'aura_stft', 'aura_mrstft', 'bleed', 'fullness', 'mel_ssim', 'mel_kl_div'):
            if key in metrics:
                metric_dict[key] = {}

        signal_threshold = 1e-6

        for source_idx in range(n_sources):
            est_source = estimated[source_idx]  # [n_channels, T]
            tgt_source = target[source_idx]     # [n_channels, T]

            tgt_rms = torch.sqrt(torch.mean(tgt_source ** 2))
            if tgt_rms <= signal_threshold:
                for key in metric_dict:
                    metric_dict[key][source_idx] = float('nan')
                continue

            # SDR
            try:
                est_np = est_source.cpu().numpy().T[np.newaxis, :, :]  # [1, T, C]
                tgt_np = tgt_source.cpu().numpy().T[np.newaxis, :, :]
                sdr, _, _, _, _ = museval.metrics.bss_eval(
                    tgt_np, est_np,
                    window=total_length, hop=total_length,
                    compute_permutation=False,
                )
                metric_dict['sdr'][source_idx] = float(np.mean(sdr))
            except Exception as e:
                logger.warning(f"Warning: Failed to compute SDR for source {source_idx}: {e}")
                metric_dict['sdr'][source_idx] = float('nan')

            # SI-SNR
            try:
                result = si_snr(est_source, tgt_source)
                metric_dict['si_snr'][source_idx] = float(result.mean().item() if result.dim() > 0 else result.item())
            except Exception as e:
                logger.warning(f"Warning: Failed to compute SI-SNR for source {source_idx}: {e}")
                metric_dict['si_snr'][source_idx] = float('nan')

            # Multi-Mel-SNR
            try:
                mm_vals = []
                for c in range(n_channels):
                    mm_vals.append(multi_mel_snr(
                        tgt_source[c].cpu().numpy(),
                        est_source[c].cpu().numpy(),
                        sr=self.sampling_rate,
                    ))
                metric_dict['multi_mel_snr'][source_idx] = float(sum(mm_vals) / len(mm_vals)) if mm_vals else float('nan')
            except Exception as e:
                logger.warning(f"Warning: Failed to compute Multi-Mel-SNR for source {source_idx}: {e}")
                metric_dict['multi_mel_snr'][source_idx] = float('nan')

            # Zimtohrli (optional)
            if 'zimtohrli' in metrics:
                try:
                    zim_scores = []
                    for c in range(n_channels):
                        d = float(self._zimt_calculator.distance(
                            tgt_source[c].detach().cpu().float().numpy(),
                            est_source[c].detach().cpu().float().numpy(),
                        ))
                        if not np.isnan(d):
                            zim_scores.append(d)
                    metric_dict['zimtohrli'][source_idx] = float(sum(zim_scores) / len(zim_scores)) if zim_scores else float('nan')
                except Exception as e:
                    logger.warning(f"Warning: Failed to compute Zimtohrli for source {source_idx}: {e}")
                    metric_dict['zimtohrli'][source_idx] = float('nan')

            # LogWMSE (optional)
            if compute_log_wmse:
                try:
                    lwmse_val = _log_wmse_module.compute_log_wmse_chunked(
                        mixture_ct=mixture,
                        estimated_ct=est_source,
                        target_ct=tgt_source,
                        sample_rate=self.sampling_rate,
                        chunk_size_seconds=total_length / self.sampling_rate,
                    )
                    metric_dict['log_wmse'][source_idx] = lwmse_val if not np.isnan(lwmse_val) else float('nan')
                except Exception as e:
                    logger.warning(f"Warning: Failed to compute LogWMSE for source {source_idx}: {e}")
                    metric_dict['log_wmse'][source_idx] = float('nan')

            # Additional spectral metrics (full audio)
            est_full = est_source.float().cpu().numpy()
            tgt_full = tgt_source.float().cpu().numpy()
            if 'l1_freq' in metrics:
                try:
                    metric_dict['l1_freq'][source_idx] = float(L1Freq_metric(tgt_full, est_full))
                except Exception as e:
                    logger.warning(f"Warning: Failed to compute L1Freq for source {source_idx}: {e}")
                    metric_dict['l1_freq'][source_idx] = float('nan')
            if 'aura_stft' in metrics:
                try:
                    metric_dict['aura_stft'][source_idx] = float(AuraSTFT_metric(tgt_full, est_full))
                except Exception as e:
                    logger.warning(f"Warning: Failed to compute AuraSTFT for source {source_idx}: {e}")
                    metric_dict['aura_stft'][source_idx] = float('nan')
            if 'aura_mrstft' in metrics:
                try:
                    metric_dict['aura_mrstft'][source_idx] = float(AuraMRSTFT_metric(tgt_full, est_full, sample_rate=self.sampling_rate))
                except Exception as e:
                    logger.warning(f"Warning: Failed to compute AuraMRSTFT for source {source_idx}: {e}")
                    metric_dict['aura_mrstft'][source_idx] = float('nan')
            if 'bleed' in metrics or 'fullness' in metrics:
                try:
                    bleedless, fullness_val = bleed_full(tgt_full, est_full, sr=self.sampling_rate)
                    if 'bleed' in metrics:
                        metric_dict['bleed'][source_idx] = float(bleedless)
                    if 'fullness' in metrics:
                        metric_dict['fullness'][source_idx] = float(fullness_val)
                except Exception as e:
                    logger.warning(f"Warning: Failed to compute bleed_full for source {source_idx}: {e}")
                    if 'bleed' in metrics:
                        metric_dict['bleed'][source_idx] = float('nan')
                    if 'fullness' in metrics:
                        metric_dict['fullness'][source_idx] = float('nan')

            # Mel-SSIM / Mel-KL-divergence (log-mel spectrogram domain)
            if 'mel_ssim' in metrics or 'mel_kl_div' in metrics:
                pred_log_mel, target_log_mel = self._compute_log_mel_pair(est_source, tgt_source)
                if 'mel_ssim' in metrics:
                    try:
                        metric_dict['mel_ssim'][source_idx] = (
                            self._ssim_from_log_mel(pred_log_mel, target_log_mel)
                            if pred_log_mel is not None else float('nan')
                        )
                    except Exception as e:
                        logger.warning(f"Warning: Failed to compute mel_ssim for source {source_idx}: {e}")
                        metric_dict['mel_ssim'][source_idx] = float('nan')
                if 'mel_kl_div' in metrics:
                    try:
                        metric_dict['mel_kl_div'][source_idx] = (
                            self._kl_from_log_mel(pred_log_mel, target_log_mel)
                            if pred_log_mel is not None else float('nan')
                        )
                    except Exception as e:
                        logger.warning(f"Warning: Failed to compute mel_kl_div for source {source_idx}: {e}")
                        metric_dict['mel_kl_div'][source_idx] = float('nan')

        return metric_dict

    def _compute_log_mel_pair(self, estimated: torch.Tensor, target: torch.Tensor):
        """Log-mel (dB) spectrograms for a [C, T] estimated/target pair, used by mel_ssim/mel_kl_div.

        Returns (pred_log_mel, target_log_mel) as numpy arrays shaped [C, n_mels, frames], or
        (None, None) if the mel backend is unavailable.
        """
        if not self._mel_metric_backend_available:
            return None, None
        try:
            pred_mel_db = self._db_transform(self._mel_transform(estimated.detach().float()))
            target_mel_db = self._db_transform(self._mel_transform(target.detach().float()))
            return pred_mel_db.cpu().numpy(), target_mel_db.cpu().numpy()
        except Exception as e:
            logger.warning(f"Warning: Failed to compute log-mel pair for mel_ssim/mel_kl_div: {e}")
            return None, None

    @staticmethod
    def _ssim_from_log_mel(pred_log_mel: np.ndarray, target_log_mel: np.ndarray) -> float:
        """Mel-SSIM between log-mel spectrograms, averaged over channels. Higher is better.

        Returns values in [-1, 1]; practically [0, 1] for audio. 1.0 = identical spectra.
        """
        if skimage_ssim is None:
            return float('nan')

        def _single_ssim(a: np.ndarray, b: np.ndarray) -> float:
            if a.shape != b.shape or a.ndim != 2:
                return float('nan')

            min_side = min(a.shape[0], a.shape[1])
            if min_side < 3:
                return float('nan')
            win_size = 7 if min_side >= 7 else min_side
            if win_size % 2 == 0:
                win_size -= 1
            if win_size < 3:
                return float('nan')

            data_min = float(min(a.min(), b.min()))
            data_max = float(max(a.max(), b.max()))
            data_range = data_max - data_min
            if data_range <= 0:
                return 1.0

            try:
                return float(skimage_ssim(a, b, data_range=data_range, win_size=win_size))
            except Exception as e:
                logger.warning(f"Warning: skimage_ssim failed for mel_ssim (shape={a.shape}, data_range={data_range}, win_size={win_size}): {e}")
                return float('nan')

        if pred_log_mel.ndim == 3:
            scores = []
            for i in range(pred_log_mel.shape[0]):
                s = _single_ssim(target_log_mel[i], pred_log_mel[i])
                if math.isfinite(s):
                    scores.append(s)
            return float(np.mean(scores)) if scores else float('nan')

        return _single_ssim(target_log_mel, pred_log_mel)

    @staticmethod
    def _softmax_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        x = x.astype(np.float64)
        x = x - np.max(x)
        x = np.clip(x, -60.0, 60.0)
        ex = np.exp(x)
        denom = float(np.sum(ex)) + eps
        return ex / denom

    @classmethod
    def _kl_from_log_mel(cls, pred_log_mel: np.ndarray, target_log_mel: np.ndarray) -> float:
        """KL(target || pred) on flattened softmax log-mel spectrograms, averaged over channels.

        Lower is better; 0 means the predicted and reference spectral distributions match.
        """
        def _single_kl(p: np.ndarray, t: np.ndarray) -> float:
            if p.shape != t.shape:
                return float('nan')
            q = cls._softmax_np(p.reshape(-1))
            r = cls._softmax_np(t.reshape(-1))
            eps = 1e-12
            value = float(np.sum(r * np.log((r + eps) / (q + eps))))
            return value if math.isfinite(value) else float('nan')

        if pred_log_mel.ndim == 3:
            scores = []
            for i in range(pred_log_mel.shape[0]):
                s = _single_kl(pred_log_mel[i], target_log_mel[i])
                if math.isfinite(s):
                    scores.append(s)
            return float(np.mean(scores)) if scores else float('nan')

        return _single_kl(pred_log_mel, target_log_mel)

    def test_step(self, batch_data_dict, batch_idx):
        """
        Process a test step with full clips/songs.

        Two dataset shapes are supported, mirroring validation_step:
        1. Segment-style datasets (per-stem waveforms still needing on-the-fly mixing) — the
           model runs create_mixtures() to build the mixture, same as training/validation.
        2. Pre-mixed, single-source-per-sample, full-clip datasets (e.g. MSRBench's
           MultiAudioFullSongDataset, or MSRTestSet2025Dataset for the MSRChallenge/2025TestSet)
           — the mixture is already provided per sample, so it's used as-is.

        Metrics are computed once over the whole (trimmed-to-actual-length) clip — there is no
        manual chunking/frame-splitting here. For datasets whose clips are already short (e.g. the
        2025 test set's fixed 10-second clips) this means the full clip is evaluated in a single,
        unsplit pass. Any internal chunking during separation itself (for long inputs) is handled
        by the model wrapper (e.g. RoformerWrapper's inference_chunk_size), independent of this.
        """
        is_msrbench_dataset = False
        test_dataset = self.trainer.datamodule.test_dataset

        if test_dataset is not None:
            unwrapped_dataset = test_dataset.dataset if isinstance(test_dataset, Subset) else test_dataset
            is_msrbench_dataset = isinstance(unwrapped_dataset, MultiAudioFullSongDatasetMSRBench)

        if not is_msrbench_dataset:
            batch_data_dict = self.create_mixtures(batch_data_dict)
            target_key_to_use = None
        else:
            batch_data_dict['mixture'] = batch_data_dict['mixture']['waveform']
            target_key_to_use = "waveform"

        batch_data_dict = self.encode_target_sources(batch_data_dict, target_key=target_key_to_use)

        batch_size = batch_data_dict['mixture'].shape[0]
        device = batch_data_dict['mixture'].device

        # Extract dataset_name from batch (if available)
        dataset_names = batch_data_dict.get('dataset_name', [None] * batch_size)
        if not isinstance(dataset_names, list):
            dataset_names = [dataset_names] if dataset_names is not None else [None] * batch_size
        
        # Get actual lengths for each sample
        if 'vocals' in batch_data_dict and 'lengths' in batch_data_dict['vocals']:
            sample_lengths = batch_data_dict['vocals']['lengths']
        else:
            # Fallback: use full mixture length
            sample_lengths = [batch_data_dict['mixture'].shape[-1]] * batch_size
        
        # Track time for periodic heartbeat logging during sample processing
        last_heartbeat_time = time.time()
        heartbeat_interval = 120.0  # Log heartbeat every 2 minutes during sample processing

        # Exclude 'other' stem from per-source/per-dataset accumulation if flag is set (parity with validation_step)
        test_other_idx = self._other_stem_idx

        # Process each sample in the batch
        for batch_idx_sample in range(batch_size):
            # Periodic heartbeat logging to prevent wandb from marking run as crashed
            current_time = time.time()
            if current_time - last_heartbeat_time >= heartbeat_interval:
                self._log_wandb_heartbeat(phase="test_sample_processing")
                last_heartbeat_time = current_time
            dataset_name = dataset_names[batch_idx_sample] if batch_idx_sample < len(dataset_names) else None
            
            # Extract full song mixture: [2, T] and trim to actual length
            actual_length = sample_lengths[batch_idx_sample]
            mixture = batch_data_dict['mixture'][batch_idx_sample, :, :actual_length]
            
            # Extract target sources: [n_sources, 2, T] and trim to actual length
            target_sources = batch_data_dict['target_sources'][batch_idx_sample, :, :, :actual_length]
            
            # Process full song (chunked if needed)
            # Chunked inference is handled by RoformerWrapper (inference_chunk_size, inference_chunk_overlap_*).
            with torch.no_grad():
                output_dict = self.model({'mixture': mixture.unsqueeze(0)})
                separated_sources = output_dict['waveform'][0]
            
            # Trim separated sources to actual length (in case of padding)
            if separated_sources.shape[-1] > actual_length:
                separated_sources = separated_sources[:, :, :actual_length]
            elif separated_sources.shape[-1] < actual_length:
                # Pad if shorter (shouldn't happen, but handle it)
                pad_length = actual_length - separated_sources.shape[-1]
                separated_sources = F.pad(separated_sources, (0, pad_length), mode='constant', value=0.0)

            # Ensure target_sources matches separated_sources length
            min_length = min(separated_sources.shape[-1], target_sources.shape[-1])
            separated_sources = separated_sources[:, :, :min_length]
            target_sources = target_sources[:, :, :min_length]

            # Update FAD-CLAP for test (expects <= 10s clips)
            if self.test_fad_clap_metric is not None:
                presence = []
                for column in list(self.columns):
                    if column in batch_data_dict and "is_present" in batch_data_dict[column]:
                        flags = batch_data_dict[column]["is_present"]
                        if isinstance(flags, (list, tuple)):
                            presence.append(float(flags[batch_idx_sample]))
                        elif torch.is_tensor(flags):
                            # flags could be [B], [] or scalar
                            if flags.dim() == 0:
                                presence.append(float(flags.item()))
                            else:
                                presence.append(float(flags[batch_idx_sample].item()))
                        else:
                            # Fallback: treat as scalar or indexable sequence
                            try:
                                presence.append(float(flags[batch_idx_sample]))
                            except Exception:
                                presence.append(float(flags))
                    else:
                        # Match validation behavior: treat missing stems as not present
                        presence.append(0.0)

                if len(presence) < self.n_sources:
                    # Match validation padding: missing sources are treated as not present
                    presence = presence + [0.0] * (self.n_sources - len(presence))

                # Exclude 'other' stem from FAD-CLAP (parity with validation_step and with
                # the per-source/per-dataset test metrics above, which skip test_other_idx).
                # Otherwise the pooled overall FAD-CLAP score below wrongly includes 'other'.
                if test_other_idx is not None and test_other_idx < len(presence):
                    presence[test_other_idx] = 0.0

                source_mask = torch.tensor(
                    presence[: self.n_sources],
                    device=separated_sources.device,
                    dtype=torch.float32,
                ).unsqueeze(0)

                self.test_fad_clap_metric.update(
                    {"waveform": separated_sources.unsqueeze(0)},
                    {"waveform": target_sources.unsqueeze(0), "source_mask": source_mask},
                )

            metrics = self._compute_metrics(separated_sources, target_sources, metrics=self.test_metrics)

            # Store per-source metrics
            for metric, per_source in metrics.items():
                if metric not in self.test_per_source_metric_values:
                    continue
                for source_idx in range(self.n_sources):
                    if test_other_idx is not None and source_idx == test_other_idx:
                        continue
                    if source_idx in per_source and not np.isnan(per_source[source_idx]):
                        self.test_per_source_metric_values[metric][source_idx].append(float(per_source[source_idx]))

            # Store per-dataset and per-dataset+per-source metrics
            if dataset_name is not None:
                for metric, per_source in metrics.items():
                    if metric not in self.test_per_dataset_metric_values:
                        continue
                    if dataset_name not in self.test_per_dataset_metric_values[metric]:
                        self.test_per_dataset_metric_values[metric][dataset_name] = []
                        self.test_per_dataset_and_source_metric_values[metric][dataset_name] = {i: [] for i in range(self.n_sources)}
                    for source_idx in range(self.n_sources):
                        if test_other_idx is not None and source_idx == test_other_idx:
                            continue
                        if source_idx in per_source and not np.isnan(per_source[source_idx]):
                            v = float(per_source[source_idx])
                            self.test_per_dataset_metric_values[metric][dataset_name].append(v)
                            self.test_per_dataset_and_source_metric_values[metric][dataset_name][source_idx].append(v)

    def on_test_epoch_start(self):
        """Initialize test metric accumulators for evaluation."""
        if self.test_fad_clap_metric is not None:
            self.test_fad_clap_metric.reset()

        self.test_per_source_metric_values = {metric: {i: [] for i in range(self.n_sources)} for metric in self.test_metrics}
        self.test_per_dataset_metric_values = {metric: {} for metric in self.test_metrics}
        self.test_per_dataset_and_source_metric_values = {metric: {} for metric in self.test_metrics}

        if self.use_ema and self.ema is not None:
            self._ema_stored_params_test = self.ema.store()
            self.ema.apply()

    def _save_test_per_stem_metrics_csv(self, output_path, per_stem_rows, overall_results, value_fad_clap):
        """Write the per-stem test metrics (mean/std/median per metric, FAD-CLAP) plus an
        "overall_mean" row (mean over the stems except "other") to a CSV, in the same format as
        the MSRBench per_stem_metrics.csv."""
        import csv
        import os

        rows = list(per_stem_rows)
        overall = {"stem": "overall_mean"}
        for metric, (mom, mome) in overall_results.items():
            overall[f"{metric}_mean"] = mome
            overall[f"{metric}_median"] = mom
        if value_fad_clap is not None:
            overall["fad_clap"] = value_fad_clap
        rows.append(overall)

        cols = []
        for row in rows:
            for col in row:
                if col not in cols:
                    cols.append(col)

        def _fmt(value):
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                return ""
            return round(value, 4) if isinstance(value, float) else value

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({col: _fmt(row.get(col)) for col in cols})
        print(f"\nTest per-stem metrics saved to: {output_path}")

    def on_test_epoch_end(self):
        """Compute and log test metrics at the end of test epoch."""
        print("\n" + "="*80)
        print("TEST RESULTS")
        print("="*80)

        source_names = [self.columns[i] for i in range(self.n_sources)]

        first_metric = self.test_metrics[0] if self.test_metrics else None
        has_data = first_metric is not None and any(
            len(self.test_per_source_metric_values[first_metric][i]) > 0
            for i in range(self.n_sources)
        )
        if not has_data:
            print("  No samples found")
            print("="*80 + "\n")
            return

        # FAD-CLAP. Overall is a single Fréchet distance computed on the pooled target/output
        # embeddings across all stems (see FADCLAPMetric.compute) — NOT an average of the
        # per-source scores, since FAD is not a per-sample/per-stem-additive metric.
        value_fad_clap = None
        per_source_fad_clap: Dict[int, float] = {}
        if self.test_fad_clap_metric is not None:
            fad_clap_score = self._compute_fad_clap_score_across_processes(self.test_fad_clap_metric)
            if self._is_rank_0():
                value_fad_clap = float(fad_clap_score.get("fad_clap", 0.0))
                self.log("epoch_test_overall/fad_clap", value_fad_clap,
                         prog_bar=False, logger=True, on_epoch=True, sync_dist=False)
                per_source_fad_clap = fad_clap_score.get("fad_clap_per_source", {}) or {}
                for source_idx, source_name in enumerate(source_names):
                    if source_idx in per_source_fad_clap:
                        self.log(f"epoch_test_sources/{source_name}/fad_clap",
                                 float(per_source_fad_clap[source_idx]),
                                 prog_bar=False, logger=True, on_epoch=True, sync_dist=False)
            self.test_fad_clap_metric.reset()

        # Overall mean-of-medians and mean-of-means (per-stem means are kept for the per-stem
        # table below).
        overall_results = {}
        source_means_by_metric: Dict[str, Dict[int, float]] = {}
        for metric in self.test_metrics:
            _, mom, mome, source_mean_by_idx = self._compute_overall_metrics_from_per_source(
                self.test_per_source_metric_values[metric], metric)
            overall_results[metric] = (mom, mome)
            source_means_by_metric[metric] = source_mean_by_idx

        if self._is_rank_0():
            for metric, (mom, mome) in overall_results.items():
                self.log(f'epoch_test_overall_mean_of_medians/{metric}', mom,
                         prog_bar=False, logger=True, on_epoch=True, sync_dist=False)
                self.log(f'epoch_test_overall_mean_of_means/{metric}', mome,
                         prog_bar=False, logger=True, on_epoch=True, sync_dist=False)

        if self._is_rank_0():
            print(f"\n  OVERALL (mean of means):")
            for metric in self.test_metrics:
                _, mome = overall_results[metric]
                print(f"    {metric}: {mome:.3f}")
            if value_fad_clap is not None:
                print(f"    fad_clap: {value_fad_clap:.3f}")
            print("---")

            # Per-stem table for multi_mel_snr, zimtohrli, fad_clap (in this order).
            table_metrics = [m for m in ("multi_mel_snr", "zimtohrli") if m in overall_results]
            if self.test_fad_clap_metric is not None:
                table_metrics.append("fad_clap")

            if table_metrics:
                other_idx = self._other_stem_idx
                col_width = 16

                def _fmt_row(label: str, values: list) -> str:
                    cells = [f"{label:<{col_width}}"]
                    cells += [f"{v:<{col_width}}" for v in values]
                    return "    " + "".join(cells)

                print(f"\n  PER-STEM RESULTS:")
                header_cells = [f"{'stem':<{col_width}}"] + [f"{m:<{col_width}}" for m in table_metrics]
                print("    " + "".join(header_cells))
                print("    " + "-" * (col_width * (len(table_metrics) + 1)))

                for source_idx, source_name in enumerate(source_names):
                    if other_idx is not None and source_idx == other_idx:
                        continue
                    row_values = []
                    for metric in table_metrics:
                        if metric == "fad_clap":
                            v = per_source_fad_clap.get(source_idx)
                        else:
                            v = source_means_by_metric.get(metric, {}).get(source_idx)
                        row_values.append(f"{v:.3f}" if v is not None else "n/a")
                    print(_fmt_row(source_name, row_values))

                print("    " + "-" * (col_width * (len(table_metrics) + 1)))
                overall_row = []
                for metric in table_metrics:
                    v = value_fad_clap if metric == "fad_clap" else overall_results.get(metric, (0.0, 0.0))[1]
                    overall_row.append(f"{v:.3f}" if v is not None else "n/a")
                print(_fmt_row("Overall", overall_row))
            print("---")

        # Per-source metrics (median logged to wandb; mean/std/median kept for the optional CSV)
        per_stem_rows = []
        for source_idx, source_name in enumerate(source_names):
            row = {"stem": source_name}
            for metric in self.test_metrics:
                all_vals = self._gather_metric_values_across_processes(
                    self.test_per_source_metric_values[metric].get(source_idx, []))
                if len(all_vals) == 0:
                    continue
                vals = torch.tensor(all_vals, dtype=torch.float64)
                median_val = vals.median().item()
                if self._is_rank_0():
                    self.log(f'epoch_test_sources/{source_name}/{metric}', median_val,
                             prog_bar=False, logger=True, on_epoch=True, sync_dist=False)
                row.setdefault("n_samples", len(all_vals))
                row[f"{metric}_mean"] = vals.mean().item()
                row[f"{metric}_std"] = vals.std().item() if len(all_vals) > 1 else 0.0
                row[f"{metric}_median"] = median_val
            if source_idx in per_source_fad_clap:
                row["fad_clap"] = float(per_source_fad_clap[source_idx])
            per_stem_rows.append(row)

        output_path = getattr(self, "test_per_stem_metrics_output_path", None)
        if output_path and self._is_rank_0():
            self._save_test_per_stem_metrics_csv(output_path, per_stem_rows, overall_results, value_fad_clap)

        # Per-dataset metrics
        local_dataset_names = set()
        for metric in self.test_metrics:
            local_dataset_names.update(self.test_per_dataset_metric_values[metric].keys())
            local_dataset_names.update(self.test_per_dataset_and_source_metric_values[metric].keys())
        all_dataset_names = self._gather_dataset_names_across_processes(list(local_dataset_names))

        if all_dataset_names:
            for dataset_name in sorted(all_dataset_names):
                for metric in self.test_metrics:
                    self._gather_and_log_metric(
                        self.test_per_dataset_metric_values[metric].get(dataset_name, []),
                        metric, f'epoch_test_datasets/{dataset_name}')
                for metric in self.test_metrics:
                    dataset_source_dict = self.test_per_dataset_and_source_metric_values[metric].get(dataset_name, {})
                    for source_idx, source_name in enumerate(source_names):
                        local_vals = dataset_source_dict.get(source_idx, [])
                        all_vals = self._gather_metric_values_across_processes(local_vals)
                        if all_vals:
                            try:
                                median_val = torch.tensor(all_vals).median().item()
                                if self._is_rank_0():
                                    self.log(f'epoch_test_datasets_and_source/{dataset_name}/{source_name}/{metric}',
                                             median_val, prog_bar=False, logger=True, on_epoch=True, sync_dist=False)
                            except Exception as e:
                                print(f"Warning: Failed to log {metric} for {dataset_name}/{source_name}: {e}")

        print("="*80 + "\n")

        if self.use_ema and self.ema is not None and hasattr(self, '_ema_stored_params_test'):
            self.ema.restore(self._ema_stored_params_test)
            del self._ema_stored_params_test
