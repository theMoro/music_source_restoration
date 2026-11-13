import random
import json
import re
import uuid
import os
import sys
import torch
import pathlib
import numpy as np
from munch import DefaultMunch
import soundfile as sf
from sacred import Experiment
from sacred import SETTINGS
from datasets import Dataset
from tqdm import tqdm

from src.utils import (
    parse_yaml,
    initialize_config,
    ignore_warnings,
)

ignore_warnings()

SETTINGS.CONFIG.READ_ONLY_CONFIG = False

from src.config_updates import add_configs, LOG_PATH, EIGHT_STEMS

# Import helper functions from refactored modules
from src.training.helpers.wandb_utils import upload_checkpoint_and_samples_to_wandb
# Note: get_wandb_logger is imported below in the wrapper function to maintain Sacred command compatibility
from src.training.helpers.dataset_generation_utils import (
    _print_rank_statistics_table,
    _share_rank_statistics_with_other_ranks,
    _share_label_distribution_with_other_ranks,
    _wait_for_all_stats_files,
    _combine_and_save_hf_datasets
)
from src.training.helpers.seeding import resolve_seeds, seed_all, derive_seed
from src.training.helpers.main_helpers import (
    setup_rank_and_environment,
    resolve_dataset_paths,
    broadcast_dataset_paths,
    sync_ranks_after_wandb_init,
    setup_checkpoint_paths,
    setup_deterministic,
    initialize_components,
    setup_callbacks,
    handle_checkpoint_loading,
    setup_model_optimizations,
    finalize_wandb_run
)

ex = Experiment('music_source_separation')


@ex.config
def default_conf():
    cmd = " ".join(sys.argv)
    cmd_short = cmd.split('\\')[-1]

    config_yaml_file = None

    batch_size = -1  # default batch size
    tqdm_rate = 1  # 100

    lightning_module = dict(
        args=dict(
            lr_scheduler=dict(
                schedule_mode=None,
                num_warmup_steps=50_000,
                lr_end=2e-7
            )
        )
    )

    locals().update(parse_yaml(config_yaml_file))

    # One master seed: manual_seed. Sacred's own global seed (logged as `seed`) is derived from it,
    # otherwise Sacred would draw and log a random one. All other seeds left null in the config are
    # filled with manual_seed by src.training.helpers.seeding.resolve_seeds.
    if 'manual_seed' not in locals() or manual_seed is None:
        manual_seed = 42
    seed = manual_seed


add_configs(ex)


def _effective_rank_world_size_for_manual_inference(trainer, main_rank=0):
    """
    Rank/world_size for generate_inference_dataset_* manual loops.

    Shard work across GPUs only when torch.distributed is actually initialized (e.g. after
    Lightning DDP / torchrun). Otherwise use a single logical worker even if Trainer was
    constructed with devices>1 — otherwise one process would take only 1/N of the indices and
    leave other GPUs idle.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    trainer_ws = int(getattr(trainer, "world_size", 1) or 1)
    if trainer_ws > 1 and main_rank == 0:
        print(
            f"\n[Inference dataset] torch.distributed is not initialized but Trainer reports "
            f"world_size={trainer_ws}. Using a single process (world_size=1) for this step so "
            f"all samples are processed on this process. To use multiple GPUs, run training "
            f"with DDP first (e.g. trainer.fit) so the process group is alive, or launch via "
            f"torchrun with proper init.\n"
        )
    return 0, 1


def generate_inference_dataset_with_trainer(
        trainer,
        pl_model,
        data_module,
        configs,
        output_dir,
        num_samples=10000,
        num_epochs=50,
        save_interval=1000,
        split='train',
        seed=42,
        rank=0,
        world_size=1,
        only_merge_existing_datasets=False
):
    """
    Generate a HuggingFace dataset from model predictions on the training set.

    This function:
    1. Uses the trainer and correct precision (matching training)
    2. Works on multiple GPUs simultaneously without processing samples twice
    3. Calculates SI-SNR to verify precision handling
    4. Stores samples (mixture, clean source, prediction, etc.) in files
    5. Creates a HuggingFace dataset containing only paths to samples
    6. Saves periodically to avoid memory issues

    Args:
        trainer: PyTorch Lightning trainer
        pl_model: Lightning module (with model and augmentation)
        data_module: Data module containing the dataset
        configs: Configuration dict
        output_dir: Directory to save the dataset and audio files
        num_samples: Target number of samples to generate
        num_epochs: Number of epochs to iterate through dataset
        save_interval: Save dataset every N samples
        split: Dataset split to use ('train' or 'val')
        rank: Current process rank (for distributed training)
        world_size: Total number of processes (for distributed training)
        only_merge_existing_datasets: If True, skip generation and only concatenate
            pre-existing Hugging Face datasets found in output_dir

    Returns:
        Path to the final saved HuggingFace dataset
    """
    # Only rank 0 should create output directory structure
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        audio_dir = os.path.join(output_dir, "audio_files")
        os.makedirs(audio_dir, exist_ok=True)
        base_temp_dir = os.path.join(output_dir, "_temp_batches")
        os.makedirs(base_temp_dir, exist_ok=True)

    # Wait for rank 0 to create directories
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    only_merge_existing_datasets = bool(only_merge_existing_datasets)
    if only_merge_existing_datasets:
        if rank == 0:
            print(
                "\nReuse-existing-batches flag enabled. Skipping sample generation and concatenating existing HF datasets.")

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        dataset_info = None
        if rank == 0:
            dataset_info = _combine_and_save_hf_datasets(
                output_dir,
                include_top_level=True
            )

        _share_label_distribution_with_other_ranks(dataset_info, rank)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if dataset_info and rank == 0:
            return dataset_info.get('path')
        return None

    # Get dataset based on split
    if split == 'train':
        dataset = data_module.train_dataset
    elif split == 'val':
        dataset = data_module.val_dataset
    else:
        raise ValueError(f"Unknown split: {split}")

    if dataset is None:
        raise ValueError(f"Dataset for split '{split}' is None")

    # Get columns and other config
    columns = pl_model.columns if hasattr(pl_model, 'columns') else None
    print("columns: ", columns)
    if columns is None:
        raise ValueError("pl_model.columns must be set")

    # Create mapping from column name to index based on pl_model.columns order
    # This ensures we index into separated_sources correctly, matching the model's output order
    column_to_idx = {col: idx for idx, col in enumerate(pl_model.columns)}

    sampling_rate = getattr(pl_model, 'sampling_rate', 48000)
    # use_augmentation = getattr(pl_model, 'augment', False)
    #
    # if not use_augmentation:
    #     raise Exception()
    #     print("WARNING: Augmentation is disabled. Setting augment=True for dataset generation.")
    #     pl_model.apply_music_effect_pipeline = True

    # Use FP16 mixed precision for inference
    use_autocast = True
    autocast_dtype = torch.float16

    # Get model and set to eval mode
    model = pl_model.model

    # Unwrap DDP if present (for inference, we don't need DDP wrapper)
    if hasattr(model, 'module'):
        # Model is wrapped in DDP or DataParallel
        print(f"[Rank {rank}] Unwrapping model from DDP/DataParallel wrapper...")
        model = model.module

    model.eval()

    # Apply EMA if enabled
    if hasattr(pl_model, 'use_ema') and pl_model.use_ema and pl_model.ema is not None:
        print(f"[Rank {rank}] Applying EMA weights for inference...")
        pl_model.ema.apply()
        print(f"[Rank {rank}] EMA weights applied successfully")

    # Get device - prioritize model's current device, then trainer, then rank-based
    if torch.cuda.is_available():
        # First, check what device the model is currently on
        model_device = next(model.parameters()).device

        # Try to get device from trainer strategy (most reliable for DDP)
        if hasattr(trainer, 'strategy') and hasattr(trainer.strategy, 'root_device'):
            device = trainer.strategy.root_device
        elif hasattr(trainer, 'devices') and trainer.devices:
            if isinstance(trainer.devices, list):
                # For multi-GPU, use the device corresponding to this rank
                device_idx = rank % len(trainer.devices) if len(trainer.devices) > 1 else trainer.devices[0]
                device = torch.device(f"cuda:{device_idx}")
            else:
                device = torch.device(f"cuda:{trainer.devices}")
        elif model_device.type == 'cuda':
            # Model is already on a GPU, use that device
            device = model_device
        else:
            # Fallback: use rank-based device assignment
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

        # Move model to device if not already there
        if model_device != device:
            print(f"[Rank {rank}] Moving model from {model_device} to {device} for inference...")
            model = model.to(device)
            pl_model.model = model  # Update reference in pl_model
        else:
            print(f"[Rank {rank}] Model already on device {device}")
    else:
        device = torch.device("cpu")
        if next(model.parameters()).device != device:
            print(f"[Rank {rank}] Moving model to CPU for inference...")
            model = model.to(device)
            pl_model.model = model
        print(f"[Rank {rank}] WARNING: CUDA not available, using CPU for inference")

    print(f"[Rank {rank}] Using device: {device} for inference")

    # Get dataset length and create indices for this rank
    dataset_length = len(dataset)
    print(f"[Rank {rank}] Dataset length: {dataset_length}")

    # Calculate indices for this rank (distribute samples across GPUs)
    all_indices = list(range(dataset_length))
    # Shuffle deterministically (same seed on every rank, so the ranks' shards below are disjoint)
    rng = random.Random(seed)
    rng.shuffle(all_indices)
    # Distribute indices across ranks
    rank_indices = [all_indices[i] for i in range(rank, len(all_indices), world_size)]
    print(f"[Rank {rank}] Processing {len(rank_indices)} samples (out of {dataset_length} total)")

    # Collect samples
    samples_collected = 0
    current_batch = []
    batch_num = 0
    temp_dirs = []
    samples_saved = 0  # Tracks how many samples this rank has persisted
    run_id = uuid.uuid4().hex  # Ensures uniqueness across reruns

    all_sdr_values = []
    all_si_snr_values = []
    all_multi_mel_snr_values = []


    def sanitize_label(label):
        """Return a filesystem-safe label string."""
        if label is None:
            return "unknown"
        label_str = str(label)
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", label_str)
        return sanitized or "unknown"

    def flush_current_batch(batch_to_process, batch_idx, saved_count):
        """Save current batch to disk and return updated counters."""
        if not batch_to_process:
            return batch_idx, saved_count

        # Create batch directory with rank prefix to avoid conflicts
        batch_dir = os.path.join(
            output_dir, "_temp_batches", f"rank_{rank}_{run_id}_batch_{batch_idx}"
        )
        os.makedirs(batch_dir, exist_ok=True)
        audio_batch_dir = os.path.join(batch_dir, "audio_files")
        os.makedirs(audio_batch_dir, exist_ok=True)

        # Process samples: save audio files and create sample dicts
        processed_samples = []
        for idx, sample_data_within_loop in enumerate(batch_to_process):
            label = sample_data_within_loop.get('label', 'unknown')
            safe_label = sanitize_label(label)
            dataset_index = sample_data_within_loop.get('dataset_index')
            if isinstance(dataset_index, int):
                dataset_index_str = f"{dataset_index:09d}"
            else:
                dataset_index_str = sanitize_label(dataset_index) if dataset_index is not None else "na"
            epoch_value = sample_data_within_loop.get('epoch', 0)
            sequence_num = saved_count + idx
            unique_id = (
                f"{run_id}_rank{rank}_epoch{epoch_value}_idx{dataset_index_str}_seq{sequence_num:010d}"
            )
            file_prefix = f"sample_{unique_id}"

            # Save audio files
            clean_path = os.path.join(audio_batch_dir, f"{file_prefix}_clean_{safe_label}.wav")
            aug_path = os.path.join(audio_batch_dir, f"{file_prefix}_aug_{safe_label}.wav")
            separated_path = os.path.join(audio_batch_dir, f"{file_prefix}_separated_{safe_label}.wav")
            mixture_path = os.path.join(audio_batch_dir, f"{file_prefix}_mixture.wav")

            # Save audio files (transpose from [2, T] to [T, 2] for soundfile)
            # Convert to float32 if needed (soundfile doesn't support float16)
            clean_stem_ = sample_data_within_loop['clean_stem'].astype(np.float32)
            aug_stem_ = sample_data_within_loop['augmented_stem'].astype(np.float32)
            separated_stem_ = sample_data_within_loop['separated_stem'].astype(np.float32)
            mixture_ = sample_data_within_loop['mixture'].astype(np.float32)

            sf.write(clean_path, clean_stem_.T, sampling_rate)
            sf.write(aug_path, aug_stem_.T, sampling_rate)
            sf.write(separated_path, separated_stem_.T, sampling_rate)
            sf.write(mixture_path, mixture_.T, sampling_rate)

            # Create sample dict with file paths (relative to batch_dir)
            processed_sample = {
                'clean_stem': clean_path,  # Absolute path
                'augmented_stem': aug_path,
                'separated_stem': separated_path,
                'mixture': mixture_path,
                'label': sample_data_within_loop['label'],
                'original_path': sample_data_within_loop.get('original_path', ''),
                'dataset_index': sample_data_within_loop['dataset_index'],
                'epoch': sample_data_within_loop['epoch'],
                'si_snr': sample_data_within_loop.get('si_snr', None),  # is None
                'offset_seconds': sample_data_within_loop.get('offset_seconds', 0.0),
                'training_chunk_size': sample_data_within_loop.get('training_chunk_size', 480000),
                'dataset_name': sample_data_within_loop.get('dataset_name', 'unknown')
            }
            processed_samples.append(processed_sample)

        # Create and save dataset for this batch
        batch_dataset = Dataset.from_list(processed_samples)
        batch_dataset.save_to_disk(batch_dir)
        temp_dirs.append(batch_dir)

        print(f"[Rank {rank}] Saved batch {batch_idx} with {len(batch_to_process)} samples to {batch_dir}")
        batch_idx += 1
        saved_count += len(batch_to_process)

        return batch_idx, saved_count

    # Process samples across epochs
    for epoch in range(num_epochs):
        if rank == 0:
            print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

        # Shuffle this rank's indices for this epoch (rng: same seed on every rank)
        epoch_indices = rank_indices.copy()
        rng.shuffle(epoch_indices)

        for idx in tqdm(epoch_indices, desc=f"[Rank {rank}] Processing epoch {epoch + 1}", disable=(rank != 0)):
            try:
                # Set epoch for dataset to get different random offsets
                if hasattr(dataset, 'set_epoch'):
                    dataset.set_epoch(epoch)
                # Seed all global RNGs (augmentations in create_mixtures, ...) per (seed, epoch, sample),
                # so every generated sample is reproducible and independent of the rank / world size.
                seed_all(derive_seed(seed, "generate", epoch, idx))

                # Load sample from dataset
                sample = dataset[idx]

                # Prepare batch data with augmentation
                waveform_length = None
                for column in columns:
                    if column in sample and sample[column].get('waveform') is not None:
                        waveform_length = sample[column]['waveform'].shape[-1]
                        break

                if waveform_length is None:
                    continue

                batch_data = {}
                original_presence = {}
                for column in columns:
                    if column in sample and sample[column].get('is_present', False):
                        batch_data[column] = {
                            "waveform": sample[column]['waveform'].unsqueeze(0),
                            "lengths": torch.tensor([sample[column]['waveform'].shape[-1]]),
                            "paths": [sample[column].get('path', '')],
                            "is_present": [sample[column].get('is_present', True)]
                        }
                        original_presence[column] = True
                    else:
                        empty_waveform = torch.zeros((1, 2, waveform_length))
                        batch_data[column] = {
                            "waveform": empty_waveform,
                            "lengths": torch.tensor([waveform_length]),
                            "paths": [""],
                            "is_present": [False]
                        }
                        original_presence[column] = False

                batch_data["dataset_name"] = [sample["dataset_name"]]

                # Apply augmentation and create mixture
                batch_data = pl_model.create_mixtures(batch_data)

                # Extract mixture (keep on device for inference, will move to CPU later)
                mixture = batch_data['mixture'].squeeze(0)  # [2, T]

                # Run inference with correct precision
                with torch.no_grad():
                    mixture_batch = mixture.unsqueeze(0).to(device)  # [1, 2, T]
                    input_dict = {'mixture': mixture_batch}

                    with torch.cuda.amp.autocast(dtype=autocast_dtype):
                        output_dict = model(input_dict)
                    separated_sources = output_dict['waveform'].squeeze(0)  # [n_sources, 2, T]
                    separated_sources = separated_sources.cpu()

                # remove "other" column
                columns_wo_other = [c for c in columns if c != "other"]

                # Create samples for each present source
                for column in columns_wo_other:
                    source_presence_batch = batch_data[column]['is_present'][0]
                    source_presence_original = original_presence.get(column, False)

                    # Only include samples where the source is actually present
                    if not (source_presence_batch and source_presence_original):
                        continue

                    # Get the correct index for this column based on model's output order
                    source_idx = column_to_idx[column]

                    # Extract clean stem (ground truth, before augmentation)
                    clean_stem = batch_data[column]["waveform"].squeeze(0).cpu()  # [2, T]

                    # Extract augmented stem
                    if 'aug_waveform' in batch_data[column]:
                        aug_stem = batch_data[column]["aug_waveform"].squeeze(0).cpu()  # [2, T]
                    else:
                        aug_stem = clean_stem

                    # Extract separated stem (prediction) - already on CPU from earlier
                    separated_stem = separated_sources[source_idx]  # [2, T]

                    # Calculate metrics to verify precision handling
                    metrics = pl_model._compute_metrics(
                        separated_stem.unsqueeze(0),
                        clean_stem.unsqueeze(0),
                        metrics=['sdr', 'si_snr', 'multi_mel_snr']
                    )

                    # Skip NaN metrics (occurs when clean_stem is near-silent after normalization,
                    # causing all frames to be skipped in the chunking loop)
                    sdr_val = metrics['sdr'][0]
                    si_snr_val = metrics['si_snr'][0]
                    multi_mel_snr_val = metrics['multi_mel_snr'][0]
                    if not (np.isnan(sdr_val) or np.isnan(si_snr_val) or np.isnan(multi_mel_snr_val)):
                        all_sdr_values.append(sdr_val)
                        all_si_snr_values.append(si_snr_val)
                        all_multi_mel_snr_values.append(multi_mel_snr_val)

                    # Get original path
                    original_path = sample[column].get('path', '') if column in sample else ""

                    if column not in original_path:
                        print(
                            f"[Rank {rank}] Warning: Possible mismatch between column '{column}' and original path '{original_path}'")

                    # Store sample data (convert to numpy, ensuring CPU)
                    mixture_cpu = mixture.cpu()
                    sample_data = {
                        'clean_stem': clean_stem.numpy(),  # Convert to numpy for saving
                        'augmented_stem': aug_stem.numpy(),
                        'separated_stem': separated_stem.numpy(),
                        'mixture': mixture_cpu.numpy(),
                        'label': column,
                        'original_path': original_path,
                        'dataset_index': idx,
                        'epoch': epoch,
                        # 'si_snr': si_snr_val,
                        'offset_seconds': sample.get('offset_seconds', 0.0),
                        'training_chunk_size': sample.get('training_chunk_size', 480000),
                        'dataset_name': sample.get('dataset_name', 'unknown')
                    }
                    current_batch.append(sample_data)
                    samples_collected += 1

                    # Print local-rank progress stats every 500 samples.
                    # No cross-rank gather here: ranks reach the 500-sample threshold at
                    # different loop iterations, so a collective all_gather would deadlock.
                    if samples_collected % 500 == 0 and all_si_snr_values:
                        median_sdr = np.median(all_sdr_values)
                        median_si_snr = np.median(all_si_snr_values)
                        median_multi_mel_snr = np.median(all_multi_mel_snr_values)

                        print(f"[Rank {rank}] After {samples_collected} samples: Median SDR = {median_sdr:.4f} dB (local)")
                        print(f"[Rank {rank}] After {samples_collected} samples: Median SI-SNR = {median_si_snr:.4f} dB (local)")
                        print(f"[Rank {rank}] After {samples_collected} samples: Median Multi-Mel-SNR = {median_multi_mel_snr:.4f} dB (local)")

                        all_sdr_values = []
                        all_si_snr_values = []
                        all_multi_mel_snr_values = []

                    # Save batch if interval reached
                    if len(current_batch) >= save_interval:
                        batch_num, samples_saved = flush_current_batch(
                            current_batch, batch_num, samples_saved
                        )
                        current_batch = []

                    # Check if we've collected enough samples
                    if samples_collected >= num_samples:
                        break

                if samples_collected >= num_samples:
                    break

            except Exception as e:
                print(f"[Rank {rank}] Error processing sample {idx} in epoch {epoch}: {e}")
                import traceback
                traceback.print_exc()
                continue

        if samples_collected >= num_samples:
            break

    # Save any remaining samples
    if current_batch:
        batch_num, samples_saved = flush_current_batch(
            current_batch, batch_num, samples_saved
        )
        current_batch = []

    print(f"[Rank {rank}] Collected {samples_collected} samples total in {len(temp_dirs)} batches")

    # Save rank statistics to file for aggregation by rank 0
    stats_file = os.path.join(output_dir, "_temp_batches", f"rank_{rank}_stats.json")
    stats_data = {
        'rank': rank,
        'samples_collected': samples_collected,
        'num_batches': len(temp_dirs),
        'num_si_snr_values': len(all_si_snr_values)
    }
    # Write stats file and ensure it's flushed to disk
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)
        f.flush()
        os.fsync(f.fileno())  # Force write to disk

    print(f"[Rank {rank}] Stats file written: {stats_file}")

    # Wait for all ranks to finish
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Only rank 0 combines all batches into final dataset
    if rank == 0:
        # Wait for all stats files with retries
        all_rank_stats = _wait_for_all_stats_files(output_dir, world_size)

        _print_rank_statistics_table(all_rank_stats)
        _share_rank_statistics_with_other_ranks(all_rank_stats, rank)

        print("\nCombining all batches from all ranks into final dataset...")
        dataset_info = _combine_and_save_hf_datasets(
            output_dir,
            include_top_level=only_merge_existing_datasets
        )

        _share_label_distribution_with_other_ranks(dataset_info, rank)

        if dataset_info:
            return dataset_info.get('path')
        return None
    else:
        _share_rank_statistics_with_other_ranks(None, rank)
        _share_label_distribution_with_other_ranks(None, rank)
        return None


# Function moved to src.training.helpers.wandb_utils - get_wandb_logger
# Note: The @ex.command decorator is kept here for Sacred experiment compatibility
@ex.command(prefix="wandb")
def get_wandb_logger(config, name=None, project="music_source_restoration", entity=None, rank0_only=True, tags=[]):
    """Wrapper for get_wandb_logger from wandb_utils to maintain Sacred command compatibility."""
    # Import here to avoid circular import issues
    from src.training.helpers.wandb_utils import get_wandb_logger as _get_wandb_logger
    return _get_wandb_logger(config, name, project, rank0_only, tags, entity=entity)


@ex.command
def main(
        _run,
        _config,
        _log,
        _rnd,
        _seed,
        rank=0
):
    # Setup rank and environment
    rank, local_rank, num_nodes, hostname = setup_rank_and_environment(_config)
    # Resolve all seeds before wandb is initialized, so the logged config holds the seeds actually used.
    master_seed = resolve_seeds(_config)
    print(f"[Rank {rank}] Master seed (manual_seed): {master_seed}")
    configs = DefaultMunch.fromDict(_config)

    # Validate dataset paths
    if 'dataset_path_8_sources' not in _config:
        raise ValueError("dataset_path_8_sources must be specified in config file")
    if 'dataset_path_msr_bench' not in _config:
        raise ValueError("dataset_path_msr_bench must be specified in config file")

    # Dataset paths from the config
    dataset_paths, prefix_paths, ir_paths = resolve_dataset_paths(_config, rank)

    # Broadcast dataset paths and prefix paths
    dataset_paths, prefix_paths, ir_paths = broadcast_dataset_paths(
        dataset_paths, prefix_paths, ir_paths, rank
    )

    dataset_path_8_sources, dataset_path_msr_bench, dataset_path_msr_test_set_2025 = dataset_paths
    prefix_path_8_sources, prefix_path_msr_bench = prefix_paths

    print(f"Final dataset paths:")
    print(f"  dataset_path_8_sources = {dataset_path_8_sources}")
    print(f"  dataset_path_msr_bench = {dataset_path_msr_bench}")
    print(f"  dataset_path_msr_test_set_2025 = {dataset_path_msr_test_set_2025}")
    print(f"  ir_paths = {ir_paths}")
    print(f"  prefix_path_8_sources = {prefix_path_8_sources}")
    print(f"  prefix_path_msr_bench = {prefix_path_msr_bench}")

    # Update config with the final dataset paths and prefix paths
    _config['dataset_path_8_sources'] = dataset_path_8_sources
    _config['dataset_path_msr_bench'] = dataset_path_msr_bench
    _config['dataset_path_msr_test_set_2025'] = dataset_path_msr_test_set_2025
    _config['prefix_path_8_sources'] = prefix_path_8_sources
    _config['prefix_path_msr_bench'] = prefix_path_msr_bench
    _config['ir_paths'] = ir_paths

    # Initialize logger
    # Note: WANDB_DIR is set inside get_wandb_logger() before wandb is imported
    logger = None
    if rank == 0:
        logger = get_wandb_logger(_config)

    # Synchronize ranks after wandb initialization
    # This ensures rank 0 has finished wandb init before other ranks proceed to DDP initialization
    sync_ranks_after_wandb_init(configs, rank, dataset_paths)

    print("Config main: ")
    print(_config)

    # Setup deterministic training (mutates _config; refresh configs so Trainer sees it)
    setup_deterministic(_config)
    configs = DefaultMunch.fromDict(_config)

    # Setup checkpoint paths
    ckpt_save_path, run_id, wandb_id = setup_checkpoint_paths(configs, rank, logger)
    config_filename = pathlib.Path(configs['config_yaml_file']).stem

    # set the correct columns
    n_stems = _config['lightning_module']['args']['n_sources']
    if n_stems == 9:
        columns = EIGHT_STEMS
    else:
        raise ValueError(f"Unsupported number of stems: {n_stems}. Supported value is 9.")

    # Broadcast columns from rank 0 to all ranks to ensure consistency
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        columns_list = [columns]
        torch.distributed.broadcast_object_list(columns_list, src=0)
        columns = columns_list[0]
        if rank != 0:
            print(f"Rank {rank}: Received columns from rank 0: {columns}")

    # Set columns in all dataloader configs (all ranks execute this)
    if 'train_dataloader' in configs['datamodule']['args']:
        _config['datamodule']['args']['train_dataloader']['dataset']['args']['columns'] = columns

    if 'val_dataloader' in configs['datamodule']['args']:
        _config['datamodule']['args']['val_dataloader']['dataset']['args']['columns'] = columns

    if 'test_dataloader' in configs['datamodule']['args']:
        _config['datamodule']['args']['test_dataloader']['dataset']['args']['columns'] = columns

    # Synchronize all ranks after setting columns
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        if rank == 0:
            print("[Rank 0] All ranks synchronized after setting columns configuration")


    # Initialize components (data module, model, callbacks)
    data_module, pl_model, configs = initialize_components(
        configs, _config, dataset_paths, prefix_paths, ir_paths
    )

    # Setup callbacks
    callbacks = setup_callbacks(configs, ckpt_save_path, wandb_id, run_id)

    # Recreate the train dataloader every epoch: DataModule.train_dataloader sets the dataset's epoch
    # (new chunk offsets / mixes every epoch), seeds the sampler from (seed, epoch) and binds the
    # workers' seed to (seed, rank, epoch) -- see src/training/helpers/seeding.py. Also required by
    # the data scheduler (new sampling weights every epoch).
    configs['train']['trainer']['args']['reload_dataloaders_every_n_epochs'] = 1

    # Initialize trainer
    configs['train']['trainer']['args']['callbacks'] = callbacks
    configs['train']['trainer']['args']['logger'] = logger
    trainer = initialize_config(configs['train']['trainer'])

    print("Run ID: ", run_id)

    if rank == 0:
        logger.experiment.config.update({'run_id': run_id})

    # Handle checkpoint loading (resume or fine-tuning or post-training)
    # Pass ckpt_save_path and wandb_id for auto-resume when SLURM restarts the job
    ckpt_path = handle_checkpoint_loading(configs, rank, pl_model)

    # Setup model optimizations (channels_last, compile, torch flags)
    pl_model = setup_model_optimizations(configs, pl_model)

    # train model
    train_config = configs.get('train_config', {})
    training_enabled = bool(train_config.get('enabled', True))

    # validate model
    validation_config = configs.get('validation_config', {})
    validation_enabled = bool(validation_config.get('enabled', False))  # validate model as an extra step (not part of training)
    validation_upload_samples = bool(validation_config.get('upload_samples', False))
    validation_upload_ckpt = bool(validation_config.get('upload_ckpt', False))
    validation_samples_path = validation_config.get('validation_samples_path', None)

    # test_model
    test_config = configs.get('test_config', {})
    testing_enabled = bool(test_config.get('enabled', False))

    # generate inference dataset (stem restoration training data from separation predictions)
    gen_inf_dataset_cfg = configs.get('generate_inference_dataset_config') or {}
    gen_inf_dataset_enabled = bool(gen_inf_dataset_cfg.get('enabled', False))

    # MSRBench inference + evaluation
    msrbench_inference_eval_cfg = configs.get('msrbench_inference_eval_config') or {}
    msrbench_inference_eval_enabled = bool(msrbench_inference_eval_cfg.get('enabled', False))

    #####################################################################
    ################### Validation, Training, Testing ###################
    #####################################################################

    # ----------- Validation -----------
    if validation_enabled:
        # Run validation before the first training epoch to establish baseline performance
        if rank == 0:
            print("\n" + "=" * 80)
            print("Running validation ...")
            print("=" * 80)

        trainer.validate(
            model=pl_model,
            datamodule=data_module,
            ckpt_path=None  # Don't load checkpoint again, we already loaded it if needed
        )

        if rank == 0:
            print("✓ Validation complete ...")
            print("=" * 80 + "\n")

    # ----------- Training (or resuming training) -----------
    if training_enabled:
        trainer.fit(
            model=pl_model,
            train_dataloaders=None,
            val_dataloaders=None,
            datamodule=data_module,
            ckpt_path=ckpt_path
        )

        # check_val_every_n_epoch may not evenly divide max_epochs, so the last epoch
        # isn't guaranteed to have been validated by trainer.fit() above — validate explicitly.
        if validation_enabled:
            if rank == 0:
                print("\n" + "=" * 80)
                print("Running final validation ...")
                print("=" * 80)

            trainer.validate(
                model=pl_model,
                datamodule=data_module,
                ckpt_path=None
            )

            if rank == 0:
                print("✓ Final validation complete ...")
                print("=" * 80 + "\n")

    # ----------- Testing -----------
    if testing_enabled:
        # optional CSV with the per-stem test metrics
        pl_model.test_per_stem_metrics_output_path = test_config.get('per_stem_metrics_output_path', None)
        trainer.test(
            model=pl_model,
            datamodule=data_module,
            ckpt_path=None
        )

    #####################################################################
    ############# Upload audio samples and / or checkpoint ##############
    #####################################################################

    # Ensure wandb_id is broadcast to all ranks (distributed might be initialized by now)
    # This is a safety check in case distributed wasn't initialized earlier when we first tried to broadcast
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # All ranks participate in broadcast - rank 0 sends, others receive
        wandb_id_list = [wandb_id] if wandb_id is not None else [None]
        torch.distributed.broadcast_object_list(wandb_id_list, src=0)
        if wandb_id_list[0] is not None:
            old_wandb_id = wandb_id
            wandb_id = wandb_id_list[0]
            if rank != 0 and old_wandb_id != wandb_id:
                print(f"Rank {rank}: Received wandb_id from rank 0 (late broadcast): {wandb_id}")

    # ----------- Upload audio samples and / or checkpoint -----------
    if validation_upload_samples or validation_upload_ckpt:
        # Upload checkpoint and audio samples to wandb (only on rank 0)
        if rank == 0 and logger is not None and wandb_id is not None:
            # Get device from model
            device = next(pl_model.model.parameters()).device
            
            # Get checkpoint path from ModelCheckpoint callback if available
            # This ensures we use the actual path where the checkpoint was saved
            checkpoint_path = None

            if not (validation_upload_samples and not validation_upload_ckpt):
                for callback in callbacks:
                    if hasattr(callback, 'last_model_path') and callback.last_model_path:
                        checkpoint_path = callback.last_model_path
                        break
                    elif hasattr(callback, 'best_model_path') and callback.best_model_path:
                        checkpoint_path = callback.best_model_path
                        break

                # Fallback: construct path manually if callback doesn't have it
                if checkpoint_path is None:
                    # Construct checkpoint path matching the callback logic:
                    # - Use wandb_id if available, otherwise use run_id
                    # Since we're here, wandb_id is not None, so use it
                    checkpoint_dir = os.path.join(ckpt_save_path, wandb_id)
                    checkpoint_path = os.path.join(str(checkpoint_dir), "last.ckpt")
                    print(f"WARNING: Could not get checkpoint path from callback, using constructed path: {checkpoint_path}")
            
            upload_checkpoint_and_samples_to_wandb(
                logger=logger,
                checkpoint_path=str(checkpoint_path),
                model=pl_model.model,
                lightning_module=pl_model,
                data_module=data_module,
                device=device,
                num_samples=15,
                config=_config,
                only_upload_samples=validation_upload_samples and not validation_upload_ckpt,
                validation_samples_path=validation_samples_path,
            )
        elif rank == 0 and logger is not None and wandb_id is None:
            print("WARNING: wandb_id is None, skipping checkpoint and audio sample upload")


    #####################################################################
    ############# Optional Inference Dataset Generation #################
    #####################################################################

    if gen_inf_dataset_enabled:
        if rank == 0:
            print("\n" + "=" * 80)
            print("Starting inference dataset generation...")
            print("=" * 80)

        inf_rank, inf_world_size = _effective_rank_world_size_for_manual_inference(trainer, main_rank=rank)

        output_dir = gen_inf_dataset_cfg.get('output_dir')
        only_merge_existing_datasets = bool(gen_inf_dataset_cfg.get('only_merge_existing_datasets', False))

        if output_dir is None:
            output_dir = os.path.join(LOG_PATH, config_filename, "inference_dataset")

        dataset_path = generate_inference_dataset_with_trainer(
            trainer=trainer,
            pl_model=pl_model,
            data_module=data_module,
            configs=configs,
            output_dir=output_dir,
            num_samples=gen_inf_dataset_cfg.get('num_samples', 10000),
            num_epochs=gen_inf_dataset_cfg.get('num_epochs', 1),
            save_interval=gen_inf_dataset_cfg.get('save_interval', 1000),
            split=gen_inf_dataset_cfg.get('split', 'train'),
            seed=gen_inf_dataset_cfg.get('seed') if gen_inf_dataset_cfg.get('seed') is not None else configs['manual_seed'],
            rank=inf_rank,
            world_size=inf_world_size,
            only_merge_existing_datasets=only_merge_existing_datasets
        )

        if rank == 0 and dataset_path:
            print(f"\n✓ Inference dataset generation complete!")
            print(f"  Dataset saved to: {dataset_path}")
            if logger is not None:
                logger.experiment.log({"inference_dataset/path": dataset_path})

    def _resolve_metric_list(
        cfg: dict,
        metric_list_key: str,
        default_metrics: list[str],
    ) -> list[str]:
        """
        Resolve metrics from a single list.
        """
        raw_metrics = cfg.get(metric_list_key)
        if raw_metrics is None:
            metrics = list(default_metrics)
        else:
            metrics = [str(m).strip() for m in raw_metrics if str(m).strip()]

        # Deduplicate while preserving order.
        seen = set()
        deduped: list[str] = []
        for metric in metrics:
            if metric not in seen:
                deduped.append(metric)
                seen.add(metric)
        return deduped

    #####################################################################
    ############ MSRBench inference + evaluation (optional saving) ######
    #####################################################################

    if msrbench_inference_eval_enabled:
        from src.training.evaluation.run_inference_and_evaluate_msrbench import (
            run_inference_and_evaluate_msrbench,
        )

        msrbench_root = msrbench_inference_eval_cfg.get('msrbench_root', dataset_path_msr_bench)
        msrbench_prefix_path = msrbench_inference_eval_cfg.get(
            'msrbench_prefix_path', prefix_path_msr_bench
        )
        prediction_metrics = _resolve_metric_list(
            msrbench_inference_eval_cfg,
            "prediction_metrics",
            default_metrics=[
                "sdr",
                "snr",
                "si_snr",
                "multi_mel_snr",
                "zimtohrli",
                "mel_ssim",
                "mel_kl_div",
                "kl",
                "pesq",
                "production_quality",
            ],
        )
        prediction_metric_stats = msrbench_inference_eval_cfg.get('prediction_metric_stats', 'both')
        prediction_pesq_sample_rate = msrbench_inference_eval_cfg.get('prediction_pesq_sample_rate', 16000)
        prediction_mel_config = msrbench_inference_eval_cfg.get('prediction_mel_config')
        run_inference_and_evaluate_msrbench(
            trainer=trainer,
            pl_model=pl_model,
            msrbench_root=msrbench_root,
            msrbench_prefix_path=msrbench_prefix_path,
            split=msrbench_inference_eval_cfg.get('split', 'train'),
            columns=msrbench_inference_eval_cfg.get('columns', None),
            sampling_rate=getattr(pl_model, 'sampling_rate', 48000),
            mono=False,
            batch_size=msrbench_inference_eval_cfg.get('batch_size', 2),
            num_workers=msrbench_inference_eval_cfg.get('num_workers', 0),
            shuffle=bool(msrbench_inference_eval_cfg.get('shuffle', False)),
            limit_batches=msrbench_inference_eval_cfg.get('limit_batches'),
            rank=rank,
            prediction_metric_stats=prediction_metric_stats,
            prediction_metrics=prediction_metrics,
            prediction_pesq_sample_rate=prediction_pesq_sample_rate,
            prediction_mel_config=prediction_mel_config,
            save_predictions=bool(msrbench_inference_eval_cfg.get('save_predictions', False)),
            output_dir=msrbench_inference_eval_cfg.get('output_dir'),
            per_sample_scores_output_path=msrbench_inference_eval_cfg.get('per_sample_scores_output_path'),
            per_stem_metrics_output_path=msrbench_inference_eval_cfg.get('per_stem_metrics_output_path'),
        )

    # Finalize wandb run
    finalize_wandb_run(logger, rank)


@ex.automain
def default_command():
    return main()
