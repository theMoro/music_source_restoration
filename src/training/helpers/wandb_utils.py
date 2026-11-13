"""
Wandb-related utility functions.
"""
import json
import os
import random
import tempfile
import pathlib
from typing import Optional

import torch
import soundfile as sf

from pytorch_lightning.loggers import WandbLogger
from src.config_updates import LOG_PATH


def upload_checkpoint_and_samples_to_wandb(
        logger,
        checkpoint_path: str,
        model: torch.nn.Module,
        lightning_module,
        data_module,
        device,
        num_samples: int = 15,
        config=None,
        only_upload_samples=False,
        validation_samples_path: Optional[str] = None,
):
    """
    Upload checkpoint as artifact and generate audio samples for wandb.

    Args:
        logger: WandbLogger instance
        checkpoint_path: Path to checkpoint file
        model: Model for inference
        lightning_module: Lightning module (for columns and augmentation)
        data_module: Data module (to access validation dataset)
        device: Device to run inference on
        num_samples: Number of samples to generate (default: 7)
        validation_samples_path: Optional path to a JSON file produced by the
            select_validation_samples_for_wandb notebook. If set, samples are
            loaded using fixed validation indices from that file. If offsets are
            present and the dataset supports get_item_with_offset, those are used;
            otherwise index-based loading is used.
    """
    import wandb

    if wandb.run is None:
        print("WARNING: wandb.run is None, skipping upload")
        return

    print(f"\n{'=' * 80}")
    print("Uploading checkpoint and audio samples to wandb...")
    print(f"{'=' * 80}")

    try:
        if not only_upload_samples:
            # 1. Upload checkpoint as artifact
            print(f"\nUploading checkpoint: {checkpoint_path}")
            if os.path.exists(checkpoint_path):
                checkpoint_artifact = wandb.Artifact(
                    name=f"checkpoint-{wandb.run.id}",
                    type="model",
                    description=f"Final checkpoint after training"
                )
                checkpoint_artifact.add_file(checkpoint_path)
                wandb.log_artifact(checkpoint_artifact)
                print("✓ Checkpoint uploaded as artifact")
            else:
                print(f"WARNING: Checkpoint not found: {checkpoint_path}")

        # 2. Generate and upload audio samples
        print(f"\nGenerating {num_samples} audio samples...")

        # Get validation dataset
        if data_module.val_datasets is not None and data_module.val_dataset is None:
            val_dataset = data_module.val_datasets[0]
        elif data_module.val_dataset is None:
            print("WARNING: Validation dataset not available, skipping audio sample upload")
            return
        else:
            val_dataset = data_module.val_dataset

        columns = lightning_module.columns if hasattr(lightning_module, 'columns') else None
        if columns is None:
            print("WARNING: Columns not found in lightning module, skipping audio sample upload")
            return

        # Create mapping from column name to index based on lightning_module.columns order
        # This ensures we index into separated_sources correctly, matching the model's output order
        print("Lightning Columns: ", lightning_module.columns)
        column_to_idx = {col: idx for idx, col in enumerate(lightning_module.columns)}
        
        # Ensure columns variable matches lightning_module.columns to avoid KeyError
        if list(columns) != list(lightning_module.columns):
            print(f"WARNING: columns variable ({columns}) doesn't match lightning_module.columns ({lightning_module.columns})")
            print("Using lightning_module.columns for consistency")
            columns = list(lightning_module.columns)

        sampling_rate = lightning_module.sampling_rate if hasattr(lightning_module, 'sampling_rate') else 48000
        use_augmentation = lightning_module.apply_music_effect_pipeline if hasattr(lightning_module, 'augment') else False

        # Select samples: either from validation_samples_path or by random/deterministic indices
        dataset_size = len(val_dataset)
        if dataset_size == 0:
            print("WARNING: Validation dataset is empty, skipping audio sample upload")
            return

        validation_samples_meta = None
        use_offset_sampling = False
        if validation_samples_path and os.path.isfile(validation_samples_path):
            with open(validation_samples_path, 'r') as f:
                data = json.load(f)
            validation_samples_meta = data.get('samples', [])
            if validation_samples_meta:
                print(f"[wandb samples] Using {num_samples} samples from {validation_samples_path} ({len(validation_samples_meta)} possible samples)")
                rng = random.Random(42)
                rng.shuffle(validation_samples_meta)
                validation_samples_meta = validation_samples_meta[:num_samples]

                offsets_available = all('offset_seconds' in entry for entry in validation_samples_meta)
                dataset_supports_offsets = hasattr(val_dataset, 'get_item_with_offset')
                use_offset_sampling = offsets_available and dataset_supports_offsets

                if offsets_available and not dataset_supports_offsets:
                    print(
                        "WARNING: validation sample file contains offset_seconds, "
                        "but dataset has no get_item_with_offset. Falling back to index-based loading."
                    )
                elif not offsets_available:
                    print(
                        "[wandb samples] validation sample file has no offset_seconds "
                        "(expected for full-song/10s fixed samples). Using index-based loading."
                    )

        if validation_samples_meta is None:
            # deterministically select samples
            sample_indices = list(range(dataset_size))
            rng = random.Random(42)
            rng.shuffle(sample_indices)
            sample_indices = sample_indices[:num_samples]
            validation_samples_meta = [{'validation_index': idx} for idx in sample_indices]


        model.eval()
        device_obj = torch.device(device) if isinstance(device, str) else device
        model.to(device_obj)

        # Use FP16 mixed precision for inference
        use_autocast = True
        autocast_dtype = torch.float16

        # Apply EMA weights if enabled (for inference, we want to use EMA model)
        if hasattr(lightning_module, 'use_ema') and lightning_module.use_ema and lightning_module.ema is not None:
            print("Applying EMA weights for sample generation...")
            lightning_module.ema.apply()
            print("EMA weights applied successfully")

        # Build list of (sample_idx, sample) to upload
        samples_to_upload = []
        for i, entry in enumerate(validation_samples_meta):
            try:
                validation_index = entry["validation_index"] % dataset_size  # fix to be able to use indices which are for the whole MSRBench, not just for family DT0
                if use_offset_sampling:
                    sample = val_dataset.get_item_with_offset(
                        validation_index, entry["offset_seconds"]
                    )
                else:
                    sample = val_dataset[validation_index]
                samples_to_upload.append((i, sample))
            except Exception as e:
                print(f"Error loading sample {i} (validation_index={entry.get('validation_index')}): {e}")
                import traceback
                traceback.print_exc()

        # Create temporary directory for audio files
        with tempfile.TemporaryDirectory() as temp_dir:
            uploaded_count = 0

            for sample_idx, sample in samples_to_upload:
                try:

                    # if it's longer than 10 seconds, randomly but deterministically trim to 10 seconds
                    max_length = sampling_rate * 10  # 10 seconds
                    if 'mixture' in sample and sample['mixture']['waveform'].shape[-1] > max_length:
                        total_length = sample['mixture']['waveform'].shape[-1]
                        start_idx = random.randint(0, total_length - max_length)
                        end_idx = start_idx + max_length
                        # Trim all relevant waveforms in the sample
                        for key in sample.keys():
                            if 'waveform' in sample[key]:
                                sample[key]['waveform'] = sample[key]['waveform'][:, start_idx:end_idx]


                    # MSRBench-style sample: mixture + one target source identified by source_label
                    source_label = sample.get('source_label', '')
                    has_single_source = (
                        'source' in sample and
                        isinstance(sample['source'], dict) and
                        sample['source'].get('waveform') is not None and
                        bool(sample['source'].get('is_present', True))
                    )
                    if has_single_source:
                        if not source_label:
                            source_label = sample['source'].get('label', '')
                        if not source_label or source_label not in column_to_idx:
                            print(
                                f"WARNING: source_label '{source_label}' not found in model columns; "
                                f"skipping sample {sample_idx}."
                            )
                            continue

                        mixture = sample['mixture']['waveform']  # [2, T]
                        target_gt = sample['source']['waveform']  # [2, T]
                        target_idx = column_to_idx[source_label]

                        with torch.no_grad():
                            mixture_batch = mixture.unsqueeze(0).to(device_obj)  # [1, 2, T]
                            input_dict = {'mixture': mixture_batch}
                            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                                output_dict = model(input_dict)
                            separated_sources = output_dict['waveform'].squeeze(0).cpu()  # [n_sources, 2, T]

                        target_pred = separated_sources[target_idx]
                        target_length = target_gt.shape[-1]
                        if target_pred.shape[-1] > target_length:
                            target_pred = target_pred[:, :target_length]
                        elif target_pred.shape[-1] < target_length:
                            pad_length = target_length - target_pred.shape[-1]
                            target_pred = torch.nn.functional.pad(target_pred, (0, pad_length), mode='constant', value=0.0)

                        mixture_path = os.path.join(temp_dir, f"sample_{sample_idx}_mixture.wav")
                        target_gt_path = os.path.join(temp_dir, f"sample_{sample_idx}_target_gt_{source_label}.wav")
                        target_pred_path = os.path.join(temp_dir, f"sample_{sample_idx}_target_pred_{source_label}.wav")

                        sf.write(mixture_path, mixture.T.cpu().numpy(), sampling_rate)
                        sf.write(target_gt_path, target_gt.T.cpu().numpy(), sampling_rate)
                        sf.write(target_pred_path, target_pred.T.cpu().numpy(), sampling_rate)

                        wandb.log({
                            f"audio_samples/sample_{sample_idx}/mixture": wandb.Audio(mixture_path, sample_rate=sampling_rate),
                            f"audio_samples/sample_{sample_idx}/target/source_label": source_label,
                            f"audio_samples/sample_{sample_idx}/target/ground_truth_{source_label}": wandb.Audio(target_gt_path, sample_rate=sampling_rate),
                            f"audio_samples/sample_{sample_idx}/target/prediction_{source_label}": wandb.Audio(target_pred_path, sample_rate=sampling_rate),
                        })
                        uploaded_count += 3
                        continue

                    # Handle standard multi-column dataset structure
                    # Prepare batch data with augmentation (like in training)
                    batch_data = {}
                    waveform_length = None
                    original_presence = {}

                    # Get waveform length from first available column
                    for column in columns:
                        if column in sample and sample[column]['waveform'] is not None:
                            waveform_length = sample[column]['waveform'].shape[-1]
                            break

                    if waveform_length is None:
                        continue

                    # Prepare batch
                    for column in columns:
                        if column in sample:
                            sample_column = sample[column]
                            column_is_present = bool(sample_column.get('is_present', True))
                        else:
                            sample_column = {}
                            column_is_present = False
                        original_presence[column] = column_is_present

                        if column in sample and column_is_present:
                            batch_data[column] = {
                                "waveform": sample[column]['waveform'].unsqueeze(0),
                                "lengths": torch.tensor([sample[column]['waveform'].shape[-1]]),
                                "paths": [sample[column].get('path', '')],
                                "is_present": [sample[column].get('is_present', True)]
                            }
                        else:
                            empty_waveform = torch.zeros((1, 2, waveform_length))
                            batch_data[column] = {
                                "waveform": empty_waveform,
                                "lengths": torch.tensor([waveform_length]),
                                "paths": [""],
                                "is_present": [False]
                            }

                    # Apply augmentation and create mixture (once, after all columns are processed)
                    batch_data = lightning_module.create_mixtures(batch_data)
                    mixture = batch_data['mixture'].squeeze(0)  # [2, T]

                    # Save mixture once per sample (not per stem)
                    mixture_path = os.path.join(temp_dir, f"sample_{sample_idx}_mixture.wav")
                    sf.write(mixture_path, mixture.T.cpu().numpy(), sampling_rate)

                    # Log mixture once per sample
                    wandb_dict = {
                        f"audio_samples/sample_{sample_idx}/mixture": wandb.Audio(mixture_path,
                                                                                  sample_rate=sampling_rate),
                    }

                    # Run inference with correct precision
                    with torch.no_grad():
                        mixture_batch = mixture.unsqueeze(0).to(device_obj)  # [1, 2, T]
                        input_dict = {'mixture': mixture_batch}

                        with torch.cuda.amp.autocast(dtype=autocast_dtype):
                            output_dict = model(input_dict)
                        separated_sources = output_dict['waveform'].squeeze(0)  # [n_sources, 2, T]
                        separated_sources = separated_sources.cpu()

                    # Save audio files for each stem that is present
                    for column in columns:
                        source_presence_batch = batch_data[column]['is_present'][0]
                        source_presence_original = original_presence.get(column, False)
                        if not (source_presence_batch and source_presence_original):
                            continue

                        # Get the correct index for this column based on model's output order
                        # Safety check: ensure column exists in mapping
                        if column not in column_to_idx:
                            print(f"WARNING: Column '{column}' not found in column_to_idx mapping. Available columns: {list(column_to_idx.keys())}")
                            continue
                        source_idx = column_to_idx[column]

                        # Get waveforms
                        clean_stem = batch_data[column]["waveform"].squeeze(0)  # [2, T]
                        if use_augmentation and 'aug_waveform' in batch_data[column]:
                            aug_stem = batch_data[column]["aug_waveform"].squeeze(0)  # [2, T]
                        else:
                            aug_stem = clean_stem  # Fallback to clean if no augmentation

                        separated_stem = separated_sources[source_idx]  # [2, T]

                        # Save audio files (transpose from [2, T] to [T, 2] for soundfile)
                        base_name = f"sample_{sample_idx}_stem_{column}"
                        clean_path = os.path.join(temp_dir, f"{base_name}_clean.wav")
                        aug_path = os.path.join(temp_dir, f"{base_name}_augmented.wav")
                        separated_path = os.path.join(temp_dir, f"{base_name}_separated.wav")

                        sf.write(clean_path, clean_stem.T.cpu().numpy(), sampling_rate)
                        sf.write(aug_path, aug_stem.T.cpu().numpy(), sampling_rate)
                        sf.write(separated_path, separated_stem.T.cpu().numpy(), sampling_rate)

                        # Add stem-specific audio to wandb dict (grouped under sample section)
                        wandb_dict[f"audio_samples/sample_{sample_idx}/{column}/clean"] = wandb.Audio(clean_path,
                                                                                                      sample_rate=sampling_rate)
                        wandb_dict[f"audio_samples/sample_{sample_idx}/{column}/augmented"] = wandb.Audio(aug_path,
                                                                                                          sample_rate=sampling_rate)
                        wandb_dict[f"audio_samples/sample_{sample_idx}/{column}/separated"] = wandb.Audio(
                            separated_path, sample_rate=sampling_rate)

                        uploaded_count += 3  # 3 files per stem (clean, augmented, separated)

                    # Log all audio files for this sample at once
                    wandb.log(wandb_dict)
                    uploaded_count += 1  # +1 for mixture

                except Exception as e:
                    print(f"Error generating sample {sample_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

            print(f"✓ Generated and uploaded {len(samples_to_upload)} audio samples ({uploaded_count} audio files)")

    except Exception as e:
        print(f"ERROR: Failed to upload checkpoint/samples to wandb: {e}")
        import traceback
        traceback.print_exc()


def get_wandb_logger(config, name=None, project="music_source_restoration", rank0_only=True, tags=[], entity=None):
    """
    Create the WandbLogger. `entity` (wandb.entity=... on the command line) defaults to None, i.e.
    the WANDB_ENTITY environment variable or the default entity of the logged-in wandb user.
    """
    if project is None:
        project = "music_source_restoration"

    resume_wandb_id = config.get('resume_wandb_id', None)

    config_filename = pathlib.Path(config.get('config_yaml_file', None)).stem
    log_dir = os.path.join(LOG_PATH, config_filename)
    os.makedirs(log_dir, exist_ok=True)
    
    # Convert to absolute path to ensure wandb uses the correct directory
    log_dir = os.path.abspath(log_dir)

    print("IMPORTANT: wandb log dir: ", log_dir)
    
    # Verify directory is writable
    if not os.access(log_dir, os.W_OK):
        raise PermissionError(f"Directory {log_dir} is not writable")

    # Set WANDB_DIR environment variable to ensure wandb folder is saved in the desired location
    # This must be set BEFORE importing wandb to take effect
    # Use absolute path to avoid any path resolution issues
    # Note: If WANDB_DIR is already set (e.g., by a launch script), it takes precedence
    if "WANDB_DIR" not in os.environ:
        os.environ["WANDB_DIR"] = log_dir
        print(f"Setting WANDB_DIR to: {log_dir}")
    else:
        print(f"WANDB_DIR already set to: {os.environ['WANDB_DIR']} (using existing value)")
        # Update log_dir to match the environment variable if it was set elsewhere
        log_dir = os.environ["WANDB_DIR"]
        # Ensure the directory exists even if it was set externally
        os.makedirs(log_dir, exist_ok=True)
        log_dir = os.path.abspath(log_dir)
    
    # Configure wandb settings to prevent hanging on finish
    import wandb
    settings = wandb.Settings(
        _disable_stats=True,  # Disable stats collection that might cause delays
        console="wrap",  # Capture stdout/stderr and send to wandb (options: "off", "wrap", "redirect")
        # Note: start_method removed as it's deprecated and non-functional
        # Note: Periodic heartbeat logging is handled in the Lightning module to prevent
        # false crash detection during long validation/test phases (see _log_wandb_heartbeat)
    )

    wandb_logger = WandbLogger(
        entity=entity,
        project=project,
        tags=tags,
        config=config,
        name=name,
        id=resume_wandb_id,
        save_dir=log_dir,  # (WandbLogger ignores `dir` because save_dir defaults to ".")
        resume="must" if resume_wandb_id else None,
        settings=settings
    )

    return wandb_logger

