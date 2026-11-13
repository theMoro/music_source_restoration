"""
Run inference on MSRBench with the trained model and evaluate metrics without saving predictions.

- Loads MSRBench via MultiAudioFullSongDataset with a shuffled DataLoader.
- For each batch: runs model inference on mixture, then evaluates predictions against
  ground-truth sources using the same metrics as evaluate_predictions (SDR, SI-SNR, Multi-Mel-SNR).
- Does not write any prediction files to disk.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import os

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from pytorch_lightning import Trainer, LightningModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from src.data.msr_bench import MultiAudioFullSongDataset
from src.training.evaluation.prediction_evaluator import PredictionEvaluator

# Folder names used when saving predictions (same layout as the MSR challenge test set).
COLUMN_TO_FOLDER = {
    "vocals": "Vocals",
    "bass": "Bass",
    "drums": "Drums",
    "guitars": "Guitars",
    "keyboards": "Keyboards",
    "synthesizers": "Synthesizers",
    "percussions": "Percussions",
    "orchestral": "Orchestral Elements",
}


def _get_world_size(trainer: Trainer) -> int:
    if hasattr(trainer, "world_size") and trainer.world_size is not None:
        return trainer.world_size
    return 1


def _get_rank(trainer: Trainer) -> int:
    """Infer current process rank from trainer or torch.distributed (for distributed-safe defaults)."""
    if hasattr(trainer, "global_rank") and trainer.global_rank is not None:
        return trainer.global_rank
    return 0


def _ensure_distributed_initialized(world_size: int) -> None:
    """
    Best-effort initialization of torch.distributed when running under SLURM.

    This allows this standalone evaluation script to participate in a real
    process group when launched with srun (ntasks > 1), enabling true
    cross-rank accumulation even outside Lightning's built-in loops.
    """
    if world_size <= 1:
        return
    if not torch.distributed.is_available() or torch.distributed.is_initialized():
        return

    # Try to infer rank/world_size from SLURM if standard env vars are missing.
    rank = os.environ.get("RANK")
    ws = os.environ.get("WORLD_SIZE")

    if rank is None or ws is None:
        slurm_procid = os.environ.get("SLURM_PROCID")
        slurm_ntasks = os.environ.get("SLURM_NTASKS")
        if slurm_procid is not None and slurm_ntasks is not None:
            os.environ.setdefault("RANK", slurm_procid)
            os.environ.setdefault("WORLD_SIZE", slurm_ntasks)
            rank = slurm_procid
            ws = slurm_ntasks

    # Set the per-process CUDA device based on local rank to avoid NCCL
    # guessing device IDs from global rank (which triggers a warning).
    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is None:
        local_rank = os.environ.get("SLURM_LOCALID")
    if local_rank is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_device(int(local_rank))
        except Exception as e:
            print(f"Warning: Failed to set CUDA device for local_rank={local_rank}: {e}")

    # MASTER_ADDR / MASTER_PORT must be set for env://
    if os.environ.get("MASTER_ADDR") is None:
        # Fallback: use first node name from SLURM, or localhost.
        master_addr = os.environ.get("SLURM_NODELIST", "127.0.0.1")
        os.environ["MASTER_ADDR"] = master_addr
    if os.environ.get("MASTER_PORT") is None:
        os.environ["MASTER_PORT"] = "12910"

    try:
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    except Exception as e:
        # If initialization fails, stay in single-rank mode; the script will
        # fall back to local-only metrics on each process.
        print(f"Warning: Failed to initialize torch.distributed process group: {e}")


def _gather_evaluator_state_via_tensors(evaluator, pl_model) -> None:
    """
    Gather evaluator metric lists from all ranks using tensor-based all_gather.
    Updates evaluator in place. Must be called on all ranks (collective).
    """
    gather_metrics = getattr(pl_model, "_gather_metric_values_across_processes", None)
    gather_names = getattr(pl_model, "_gather_dataset_names_across_processes", None)
    if gather_metrics is None or gather_names is None:
        raise RuntimeError(
            "pl_model must have _gather_metric_values_across_processes and "
            "_gather_dataset_names_across_processes (e.g. MultiSourceSeparationLightning)"
        )
    n_sources = evaluator.n_sources

    # total_samples_processed: gather counts and sum
    gathered_counts = gather_metrics([float(evaluator.total_samples_processed)])
    evaluator.total_samples_processed = int(sum(gathered_counts))

    # Per-source dicts (10s and 1s)
    for per_source_store in (evaluator.per_source_metric_values, evaluator.per_source_metric_values_1sec):
        for metric, by_source in per_source_store.items():
            for source_idx in range(n_sources):
                by_source[source_idx] = gather_metrics(by_source.get(source_idx, []))

    # Union of family names across ranks (use 10s as the authoritative source)
    all_family_names = set()
    for metric_families in evaluator.per_family_metric_values.values():
        all_family_names.update(metric_families.keys())
    all_families = gather_names(sorted(all_family_names))

    # Per-family dicts
    for family_store in (evaluator.per_family_metric_values, evaluator.per_family_metric_values_1sec):
        for metric, by_family in family_store.items():
            for family in all_families:
                by_family[family] = gather_metrics(by_family.get(family, []))

    # Per-family-and-source dicts
    for family_source_store in (
        evaluator.per_family_and_source_metric_values,
        evaluator.per_family_and_source_metric_values_1sec,
    ):
        for metric, by_family in family_source_store.items():
            for family in all_families:
                if family not in by_family:
                    by_family[family] = {i: [] for i in range(n_sources)}
                for source_idx in range(n_sources):
                    by_family[family][source_idx] = gather_metrics(by_family[family].get(source_idx, []))


def _dedupe_per_sample_scores_by_clip_id(evaluator: PredictionEvaluator) -> None:
    """
    Drop duplicate per-sample rows caused by DistributedSampler padding: when the
    dataset size isn't divisible by world_size, a few samples get assigned to more
    than one rank and are evaluated (and saved) twice.
    """
    seen: set = set()
    deduped: list[dict] = []
    for row in evaluator.per_sample_scores:
        clip_id = row.get("clip_id")
        if clip_id in seen:
            continue
        seen.add(clip_id)
        deduped.append(row)
    evaluator.per_sample_scores = deduped


def run_inference_and_evaluate_msrbench(
    trainer: Trainer,
    pl_model: LightningModule,
    msrbench_root: str | Path,
    msrbench_prefix_path: Optional[str | Path] = None,
    split: str = "train",
    columns: Optional[list[str]] = None,
    sampling_rate: int = 48000,
    mono: bool = False,
    batch_size: int = 2,
    num_workers: int = 0,
    shuffle: bool = True,
    limit_batches: Optional[int] = None,
    rank: int = 0,
    world_size: Optional[int] = None,
    prediction_metric_stats: str = "both",
    prediction_metrics: Optional[list[str]] = None,
    prediction_pesq_sample_rate: int = 16000,
    prediction_mel_config: Optional[dict] = None,
    save_predictions: bool = False,
    output_dir: Optional[str | Path] = None,
    per_sample_scores_output_path: Optional[str | Path] = None,
    per_stem_metrics_output_path: Optional[str | Path] = None,
) -> None:
    """
    Run inference on MSRBench with the given model and evaluate metrics in-memory (no saving).

    Uses MSRBench dataset with a shuffled DataLoader. For each batch, runs the separation
    model on the mixture, then computes SDR, SI-SNR, and Multi-Mel-SNR against the
    ground-truth source for each sample (MSRBench has one source per sample).

    Args:
        trainer: PyTorch Lightning Trainer (used for device/precision context).
        pl_model: Lightning module with the separation model (must have .model and .columns).
        msrbench_root: Root directory of MSRBench dataset.
        msrbench_prefix_path: Optional prefix path for MSRBench audio files.
        split: MSRBench split to use (e.g. "train", "validation", "test").
        columns: List of stem names (must match pl_model.columns). If None, uses pl_model.columns.
        sampling_rate: Target sampling rate (default: 48000).
        mono: Whether to load audio as mono (default: False).
        batch_size: Batch size for inference and evaluation.
        num_workers: DataLoader num_workers.
        shuffle: Whether to shuffle the MSRBench dataloader (default: True).
        limit_batches: If set, run only this many batches per rank (for faster sanity checks).
        rank: Current rank for distributed (only rank 0 prints).
        world_size: Total number of processes. If None, inferred from trainer or torch.distributed.
        save_predictions: If True, also save separated stem predictions to disk under output_dir
            (each rank writes only the clip_ids in its own DistributedSampler shard).
        output_dir: Output directory for saved predictions. Required if save_predictions is True.
        per_sample_scores_output_path: If set, write per-sample metric scores to this CSV path
            (written once, on rank 0, after gathering scores from all ranks).
        per_stem_metrics_output_path: If set, write a second CSV with all metrics (median/mean/std,
            depending on prediction_metric_stats) aggregated per stem, plus an "overall_mean" row
            (all stems except "other"). Also includes FAD-CLAP per stem / overall when "fad_clap" is
            enabled.
    """
    if world_size is None:
        world_size = _get_world_size(trainer)

    # Try to initialize a real process group when running under SLURM with multiple tasks.
    _ensure_distributed_initialized(world_size)

    print(f"rank: {rank}")
    print(f"world_size: {world_size}")
    print(f"torch.distributed.is_available(): {torch.distributed.is_available()}")
    print(f"torch.distributed.is_initialized(): {torch.distributed.is_initialized()}")

    # Treat as truly distributed only if Torch distributed is actually initialized.
    distributed = (
        world_size > 1
        and torch.distributed.is_available()
        and torch.distributed.is_initialized()
    )
    if distributed:
        # Use actual rank from trainer/distributed so default rank=0 does not break multi-GPU
        rank = _get_rank(trainer)

    if columns is None:
        if hasattr(pl_model, "columns") and pl_model.columns is not None:
            columns = pl_model.columns
        else:
            raise ValueError("columns must be specified or pl_model must have .columns")

    if rank == 0:
        print("\n" + "=" * 80)
        print(f"MSRBench inference + evaluation ({'saving predictions' if save_predictions else 'no saving'})")
        print("=" * 80)
        print(f"MSRBench root: {msrbench_root}")
        print(f"Split: {split}")
        print(f"Shuffle: {shuffle}")
        print(f"Batch size: {batch_size}")
        if distributed:
            print(f"Distributed: {world_size} ranks (torch.distributed initialized)")

    column_to_folder = COLUMN_TO_FOLDER
    output_base: Optional[Path] = Path(output_dir) if output_dir is not None else None
    stem_folders: dict[str, Path] = {}
    if save_predictions:
        if output_base is None:
            raise ValueError(
                "output_dir must be set in msrbench_inference_eval_config when save_predictions is true"
            )
        output_base.mkdir(parents=True, exist_ok=True)
        for column in columns:
            folder_name = column_to_folder.get(column, column.capitalize())
            stem_folder = output_base / folder_name
            stem_folder.mkdir(parents=True, exist_ok=True)
            stem_folders[column] = stem_folder
        if rank == 0:
            print(f"Saving predictions to: {output_dir}")

    pl_model.eval()
    model = pl_model.model
    model.eval()

    device = next(pl_model.parameters()).device
    if torch.cuda.is_available() and device.type != "cuda":
        if distributed:
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        else:
            device = torch.device("cuda")
        pl_model = pl_model.to(device)
        model = model.to(device)

    autocast_dtype = torch.float16

    # Load MSRBench (mixture + source only)
    inference_dataset = MultiAudioFullSongDataset(
        split=split,
        columns=None,
        sr=sampling_rate,
        mono=mono,
        root=msrbench_root,
        prefix_path=msrbench_prefix_path,
    )

    sampler = None
    shuffle_dl = shuffle
    if distributed:
        sampler = DistributedSampler(
            inference_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle_dl = False  # sampler handles shuffling

    dataloader = DataLoader(
        inference_dataset,
        batch_size=batch_size,
        shuffle=shuffle_dl,
        sampler=sampler,
        collate_fn=inference_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False,
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None,
    )

    evaluator = PredictionEvaluator(
        columns=columns,
        sampling_rate=sampling_rate,
        prediction_metric_stats=prediction_metric_stats,
        prediction_metrics=prediction_metrics,
        prediction_pesq_sample_rate=prediction_pesq_sample_rate,
        prediction_mel_config=prediction_mel_config,
        per_sample_scores_output_path=per_sample_scores_output_path,
        per_stem_metrics_output_path=per_stem_metrics_output_path,
    )
    evaluator = evaluator.to(device)
    # Attach trainer so self.log() in on_test_epoch_end uses trainer.logger (e.g. wandb)
    evaluator.trainer = trainer

    pl_model.trainer = trainer

    if rank == 0:
        print(f"Loaded {len(inference_dataset)} samples from MSRBench")
        if distributed:
            per_rank = (len(inference_dataset) + world_size - 1) // world_size
            print(f"Per-rank shard size: ~{per_rank} samples")
        if limit_batches is not None:
            print(f"Limiting to {limit_batches} batches per rank (faster run)")
        print(f"Running inference + evaluation...")

    # If using a distributed Trainer strategy (e.g., DDP), all ranks will enter
    # this function together via the Lightning launch mechanism, so no explicit
    # torch.distributed.barrier() is required here.

    with torch.no_grad():
        for batch_idx, batch in enumerate(
            tqdm(dataloader, desc=f"[Rank {rank}] MSRBench inference+eval", disable=(rank != 0))
        ):
            if limit_batches is not None and batch_idx >= limit_batches:
                break

            mixture = batch["mixture"]["waveform"].to(device)
            B = mixture.shape[0]

            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                output_dict = model({"mixture": mixture})

            # output_dict["waveform"]: [B, n_sources, 2, T]
            pred_waveform_all = output_dict["waveform"]
            mixture_lengths = batch["mixture"]["lengths"]
            source_waveform = batch["source"]["waveform"]
            source_lengths = batch["source"]["lengths"]
            source_labels = batch["source_label"]
            families = batch.get("family", [None] * B)
            if not isinstance(families, list):
                families = [families] * B

            # Build batch in the format PredictionEvaluator.test_step expects (vectorized).
            # ground_truth: { waveform [B,2,T], lengths }; prediction: { waveform [B,2,T], lengths }
            max_len = source_waveform.shape[-1]
            ref_device = pred_waveform_all.device
            T_pred = pred_waveform_all.shape[-1]
            # Indices: source column index per sample, or -1 if label not in columns
            indices = [
                columns.index(source_labels[i]) if i < len(source_labels) and source_labels[i] in columns else -1
                for i in range(B)
            ]
            indices_t = torch.tensor(indices, device=ref_device, dtype=torch.long)
            valid = indices_t >= 0
            indices_clamped = indices_t.clamp(min=0)
            # [B, 2, T_pred] then trim/pad to max_len
            selected = pred_waveform_all[torch.arange(B, device=ref_device), indices_clamped, :, :]
            if T_pred >= max_len:
                pred_stack = selected[:, :, :max_len].clone()
            else:
                pred_stack = F.pad(selected, (0, max_len - T_pred), value=0.0)
            pred_stack[~valid] = 0
            # Lengths: min(T_pred, mixture_lengths) for valid, 0 otherwise
            max_len_t = torch.full((B,), T_pred, device=mixture_lengths.device, dtype=mixture_lengths.dtype)
            pred_lengths_t = torch.where(valid.cpu(), torch.minimum(max_len_t, mixture_lengths), torch.zeros(B, device=mixture_lengths.device, dtype=mixture_lengths.dtype))

            clip_ids = batch["clip_id"]

            if save_predictions:
                selected_cpu = selected.detach().cpu().numpy().astype(np.float32)  # [B, 2, T_pred]
                for i in range(B):
                    if not valid[i]:
                        continue
                    length = int(pred_lengths_t[i].item())
                    if length <= 0:
                        continue
                    audio = selected_cpu[i, :, :length]  # [2, length]
                    output_path = stem_folders[source_labels[i]] / f"{clip_ids[i]}.wav"
                    sf.write(str(output_path), audio.T, sampling_rate)

            eval_batch = {
                "ground_truth": {
                    "waveform": source_waveform,
                    "lengths": source_lengths,
                },
                "prediction": {
                    "waveform": pred_stack.cpu(),
                    "lengths": pred_lengths_t,
                },
                # Unprocessed mixture for LogWMSE (shape [B, C, T]).
                "unprocessed": {
                    "waveform": mixture.cpu(),
                    "lengths": mixture_lengths,
                },
                "source_label": source_labels,
                "family": families,
                "clip_id": clip_ids,
            }

            evaluator.test_step(eval_batch, batch_idx)

    # Summary of distributed metrics gathering:
    # - Prefer simple, robust torch.distributed.all_gather_object for this standalone
    #   MSRBench script, instead of Lightning's all_gather helpers on the pl_model.
    # - If torch.distributed is not initialized, fall back to single-rank behavior.

    # First, if we are in a real distributed process group, synchronize before gather.
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        if rank == 0:
            print("[MSRBench] All ranks finished inference loop, gathering metrics...")

    # Gather metrics from all ranks and report only on rank 0.
    # Use object-based all_gather, which is safe outside Lightning loops.
    if distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
        if rank == 0:
            print(f"[MSRBench gather] Distributed path: world_size={world_size}, gathering from all ranks...")

        # Collect evaluator states from all ranks
        all_states = [None] * world_size
        torch.distributed.all_gather_object(all_states, evaluator.get_state_for_gather())

        if rank == 0:
            print(f"[MSRBench gather] Rank 0: merging {world_size} gathered states...")
            evaluator.reset_metrics()
            evaluator.total_samples_processed = 0
            for r in range(world_size):
                evaluator.merge_state_from_other(all_states[r])
            _dedupe_per_sample_scores_by_clip_id(evaluator)
            evaluator.on_test_epoch_end()

        torch.distributed.barrier()
        if rank == 0:
            print("[MSRBench gather] Barrier done, evaluation complete on rank 0.")
    else:
        # Single-rank (or no initialized process group): use local evaluator state only.
        if rank == 0:
            print("[MSRBench gather] Single-rank path (no distributed gather), calling on_test_epoch_end().")
        _dedupe_per_sample_scores_by_clip_id(evaluator)
        evaluator.on_test_epoch_end()

    if rank == 0:
        print(
            "\n✓ MSRBench inference + evaluation complete"
            + (f" (predictions saved to {output_dir})" if save_predictions else " (no predictions saved)")
        )
        print("=" * 80 + "\n")
