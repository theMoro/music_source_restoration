"""
Dataset generation utility functions.
"""
import os
import json
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset, concatenate_datasets


def _collect_hf_batch_dirs(output_dir: str, include_top_level: bool = False):
    """
    Collect Hugging Face dataset directories under output_dir that can be concatenated.
    """
    batch_dirs = []
    seen = set()

    temp_batches_dir = os.path.join(output_dir, "_temp_batches")
    if os.path.exists(temp_batches_dir):
        for item in sorted(os.listdir(temp_batches_dir)):
            if not item.startswith("rank_") or "_batch_" not in item:
                continue
            candidate = os.path.join(temp_batches_dir, item)
            if os.path.isdir(candidate) and candidate not in seen:
                batch_dirs.append(candidate)
                seen.add(candidate)

    if include_top_level and os.path.exists(output_dir):
        excluded = {"_temp_batches", "audio_files", "final_dataset"}
        dataset_markers = ("dataset_info.json", "dataset_dict.json", "state.json")
        for item in sorted(os.listdir(output_dir)):
            if item in excluded:
                continue
            candidate = os.path.join(output_dir, item)
            if not os.path.isdir(candidate):
                continue
            if candidate in seen:
                continue
            if any(os.path.exists(os.path.join(candidate, marker)) for marker in dataset_markers):
                batch_dirs.append(candidate)
                seen.add(candidate)

    return batch_dirs


def _print_label_distribution(label_counts, total_samples, final_dataset_path=None):
    """
    Pretty-print the label distribution and total sample count.
    """
    if not label_counts:
        print("No label distribution available.")
        return

    print("\nSample distribution by label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")

    print(f"Total samples: {total_samples}")
    if final_dataset_path:
        print(f"Final dataset saved to: {final_dataset_path}")


def _print_rank_statistics_table(all_rank_stats):
    """
    Pretty-print rank-level statistics summary.
    """
    if not all_rank_stats:
        print("Warning: No rank statistics found!")
        return

    print("\n" + "=" * 80)
    print("RANK STATISTICS SUMMARY")
    print("=" * 80)
    print(f"\n{'Rank':<6} {'Samples':<10} {'Batches':<10} {'Avg SI-SNR (dB)':<20} {'Std SI-SNR (dB)':<20}")
    print("-" * 80)

    total_samples = 0
    all_avg_si_snr = []
    for stats in sorted(all_rank_stats, key=lambda x: x['rank']):
        rank_num = stats['rank']
        samples = stats['samples_collected']
        batches = stats['num_batches']

        total_samples += samples

        print(f"{rank_num:<6} {samples:<10} {batches:<10}")

    print("-" * 80)
    print(f"{'TOTAL':<6} {total_samples:<10} {'':<10} {'':<20} {'':<20}")

    if all_avg_si_snr:
        overall_avg = np.mean(all_avg_si_snr)
        overall_std = np.std(all_avg_si_snr)
        print(f"\nOverall Average SI-SNR across all ranks: {overall_avg:.4f} ± {overall_std:.4f} dB")
    print("=" * 80 + "\n")


def _broadcast_object_to_ranks(obj, rank, callback=None, include_rank0=False):
    """
    Broadcast a Python object from rank 0 to all other ranks and optionally execute a callback.
    """
    distributed = torch.distributed.is_available() and torch.distributed.is_initialized()

    if distributed:
        obj_list = [obj]
        torch.distributed.broadcast_object_list(obj_list, src=0)
        shared_obj = obj_list[0]
        if callback is not None and shared_obj is not None:
            if include_rank0 or rank != 0:
                callback(shared_obj)
        return shared_obj
    else:
        if callback is not None and obj is not None and (include_rank0 or rank != 0):
            callback(obj)
        return obj


def _share_rank_statistics_with_other_ranks(all_rank_stats, rank):
    """
    Share rank statistics summary with other ranks for consistent logging.
    """
    _broadcast_object_to_ranks(
        all_rank_stats if rank == 0 else None,
        rank,
        callback=_print_rank_statistics_table
    )


def _share_label_distribution_with_other_ranks(dataset_info, rank):
    """
    Share label distribution summary so every rank sees it.
    """

    def _callback(info):
        if info is None:
            return
        _print_label_distribution(
            info.get('label_counts', {}),
            info.get('num_samples', 0),
            info.get('path')
        )

    _broadcast_object_to_ranks(
        dataset_info if rank == 0 else None,
        rank,
        callback=_callback
    )


def _wait_for_all_stats_files(output_dir, world_size, max_retries=30, retry_delay=0.5):
    """
    Wait for all rank stats files to be available with retries.

    Args:
        output_dir: Directory containing _temp_batches folder
        world_size: Total number of ranks
        max_retries: Maximum number of retry attempts
        retry_delay: Delay between retries in seconds

    Returns:
        List of stats dictionaries, one per rank
    """
    import time
    all_rank_stats = []
    missing_ranks = []

    for attempt in range(max_retries):
        all_rank_stats = []
        missing_ranks = []

        for r in range(world_size):
            stats_file = os.path.join(output_dir, "_temp_batches", f"rank_{r}_stats.json")
            if os.path.exists(stats_file):
                try:
                    # Try to read the file to ensure it's complete
                    with open(stats_file, 'r') as f:
                        stats = json.load(f)
                        # Verify the stats contain expected fields
                        if 'rank' in stats and stats['rank'] == r:
                            all_rank_stats.append(stats)
                        else:
                            missing_ranks.append(r)
                except (json.JSONDecodeError, IOError, OSError) as e:
                    # File exists but not readable yet or incomplete
                    missing_ranks.append(r)
            else:
                missing_ranks.append(r)

        if len(all_rank_stats) == world_size:
            # All stats files found and readable
            return all_rank_stats

        if attempt < max_retries - 1:
            if missing_ranks:
                print(
                    f"[Rank 0] Waiting for stats files from ranks {missing_ranks} (attempt {attempt + 1}/{max_retries})...")
            time.sleep(retry_delay)

    # Final attempt - return what we have
    if missing_ranks:
        print(f"[Rank 0] WARNING: Could not find stats files for ranks {missing_ranks} after {max_retries} attempts")
    return all_rank_stats


def _combine_and_save_hf_datasets(output_dir: str, include_top_level: bool = False):
    """
    Load, concatenate, and save Hugging Face datasets present in output_dir.
    """
    all_batch_dirs = _collect_hf_batch_dirs(output_dir, include_top_level=include_top_level)

    if not all_batch_dirs:
        print("No batches found to combine!")
        return None

    print(f"Found {len(all_batch_dirs)} batches to combine")

    batch_datasets = []
    for batch_dir in tqdm(all_batch_dirs, desc="Loading batches"):
        try:
            batch_dataset = Dataset.load_from_disk(batch_dir)
            batch_datasets.append(batch_dataset)
        except Exception as e:
            print(f"Error loading batch {batch_dir}: {e}")

    if not batch_datasets:
        print("No valid batches to combine!")
        return None

    print("Concatenating datasets...")
    final_dataset = concatenate_datasets(batch_datasets)

    label_counts = {}
    for sample in tqdm(final_dataset, desc="Computing statistics"):
        label = sample.get('label', 'unknown')
        label_counts[label] = label_counts.get(label, 0) + 1

    final_dataset_path = os.path.join(output_dir, "final_dataset")
    print(f"Saving final dataset with {len(final_dataset)} samples to {final_dataset_path}...")
    final_dataset.save_to_disk(final_dataset_path)

    dataset_info = {
        'path': final_dataset_path,
        'label_counts': label_counts,
        'num_samples': len(final_dataset)
    }

    _print_label_distribution(label_counts, len(final_dataset), final_dataset_path)

    return dataset_info

