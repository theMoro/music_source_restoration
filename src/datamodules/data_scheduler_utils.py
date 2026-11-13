"""
Utility functions for data scheduler percentage computation.

This module provides helper functions that can be used with the data scheduler
to define how dataset sampling percentages change over training epochs.

Main Functions:
--------------
- size_proportional_with_decay: Start with size-proportional percentages, then decay
  specific datasets to 0. This is the recommended approach.

Standard Behavior (normalize=True):
----------------------------------
When datasets decay, their percentage is redistributed to remaining datasets
proportionally. This means:
- Percentages always sum to 1.0
- Total samples per epoch stays constant
- Remaining datasets get more samples as others decay
- This is the recommended approach

Example Configuration:
----------------------
```yaml
datamodule:
    args:
        data_scheduler:
            enabled: true
            schedule_function:
                module: src.datamodules.data_scheduler_utils
                function: size_proportional_with_decay
                args:
                    decay_datasets:
                        slakh2100:
                            decay_type: linear
                            zero_at_epoch: 50  # Reaches 0 at epoch 50
                        raw_stems:
                            decay_type: exponential
                            zero_at_epoch: 30  # Reaches 0 at epoch 30
                    # normalize: true  # Default: redistribute freed percentage
```

Decay Speed Control:
-------------------
You can control decay speed in two ways:
1. zero_at_epoch: Specify exact epoch when percentage reaches 0
2. decay_rate: Higher values = faster decay (for exponential/linear)

Decay Types:
-----------
- linear: Constant rate decrease
- exponential: Fast initial decay, slower later
- cosine: Smooth S-curve transition
- step: Stays at 1.0 until near end, then drops quickly
"""

import math
from typing import Dict, List, Optional, Any


def _compute_decay_factor(
    epoch: int,
    decay_type: str = "linear",
    zero_at_epoch: Optional[int] = None,
    decay_rate: Optional[float] = None,
    max_epochs: Optional[int] = None,
) -> float:
    """
    Compute decay factor based on epoch and decay type.
    
    Args:
        epoch: Current epoch number
        decay_type: Type of decay ("linear", "exponential", "cosine", "step")
        zero_at_epoch: Epoch at which decay reaches 0 (takes precedence over decay_rate)
        decay_rate: Decay rate factor (higher = faster decay)
        max_epochs: Maximum number of epochs (used if zero_at_epoch not specified)
    
    Returns:
        Decay factor between 0.0 (fully decayed) and 1.0 (no decay)
    """
    if zero_at_epoch is not None:
        # Compute based on target epoch
        if epoch >= zero_at_epoch:
            return 0.0
        progress = epoch / zero_at_epoch
    elif decay_rate is not None:
        # Compute based on decay rate
        if max_epochs is None or max_epochs == 0:
            max_epochs = 1
        progress = epoch / max_epochs
        # Apply decay rate: higher rate = faster decay
        progress = progress * decay_rate
    else:
        # Default: linear decay over all epochs
        if max_epochs is None or max_epochs == 0:
            max_epochs = 1
        progress = min(epoch / max_epochs, 1.0)
    
    progress = min(progress, 1.0)
    
    if decay_type == "linear":
        return max(0.0, 1.0 - progress)
    elif decay_type == "exponential":
        # Exponential decay: e^(-decay_rate * progress)
        # For zero_at_epoch: e^(-progress * log(2)) gives 0.5 at halfway
        if zero_at_epoch is not None:
            # Scale so it reaches ~0 at zero_at_epoch
            exp_factor = -math.log(0.01) / zero_at_epoch  # Reaches 0.01 at zero_at_epoch  # +2 / 100 = 0.02 
            return max(0.0, math.exp(-exp_factor * epoch))  # --> e^(-0.02 * 90) = 0.1653
        else:
            # Use decay_rate or default
            rate = decay_rate if decay_rate is not None else 2.0
            return max(0.0, math.exp(-rate * progress))
    elif decay_type == "cosine":
        # Cosine decay: smooth transition
        return max(0.0, 0.5 * (1 + math.cos(math.pi * progress)))
    elif decay_type == "step":
        # Step decay: stays at 1.0 until near the end, then drops quickly
        if zero_at_epoch is not None:
            step_threshold = zero_at_epoch * 0.8  # Start decaying at 80% of way
            if epoch < step_threshold:
                return 1.0
            else:
                # Linear decay in the last 20%
                remaining_progress = (epoch - step_threshold) / (zero_at_epoch - step_threshold)
                return max(0.0, 1.0 - remaining_progress)
        else:
            # Default: step at 80% of max_epochs
            step_threshold = max_epochs * 0.8 if max_epochs else 0
            if epoch < step_threshold:
                return 1.0
            else:
                remaining_progress = (epoch - step_threshold) / (max_epochs - step_threshold) if max_epochs else 1.0
                return max(0.0, 1.0 - remaining_progress)
    else:
        raise ValueError(f"Unknown decay_type: {decay_type}. Must be one of: linear, exponential, cosine, step")


def size_proportional_with_decay(
    epoch: int,
    dataset_names: List[str],
    max_epochs: Optional[int],
    dataset_sizes: Optional[Dict[str, int]] = None,
    decay_datasets: Optional[Dict[str, Dict]] = None,
    normalize: bool = True,
    base_percentages: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Start with size-proportional percentages, then decay specified datasets to 0.

    This is the standard approach: percentages start proportional to dataset sizes
    (each sample has equal probability), then specific datasets decay to 0 over time.
    The freed percentage is redistributed to remaining datasets proportionally,
    keeping total samples per epoch constant.

    Standard Behavior (normalize=True):
    - Percentages always sum to 1.0
    - As datasets decay, their percentage is redistributed to others
    - Total samples per epoch stays constant
    - This is the recommended approach

    Alternative (normalize=False):
    - Percentages can sum to less than 1.0
    - Total samples per epoch decreases as datasets decay
    - Not recommended for most use cases

    Args:
        epoch: Current epoch number
        dataset_names: List of all dataset names
        max_epochs: Maximum number of epochs
        dataset_sizes: Dictionary mapping dataset names to their sample counts.
                      If None, assumes uniform distribution (not recommended).
        decay_datasets: Dictionary mapping dataset names to decay configs.
                       Format: {
                           "dataset_name": {
                               "decay_type": "linear" | "exponential" | "cosine" | "step",
                               "zero_at_epoch": int,  # Epoch when percentage reaches 0 (optional)
                               "decay_rate": float,   # Decay speed factor (optional, higher = faster)
                           }
                       }
                       If both zero_at_epoch and decay_rate are None, uses linear decay over all epochs.
        normalize: If True (default), redistribute freed percentage to remaining datasets.
                  If False, let total percentage decrease.
        base_percentages: If provided (passed automatically by the datamodule when both
                         'percentages' and 'schedule_function' are defined in the config),
                         use these as the starting percentages instead of size-proportional ones.
                         Decay is then applied relative to these base values.

    Returns:
        Dictionary mapping dataset names to their current percentages

    Example:
        # Decay slakh2100 and raw_stems to 0 by epoch 50, using linear decay
        schedule_func = lambda e, names, max_e, **kwargs: size_proportional_with_decay(
            e, names, max_e,
            dataset_sizes=kwargs.get('dataset_sizes'),
            decay_datasets={
                "slakh2100": {
                    "decay_type": "linear",
                    "zero_at_epoch": 50
                },
                "raw_stems": {
                    "decay_type": "linear",
                    "zero_at_epoch": 50
                }
            }
        )
    """
    # Use base_percentages as starting point if provided, otherwise compute size-proportional
    if base_percentages is not None:
        # Ensure all dataset_names are covered (fill missing with 0)
        initial_percentages = {name: base_percentages.get(name, 0.0) for name in dataset_names}
        # Normalize in case they don't sum to 1.0
        total = sum(initial_percentages.values())
        if total > 0:
            initial_percentages = {name: p / total for name, p in initial_percentages.items()}
        else:
            initial_percentages = {name: 1.0 / len(dataset_names) for name in dataset_names}
    elif dataset_sizes is not None:
        total_samples = sum(dataset_sizes.get(name, 0) for name in dataset_names)
        if total_samples > 0:
            initial_percentages = {
                name: dataset_sizes.get(name, 0) / total_samples
                for name in dataset_names
            }
        else:
            # Fallback to uniform
            initial_percentages = {name: 1.0 / len(dataset_names) for name in dataset_names}
    else:
        # Fallback to uniform if sizes not provided
        initial_percentages = {name: 1.0 / len(dataset_names) for name in dataset_names}
    
    # Apply decay to specified datasets
    percentages = {}
    decayed_datasets = set()
    
    for name in dataset_names:
        if name in (decay_datasets or {}):
            decay_config = decay_datasets[name]
            decay_type = decay_config.get("decay_type", "linear")
            zero_at_epoch = decay_config.get("zero_at_epoch")
            decay_rate = decay_config.get("decay_rate")
            
            decay_factor = _compute_decay_factor(
                epoch=epoch,
                decay_type=decay_type,
                zero_at_epoch=zero_at_epoch,
                decay_rate=decay_rate,
                max_epochs=max_epochs,
            )
            
            percentages[name] = initial_percentages[name] * decay_factor
            
            if decay_factor == 0.0:
                decayed_datasets.add(name)
        else:
            # No decay for this dataset
            percentages[name] = initial_percentages[name]
    
    # Normalize: redistribute freed percentage to remaining datasets
    if normalize:
        total_percentage = sum(percentages.values())
        if total_percentage > 0:
            # Redistribute proportionally to remaining datasets
            # Keep the relative proportions of non-decayed datasets
            remaining_datasets = [name for name in dataset_names if name not in decayed_datasets]
            
            if len(remaining_datasets) > 0:
                # Compute how much percentage was freed
                freed_percentage = 1.0 - total_percentage
                
                # Redistribute proportionally to remaining datasets
                remaining_total = sum(percentages[name] for name in remaining_datasets)
                if remaining_total > 0:
                    for name in remaining_datasets:
                        # Scale up proportionally
                        percentages[name] = percentages[name] * (1.0 + freed_percentage / remaining_total)
                else:
                    # All remaining datasets have 0, distribute evenly
                    per_dataset = freed_percentage / len(remaining_datasets)
                    for name in remaining_datasets:
                        percentages[name] += per_dataset
            else:
                # All datasets decayed, fallback to uniform
                percentages = {name: 1.0 / len(dataset_names) for name in dataset_names}
        else:
            # All percentages are 0, fallback to uniform
            percentages = {name: 1.0 / len(dataset_names) for name in dataset_names}
    
    # Final normalization to ensure sum is exactly 1.0
    total = sum(percentages.values())
    if total > 0:
        percentages = {name: p / total for name, p in percentages.items()}
    else:
        # Fallback to uniform
        percentages = {name: 1.0 / len(dataset_names) for name in dataset_names}
    
    return percentages


def decay_to_single_dataset(
    epoch: int,
    dataset_names: List[str],
    max_epochs: Optional[int],
    dataset_sizes: Optional[Dict[str, int]] = None,
    keep_dataset: str = "raw_stems",
    decay_type: str = "linear",
    zero_at_epoch: Optional[int] = None,
    decay_rate: Optional[float] = None,
    base_percentages: Optional[Dict[str, float]] = None,
    **kwargs: Any,
) -> Dict[str, float]:
    """
    Convenience wrapper around size_proportional_with_decay that decays every dataset_name
    except `keep_dataset` to 0, so a training run gradually narrows down to a single dataset.

    Built for stem-wise restoration fine-tuning (configs/stem_restoration/apollo_stem_restoration.yaml):
    start by training on all paired (separated_stem, clean_stem) samples available for a stem,
    then decay every dataset_name other than "raw_stems" out over `zero_at_epoch` epochs, so the
    back half of training is raw_stems-only (the only dataset_name with genuinely unaugmented
    ground-truth stems). If `dataset_names` only contains `keep_dataset` to begin with (the
    current generate_restoration_dataset.sh output, which already filters to raw_stems only),
    there is nothing to decay and this is a no-op — safe to leave enabled by default.

    Args:
        epoch, dataset_names, max_epochs, dataset_sizes, base_percentages: forwarded to
            size_proportional_with_decay, see that function.
        keep_dataset: the one dataset_name that should end up at 100% once decay completes.
        decay_type / zero_at_epoch / decay_rate: applied identically to every decayed dataset,
            see size_proportional_with_decay / _compute_decay_factor.
    """
    decay_datasets = {
        name: {"decay_type": decay_type, "zero_at_epoch": zero_at_epoch, "decay_rate": decay_rate}
        for name in dataset_names
        if name != keep_dataset
    }
    return size_proportional_with_decay(
        epoch=epoch,
        dataset_names=dataset_names,
        max_epochs=max_epochs,
        dataset_sizes=dataset_sizes,
        decay_datasets=decay_datasets,
        normalize=True,
        base_percentages=base_percentages,
    )


# def linear_decay_schedule(
#     epoch: int,
#     dataset_names: List[str],
#     max_epochs: Optional[int],
#     dataset_sizes: Optional[Dict[str, int]] = None,
#     decay_datasets: Optional[Dict[str, Dict]] = None,
# ) -> Dict[str, float]:
#     """
#     Convenience function for linear decay.
#
#     Same as size_proportional_with_decay with decay_type="linear" for all decay_datasets.
#     """
#     if decay_datasets is not None:
#         # Ensure all use linear decay
#         decay_datasets = {
#             name: {**config, "decay_type": "linear"}
#             for name, config in decay_datasets.items()
#         }
#
#     return size_proportional_with_decay(
#         epoch=epoch,
#         dataset_names=dataset_names,
#         max_epochs=max_epochs,
#         dataset_sizes=dataset_sizes,
#         decay_datasets=decay_datasets,
#         normalize=True,
#     )
#
#
# def exponential_decay_schedule(
#     epoch: int,
#     dataset_names: List[str],
#     max_epochs: Optional[int],
#     dataset_sizes: Optional[Dict[str, int]] = None,
#     decay_datasets: Optional[Dict[str, Dict]] = None,
# ) -> Dict[str, float]:
#     """
#     Convenience function for exponential decay.
#
#     Same as size_proportional_with_decay with decay_type="exponential" for all decay_datasets.
#     """
#     if decay_datasets is not None:
#         # Ensure all use exponential decay
#         decay_datasets = {
#             name: {**config, "decay_type": "exponential"}
#             for name, config in decay_datasets.items()
#         }
#
#     return size_proportional_with_decay(
#         epoch=epoch,
#         dataset_names=dataset_names,
#         max_epochs=max_epochs,
#         dataset_sizes=dataset_sizes,
#         decay_datasets=decay_datasets,
#         normalize=True,
#     )
#
#
# def cosine_decay_schedule(
#     epoch: int,
#     dataset_names: List[str],
#     max_epochs: Optional[int],
#     dataset_sizes: Optional[Dict[str, int]] = None,
#     decay_datasets: Optional[Dict[str, Dict]] = None,
# ) -> Dict[str, float]:
#     """
#     Convenience function for cosine decay.
#
#     Same as size_proportional_with_decay with decay_type="cosine" for all decay_datasets.
#     """
#     if decay_datasets is not None:
#         # Ensure all use cosine decay
#         decay_datasets = {
#             name: {**config, "decay_type": "cosine"}
#             for name, config in decay_datasets.items()
#         }
#
#     return size_proportional_with_decay(
#         epoch=epoch,
#         dataset_names=dataset_names,
#         max_epochs=max_epochs,
#         dataset_sizes=dataset_sizes,
#         decay_datasets=decay_datasets,
#         normalize=True,
#     )