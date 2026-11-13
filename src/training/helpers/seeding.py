"""
Seeding utilities shared by all training pipelines (separation, mixture restoration, stem
restoration, stem restoration dataset generation).

One master seed, the top-level `manual_seed` config key, drives every random number generator of a
run. `seed` (Sacred's own global seed, logged to wandb) is derived from it in src/train.py's
default_conf, and `resolve_seeds` fills every other seed that is left `null` in the config with it.

Determinism scheme (everything is keyed by (master seed, epoch, batch / item index), never by how
far a random stream has advanced, so a run resumed from a checkpoint draws the same random numbers
as an uninterrupted run):
  - model initialization: `pl.seed_everything(manual_seed)` before the model is built.
  - training/validation batches: `PerBatchSeedCallback` reseeds python / numpy / torch (CPU + CUDA)
    at the start of every batch from (seed, stage, epoch, batch_idx, rank). This covers on-GPU
    augmentations (separation), the degradation pipeline's internal torch draws and the mastering
    pipeline (mixture restoration), and dropout. Different ranks get different streams.
  - per-item dataset randomness (chunk offsets, stem mixing, ...): seeded per (seed, epoch, index)
    inside the datasets, so it does not depend on which DataLoader worker or rank loads the item.
  - DataLoader workers: `seed_dataloader_worker` seeds python / numpy / torch of every worker from
    (seed, rank, epoch, worker_id), for any randomness a dataset draws from the global RNGs.
  - shuffling: Lightning's DistributedSampler (seed = manual_seed, set_epoch every epoch) under DDP;
    otherwise a torch.Generator seeded from (seed, epoch) in DataModule.train_dataloader.
"""

import random
import zlib
from typing import Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback


def _key_to_int(key: Union[int, str]) -> int:
    if isinstance(key, str):
        return zlib.crc32(key.encode("utf-8"))
    return int(key) & 0xFFFFFFFF


def derive_seed(*keys: Union[int, str]) -> int:
    """Deterministic 32-bit seed from a tuple of ints / strings (independent of PYTHONHASHSEED)."""
    seq = np.random.SeedSequence([_key_to_int(k) for k in keys])
    return int(seq.generate_state(1, dtype=np.uint32)[0])


def seed_all(seed: int) -> None:
    """Seed python's `random`, numpy's global RNG and torch (CPU + all CUDA devices)."""
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices


def seed_dataloader_worker(worker_id: int, base_seed: int, rank: int, epoch: Union[int, str]) -> None:
    """worker_init_fn (bound with functools.partial in DataModule): seeds every DataLoader worker
    from (base_seed, rank, epoch, worker_id)."""
    seed_all(derive_seed(base_seed, "worker", rank, epoch, worker_id))


def _get(d: dict, *path):
    for key in path:
        if not isinstance(d, dict) or key not in d:
            return None
        d = d[key]
    return d


def resolve_seeds(_config: dict) -> int:
    """Fill every seed left `null` in the config with the master seed `manual_seed` (mutates
    `_config` in place, so the resolved values are what gets logged to wandb). Returns the master
    seed. Seeds that select data rather than drive training randomness (e.g. the MSRBench validation
    subset `subset_seed` and the stem restoration train/val split `seed`) are fixed numbers in the
    configs and are not touched, so different training seeds are validated on the same data."""
    if _config.get("manual_seed") is None:
        _config["manual_seed"] = 42
    master = int(_config["manual_seed"])
    if _config.get("seed") is not None and int(_config["seed"]) != master:
        raise ValueError(
            f"seed={_config['seed']} differs from manual_seed={master}. Only set manual_seed: "
            "Sacred's `seed` is derived from it (src/train.py default_conf)."
        )
    _config["seed"] = master

    # Final-system runs: manual_seed must be 100 * system_seed + stage_idx (see the configs).
    system_seed, stage_idx = _config.get("system_seed"), _config.get("stage_idx")
    if system_seed is not None and stage_idx is not None:
        expected = 100 * int(system_seed) + int(stage_idx)
        if master != expected:
            raise ValueError(
                f"manual_seed={master} but system_seed={system_seed}, stage_idx={stage_idx} "
                f"require manual_seed = 100 * system_seed + stage_idx = {expected}."
            )

    dm_args = _get(_config, "datamodule", "args")
    if isinstance(dm_args, dict) and dm_args.get("dataloader_generator_seed") is None:
        dm_args["dataloader_generator_seed"] = master

    train_ds_args = _get(_config, "datamodule", "args", "train_dataloader", "dataset", "args")
    if isinstance(train_ds_args, dict):
        # dataset_8s: `seed` drives chunk offsets / stem mixing; StemRestorationDataset: `chunk_seed`
        # drives chunk offsets (its `seed` is the train/val split seed and stays fixed).
        for key in ("seed", "chunk_seed"):
            if key in train_ds_args and train_ds_args[key] is None:
                train_ds_args[key] = master

    lm_args = _get(_config, "lightning_module", "args")
    if isinstance(lm_args, dict) and "seed" in lm_args and lm_args["seed"] is None:
        lm_args["seed"] = master

    gen_cfg = _config.get("generate_inference_dataset_config")
    if isinstance(gen_cfg, dict) and gen_cfg.get("seed") is None:
        gen_cfg["seed"] = master

    return master


class PerBatchSeedCallback(Callback):
    """Reseeds python / numpy / torch at the start of every train / validation / test batch from
    (seed, stage, epoch, batch_idx, rank), see the module docstring."""

    def __init__(self, seed: int):
        super().__init__()
        self.seed = int(seed)

    def _reseed(self, trainer: "pl.Trainer", stage: str, batch_idx: int, dataloader_idx: int = 0) -> None:
        seed_all(derive_seed(self.seed, stage, trainer.current_epoch, dataloader_idx, batch_idx, trainer.global_rank))

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._reseed(trainer, "train", batch_idx)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._reseed(trainer, "val", batch_idx, dataloader_idx)

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._reseed(trainer, "test", batch_idx, dataloader_idx)


def batch_rng(seed: Optional[int], *keys: Union[int, str]) -> np.random.Generator:
    """numpy Generator seeded from (seed, *keys), for explicit per-batch draws (e.g. the
    degradation category / mastering draws in MixtureRestorationGANLightning)."""
    return np.random.default_rng(derive_seed(0 if seed is None else seed, *keys))
