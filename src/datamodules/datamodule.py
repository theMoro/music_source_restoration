import functools
from typing import Dict, Optional, NoReturn
import torch
import lightning.pytorch as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
import importlib
import numpy as np

from src.training.helpers.seeding import derive_seed, seed_dataloader_worker

def initialize_config(module_cfg):
    module = importlib.import_module(module_cfg["module"])
    if 'args' in module_cfg.keys(): return getattr(module, module_cfg["main"])(**module_cfg["args"])
    else: return getattr(module, module_cfg["main"])()

class DataModule(pl.LightningDataModule):
    """
    Lightning DataModule with optional data scheduler support.
    
    The data scheduler allows you to dynamically adjust the sampling percentages
    of different datasets over training epochs. This is useful for curriculum learning
    or adaptive sampling strategies.
    
    Data Scheduler Usage:
    --------------------
    To enable the data scheduler, add a `data_scheduler` argument to the datamodule config:
    
    ```yaml
    datamodule:
        module: src.datamodules.datamodule
        main: DataModule
        args:
            # ... train_dataloader, val_dataloader, etc. ...
            data_scheduler:
                enabled: true
                # Option 1: Use constant percentages
                percentages:
                    dataset_A: 0.5
                    dataset_B: 0.3
                    dataset_C: 0.2
                
                # Option 2: Use a schedule function (dynamic over epochs)
                schedule_function:
                    module: src.datamodules.data_scheduler_utils
                    function: linear_ramp_schedule
                    args:
                        start_percentages:
                            dataset_A: 0.1
                            dataset_B: 0.3
                        end_percentages:
                            dataset_A: 0.7
                            dataset_B: 0.1
                
                # Option 3: Use uniform distribution across datasets (each dataset gets equal percentage)
                # uniform_datasets: true
                
                # Default (if no percentages or schedule_function): Size-proportional distribution
                # Each sample has equal probability, regardless of dataset size
    ```
    
    Default Behavior:
    - If no `percentages` or `schedule_function` is specified, uses size-proportional distribution.
      This means each sample has equal probability of being selected, regardless of which dataset
      it comes from. Larger datasets will naturally contribute more samples to batches.
    - Set `uniform_datasets: true` to use uniform distribution across datasets (each dataset gets
      equal percentage, regardless of size).
    
    When enabled, the scheduler:
    - Uses WeightedRandomSampler instead of standard shuffling
    - Automatically reloads the dataloader every epoch (reload_dataloaders_every_n_epochs=1)
    - Computes per-sample weights based on dataset_name
    - Logs percentages every epoch for monitoring
    
    If disabled (default), standard shuffle=True behavior is used.
    
    Schedule Functions:
    -------------------
    You can use predefined functions from `src.datamodules.data_scheduler_utils`:
    - For dataset_name: size_proportional_with_decay
    
    Or provide your own callable that takes (epoch, names, max_epochs) and returns
    a dict mapping names to percentages.
    """
    def __init__(
        self,
        train_dataloader: dict,
        val_dataloader: dict = None,
        test_dataloader: dict = None,
        data_scheduler: Optional[dict] = None,
        dataloader_generator_seed: Optional[int] = None,
    ):
        super().__init__()
        self.train_config = train_dataloader
        # Seed of the shuffle / WeightedRandomSampler generator and of the DataLoader workers (resolved
        # from manual_seed if null, see src.training.helpers.seeding.resolve_seeds). Everything is
        # derived per epoch in train_dataloader() (which Lightning calls every epoch, see
        # reload_dataloaders_every_n_epochs in src/train.py), so a resumed run gets the same order and
        # worker seeds as an uninterrupted one.
        self.dataloader_generator_seed = (
            int(dataloader_generator_seed) if dataloader_generator_seed is not None else None
        )
        self.val_config = val_dataloader
        self.test_config = test_dataloader
        
        # Data scheduler configuration
        self.data_scheduler_config = data_scheduler or {}
        self.data_scheduler_enabled = self.data_scheduler_config.get('enabled', False)
        
        self.train_dataset = initialize_config(self.train_config['dataset'])
        
        # Precompute indices grouped by dataset_name if scheduler is enabled
        self.indices_by_dataset = {}
        self.dataset_names = None
        if self.data_scheduler_enabled:
            self._prepare_dataset_indices()
        
        # Handle val_dataloader as either a single dict or a list of dicts
        if val_dataloader is not None:
            if isinstance(val_dataloader, list):
                # Multiple validation dataloaders
                self.val_datasets = [initialize_config(config['dataset']) for config in val_dataloader]
                self.val_dataset = None  # Keep for backward compatibility, but won't be used
            else:
                # Single validation dataloader (backward compatibility)
                self.val_dataset = initialize_config(self.val_config['dataset'])
                self.val_datasets = None
        else:
            self.val_dataset = None
            self.val_datasets = None
            self.val_configs = None
        
        if test_dataloader is not None:
            self.test_dataset = initialize_config(self.test_config['dataset'])
        else:
            self.test_dataset = None


    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None) -> NoReturn:
        pass
    
    def _prepare_dataset_indices(self):
        """Precompute indices grouped by dataset_name for efficient weight computation."""
        if not hasattr(self.train_dataset, 'ds'):
            raise ValueError(
                "Data scheduler requires a dataset with 'ds' attribute (HuggingFace dataset). "
                "Make sure you're using MultiAudioRandomSegmentDataset or similar."
            )
        
        hf_dataset = self.train_dataset.ds
        if 'dataset_name' not in hf_dataset.column_names:
            raise ValueError(
                "Data scheduler requires 'dataset_name' column in the dataset. "
                "The dataset does not have this column."
            )
        
        # Get all dataset names
        all_names = np.array(hf_dataset['dataset_name'])
        self.dataset_names = np.unique(all_names).tolist()
        
        # Group indices by dataset name
        self.indices_by_dataset = {
            name: np.where(all_names == name)[0] for name in self.dataset_names
        }
        
        print(f"Data scheduler enabled. Found {len(self.dataset_names)} datasets (dataset_name): {self.dataset_names}")
        total_samples = sum(len(indices) for indices in self.indices_by_dataset.values())
        for name, indices in self.indices_by_dataset.items():
            num_samples = len(indices)
            percentage = (num_samples / total_samples * 100) if total_samples > 0 else 0.0
            print(f"  {name}: {num_samples} samples ({percentage:.1f}% of total)")
    
    def _get_percentages(self, epoch: int) -> Dict[str, float]:
        """
        Compute dataset percentages for the current epoch.
        
        This method can be customized via the data_scheduler config:
        - If 'schedule_function' is provided, it should be a callable that takes (epoch, dataset_names, max_epochs) and returns a dict
        - If both 'schedule_function' and 'percentages' are provided, the resolved percentages are passed to the schedule
          function as 'base_percentages' (if the function's signature accepts it), so the function can use them as a starting point
        - If only 'percentages' is provided, it should be a dict mapping dataset names to percentages (constant over epochs)
        - If 'uniform_datasets' is True, uses uniform distribution across datasets (each dataset gets equal percentage)
        - Otherwise, defaults to size-proportional distribution (each sample has equal probability, regardless of dataset)
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Dictionary mapping dataset names to their sampling percentages (should sum to 1.0)
        """
        if not self.data_scheduler_enabled or self.dataset_names is None:
            # Fallback: size-proportional distribution (each sample has equal probability)
            return self._get_size_proportional_percentages()
        
        max_epochs = self.trainer.max_epochs if self.trainer is not None else None
        
        # Check if custom schedule function is provided
        if 'schedule_function' in self.data_scheduler_config:
            schedule_func = self.data_scheduler_config['schedule_function']

            # Compute dataset sizes for functions that need them
            dataset_sizes = {
                name: len(indices)
                for name, indices in self.indices_by_dataset.items()
            }

            # Resolve base_percentages from 'percentages' if both are defined (size-proportional fill for missing datasets)
            base_percentages = None
            if 'percentages' in self.data_scheduler_config:
                base_percentages = self.data_scheduler_config['percentages'].copy()
                total_specified = sum(base_percentages.get(name, 0) for name in self.dataset_names)
                remaining = 1.0 - total_specified
                if remaining > 0:
                    size_proportional = self._get_size_proportional_percentages()
                    total_unspecified_size = sum(size_proportional.get(name, 0) for name in self.dataset_names if name not in base_percentages)
                    for name in self.dataset_names:
                        if name not in base_percentages:
                            if total_unspecified_size > 0:
                                base_percentages[name] = remaining * (size_proportional.get(name, 0) / total_unspecified_size)
                            else:
                                base_percentages[name] = remaining / len([n for n in self.dataset_names if n not in base_percentages])
                bp_total = sum(base_percentages.values())
                if bp_total > 0:
                    base_percentages = {name: p / bp_total for name, p in base_percentages.items()}

            import inspect

            if callable(schedule_func):
                sig = inspect.signature(schedule_func)
                kwargs = {}
                if 'dataset_sizes' in sig.parameters:
                    kwargs['dataset_sizes'] = dataset_sizes
                if 'base_percentages' in sig.parameters and base_percentages is not None:
                    kwargs['base_percentages'] = base_percentages
                percs = schedule_func(epoch, self.dataset_names, max_epochs, **kwargs)
            elif isinstance(schedule_func, dict):
                # Try to import from module path (e.g., {"module": "src.datamodules.data_scheduler_utils", "function": "size_proportional_with_decay", "args": {...}})
                if 'module' in schedule_func and 'function' in schedule_func:
                    module = importlib.import_module(schedule_func['module'])
                    func = getattr(module, schedule_func['function'])
                    # Get additional args if provided
                    func_args = schedule_func.get('args', {}).copy()
                    sig = inspect.signature(func)
                    if 'dataset_sizes' in sig.parameters and 'dataset_sizes' not in func_args:
                        func_args['dataset_sizes'] = dataset_sizes
                    if 'base_percentages' in sig.parameters and 'base_percentages' not in func_args and base_percentages is not None:
                        func_args['base_percentages'] = base_percentages
                    # Call the function with epoch, dataset_names, max_epochs, and any additional args
                    percs = func(epoch, self.dataset_names, max_epochs, **func_args)
                else:
                    raise ValueError("schedule_function dict must have 'module' and 'function' keys")
            else:
                raise ValueError("schedule_function must be a callable or a dict with 'module' and 'function' keys")
        # Check if constant percentages are provided
        elif 'percentages' in self.data_scheduler_config:
            percs = self.data_scheduler_config['percentages'].copy()
            # Fill in missing datasets proportionally to their sizes
            total_specified = sum(percs.get(name, 0) for name in self.dataset_names)
            remaining = 1.0 - total_specified
            if remaining > 0:
                # Distribute remaining proportionally to dataset sizes
                size_proportional = self._get_size_proportional_percentages()
                total_unspecified_size = sum(size_proportional.get(name, 0) for name in self.dataset_names if name not in percs)
                for name in self.dataset_names:
                    if name not in percs:
                        if total_unspecified_size > 0:
                            percs[name] = remaining * (size_proportional.get(name, 0) / total_unspecified_size)
                        else:
                            percs[name] = remaining / len([n for n in self.dataset_names if n not in percs])
        else:
            # Default: size-proportional distribution (each sample has equal probability)
            # Unless uniform_datasets is explicitly set to True
            if self.data_scheduler_config.get('uniform_datasets', False):
                # Uniform across datasets (each dataset gets equal percentage)
                percs = {name: 1.0 / len(self.dataset_names) for name in self.dataset_names}
            else:
                # Size-proportional (each sample has equal probability)
                percs = self._get_size_proportional_percentages()
        
        # Normalize to ensure sum is 1.0
        total = sum(percs.values())
        if total > 0:
            percs = {name: p / total for name, p in percs.items()}
        else:
            # Fallback to size-proportional if all percentages are 0
            percs = self._get_size_proportional_percentages()
        
        return percs
    
    def _get_size_proportional_percentages(self) -> Dict[str, float]:
        """
        Compute percentages proportional to dataset sizes.
        
        This ensures each sample has equal probability of being selected,
        regardless of which dataset it comes from.
        
        Returns:
            Dictionary mapping dataset names to percentages proportional to their sizes
        """
        if self.dataset_names is None or len(self.indices_by_dataset) == 0:
            # Fallback to uniform if indices not available
            return {name: 1.0 / len(self.dataset_names) for name in self.dataset_names} if self.dataset_names else {}
        
        # Count total samples
        total_samples = sum(len(indices) for indices in self.indices_by_dataset.values())
        
        if total_samples == 0:
            # Fallback to uniform
            return {name: 1.0 / len(self.dataset_names) for name in self.dataset_names}
        
        # Compute percentages proportional to dataset sizes
        percs = {}
        for name in self.dataset_names:
            if name in self.indices_by_dataset:
                num_samples = len(self.indices_by_dataset[name])
                percs[name] = num_samples / total_samples
            else:
                percs[name] = 0.0
        
        return percs

    def _rank(self) -> int:
        return int(self.trainer.global_rank) if self.trainer is not None else 0

    def _worker_init_fn(self, epoch):
        """worker_init_fn seeding every worker from (seed, rank, epoch, worker_id)."""
        if self.dataloader_generator_seed is None:
            return None
        return functools.partial(
            seed_dataloader_worker, base_seed=self.dataloader_generator_seed, rank=self._rank(), epoch=epoch
        )

    def _train_generator(self, epoch: int) -> Optional[torch.Generator]:
        """Shuffle / sampler generator for this epoch. Same on every rank (required for
        WeightedRandomSampler under Lightning's DistributedSamplerWrapper); under DDP the shuffle of
        a plain dataset is done by Lightning's DistributedSampler (seed = manual_seed, per epoch)."""
        if self.dataloader_generator_seed is None:
            return None
        return torch.Generator().manual_seed(derive_seed(self.dataloader_generator_seed, "sampler", epoch))

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        r"""Get train loader (called every epoch, see reload_dataloaders_every_n_epochs in src/train.py)."""
        epoch = self.trainer.current_epoch if self.trainer is not None else 0
        # The workers are started from this DataLoader before on_train_epoch_start runs, so the
        # dataset's epoch (chunk offsets / mixing RNG) must be set here, not in a LightningModule hook.
        if hasattr(self.train_dataset, 'set_epoch'):
            self.train_dataset.set_epoch(epoch)
        generator = self._train_generator(epoch)
        worker_init_fn = self._worker_init_fn(epoch)
        # Persistent workers would keep a stale copy of the dataset (old epoch) and stale worker seeds.
        persistent_workers = False
        if self.train_config.get('persistent_workers', False):
            print("[DataModule] train_dataloader: persistent_workers=True ignored (the dataloader is "
                  "recreated every epoch so that the dataset epoch and the worker seeds are updated).")

        # Use data scheduler if enabled, otherwise use standard shuffling
        if self.data_scheduler_enabled and self.dataset_names is not None:
            # Get percentages for dataset_name
            name_percs = self._get_percentages(epoch)
            
            dataset_len = len(self.train_dataset)
            weights = np.zeros(dataset_len, dtype=np.float32)
            
            for name, perc in name_percs.items():
                if name in self.indices_by_dataset:
                    idxs = self.indices_by_dataset[name]
                    if len(idxs) > 0:
                        weights[idxs] = perc / len(idxs)
            
            # Normalize so sum(weights) is positive (WeightedRandomSampler uses relative weights)
            w_sum = weights.sum()
            if w_sum > 0:
                weights = weights / w_sum
            
            # Let the dataset use the same weights for internal sampling (e.g. get_stem and
            # __getitem__ retry in MultiAudioRandomSegmentMixDataset), so dataset-wise
            # distribution is consistent with the sampler.
            if hasattr(self.train_dataset, 'set_sampler_weights'):
                self.train_dataset.set_sampler_weights(weights)

            # Create weighted random sampler
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=dataset_len,
                replacement=True,
                generator=generator,
            )
            
            # Log percentages for monitoring every epoch
            print(f"\n[Epoch {epoch}] Data scheduler percentages:")
            
            log_dict = {}
            print("  By dataset_name:")
            for name, perc in sorted(name_percs.items()):
                print(f"    {name}: {perc:.3f} ({perc*100:.1f}%)")
                log_dict[f"data_scheduler/percentage_{name}"] = perc
            
            # Log via trainer's logger if available
            if self.trainer is not None and self.trainer.logger is not None:
                try:
                    log_dict["epoch"] = epoch  # Include epoch as a metric for reference
                    
                    # Use logger's experiment interface (works for WandB, TensorBoard, etc.)
                    if hasattr(self.trainer.logger, 'experiment'):
                        self.trainer.logger.experiment.log(log_dict)
                except Exception as e:
                    # Silently fail if logging is not available (e.g., during initialization)
                    print(f"Failed to log dataset percentages (data scheduler): {e}")
            
            train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.train_config['batch_size'],
                collate_fn=self.train_dataset.collate_fn,
                num_workers=self.train_config['num_workers'],
                pin_memory=True,
                persistent_workers=persistent_workers,
                sampler=sampler,
                shuffle=False,  # Sampler + its generator handle randomness
                worker_init_fn=worker_init_fn,
            )
        else:
            # Standard behavior: use shuffle; clear any previous sampler weights
            if hasattr(self.train_dataset, 'set_sampler_weights'):
                self.train_dataset.set_sampler_weights(None)
            train_loader = DataLoader(
                dataset=self.train_dataset,
                batch_size=self.train_config['batch_size'],
                collate_fn=self.train_dataset.collate_fn,
                num_workers=self.train_config['num_workers'],
                pin_memory=True,
                persistent_workers=persistent_workers,
                shuffle=True,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )

        return train_loader

    def val_dataloader(self):
        # Handle list of validation dataloaders
        if self.val_datasets is not None:
            dataloaders = []
            for dataset, config in zip(self.val_datasets, self.val_config):
                dataloader = DataLoader(
                    dataset=dataset,
                    batch_size=config['batch_size'],
                    collate_fn=dataset.collate_fn,
                    num_workers=config['num_workers'],
                    pin_memory=True,
                    persistent_workers=config['persistent_workers'],
                    shuffle=False,
                    worker_init_fn=self._worker_init_fn('eval'),
                )
                dataloaders.append(dataloader)
            return dataloaders
        
        # Handle single validation dataloader
        if self.val_dataset is not None:
            val_dataloder = DataLoader(
                dataset=self.val_dataset,
                batch_size=self.val_config['batch_size'],
                collate_fn=self.val_dataset.collate_fn,
                num_workers=self.val_config['num_workers'],
                pin_memory=True,
                persistent_workers=self.val_config['persistent_workers'],
                shuffle=False,
                worker_init_fn=self._worker_init_fn('eval'),
            )
            return val_dataloder

    def test_dataloader(self):
        # Use explicit test_dataloader config if provided
        if self.test_config is not None:
            test_dataloader = DataLoader(
                dataset=self.test_dataset,
                batch_size=self.test_config['batch_size'],
                collate_fn=self.test_dataset.collate_fn,
                num_workers=self.test_config['num_workers'],
                pin_memory=True,
                worker_init_fn=self._worker_init_fn('eval'),
                persistent_workers=self.test_config.get('persistent_workers', False),
                shuffle=False
            )
            return test_dataloader

    def teardown(self, stage=None):
        pass
