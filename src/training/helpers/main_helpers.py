"""
Helper functions for main() to simplify its structure.
"""
import os
import socket
import pathlib
import time
import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import Callback
from munch import DefaultMunch
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from src.config_updates import CKPT_PATH
from src.utils import initialize_config, inject_dataset_paths_into_config
from src.training.lightningmodule.helper import load_ckpt
from src.training.helpers.checkpoint_utils import restore_columns_from_checkpoint
from src.training.helpers.seeding import PerBatchSeedCallback


def setup_rank_and_environment(_config):
    """Setup rank, local rank, nodes, and hostname."""
    rank = rank_zero_only.rank
    print("rank:", rank)

    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    print("local_rank:", local_rank)

    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    print("num_nodes:", num_nodes)
    if num_nodes > 1:
        print("Updating number of nodes:", num_nodes)
        _config['train']['trainer']['args']['num_nodes'] = num_nodes

    hostname = socket.gethostname()
    print("hostname:", hostname)
    
    # Print distributed training environment variables for debugging
    print("\n" + "="*80)
    print("Distributed Training Environment Variables:")
    print("="*80)
    dist_env_vars = [
        'MASTER_ADDR', 'MASTER_PORT', 'RANK', 'LOCAL_RANK', 'WORLD_SIZE',
        'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NNODES', 'SLURM_NTASKS',
        'SLURM_NODEID', 'SLURM_JOB_NODELIST', 'SLURM_STEP_NODELIST'
    ]
    for var in dist_env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"  {var}: {value}")
    print("="*80 + "\n")

    return rank, local_rank, num_nodes, hostname


def _wait_for_marker(marker_file, timeout=3600, check_interval=5, process="initialization"):
    """
    Wait for rank 0 to signal completion by checking for a marker file.

    Args:
        marker_file: Path to the marker file
        timeout: Maximum time to wait in seconds (default: 1 hour)
        check_interval: How often to check for the file in seconds (default: 5)

    Returns:
        True if file appeared, False if timeout
    """
    marker_file = pathlib.Path(marker_file)
    start_time = time.time()
    last_status_time = 0

    print(f"[Rank {os.environ.get('SLURM_PROCID', '?')}] Waiting for rank 0 to finish {process}...")
    print(f"  Marker: {marker_file}")
    print(f"  Timeout: {timeout} seconds")

    while not marker_file.exists():
        elapsed = time.time() - start_time
        if elapsed > timeout:
            print(f"  ERROR: Timeout after {elapsed:.1f} seconds waiting for {process}")
            return False

        # Print status every 30 seconds
        if elapsed - last_status_time >= 30:
            remaining = timeout - elapsed
            print(f"  Still waiting... ({elapsed:.0f}s elapsed, {remaining:.0f}s remaining)")
            last_status_time = elapsed

        time.sleep(check_interval)

    elapsed = time.time() - start_time
    print(f"  ✓ {process} completion detected after {elapsed:.1f} seconds")
    return True


def _create_marker(marker_file):
    """Create a marker file to signal that a step is done."""
    marker_file = pathlib.Path(marker_file)
    marker_file.parent.mkdir(parents=True, exist_ok=True)
    marker_file.touch()
    print(f"Created marker: {marker_file}")


def resolve_dataset_paths(_config, rank):
    """
    Read the dataset, prefix and IR paths from the config (see the "Dataset paths" section of the
    yaml configs and SETUP.md).

    Args:
        _config: Configuration dictionary
        rank: Process rank in distributed training

    Returns:
        tuple: ((dataset_path_8_sources, dataset_path_msr_bench, dataset_path_msr_test_set_2025),
                (prefix_path_8_sources, prefix_path_msr_bench),
                ir_paths)
    """
    dataset_path_8_sources = _config['dataset_path_8_sources']
    dataset_path_msr_bench = _config['dataset_path_msr_bench']
    dataset_path_msr_test_set_2025 = _config.get('dataset_path_msr_test_set_2025')
    ir_paths = _config.get('ir_paths', None)

    # Prefixes for relative audio paths stored in the HF datasets. The conversion scripts in
    # preprocessing/ store absolute paths, so these are only needed for datasets that were moved.
    prefix_path_8_sources = _config.get('prefix_path_8_sources')
    prefix_path_msr_bench = _config.get('prefix_path_msr_bench')

    # Synchronize ranks (prevents race conditions where ranks proceed at different speeds).
    # Additional synchronization after wandb initialization happens in sync_ranks_after_wandb_init().
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        if rank == 0:
            print("[Rank 0] All ranks synchronized after dataset path setup")

    dataset_paths = (dataset_path_8_sources, dataset_path_msr_bench, dataset_path_msr_test_set_2025)
    prefix_paths = (prefix_path_8_sources, prefix_path_msr_bench)

    return (dataset_paths, prefix_paths, ir_paths)


def broadcast_dataset_paths(dataset_paths, prefix_paths, ir_paths, rank):
    """
    Broadcast dataset paths, prefix paths, and IR paths from rank 0 to all ranks.

    Returns:
        tuple: ((dataset_paths), (prefix_paths), ir_paths) - same structure as input
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        print("Broadcasting dataset paths and prefix paths from rank 0 to all ranks...")
        data_paths_list = [*dataset_paths, *prefix_paths, ir_paths]
        torch.distributed.broadcast_object_list(data_paths_list, src=0)
        dataset_paths = tuple(data_paths_list[:len(dataset_paths)])
        prefix_paths = tuple(data_paths_list[len(dataset_paths):len(dataset_paths) + len(prefix_paths)])
        ir_paths = data_paths_list[-1]

        if rank != 0:
            print(f"Rank {rank}: Received dataset paths from rank 0: {dataset_paths}")
            print(f"Rank {rank}: Received prefix paths from rank 0: {prefix_paths}")
            print(f"Rank {rank}: Received ir_paths from rank 0: {ir_paths}")

    return (dataset_paths, prefix_paths, ir_paths)


def sync_ranks_after_wandb_init(configs, rank, dataset_paths):
    """
    Synchronize all ranks after wandb initialization.

    This ensures rank 0 has finished wandb initialization before other ranks
    proceed to DDP initialization, preventing connection timeouts.

    Args:
        configs: Munch object with configuration
        rank: Process rank in distributed training
        dataset_paths: Tuple of dataset paths to determine marker location
    """
    dataset_path_8_sources, dataset_path_msr_bench, dataset_path_msr_test_set_2025 = dataset_paths

    # Determine marker location based on dataset type
    dataset_type = configs['datamodule']['args']['train_dataloader']['dataset']['module']
    if dataset_type == "src.data.dataset_8s":
        marker_base_path = dataset_path_8_sources
    elif dataset_type == "src.data.msr_bench":
        marker_base_path = dataset_path_msr_bench
    else:
        marker_base_path = None

    if marker_base_path:
        # Use parent directory of dataset path for marker (shared location)
        init_marker = pathlib.Path(marker_base_path).parent / ".mss_init_complete"

        if rank == 0:
            # Rank 0 creates the marker after wandb initialization is complete
            _create_marker(init_marker)
            print(f"[Rank {rank}] Created initialization completion marker after wandb init")
        else:
            # Non-rank-0 processes wait for rank 0 to signal completion
            if not init_marker.exists():
                print(f"[Rank {rank}] Waiting for rank 0 to complete wandb initialization...")
                success = _wait_for_marker(init_marker, timeout=300, process="wandb initialization")  # 5 min timeout
                if not success:
                    print(f"[Rank {rank}] WARNING: Timeout waiting for rank 0 initialization marker")
                    print(f"[Rank {rank}] Proceeding anyway (this may cause DDP initialization issues)")
                else:
                    print(f"[Rank {rank}] Rank 0 initialization complete, proceeding...")


def setup_checkpoint_paths(configs, rank, logger):
    """Setup checkpoint save paths and wandb_id."""
    config_filename = pathlib.Path(configs['config_yaml_file']).stem

    ckpt_save_path = os.path.join(CKPT_PATH, config_filename)
    os.makedirs(ckpt_save_path, exist_ok=True)
    run_id = len([f for f in os.listdir(ckpt_save_path) if f.startswith('run_')])

    wandb_id = None

    if rank == 0:
        if logger is not None:
            wandb_id = logger.experiment.id
            if wandb_id is not None:
                os.makedirs(os.path.join(ckpt_save_path, f"{wandb_id}"), exist_ok=True)

    # Broadcast wandb_id from rank 0 to all ranks for multi-GPU training
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        wandb_id_list = [wandb_id] if wandb_id is not None else [None]
        torch.distributed.broadcast_object_list(wandb_id_list, src=0)
        wandb_id = wandb_id_list[0]
        if rank != 0 and wandb_id is not None:
            print(f"Rank {rank}: Received wandb_id from rank 0: {wandb_id}")

    return ckpt_save_path, run_id, wandb_id


def _disable_deterministic_memory_fill():
    """Turn off the NaN-filling of uninitialized tensors that deterministic mode enables.

    torch.use_deterministic_algorithms(True) also sets
    torch.utils.deterministic.fill_uninitialized_memory, which fills every torch.empty() result
    with NaN. That fill is not implemented for ComplexHalf, so the STFT backward under
    16-mixed precision fails ("fill_empty_deterministic_" not implemented for 'ComplexHalf').
    The fill is only a debugging aid for reads of uninitialized memory; it does not affect
    results or reproducibility.
    """
    torch.utils.deterministic.fill_uninitialized_memory = False


def setup_deterministic(_config):
    """Seed all RNGs with ``manual_seed`` (always, before the model is built) and set up
    deterministic training if ``deterministic`` is enabled.

    Mutates ``_config`` so later ``DefaultMunch.fromDict(_config)`` (e.g. in
    ``initialize_components``) preserves ``Trainer(..., deterministic=True)``.
    """
    manual_seed = int(_config.get('manual_seed', 42))
    # workers=False: DataModule seeds its workers itself (src.training.helpers.seeding.seed_dataloader_worker).
    # Also sets PL_GLOBAL_SEED, which Lightning uses as the DistributedSampler shuffle seed.
    pl.seed_everything(manual_seed, workers=False)
    if not _config.get('deterministic'):
        return
    # Required for deterministic cuBLAS matmul on CUDA (safe if set late for some ops).
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=True)
    _disable_deterministic_memory_fill()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    _config.setdefault('train', {}).setdefault('trainer', {}).setdefault('args', {})['deterministic'] = True


class DeterministicAlgorithmsWarnOnlyCallback(Callback):
    """
    Lightning's Trainer(deterministic=True) enables strict deterministic algorithms, which
    overrides ``torch.use_deterministic_algorithms(..., warn_only=True)`` and errors on CUDA
    ops that have no deterministic implementation (e.g. reflection_pad backward). Re-apply
    warn-only mode so training can proceed while cudnn + seeds still provide most
    reproducibility.
    """

    def _apply(self) -> None:
        torch.use_deterministic_algorithms(True, warn_only=True)
        _disable_deterministic_memory_fill()

    def on_fit_start(self, trainer, pl_module) -> None:
        self._apply()

    def on_validation_start(self, trainer, pl_module) -> None:
        self._apply()

    def on_test_start(self, trainer, pl_module) -> None:
        self._apply()

    def on_predict_start(self, trainer, pl_module) -> None:
        self._apply()

    def on_before_backward(self, trainer, pl_module, loss) -> None:
        self._apply()


def initialize_components(configs, _config, dataset_paths, prefix_paths, ir_paths):
    """
    Initialize data module, model, and callbacks.
    
    Args:
        configs: Munch object with configuration
        _config: Configuration dictionary
        dataset_paths: Tuple of (dataset_path_8_sources, dataset_path_msr_bench,
                     dataset_path_msr_test_set_2025)
        prefix_paths: Tuple of (prefix_path_8_sources, prefix_path_msr_bench)
        ir_paths: IR paths string
    """
    # Unpack grouped tuples
    dataset_path_8_sources, dataset_path_msr_bench, dataset_path_msr_test_set_2025 = dataset_paths
    prefix_path_8_sources, prefix_path_msr_bench = prefix_paths
    
    # Update batch size if specified
    if configs['batch_size'] > 0:
        print(f"Use batchsize of {configs['batch_size']}")
        configs['datamodule']['args']['train_dataloader']['batch_size'] = configs['batch_size']
        if 'val_dataloader' in configs['datamodule']['args']:
            configs['datamodule']['args']['val_dataloader']['batch_size'] = configs['batch_size']

    # Inject dataset paths and prefix paths into dataset configs
    inject_dataset_paths_into_config(
        _config,
        dataset_path_8_sources=dataset_path_8_sources,
        dataset_path_msr_bench=dataset_path_msr_bench,
        dataset_path_msr_test_set_2025=dataset_path_msr_test_set_2025,
        prefix_path_8_sources=prefix_path_8_sources,
        prefix_path_msr_bench=prefix_path_msr_bench
    )
    
    # Inject ir_paths into lightning module config
    if 'lightning_module' in _config and 'args' in _config['lightning_module']:
        _config['lightning_module']['args']['ir_paths'] = ir_paths

    # Propagate top-level `stem` / `dataset_root` (stem-wise restoration fine-tuning, see
    # configs/stem_restoration/apollo_stem_restoration.yaml) into both train/val dataset args, so launch
    # scripts only need to override the top-level keys once instead of duplicating them under
    # both datamodule.args.train_dataloader.dataset.args and .val_dataloader.dataset.args.
    for _dl_key in ('train_dataloader', 'val_dataloader'):
        _dl = _config.get('datamodule', {}).get('args', {}).get(_dl_key)
        if not _dl or 'dataset' not in _dl or 'args' not in _dl['dataset']:
            continue
        _ds_args = _dl['dataset']['args']
        if 'stem' in _ds_args and _config.get('stem') is not None:
            _ds_args['stem'] = _config['stem']
        if 'dataset_root' in _ds_args and _config.get('dataset_root') is not None:
            _ds_args['dataset_root'] = _config['dataset_root']

    # All seeds (dataloader_generator_seed, dataset / lightning module seeds) were already resolved
    # from manual_seed by src.training.helpers.seeding.resolve_seeds at the start of main().

    configs = DefaultMunch.fromDict(_config)  # Recreate configs with injected paths

    # Get rank for logging (all ranks print these messages)
    try:
        rank = int(os.environ.get('SLURM_PROCID', os.environ.get('RANK', 0)))
    except:
        rank = 0
    print(f'[Rank {rank}] Initialize data module')
    data_module = initialize_config(configs['datamodule'])

    print(f'[Rank {rank}] Initialize lightning module')
    pl_model = initialize_config(configs['lightning_module'])

    return data_module, pl_model, configs


def setup_callbacks(configs, ckpt_save_path, wandb_id, run_id):
    """Setup and initialize callbacks."""
    print('Initialize callbacks')
    callbacks_configs = configs['train']['callbacks']
    callbacks = []
    # Stem-wise restoration fine-tuning (configs/stem_restoration/apollo_stem_restoration.yaml) wants a
    # deterministic <ckpt_root>/<version>/<stem>/ checkpoint path instead of the usual
    # <ckpt_save_path>/<wandb_id>/ one, so inference can be pointed at a single parent folder
    # containing every stem's checkpoint. Only takes effect when both keys are explicitly set;
    # every other training pipeline in the repo leaves them null and is unaffected.
    ckpt_root_override = configs.get('stem_restoration_ckpt_root', None)
    version_override = configs.get('stem_restoration_version', None)
    stem_override = configs.get('stem', None)
    if ckpt_root_override and version_override and stem_override:
        ckpt_dirpath = os.path.join(ckpt_root_override, version_override, stem_override, f"{wandb_id}")
    else:
        ckpt_dirpath = os.path.join(ckpt_save_path, f"{wandb_id}") if wandb_id else os.path.join(ckpt_save_path, f"{run_id}")
    # Single top-level knob that drives both DelayedModelCheckpoint
    # and DelayedEarlyStopping, so a script only needs to override one CLI arg to delay both
    # checkpoint saving and early-stopping tracking by the same number of epochs.
    min_epochs_before_ckpt_saving = configs.get('min_epochs_before_ckpt_saving', 0)
    # Top-level knob to keep more than the config's default save_top_k best checkpoints (e.g. to
    # compare a few late-training checkpoints per run rather than just the single best one). Left null by
    # default so every existing pipeline keeps its own configured save_top_k unless a script
    # explicitly overrides checkpoint_save_top_k.
    save_top_k_override = configs.get('checkpoint_save_top_k', None)
    for callback_config in callbacks_configs:
        if callback_config['name'] == 'checkpoint':
            callback_config['args']['dirpath'] = ckpt_dirpath
            if 'min_epochs_before_ckpt_saving' in callback_config['args']:
                callback_config['args']['min_epochs_before_ckpt_saving'] = min_epochs_before_ckpt_saving
            if save_top_k_override is not None:
                callback_config['args']['save_top_k'] = save_top_k_override
        if callback_config['name'] == 'early_stopping' and 'min_epochs_before_ckpt_saving' in callback_config['args']:
            callback_config['args']['min_epochs_before_ckpt_saving'] = min_epochs_before_ckpt_saving
        if callback_config['name'] == 'tqdm' and configs['tqdm_rate'] > 0:
            callback_config['args']['refresh_rate'] = configs['tqdm_rate']
        callback = initialize_config(callback_config)
        callbacks.append(callback)

    if configs.get('deterministic', False):
        callbacks.append(DeterministicAlgorithmsWarnOnlyCallback())

    # Reseed python / numpy / torch at the start of every batch from (seed, epoch, batch, rank):
    # makes on-GPU augmentations / degradations reproducible and independent of resumes.
    callbacks.append(PerBatchSeedCallback(int(configs.get('manual_seed', 42))))

    return callbacks


def _resolve_local_ckpt_path(config_filename, wandb_id, ckpt_filename):
    """
    Local path of a checkpoint identified by its wandb run id:
    <CKPT_PATH>/<config_filename>/<wandb_id>/<ckpt_filename>. If it does not exist, the caller
    falls back to downloading the checkpoint from wandb.
    """
    return os.path.join(CKPT_PATH, config_filename, wandb_id, ckpt_filename)


def _wandb_cfg(configs):
    """The `wandb.*` settings given on the command line (project, entity, ...), if any."""
    return configs.get('wandb') or {}


def handle_checkpoint_loading(configs, rank, pl_model):
    """
    Handle checkpoint loading for resume or fine-tuning or post-training processes.

    If we resume training, a path is returned for Trainer.fit(...,ckpt_path=...).
    Otherwise, None is returned and the corresponding checkpoint is loaded into the model directly.
    
    Args:
        configs: Configuration object
        rank: Process rank
        pl_model: Lightning module
    """


    from src.utils import download_checkpoint_from_wandb

    # 1. Case: resume training from a checkpoint
    if configs['resume_wandb_id'] is not None or configs['resume_ckpt_path']:
        ckpt_resume_path = None

        if configs['resume_wandb_id'] is not None:
            resume_ckpt_name = configs['resume_ckpt_name'] if configs['resume_ckpt_name'] else 'last'
            resume_ckpt_name = resume_ckpt_name + '.ckpt' if resume_ckpt_name[:-5] != '.ckpt' else resume_ckpt_name
            config_filename = pathlib.Path(configs['config_yaml_file']).stem
            ckpt_resume_path = _resolve_local_ckpt_path(config_filename, configs['resume_wandb_id'], resume_ckpt_name)

            print(f'Resuming run with wandb_id {configs["resume_wandb_id"]}')
            print(f'Resume checkpoint: {ckpt_resume_path}')

            if rank == 0:
                if not os.path.exists(ckpt_resume_path):
                    print(f'Checkpoint not found locally, attempting to download from wandb...')
                    try:
                        ckpt_resume_path = download_checkpoint_from_wandb(
                            wandb_id=configs['resume_wandb_id'],
                            checkpoint_path=str(ckpt_resume_path),
                            project=_wandb_cfg(configs).get('project') or "music_source_restoration",
                            entity=_wandb_cfg(configs).get('entity'),
                            artifact_name=f"checkpoint-{configs['resume_wandb_id']}"
                        )
                    except FileNotFoundError as e:
                        raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_resume_path}. Error: {e}")

            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()

        elif configs['resume_ckpt_path']:
            ckpt_resume_path = configs['resume_ckpt_path']
            print(f'Resuming run.')
            print(f'Resume checkpoint: {ckpt_resume_path}')

        return ckpt_resume_path

    # 2. Case: loading a checkpoint for fine-tuning or post-training processes (without optimizer states)
    finetune_wandb_id = configs.get('finetune_wandb_id', None)
    post_training_wandb_id = configs.get('post_training_wandb_id', None)
    post_training_ckpt_path = configs.get('post_training_ckpt_path', None)

    finetune_config_filename = configs.get('finetune_config_filename', None)
    post_training_config_filename = configs.get('post_training_config_filename', None)

    if post_training_wandb_id is not None and post_training_ckpt_path is not None:
        raise ValueError("Cannot specify both 'post_training_wandb_id' and 'post_training_ckpt_path'. Please choose one.")
    if post_training_wandb_id is not None and finetune_wandb_id is not None:
        raise ValueError("Cannot specify both 'post_training_wandb_id' and 'finetune_wandb_id'. Please choose one.")
    if post_training_ckpt_path is not None and finetune_wandb_id is not None:
        raise ValueError("Cannot specify both 'post_training_ckpt_path' and 'finetune_wandb_id'. Please choose one.")
    if post_training_wandb_id is None and post_training_ckpt_path is None and finetune_wandb_id is None:
        print("IMPORTANT: No checkpoint loaded for fine-tuning or post-training processes.")
        return None

    if post_training_ckpt_path is not None:
        ckpt_path = post_training_ckpt_path
        context_name = "post-training"

        print(f'Post-training process from checkpoint path {post_training_ckpt_path}')
        print(f'Post-training checkpoint: {ckpt_path}')
    else:
        wandb_id_of_ckpt = finetune_wandb_id if finetune_wandb_id is not None else post_training_wandb_id
        config_filename_of_ckpt = finetune_config_filename if finetune_wandb_id is not None else post_training_config_filename

        ckpt_path = _resolve_local_ckpt_path(config_filename_of_ckpt, wandb_id_of_ckpt, 'last.ckpt')
        context_name = "context-name"

        if finetune_wandb_id is not None:
            print(f'Fine-tuning from checkpoint with wandb_id {finetune_wandb_id}')
            print(f'Fine-tuning config filename: {finetune_config_filename}')
            print(f'Fine-tuning checkpoint: {ckpt_path}')

            context_name = "8-stem fine-tuning"

        elif post_training_wandb_id is not None:
            print(f'Post-training process from checkpoint with wandb_id {post_training_wandb_id}')
            print(f'Post-training config filename: {post_training_config_filename}')
            print(f'Post-training checkpoint: {ckpt_path}')

            context_name = "post-training"

        if rank == 0:
            if not os.path.exists(ckpt_path):
                print(f'Checkpoint not found locally, attempting to download from wandb...')
                try:
                    ckpt_path = download_checkpoint_from_wandb(
                        wandb_id=wandb_id_of_ckpt,
                        checkpoint_path=str(ckpt_path),
                        project=_wandb_cfg(configs).get('project') or "music_source_restoration",
                        entity=_wandb_cfg(configs).get('entity')
                    )
                except FileNotFoundError as e:
                    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Error: {e}")

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()

    # Load model weights for fine-tuning or post-training processes
    if ckpt_path is not None:
        print(f'Loading model weights from checkpoint (without optimizer states)...')
        full_ckpt = load_ckpt(
            ckpt_path,
            pl_model.model
        )

        restore_columns_from_checkpoint(pl_model, full_ckpt, context_name)

        if hasattr(pl_model, 'use_ema') and pl_model.use_ema and pl_model.ema is not None:
            if 'ema_state_dict' in full_ckpt:
                pl_model.ema.load_state_dict(full_ckpt['ema_state_dict'])
                print("Loaded EMA state from checkpoint")
            else:
                print("Warning: EMA enabled but no EMA state found in checkpoint. EMA will use current model weights.")
        print('Model weights loaded successfully. Starting fresh training with new optimizer and epoch counter.')

    return None


def setup_model_optimizations(configs, pl_model):
    """Setup model optimizations (channels_last, compile, etc.)."""
    # Channels Last
    if configs['train']['trainer'].get('channels_last', False):
        pl_model = pl_model.to(memory_format=torch.channels_last)

    # Compile (skipped under deterministic training — torch.compile is not reliably reproducible)
    if configs['train']['trainer'].get('compile', False):
        if configs.get('deterministic', False):
            print(
                "⚠️ Skipping torch.compile while deterministic=True "
                "(compile is not reliably reproducible across runs)."
            )
        else:
            try:
                print("🔧 Compiling model with torch.compile...")
                pl_model = torch.compile(pl_model, mode="default", fullgraph=False)
            except Exception as e:
                print(f"⚠️ torch.compile failed: {e}")

    # Torch Flags
    if configs['train']['trainer'].get('torch_flags', False):
        if configs.get('deterministic', False):
            print(
                "⚡ torch_flags with deterministic=True: math SDP only, cudnn.benchmark off "
                "(flash / mem-efficient attention are non-deterministic on CUDA)"
            )
            if torch.cuda.is_available():
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cudnn.benchmark = False
        else:
            print("⚡ Mixed precision enabled — enabling all attention backends (Flash, Mem-Efficient, Math)")
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    torch.cuda.empty_cache()

    return pl_model


def finalize_wandb_run(logger, rank):
    """Finalize wandb run with timeout and error handling."""
    if rank != 0 or logger is None:
        return

    try:
        import wandb
        import threading
        import time

        print("\n" + "=" * 80)
        print("Finalizing wandb run and waiting for uploads to complete...")
        print("=" * 80)

        print("\nWaiting for wandb uploads to complete (checkpoint + audio samples)...")
        max_upload_wait = 300
        upload_check_interval = 5
        elapsed_time = 0

        print(f"  Waiting {max_upload_wait} seconds for uploads to complete...")
        print(f"  (This ensures checkpoint artifact and audio files are uploaded)")

        while elapsed_time < max_upload_wait:
            try:
                if wandb.run is not None:
                    time.sleep(upload_check_interval)
                    elapsed_time += upload_check_interval

                    if elapsed_time % 30 == 0:
                        remaining = max_upload_wait - elapsed_time
                        print(f"  Waiting... {elapsed_time}/{max_upload_wait}s elapsed ({remaining}s remaining)")
                else:
                    print("  wandb.run is None, proceeding...")
                    break
            except Exception as e:
                print(f"  Note: Error during upload wait: {e}")
                break

        if elapsed_time >= max_upload_wait:
            print(f"✓ Maximum upload wait time ({max_upload_wait}s) reached. Proceeding with finish...")
        else:
            print(f"✓ Upload wait completed ({elapsed_time} seconds)")

        print("  Giving wandb additional 10 seconds for final sync...")
        time.sleep(10)

        finished = threading.Event()
        finish_error = None
        finish_success = False

        def finish_wandb():
            nonlocal finish_error, finish_success
            try:
                logger.experiment.finish(exit_code=0)
                finish_success = True
            except Exception as e:
                finish_error = e
            finally:
                finished.set()

        finish_thread = threading.Thread(target=finish_wandb, daemon=False)
        finish_thread.start()

        if finished.wait(timeout=30):
            if finish_error is not None:
                print(f"WARNING: Error during wandb.finish(): {finish_error}")
                print("Training completed successfully, but wandb sync may be incomplete")
            elif finish_success:
                print("✓ wandb run finalized successfully")
        else:
            print("WARNING: wandb.finish() timed out after 30 seconds")
            print("Force closing wandb connections...")

        try:
            if wandb.run is not None:
                try:
                    if hasattr(wandb.run, '_backend'):
                        wandb.run._backend._shutdown()
                except:
                    pass
                try:
                    wandb.finish(exit_code=0)
                except:
                    pass
        except Exception as e:
            print(f"Note: Error closing wandb backend: {e}")

        if finish_thread.is_alive():
            finish_thread.join(timeout=5)

        print("\nTraining completed. Exiting...")

    except Exception as e:
        print(f"ERROR: Failed to finalize wandb run: {e}")
        print("The training completed successfully, but wandb sync may be incomplete")
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish(exit_code=0)
        except:
            pass
    finally:
        print("Waiting 5 seconds for final sync...")
        import time
        time.sleep(5)

        print("Exiting process...")
        os._exit(0)

