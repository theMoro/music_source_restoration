import os
import sys
from typing import Dict
import yaml
import importlib

def ignore_warnings():
    import warnings

    # Filter out warnings containing "deprecated"
    warnings.filterwarnings("ignore", message=".*invalid value encountered in cast.*")
    warnings.filterwarnings("ignore", message=".*deprecated.*")
    warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
    warnings.filterwarnings("ignore", message=".*that has Tensor Cores.*")
    warnings.filterwarnings("ignore", message=".*cudnnFinalize Descriptor Failed.*")
    warnings.filterwarnings("ignore", message=".*torch.use_deterministic_algorithms(True, warn_only=True)*")
    from transformers import logging
    logging.set_verbosity_error()


def parse_yaml(config_yaml) -> Dict:
    r"""Parse yaml file.

    Args:
        config_yaml (str or None): config yaml path

    Returns:
        yaml_dict (Dict): parsed yaml file
    """

    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


def initialize_config(module_cfg, reload=False):
    if reload and module_cfg["module"] in sys.modules:
        module = importlib.reload(sys.modules[module_cfg["module"]])
    else:
        module = importlib.import_module(module_cfg["module"])
    if 'args' in module_cfg.keys(): return getattr(module, module_cfg["main"])(**module_cfg["args"])
    return getattr(module, module_cfg["main"])()


def inject_dataset_paths_into_config(config, dataset_path_8_sources=None, dataset_path_msr_bench=None,
                                     dataset_path_msr_test_set_2025=None,
                                     prefix_path_8_sources=None, prefix_path_msr_bench=None):
    """
    Inject dataset paths and prefix paths into dataset configs.
    """
    # Get paths from provided values or config (required, no defaults)
    if dataset_path_8_sources is None:
        if 'dataset_path_8_sources' not in config:
            raise ValueError("dataset_path_8_sources must be provided or specified in config")
        dataset_path_8_sources = config['dataset_path_8_sources']

    if dataset_path_msr_bench is None:
        if 'dataset_path_msr_bench' not in config:
            raise ValueError("dataset_path_msr_bench must be provided or specified in config")
        dataset_path_msr_bench = config['dataset_path_msr_bench']

    # Optional, eval-only: not required for every run (unlike the training dataset paths above).
    if dataset_path_msr_test_set_2025 is None:
        dataset_path_msr_test_set_2025 = config.get('dataset_path_msr_test_set_2025')

    # Get prefix paths from provided values or config
    if prefix_path_8_sources is None:
        prefix_path_8_sources = config.get('prefix_path_8_sources')
    if prefix_path_msr_bench is None:
        prefix_path_msr_bench = config.get('prefix_path_msr_bench')
    
    def inject_path(dataset_config):
        """Inject the appropriate dataset path and prefix path into a dataset config."""
        if 'args' not in dataset_config:
            return
        
        module_name = dataset_config.get('module', '')
        if 'dataset_8s' in module_name:
            dataset_config['args']['root'] = dataset_path_8_sources
            if prefix_path_8_sources is not None:
                dataset_config['args']['prefix_path'] = prefix_path_8_sources
        elif 'msr_test_set_2025' in module_name:
            if dataset_path_msr_test_set_2025 is not None:
                dataset_config['args']['root'] = dataset_path_msr_test_set_2025
        elif 'msr_bench' in module_name:
            dataset_config['args']['root'] = dataset_path_msr_bench
            if prefix_path_msr_bench is not None:
                dataset_config['args']['prefix_path'] = prefix_path_msr_bench
    
    # Inject paths into all dataset configs
    if 'datamodule' in config and 'args' in config['datamodule']:
        if 'train_dataloader' in config['datamodule']['args'] and 'dataset' in config['datamodule']['args']['train_dataloader']:
            inject_path(config['datamodule']['args']['train_dataloader']['dataset'])
        if 'val_dataloader' in config['datamodule']['args']:
            val_dl = config['datamodule']['args']['val_dataloader']
            if isinstance(val_dl, list):
                for dl in val_dl:
                    if isinstance(dl, dict) and 'dataset' in dl:
                        inject_path(dl['dataset'])
            elif isinstance(val_dl, dict) and 'dataset' in val_dl:
                inject_path(val_dl['dataset'])
        if 'test_dataloader' in config['datamodule']['args'] and 'dataset' in config['datamodule']['args']['test_dataloader']:
            inject_path(config['datamodule']['args']['test_dataloader']['dataset'])


def download_checkpoint_from_wandb(wandb_id: str, checkpoint_path: str, project: str = "music_source_restoration",
                                   entity: str = None, artifact_name: str = None) -> str:
    """
    Download checkpoint from wandb artifact if it's not available locally.
    
    Args:
        wandb_id: The wandb run ID
        checkpoint_path: The expected local path where the checkpoint should be
        project: The wandb project name (default: "music_source_restoration")
        entity: The wandb entity name (default: None = WANDB_ENTITY or the default entity of the logged-in user)
        artifact_name: Optional custom artifact name. If None, uses f"checkpoint-{wandb_id}"
    
    Returns:
        The path to the checkpoint file (either existing or downloaded)
    
    Raises:
        FileNotFoundError: If checkpoint is not found locally and cannot be downloaded from wandb
    """
    import wandb
    
    # Check if checkpoint already exists locally
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    # Check if the checkpoint directory exists (might have been downloaded before)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.exists(checkpoint_dir):
        # Check if there's a checkpoint file in the directory
        checkpoint_filename = os.path.basename(checkpoint_path)
        for file in os.listdir(checkpoint_dir):
            if file.endswith('.ckpt'):
                # Found a checkpoint file, use it
                found_path = os.path.join(checkpoint_dir, file)
                print(f"Found existing checkpoint in cache: {found_path}")
                return found_path
    
    # Try to download from wandb
    print(f"Checkpoint not found locally at {checkpoint_path}")
    print(f"Attempting to download from wandb artifact for run {wandb_id}...")
    
    try:
        # Initialize wandb API (no need to start a run)
        api = wandb.Api()
        if entity is None:
            entity = os.environ.get("WANDB_ENTITY") or api.default_entity
        
        # Construct artifact name if not provided
        if artifact_name is None:
            artifact_name = f"checkpoint-{wandb_id}"
        
        # Try to get the artifact
        artifact_path = f"{entity}/{project}/{artifact_name}:latest"
        try:
            artifact = api.artifact(artifact_path)
        except Exception as e:
            # Try without :latest tag
            try:
                artifact_path = f"{entity}/{project}/{artifact_name}"
                artifact = api.artifact(artifact_path)
            except Exception as e2:
                raise FileNotFoundError(
                    f"Could not find wandb artifact '{artifact_path}'. "
                    f"Original error: {e}. Alternative error: {e2}"
                )
        
        # Create directory for checkpoint if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Download artifact to a temporary location first
        import tempfile
        temp_download_dir = tempfile.mkdtemp()
        artifact_dir = artifact.download(root=temp_download_dir)
        
        # Find the checkpoint file in the downloaded artifact
        checkpoint_filename = os.path.basename(checkpoint_path)
        downloaded_path = None
        
        # Look for .ckpt files in the artifact directory
        for root, dirs, files in os.walk(artifact_dir):
            for file in files:
                if file.endswith('.ckpt'):
                    candidate_path = os.path.join(root, file)
                    if downloaded_path is None or file == checkpoint_filename:
                        downloaded_path = candidate_path
                        # If the expected filename matches, use it directly
                        if file == checkpoint_filename:
                            break
        
        if downloaded_path is None:
            import shutil
            shutil.rmtree(temp_download_dir, ignore_errors=True)
            raise FileNotFoundError(
                f"Downloaded artifact from wandb but no .ckpt file found in {artifact_dir}"
            )
        
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Copy to expected location
        import shutil
        shutil.copy2(downloaded_path, checkpoint_path)
        print(f"Copied checkpoint to expected location: {checkpoint_path}")
        
        # Clean up temporary download directory
        shutil.rmtree(temp_download_dir, ignore_errors=True)
        
        print(f"✓ Successfully downloaded checkpoint from wandb artifact: {checkpoint_path}")
        return checkpoint_path
        
    except Exception as e:
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path} and could not download from wandb: {e}"
        )