"""
Checkpoint-related utility functions.
"""


def restore_columns_from_checkpoint(lightning_module, full_ckpt, context_name="checkpoint"):
    """
    Restore columns from checkpoint to ensure correct indexing.

    The model output order matches the training column order (from encode_target_sources),
    so we MUST use the same column order that was used during training.
    When a checkpoint is loaded from a previous training stage, the columns from that stage
    are what the model was trained with, and we must use the same order for indexing.

    Args:
        lightning_module: The lightning module to restore columns for
        full_ckpt: Full checkpoint dictionary from load_ckpt()
        context_name: Context name for logging (e.g., "MSRBench evaluation", "validation")

    Returns:
        bool: True if columns were restored from checkpoint, False if using config columns
    """
    try:
        # PyTorch Lightning stores hyper_parameters in the checkpoint
        if 'hyper_parameters' in full_ckpt:
            hparams = full_ckpt['hyper_parameters']
            if 'columns' in hparams:
                ckpt_columns = hparams['columns']
                if ckpt_columns is not None:
                    # Convert to list if needed (might be stored as tuple or other type)
                    ckpt_columns_list = list(ckpt_columns) if not isinstance(ckpt_columns, list) else ckpt_columns
                    if len(ckpt_columns_list) > 0:
                        print(f"[{context_name}] ✓ Restoring columns from checkpoint: {ckpt_columns_list}")
                        lightning_module.columns = ckpt_columns_list
                        return True

        else:
            print(f"[{context_name}] ⚠ Warning: No hyper_parameters in checkpoint")
    except Exception as e:
        pass

    return False

