from typing import Dict
import lightning.pytorch as pl
from huggingface_hub import PyTorchModelHubMixin
import transformers

from src.utils import initialize_config

def get_lr_scheduler(
        optimizer,
        num_training_steps,
        schedule_mode="linear",
        num_warmup_steps=20000,
        lr_end=2e-7,
):
    if schedule_mode in {"linear"}:
        print("Linear schedule!")
        return transformers.get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=1.0,
            lr_end=lr_end,
        )
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")


class BaseLightningModule(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        model: Dict,
        loss: Dict,
        optimizer: Dict,
        lr_scheduler:Dict=None,
        use_ema: bool = False,
        ema_decay: float = 0.9999,
    ):

        super().__init__()
        self.model_config = model
        self.model = initialize_config(self.model_config)

        self.loss_config = loss
        self.loss_func = initialize_config(self.loss_config)

        self.optimizer_config = optimizer

        # Always use parameter grouping if the model supports it
        # This ensures lr_decay_factor and weight_decay are always applied when possible
        # If lr_pretrained_masks and lr_new_masks are not specified, they default to the standard LR
        if hasattr(self.model, 'parameter_groups'):
            lr = self.optimizer_config['args']['lr']
            # Default mask learning rates to the standard LR if not specified
            lr_pretrained_masks = self.optimizer_config.get('lr_pretrained_masks', lr)
            lr_new_masks = self.optimizer_config.get('lr_new_masks', lr)

            # LR decay factor
            lr_decay_factor = self.optimizer_config.get('lr_decay_factor', 0.95)

            # Weight decay
            weight_decay = self.optimizer_config['args'].get('weight_decay', 0.0)
            
            # Check if this is MultipleModelsWrapper and extract per-model learning rates
            from src.models.multiple_models_wrapper import MultipleModelsWrapper
            if isinstance(self.model, MultipleModelsWrapper):
                # Extract per-model learning rates if specified
                lr_separation = self.optimizer_config.get('lr_separation', lr)
                
                params = self.model.parameter_groups(
                    lr,
                    lr_pretrained_masks,
                    lr_new_masks,
                    lr_decay_factor,
                    weight_decay,
                    lr_separation=lr_separation,
                )

        else:
            # Fallback: use simple parameter iterator if model doesn't support parameter grouping
            # Note: weight_decay from optimizer args will still be applied by the optimizer,
            # but lr_decay_factor cannot be applied without parameter groups
            print("WARNING: optimizer: parameter_groups not available, applying weight_decay but NOT lr_decay_factor!")
            params = list(self.model.parameters())

        # Skip optimizer when there are no trainable parameters (e.g. inference-only runs)
        if len(params) == 0:
            params_empty = True
        elif isinstance(params[0], dict):
            params_empty = all(len(g.get('params', [])) == 0 for g in params)
        else:
            params_empty = False
            
        if params_empty:
            self.optimizer = None
        else:
            self.optimizer_config['args']['params'] = params  # modify if some parts are frozen
            self.optimizer = initialize_config(self.optimizer_config)

        self.lr_scheduler_config = lr_scheduler

        # Initialize EMA if enabled
        self.use_ema = use_ema
        if self.use_ema:
            from .ema import ExponentialMovingAverage
            self.ema = ExponentialMovingAverage(self.model, decay=ema_decay)
            print(f"EMA enabled with decay={ema_decay}")
        else:
            self.ema = None

    def _normalize_learning_rates(self):
        """Ensure all learning rates in parameter groups are floats (not lists/tuples).
        
        This can happen when resuming from checkpoints that were saved incorrectly
        or when learning rates are set as sequences instead of scalars.
        """
        for group in self.optimizer.param_groups:
            if 'lr' in group:
                lr = group['lr']
                if isinstance(lr, (list, tuple)):
                    # If lr is a sequence, take the first element
                    if len(lr) > 0:
                        group['lr'] = float(lr[0])
                        print(f"WARNING: Learning rate was a sequence {lr}, converted to float: {group['lr']}")
                    else:
                        raise ValueError(f"Learning rate sequence is empty: {lr}")
                elif not isinstance(lr, (int, float)):
                    # Convert to float if it's some other numeric type
                    group['lr'] = float(lr)

    def forward(self, x):
        raise NotImplementedError()

    def training_step(self, batch_data_dict, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch_data_dict, batch_idx):
        raise NotImplementedError()

    def configure_optimizers(self):
        r"""Configure optimizer.
            will be called automatically
        """
        # No optimizer when there are no trainable parameters (e.g. inference-only setup)
        if self.optimizer is None:
            return []

        # Ensure all learning rates in parameter groups are floats (not lists/tuples)
        # This can happen when resuming from checkpoints that were saved incorrectly
        self._normalize_learning_rates()

        if self.lr_scheduler_config and self.lr_scheduler_config['schedule_mode'] is not None:
            # Get estimated stepping batches - this might be None during initialization
            # PyTorch Lightning will restore scheduler state from checkpoint when resuming
            num_training_steps = self.trainer.estimated_stepping_batches
            
            # If estimated_stepping_batches is None (can happen during initialization),
            # use a placeholder value. PyTorch Lightning will restore the actual scheduler
            # state from checkpoint when resuming, so this is only used for fresh runs.
            if num_training_steps is None:
                # Estimate based on max_epochs if available
                max_epochs = getattr(self.trainer, 'max_epochs', None)
                if max_epochs is not None:
                    # Rough estimate: assume ~6 steps per epoch (will be updated when trainer is ready)
                    num_training_steps = max_epochs * 6
                    print(f"WARNING: estimated_stepping_batches is None, using placeholder: {num_training_steps}")
                else:
                    num_training_steps = 120*6  # Fallback placeholder
                    print(f"WARNING: estimated_stepping_batches is None and max_epochs unavailable, using placeholder: {num_training_steps}")

            schedule_mode = self.lr_scheduler_config['schedule_mode']
            num_warmup_steps = self.lr_scheduler_config.get('num_warmup_steps', 0)
            lr_end = self.lr_scheduler_config.get('lr_end', None)

            print("schedule_mode: ", schedule_mode)
            print("num_warmup_steps: ", num_warmup_steps)
            print("lr_end: ", lr_end)
            print("num_training_steps: ", num_training_steps)

            scheduler = get_lr_scheduler(
                self.optimizer,
                num_training_steps,
                schedule_mode=schedule_mode,
                num_warmup_steps=num_warmup_steps,
                lr_end=lr_end,
            )

            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    'scheduler': scheduler,
                    'interval': "step",
                    'frequency': 1,
                }
            }
        else:
            return self.optimizer

    def setup(self, stage):
        """
        Called at the beginning of fit (train + validate), validate, test, or predict.
        This is a good hook when you need to build models dynamically or adjust
        something about them. This hook is called on every process when using DDP.
        """
        # Normalize learning rates after checkpoint loading (this runs after state restoration)
        # This ensures learning rates are floats even if checkpoint restored them as sequences
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            self._normalize_learning_rates()
        
        # Reinitialize EMA after model has been moved to the correct device
        # Only if EMA state wasn't loaded from checkpoint (on_load_checkpoint handles that)
        if self.use_ema and self.ema is not None:
            # Check if EMA was already initialized (from checkpoint)
            if not self.ema.ema_params:
                # Reinitialize EMA parameters on the correct device
                for name, param in self.model.named_parameters():
                    if param.requires_grad:
                        self.ema.ema_params[name] = param.data.clone().to(param.device)
            else:
                # EMA was loaded from checkpoint, just ensure it's on the correct device
                for name, param in self.model.named_parameters():
                    if name in self.ema.ema_params:
                        self.ema.ema_params[name] = self.ema.ema_params[name].to(param.device)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """
        Perform a single optimization step.
        
        Override to add custom behavior like gradient clipping or EMA updates.
        """
        # Call the actual optimizer step
        optimizer.step(closure=optimizer_closure)
        
        # Update EMA after optimizer step
        if self.use_ema and self.ema is not None:
            self.ema.update()

    def on_train_epoch_end(self):
        """Called at the end of training epoch."""

    def on_validation_epoch_end(self):
        """Called at the end of validation epoch."""

    def on_save_checkpoint(self, checkpoint):
        """Save EMA state to checkpoint."""
        if self.use_ema and self.ema is not None:
            checkpoint['ema_state_dict'] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        """Load EMA state from checkpoint."""
        if self.use_ema and self.ema is not None:
            if 'ema_state_dict' in checkpoint:
                self.ema.load_state_dict(checkpoint['ema_state_dict'])
                print("Loaded EMA state from checkpoint")
            else:
                print("Warning: EMA enabled but no EMA state found in checkpoint. EMA will be reinitialized.")
