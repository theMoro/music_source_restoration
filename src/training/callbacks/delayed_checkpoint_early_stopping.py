from typing import Any

from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


class DelayedModelCheckpoint(ModelCheckpoint):
    """ModelCheckpoint that writes nothing (best or last) before a minimum epoch.

    Pairs with DelayedEarlyStopping (same `min_epochs_before_ckpt_saving` value)
    so early stopping only starts tracking once checkpoints start being written --
    see that class for how this determines the earliest possible stop epoch.
    """

    def __init__(self, *args: Any, min_epochs_before_ckpt_saving: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.min_epochs_before_ckpt_saving = min_epochs_before_ckpt_saving

    def _should_skip_saving_checkpoint(self, trainer: Any) -> bool:
        return (
            trainer.current_epoch < self.min_epochs_before_ckpt_saving
            or super()._should_skip_saving_checkpoint(trainer)
        )


class DelayedEarlyStopping(EarlyStopping):
    """EarlyStopping that only starts tracking (best score / wait_count) once
    `trainer.current_epoch` reaches `min_epochs_before_ckpt_saving`.

    Before that epoch, checks are skipped entirely -- best_score and wait_count
    stay untouched. The first check at `min_epochs_before_ckpt_saving` always
    counts as an improvement (best_score is still at its initial +/-inf), so
    patience only starts being spent from there. Earliest possible stop is
    therefore `min_epochs_before_ckpt_saving + patience` epochs.
    """

    def __init__(self, *args: Any, min_epochs_before_ckpt_saving: int = 0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.min_epochs_before_ckpt_saving = min_epochs_before_ckpt_saving

    def _run_early_stopping_check(self, trainer: Any) -> None:
        if trainer.current_epoch < self.min_epochs_before_ckpt_saving:
            return
        super()._run_early_stopping_check(trainer)
