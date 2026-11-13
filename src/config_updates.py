import os


# Order of the stems (= order of the mask estimators of the separation model)
EIGHT_STEMS = ["bass", "drums", "other", "vocals", "guitars", "keyboards", "synthesizers", "percussions", "orchestral"]

# Pretrained checkpoints (relative to RESOURCES_FOLDER). The 9-stem separation model is
# initialized from the 6-stem BS-RoFormer-SW checkpoint (see SETUP.md for the download link).
CKPT_DICT = {
    "8s_separation": "6stem/BS-Rofo-SW-Fixed.ckpt",
}

# ============================================================================
# Output / resource paths
# ============================================================================
# All paths are relative to the repository root (the working directory of `python -m src.train`)
# unless an absolute path is given via the environment variables below (see SETUP.md).

# Folder with the pretrained checkpoints (see CKPT_DICT)
RESOURCES_FOLDER = os.environ.get("MSR_RESOURCES_FOLDER", os.path.join("checkpoints", "pretrained"))

# Checkpoints are written to <CKPT_PATH>/<config file name>/<wandb run id>/
CKPT_PATH = os.environ.get("MSR_CKPT_PATH", os.path.join("outputs", "checkpoints"))

# wandb run files and generated datasets are written to <LOG_PATH>/<config file name>/
LOG_PATH = os.environ.get("MSR_LOG_PATH", os.path.join("outputs", "logs"))


# ============================================================================
# Sacred Experiment Configuration
# ============================================================================

def add_configs(ex):
    """
    Adds named configs that switch the separation training config
    (configs/separation/bs_roformer_8stem.yaml) to the post-training pipelines (stem restoration
    dataset generation, evaluation of stage 2, 1+2, 1+2+3 or 2+3). Training-only settings of that
    config (optimizer, LR schedule, data scheduler, ...) have no effect in these pipelines.
    """

    @ex.named_config
    def generate_restoration_dataset():
        """
        Generate the stem restoration dataset: mixtures of the mss8s training set (raw_stems
        only, no stem mixing) are restored by the mixture restoration model (Apollo) and separated
        by the separation model; the (separated stem, clean stem) pairs are stored as a
        HuggingFace dataset. Set the checkpoints with post_training_wandb_id (separation) and
        lightning_module.args.model.args.apollo_model.args.ckpt_path (mixture restoration).
        """
        train_config = dict(enabled=False)
        validation_config = dict(enabled=False, upload_samples=False, upload_ckpt=False)
        test_config = dict(enabled=False, per_stem_metrics_output_path=None)

        generate_inference_dataset_config = dict(
            enabled=True
        )

        lightning_module = dict(
            args=dict(
                val_metrics=["sdr", "si_snr", "multi_mel_snr", "zimtohrli"],
                model=dict(
                    args=dict(
                        separate=True,
                        train_separation_model=False,
                        apollo=True,
                        separation_model=dict(
                            args=dict(
                                finetune_feedforward_layers=False,
                            )
                        ),
                    )
                )
            )
        )

        datamodule = dict(
            args=dict(
                train_dataloader=dict(
                    dataset=dict(
                        args=dict(
                            dataset_names=['raw_stems'],
                            all_dataset_names_except=None,
                            mix_probability_start=0.0,
                            mix_probability_end=0.0,
                        )
                    )
                ),
                data_scheduler=dict(
                    enabled=False
                )
            )
        )

        train = dict(
            trainer=dict(
                args=dict(
                    devices=1
                )
            )
        )

    @ex.named_config
    def inference():
        """
        Evaluation of the separation model (stage 2) without training; enable test_config (MSR
        challenge test set) or msrbench_inference_eval_config (MSRBench) on the command line and
        select the separation checkpoint with post_training_wandb_id / post_training_ckpt_path.
        Combine with `stage1` and/or `stage3` to evaluate the combined system.
        """
        train_config = dict(enabled=False)
        validation_config = dict(enabled=False, upload_samples=False, upload_ckpt=False)

        lightning_module = dict(
            args=dict(
                val_metrics=["sdr", "si_snr", "multi_mel_snr", "zimtohrli"],
                model=dict(
                    args=dict(
                        separate=True,
                        train_separation_model=False,
                        separation_model=dict(
                            args=dict(
                                finetune_feedforward_layers=False,
                            )
                        ),
                    )
                )
            )
        )

        datamodule = dict(
            args=dict(
                data_scheduler=dict(
                    enabled=False
                )
            )
        )

    @ex.named_config
    def stage1():
        """
        Adds the mixture restoration model (stage 1) before the separation model; set its
        checkpoint with lightning_module.args.model.args.apollo_model.args.ckpt_path.
        """
        lightning_module = dict(args=dict(model=dict(args=dict(apollo=True))))

    @ex.named_config
    def stage3():
        """
        Adds the stem-wise restoration models (stage 3) after the separation model; set the
        checkpoints with lightning_module.args.model.args.apollo_stemwise_model.args.ckpt_root /
        ckpt_ids (see ApolloStemwiseRestorationWrapper).
        """
        lightning_module = dict(args=dict(model=dict(args=dict(apollo_stemwise=True))))
