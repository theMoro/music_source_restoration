#!/bin/bash
# Shared settings of the experiment scripts in scripts/ (sourced by them, not meant to be run).
# Every path can be set with an environment variable; the defaults are relative to the repository
# root (see the "Paths" section of SETUP.md).
#
#   MSR_CKPT_PATH   checkpoints of all runs               (default: outputs/checkpoints)
#   MSR_LOG_PATH    wandb run files                       (default: outputs/logs)
#   MSR_DATA_PATH   generated stem restoration datasets   (default: data/hf)
#   MSR_LAUNCHER    command prefix of the python call     (default: none; e.g. "srun --gpu-bind=none"
#                   when the script is submitted with sbatch for multi-GPU / multi-node training)
#   WANDB_ENTITY    wandb entity (team / user) to log to  (default: your default wandb entity)

# All scripts must be started from the repository root.
if [ ! -f "src/train.py" ]; then
  echo "ERROR: run the scripts from the repository root (e.g. bash scripts/separation/train_bs_roformer.sh 41)." >&2
  exit 1
fi

export MSR_CKPT_PATH="${MSR_CKPT_PATH:-outputs/checkpoints}"
export MSR_LOG_PATH="${MSR_LOG_PATH:-outputs/logs}"
export MSR_DATA_PATH="${MSR_DATA_PATH:-data/hf}"
MSR_LAUNCHER="${MSR_LAUNCHER:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
