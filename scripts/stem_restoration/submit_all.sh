#!/bin/bash
# Stage 3b of the final system: runs the stem-wise restoration fine-tuning (train_apollo.sh) for all
# 8 stems and every given generator learning rate -- one run per (stem, learning rate) pair.
# The learning rates have no default on purpose: always pass them explicitly.
#
# By default the runs are executed one after another (MSR_SUBMIT="bash"). Set MSR_SUBMIT to a job
# submission command to run them in parallel, e.g. on SLURM:
#   MSR_SUBMIT="sbatch --gres=gpu:1" MSR_LAUNCHER="srun" bash scripts/stem_restoration/submit_all.sh ...
#
# Usage: bash scripts/stem_restoration/submit_all.sh <system_seed> <mixture_restoration_wandb_id> <separation_wandb_id> <lr> [<lr> ...]
#   e.g. bash scripts/stem_restoration/submit_all.sh 41 abcd1234 efgh5678 3e-5 1e-5

SYSTEM_SEED="$1"
MIXTURE_RESTORATION_WANDB_ID="$2"
SEPARATION_WANDB_ID="$3"
if ! [[ "$SYSTEM_SEED" =~ ^[0-9]+$ ]] || [ -z "$MIXTURE_RESTORATION_WANDB_ID" ] || [ -z "$SEPARATION_WANDB_ID" ] || [ $# -lt 4 ]; then
  echo "Usage: bash scripts/stem_restoration/submit_all.sh <system_seed> <mixture_restoration_wandb_id> <separation_wandb_id> <lr> [<lr> ...]"
  exit 1
fi
LEARNING_RATES=("${@:4}")

source scripts/common.sh
MSR_SUBMIT="${MSR_SUBMIT:-bash}"

STEMS=(
  bass
  drums
  vocals
  guitars
  keyboards
  synthesizers
  percussions
  orchestral
)

for STEM in "${STEMS[@]}"; do
  for LR in "${LEARNING_RATES[@]}"; do
    echo "Running system ${SYSTEM_SEED}, stem ${STEM}, lr ${LR} (mixres ${MIXTURE_RESTORATION_WANDB_ID}, sep ${SEPARATION_WANDB_ID})"
    ${MSR_SUBMIT} scripts/stem_restoration/train_apollo.sh "${SYSTEM_SEED}" "${MIXTURE_RESTORATION_WANDB_ID}" "${SEPARATION_WANDB_ID}" "${STEM}" "${LR}"
  done
done
