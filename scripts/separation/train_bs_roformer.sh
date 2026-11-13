#!/bin/bash
# Stage 2 of the final system -- separation model training: BS-RoFormer (initialized from the
# BS-RoFormer-SW checkpoint) on the mss8s dataset with random stem mixing. All settings are in
# configs/separation/bs_roformer_8stem.yaml.
#
# manual_seed is the master seed: weight init, dataloader shuffling, the dataset's segment/mixing RNG
# and the augmentations are all derived from it (see src/training/helpers/seeding.py).
#
# Usage: bash scripts/separation/train_bs_roformer.sh <system_seed> [extra Sacred overrides ...]
#   <system_seed>  seed of the final system (41, 42 or 43); the run uses
#                  manual_seed = 100 * <system_seed> + 1 (stage_idx 1 = separation)
#   extra overrides, e.g. resume_wandb_id=<run id> to resume after the time limit (see TRAINING.md)
#   e.g. bash scripts/separation/train_bs_roformer.sh 41
# Checkpoints: <MSR_CKPT_PATH>/bs_roformer_8stem/<wandb run id>/last.ckpt

source scripts/common.sh

SYSTEM_SEED="$1"
if ! [[ "$SYSTEM_SEED" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/separation/train_bs_roformer.sh <system_seed> [extra Sacred overrides ...]"
  exit 1
fi
shift
STAGE_IDX=1
SEED=$((100 * SYSTEM_SEED + STAGE_IDX))
echo "[train_bs_roformer] system_seed=${SYSTEM_SEED} stage_idx=${STAGE_IDX} manual_seed=${SEED} extra args: $*"

${MSR_LAUNCHER} python -m src.train with config_yaml_file="configs/separation/bs_roformer_8stem.yaml" \
                  system_seed=${SYSTEM_SEED} \
                  stage_idx=${STAGE_IDX} \
                  manual_seed=${SEED} \
                  wandb.project="separation" \
                  wandb.name="bs_roformer_separation_system${SYSTEM_SEED}_seed${SEED}_final" \
                  wandb.tags="['final', 'system${SYSTEM_SEED}', 'separation', 'seed${SEED}']" \
                  "$@"

echo "Job finished successfully"
