#!/bin/bash
# Stage 1 of the final system -- mixture restoration: fine-tunes the pretrained Apollo model (first 3
# BSNet blocks + band-split frozen) with the degradation pipeline incl. AAC/DAC/EnCodec codecs, GAN loss
# weight 0.2, no identity loss, lr 1.5e-6 (generator) / 3e-7 (discriminator), 220 epochs with validation
# + a checkpoint every 11 epochs (20 checkpoints are kept). All settings are in
# configs/mixture_restoration/apollo_mixture_restoration.yaml.
#
# AAC/DAC/EnCodec need ffmpeg / descript-audio-codec / encodec at runtime; if one is missing, that
# degradation silently passes the audio through (with a one-time warning in the job's log).
#
# Usage: bash scripts/mixture_restoration/train_apollo.sh <system_seed> [extra Sacred overrides ...]
#   <system_seed>  seed of the final system (41, 42 or 43); the run uses
#                  manual_seed = 100 * <system_seed> + 0 (stage_idx 0 = mixture restoration)
#   extra overrides, e.g. resume_wandb_id=<run id> to resume after the time limit (see TRAINING.md)
#   e.g. bash scripts/mixture_restoration/train_apollo.sh 41
# Checkpoints: <MSR_CKPT_PATH>/apollo_mixture_restoration/<wandb run id>/

source scripts/common.sh

SYSTEM_SEED="$1"
if ! [[ "$SYSTEM_SEED" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/mixture_restoration/train_apollo.sh <system_seed> [extra Sacred overrides ...]"
  exit 1
fi
shift
STAGE_IDX=0
SEED=$((100 * SYSTEM_SEED + STAGE_IDX))
echo "[train_apollo mixture_restoration] system_seed=${SYSTEM_SEED} stage_idx=${STAGE_IDX} manual_seed=${SEED} extra args: $*"

${MSR_LAUNCHER} python -m src.train with config_yaml_file="configs/mixture_restoration/apollo_mixture_restoration.yaml" \
                  system_seed=${SYSTEM_SEED} \
                  stage_idx=${STAGE_IDX} \
                  manual_seed=${SEED} \
                  wandb.project="mixture_restoration" \
                  wandb.name="apollo_mixture_restoration_system${SYSTEM_SEED}_seed${SEED}_final" \
                  wandb.tags="['final', 'system${SYSTEM_SEED}', 'mixture_restoration', 'seed${SEED}']" \
                  "$@"

echo "Job finished successfully"
