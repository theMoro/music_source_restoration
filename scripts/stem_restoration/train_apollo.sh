#!/bin/bash
# Stage 3b of the final system -- stem-wise restoration: fine-tunes the system's mixture restoration
# model (the same checkpoint used to generate the dataset, see generate_dataset.sh) on one stem of the
# system's stem restoration dataset (GAN loss weight 0.5, feature matching weight 0.1). All settings are
# in configs/stem_restoration/apollo_stem_restoration.yaml; the generator learning rate is given as an
# argument, the discriminator learning rate is 0.6 * the generator learning rate.
#
# Usage: bash scripts/stem_restoration/train_apollo.sh <system_seed> <mixture_restoration_wandb_id> <separation_wandb_id> <stem> <lr_g> [extra Sacred overrides ...]
#        (or bash scripts/stem_restoration/submit_all.sh for all stems and the given learning rates)
#   <system_seed>  seed of the final system (41, 42 or 43); the run uses
#                  manual_seed = 100 * <system_seed> + 3 (stage_idx 3 = stem-wise restoration)
#   <mixture_restoration_wandb_id> / <separation_wandb_id>: the system's stage-1 / stage-2 runs (the
#       models the dataset was generated with). They select the dataset written by generate_dataset.sh
#       and the initialization checkpoint apollo_mixture_restoration/<mixres id>/last.ckpt.
#   <stem> is one of: bass, drums, vocals, guitars, keyboards, synthesizers, percussions, orchestral
#   <lr_g> is the generator learning rate, e.g. 1e-5
#   e.g. bash scripts/stem_restoration/train_apollo.sh 41 abcd1234 efgh5678 vocals 1e-5
# Checkpoints: ${CKPT_ROOT}/<version>/<stem>/<wandb run id>/last.ckpt with
#   <version> = system<seed>_lr<lr>_sep-<sep id>_mixres-<mixres id> (unique per system and learning rate).

source scripts/common.sh

SYSTEM_SEED="$1"
MIXTURE_RESTORATION_WANDB_ID="$2"
SEPARATION_WANDB_ID="$3"
STEM="$4"
LR_G="$5"
STEMS_ALLOWED=" bass drums vocals guitars keyboards synthesizers percussions orchestral "
if ! [[ "$SYSTEM_SEED" =~ ^[0-9]+$ ]] || [ -z "$MIXTURE_RESTORATION_WANDB_ID" ] || [ -z "$SEPARATION_WANDB_ID" ] \
   || [[ "$STEMS_ALLOWED" != *" ${STEM} "* ]] || [ -z "$LR_G" ]; then
  echo "Usage: bash scripts/stem_restoration/train_apollo.sh <system_seed> <mixture_restoration_wandb_id> <separation_wandb_id> <stem> <lr_g> [extra Sacred overrides ...]"
  echo "  <stem> is one of:${STEMS_ALLOWED}"
  exit 1
fi
shift 5
STAGE_IDX=3
SEED=$((100 * SYSTEM_SEED + STAGE_IDX))
LR_D=$(awk -v lr="$LR_G" 'BEGIN { printf "%g", 0.6 * lr }')

CKPT_BASE="${MSR_CKPT_PATH}"
DATASET_BASE="${MSR_DATA_PATH}"
VERSION="system${SYSTEM_SEED}_lr${LR_G}_sep-${SEPARATION_WANDB_ID}_mixres-${MIXTURE_RESTORATION_WANDB_ID}"
# Same path as OUTPUT_DIR in generate_dataset.sh
DATASET_ROOT="${DATASET_BASE}/stem_restoration_dataset_system${SYSTEM_SEED}_sep-${SEPARATION_WANDB_ID}_mixres-${MIXTURE_RESTORATION_WANDB_ID}/final_dataset"
CKPT_ROOT="${CKPT_BASE}/stem_restoration"
APOLLO_CKPT_PATH="${CKPT_BASE}/apollo_mixture_restoration/${MIXTURE_RESTORATION_WANDB_ID}/last.ckpt"
echo "[train_apollo stem_restoration] system_seed=${SYSTEM_SEED} stage_idx=${STAGE_IDX} manual_seed=${SEED} stem=${STEM} lr_g=${LR_G} lr_d=${LR_D} dataset=${DATASET_ROOT} init=${APOLLO_CKPT_PATH} extra args: $*"
if [ ! -d "${DATASET_ROOT}" ]; then
  echo "ERROR: stem restoration dataset not found: ${DATASET_ROOT}"
  exit 1
fi
if [ ! -f "${APOLLO_CKPT_PATH}" ]; then
  echo "ERROR: mixture-restoration checkpoint not found: ${APOLLO_CKPT_PATH}"
  exit 1
fi

${MSR_LAUNCHER} python -m src.train with config_yaml_file="configs/stem_restoration/apollo_stem_restoration.yaml" \
                  system_seed=${SYSTEM_SEED} \
                  stage_idx=${STAGE_IDX} \
                  manual_seed=${SEED} \
                  stem="${STEM}" \
                  dataset_root="${DATASET_ROOT}" \
                  stem_restoration_ckpt_root="${CKPT_ROOT}" \
                  stem_restoration_version="${VERSION}" \
                  lightning_module.args.generator.args.ckpt_path="${APOLLO_CKPT_PATH}" \
                  lightning_module.args.optimizer_g.args.lr=${LR_G} \
                  lightning_module.args.optimizer_d.args.lr=${LR_D} \
                  wandb.project="stem_restoration" \
                  wandb.name="stem_restoration_apollo_${STEM}_lr${LR_G}_system${SYSTEM_SEED}_seed${SEED}_final" \
                  wandb.tags="['final', 'system${SYSTEM_SEED}', 'stem_restoration', 'seed${SEED}', '${STEM}', 'lr${LR_G}']" \
                  "$@"

echo "Job finished successfully"
