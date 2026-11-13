#!/bin/bash
# Stage 3a of the final system -- generates the training data for the stem-wise restoration models:
# mixtures of the mss8s training set (raw_stems only) are restored by the system's mixture
# restoration model and separated by the system's separation model; the (separated stem, clean stem)
# pairs are stored as a HuggingFace dataset under OUTPUT_DIR (final dataset: OUTPUT_DIR/final_dataset).
#
# Usage: bash scripts/stem_restoration/generate_dataset.sh <system_seed> <mixture_restoration_wandb_id> <separation_wandb_id> [extra Sacred overrides ...]
#   <system_seed>  seed of the final system (41, 42 or 43); the run uses
#                  manual_seed = 100 * <system_seed> + 2 (stage_idx 2 = dataset generation)
#   <mixture_restoration_wandb_id> / <separation_wandb_id>: the system's stage-1 / stage-2 runs
#   e.g. bash scripts/stem_restoration/generate_dataset.sh 41 abcd1234 efgh5678
# Output (derived from the arguments, so several systems can be generated at the same time; must not
# exist yet): ${DATASET_BASE}/stem_restoration_dataset_system<seed>_sep-<sep id>_mixres-<mixres id>
# -- scripts/stem_restoration/train_apollo.sh derives the same path from the same arguments.

source scripts/common.sh

SYSTEM_SEED="$1"
MIXTURE_RESTORATION_WANDB_ID="$2"
SEPARATION_WANDB_ID="$3"
if ! [[ "$SYSTEM_SEED" =~ ^[0-9]+$ ]] || [ -z "$MIXTURE_RESTORATION_WANDB_ID" ] || [ -z "$SEPARATION_WANDB_ID" ]; then
  echo "Usage: bash scripts/stem_restoration/generate_dataset.sh <system_seed> <mixture_restoration_wandb_id> <separation_wandb_id> [extra Sacred overrides ...]"
  exit 1
fi
shift 3
STAGE_IDX=2  # to generate the dataset is technically not an extra stage, however, for setting the seed we act like it is
SEED=$((100 * SYSTEM_SEED + STAGE_IDX))

CKPT_BASE="${MSR_CKPT_PATH}"
DATASET_BASE="${MSR_DATA_PATH}"
# Mixture restoration model; the stem-wise restoration models are initialized from the same checkpoint
MIXTURE_RESTORATION_MODEL_CKPT_PATH="${CKPT_BASE}/apollo_mixture_restoration/${MIXTURE_RESTORATION_WANDB_ID}/last.ckpt"
SEPARATION_MODEL_CKPT_PATH="${CKPT_BASE}/bs_roformer_8stem/${SEPARATION_WANDB_ID}/last.ckpt"
OUTPUT_DIR="${DATASET_BASE}/stem_restoration_dataset_system${SYSTEM_SEED}_sep-${SEPARATION_WANDB_ID}_mixres-${MIXTURE_RESTORATION_WANDB_ID}"
echo "[generate_dataset] system_seed=${SYSTEM_SEED} stage_idx=${STAGE_IDX} manual_seed=${SEED} mixres=${MIXTURE_RESTORATION_WANDB_ID} sep=${SEPARATION_WANDB_ID} output_dir=${OUTPUT_DIR}"

if [ ! -f "${MIXTURE_RESTORATION_MODEL_CKPT_PATH}" ]; then
  echo "ERROR: mixture-restoration checkpoint not found: ${MIXTURE_RESTORATION_MODEL_CKPT_PATH}"
  exit 1
fi
if [ ! -f "${SEPARATION_MODEL_CKPT_PATH}" ]; then
  echo "ERROR: separation checkpoint not found: ${SEPARATION_MODEL_CKPT_PATH}"
  exit 1
fi
# Temporary batches of an earlier (e.g. interrupted) run in the same folder would be merged into the
# dataset (duplicates), so the output folder must be new.
if [ -e "${OUTPUT_DIR}" ] && [ -n "$(ls -A "${OUTPUT_DIR}" 2>/dev/null)" ]; then
  echo "ERROR: output folder ${OUTPUT_DIR} exists and is not empty -- delete it and restart."
  exit 1
fi

${MSR_LAUNCHER} python -m src.train with config_yaml_file="configs/separation/bs_roformer_8stem.yaml" generate_restoration_dataset \
                  system_seed=${SYSTEM_SEED} \
                  stage_idx=${STAGE_IDX} \
                  manual_seed=${SEED} \
                  generate_inference_dataset_config.output_dir="${OUTPUT_DIR}" \
                  post_training_config_filename="bs_roformer_8stem" \
                  post_training_wandb_id="${SEPARATION_WANDB_ID}" \
                  lightning_module.args.model.args.apollo_model.args.ckpt_path="${MIXTURE_RESTORATION_MODEL_CKPT_PATH}" \
                  wandb.project="stem_restoration" \
                  wandb.name="generate_stem_restoration_dataset_system${SYSTEM_SEED}_seed${SEED}_sep-${SEPARATION_WANDB_ID}_mixres-${MIXTURE_RESTORATION_WANDB_ID}_final" \
                  wandb.tags="['final', 'system${SYSTEM_SEED}', 'dataset_generation', 'seed${SEED}']" \
                  "$@"

echo "Job finished successfully"
