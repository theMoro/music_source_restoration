#!/bin/bash
# Evaluation of the separation model alone or combined with the restoration models:
#   stages 2      : separation (BS-RoFormer)
#   stages 1+2    : mixture restoration (Apollo) -> separation
#   stages 2+3    : separation -> stem-wise restoration (one Apollo model per stem)
#   stages 1+2+3  : mixture restoration -> separation -> stem-wise restoration (full system)
# on the MSR Challenge 2025 test set (non-blind + streaming subsets, metrics logged to wandb) or on
# MSRBench (metrics logged to wandb and written to results/<EVAL_LABEL>/).
#
# Stem-wise restoration: the checkpoint of every stem is given by its wandb run id in
# STEM_RESTORATION_CKPT_IDS (looked up as <STEM_RESTORATION_CKPT_ROOT>/*/<stem>/<wandb_id>/last.ckpt);
# stems in SKIP_STEMS are not restored.
#
# Usage: bash scripts/evaluation/evaluate.sh <system_seed> <stages> <benchmark> <separation_wandb_id> [mixture_restoration_wandb_id]
#                           [stem_restoration_ckpt_ids] [stem_set_label] [skip_stems]
#   <system_seed> seed of the final system (41, 42 or 43); names the results folder and the wandb run
#   <stages>      one of: 2, 1+2, 2+3, 1+2+3
#   <benchmark>   one of: test_set, msrbench
#   <separation_wandb_id>          run id of the system's separation model
#   [mixture_restoration_wandb_id] run id of the system's mixture restoration model (required for
#                                  stages 1+*, pass "" for stages 2 / 2+3); uses last.ckpt, or
#                                  <run id>@<epoch> for the checkpoint of that epoch (0-based epoch as in
#                                  the checkpoint name, e.g. y8c8uaej@98 -> <run id>/epoch=98-val/mel_snr=*.ckpt)
#   Stages *+3 only:
#   [stem_restoration_ckpt_ids]    dict of stem -> wandb run id of the stem-wise restoration models
#                                  (required), e.g. "{'bass': 'id1', 'drums': 'id2', ...}"
#   [stem_set_label]               name of the set of stem restoration models, appended to the results
#                                  folder / wandb name (default: none; "" = none)
#   [skip_stems]                   stems without stem-wise restoration (default: test_set:
#                                  other, bass, drums, keyboards; msrbench: other)
#
# Results (both benchmarks): results/<EVAL_LABEL>/per_stem_metrics.csv (+ per_sample_scores.csv for
# MSRBench), EVAL_LABEL = system<seed>_stages_<stages>_<benchmark>_sep_<id>[_mixres_<id>][_stemres_<label>].

source scripts/common.sh

SYSTEM_SEED="$1"
STAGES="$2"
BENCHMARK="$3"
SEPARATION_WANDB_ID="$4"
ENHANCEMENT_WANDB_ID="$5"
STEM_RESTORATION_CKPT_IDS="$6"
STEM_SET_LABEL="$7"
SKIP_STEMS="$8"
USAGE="Usage: bash scripts/evaluation/evaluate.sh <system_seed> <stages: 2 | 1+2 | 2+3 | 1+2+3> <benchmark: test_set | msrbench> <separation_wandb_id> [mixture_restoration_wandb_id] [stem_restoration_ckpt_ids] [stem_set_label] [skip_stems]"
if ! [[ "$SYSTEM_SEED" =~ ^[0-9]+$ ]] || [ -z "$STAGES" ] || [ -z "$BENCHMARK" ] || [ -z "$SEPARATION_WANDB_ID" ]; then
  echo "$USAGE"; exit 1
fi
if [[ "$STAGES" == 1+* ]] && [ -z "$ENHANCEMENT_WANDB_ID" ]; then
  echo "ERROR: stages ${STAGES} need the mixture restoration wandb id."; echo "$USAGE"; exit 1
fi
if [[ "$STAGES" == *+3 ]] && [ -z "$STEM_RESTORATION_CKPT_IDS" ]; then
  echo "ERROR: stages ${STAGES} need the stem restoration checkpoint ids."; echo "$USAGE"; exit 1
fi

# Separation model (stage 2): checkpoint <CKPT_PATH>/bs_roformer_8stem/<SEPARATION_WANDB_ID>/last.ckpt (or
# downloaded from wandb); alternatively set post_training_ckpt_path=<path> instead of the two options below.
SEPARATION_CKPT_ARGS="post_training_config_filename=bs_roformer_8stem post_training_wandb_id=${SEPARATION_WANDB_ID}"
# Mixture restoration model (stage 1)
ENHANCEMENT_CKPT_DIR="${MSR_CKPT_PATH}/apollo_mixture_restoration"
if [[ "$ENHANCEMENT_WANDB_ID" == *@* ]]; then
  ENHANCEMENT_EPOCH="${ENHANCEMENT_WANDB_ID#*@}"
  ENHANCEMENT_WANDB_ID="${ENHANCEMENT_WANDB_ID%@*}"
  ENHANCEMENT_MATCHES=( "${ENHANCEMENT_CKPT_DIR}/${ENHANCEMENT_WANDB_ID}"/epoch=${ENHANCEMENT_EPOCH}-*.ckpt
                        "${ENHANCEMENT_CKPT_DIR}/${ENHANCEMENT_WANDB_ID}"/epoch=${ENHANCEMENT_EPOCH}-*/*.ckpt )
  ENHANCEMENT_MATCHES=( $(ls -1 "${ENHANCEMENT_MATCHES[@]}" 2>/dev/null) )
  if [ "${#ENHANCEMENT_MATCHES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly one checkpoint of epoch ${ENHANCEMENT_EPOCH} in ${ENHANCEMENT_CKPT_DIR}/${ENHANCEMENT_WANDB_ID}, found ${#ENHANCEMENT_MATCHES[@]}: ${ENHANCEMENT_MATCHES[*]}"
    exit 1
  fi
  ENHANCEMENT_CKPT="${ENHANCEMENT_MATCHES[0]}"
  ENHANCEMENT_LABEL="${ENHANCEMENT_WANDB_ID}_ep${ENHANCEMENT_EPOCH}"
else
  ENHANCEMENT_CKPT="${ENHANCEMENT_CKPT_DIR}/${ENHANCEMENT_WANDB_ID}/last.ckpt"
  ENHANCEMENT_LABEL="${ENHANCEMENT_WANDB_ID}"
fi
# Stem-wise restoration models (stage 3)
STEM_RESTORATION_CKPT_ROOT="${MSR_CKPT_PATH}/stem_restoration"
if [ -z "$SKIP_STEMS" ]; then
  if [ "$BENCHMARK" = "test_set" ]; then
    SKIP_STEMS="['other', 'bass', 'drums', 'keyboards']"
  else
    SKIP_STEMS="['other']"
  fi
fi

echo "[eval] system_seed=${SYSTEM_SEED} stages=${STAGES} benchmark=${BENCHMARK} sep=${SEPARATION_WANDB_ID} mixres=${ENHANCEMENT_LABEL}"
if [[ "$STAGES" == *+3 ]]; then
  echo "[eval] stem_set=${STEM_SET_LABEL:-<none>} skip_stems=${SKIP_STEMS} ckpt_ids=${STEM_RESTORATION_CKPT_IDS}"
fi

EVAL_LABEL="system${SYSTEM_SEED}_stages_${STAGES}_${BENCHMARK}_sep_${SEPARATION_WANDB_ID}"
if [[ "$STAGES" == 1+* ]]; then
  EVAL_LABEL="${EVAL_LABEL}_mixres_${ENHANCEMENT_LABEL}"
fi
if [[ "$STAGES" == *+3 ]] && [ -n "$STEM_SET_LABEL" ]; then
  EVAL_LABEL="${EVAL_LABEL}_stemres_${STEM_SET_LABEL}"
fi

# Named configs and arguments of the selected stages
STAGE_ARGS=()
case "$STAGES" in
  2)     NAMED_CONFIGS="inference" ;;
  1+2)   NAMED_CONFIGS="inference stage1" ;;
  2+3)   NAMED_CONFIGS="inference stage3" ;;
  1+2+3) NAMED_CONFIGS="inference stage1 stage3" ;;
  *) echo "Unknown stages: $STAGES"; exit 1 ;;
esac
if [[ "$STAGES" == 1+* ]]; then
  if [ ! -f "${ENHANCEMENT_CKPT}" ]; then
    echo "ERROR: mixture-restoration checkpoint not found: ${ENHANCEMENT_CKPT}"
    exit 1
  fi
  STAGE_ARGS+=("lightning_module.args.model.args.apollo_model.args.ckpt_path=${ENHANCEMENT_CKPT}")
fi
if [[ "$STAGES" == *+3 ]]; then
  python3 scripts/evaluation/check_stem_restoration_ckpts.py "${STEM_RESTORATION_CKPT_ROOT}" "${STEM_RESTORATION_CKPT_IDS}" \
    || { echo "Checkpoint check failed -- fix STEM_RESTORATION_CKPT_IDS."; exit 1; }
  STAGE_ARGS+=("lightning_module.args.model.args.apollo_stemwise_model.args.ckpt_root=${STEM_RESTORATION_CKPT_ROOT}")
  STAGE_ARGS+=("lightning_module.args.model.args.apollo_stemwise_model.args.ckpt_ids=${STEM_RESTORATION_CKPT_IDS}")
  STAGE_ARGS+=("lightning_module.args.model.args.apollo_stemwise_model.args.skip_stems=${SKIP_STEMS}")
fi

# Benchmark
case "$BENCHMARK" in
  test_set) BENCHMARK_ARGS=(
              "test_config.enabled=True"
              "test_config.per_stem_metrics_output_path=results/${EVAL_LABEL}/per_stem_metrics.csv"
            ) ;;
  msrbench) BENCHMARK_ARGS=(
              "msrbench_inference_eval_config.enabled=True"
              "msrbench_inference_eval_config.per_sample_scores_output_path=results/${EVAL_LABEL}/per_sample_scores.csv"
              "msrbench_inference_eval_config.per_stem_metrics_output_path=results/${EVAL_LABEL}/per_stem_metrics.csv"
            ) ;;
  *) echo "Unknown benchmark: $BENCHMARK"; exit 1 ;;
esac

${MSR_LAUNCHER} python -m src.train with config_yaml_file="configs/separation/bs_roformer_8stem.yaml" ${NAMED_CONFIGS} \
                  ${SEPARATION_CKPT_ARGS} \
                  "${STAGE_ARGS[@]}" \
                  "${BENCHMARK_ARGS[@]}" \
                  wandb.project="msr_evaluation" \
                  system_seed=${SYSTEM_SEED} \
                  wandb.name="eval_${EVAL_LABEL}_final" \
                  wandb.tags="['final', 'system${SYSTEM_SEED}', 'evaluation', 'stages_${STAGES}', '${BENCHMARK}']"

echo "Job finished successfully"
