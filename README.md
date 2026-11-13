# Music Source Restoration

Code for our music source restoration system, which restores degraded music mixtures and separates
them into the stems vocals, guitars, bass, keyboards, synthesizers, drums, percussions and
orchestral. The system consists of three models that are trained separately:

```
degraded mixture ──► (1) mixture restoration ──► (2) source separation ──► (3) stem-wise restoration ──► stems
                         Apollo                      BS-RoFormer                one Apollo model per stem
```

1. **Mixture restoration**: fine-tunes the pretrained [Apollo](https://github.com/JusperLee/Apollo)
   model on mixtures that are mastered and degraded with a degradation pipeline covering the
   MSRBench degradation types (bandwidth limitation, MP3/AAC, DAC/EnCodec neural codecs, radio,
   cassette, vinyl, live recording). GAN training with a multi-resolution spectral discriminator.
2. **Source separation**: fine-tunes BS-RoFormer (initialized from the 6-stem BS-RoFormer-SW
   checkpoint, with new mask estimators for synthesizers, percussions and orchestral) on the 8-stem
   mss8s dataset (MoisesDB, MUSDB25, RawStems, Slakh2100) with random stem mixing and a data scheduler.
3. **Stem-wise restoration**: fine-tunes the mixture restoration model separately for each stem
   on (separated stem, clean stem) pairs, which are generated with models (1) and (2).

## Getting started

1. [SETUP.md](SETUP.md): conda environment, pretrained checkpoints, download and conversion of the
   datasets, activity masks.
2. [TRAINING.md](TRAINING.md): training of the three stages and evaluation of the stages alone and
   of the combined systems (1+2, 1+2+3) on MSRBench and the MSR Challenge 2025 test set.

All metrics are logged to [Weights & Biases](https://wandb.ai); see [METRICS.md](METRICS.md) for
the metric definitions.

## Paths

All paths are relative to the repository root (run every command from there) and have defaults,
so with the folder layout of [SETUP.md](SETUP.md) nothing has to be configured:

| What | Default | Set with |
|---|---|---|
| mss8s training dataset | `data/hf/mss8s` | config key `dataset_path_8_sources` |
| MSRBench | `data/hf/msrbench` | config key `dataset_path_msr_bench` |
| MSR Challenge 2025 test set | `data/raw/msr_test_set_2025` | config key `dataset_path_msr_test_set_2025` |
| pretrained BS-RoFormer-SW checkpoint | `checkpoints/pretrained/6stem/BS-Rofo-SW-Fixed.ckpt` | env `MSR_RESOURCES_FOLDER` |
| checkpoints of all runs | `outputs/checkpoints` | env `MSR_CKPT_PATH` |
| wandb run files | `outputs/logs` | env `MSR_LOG_PATH` |
| generated stem restoration datasets (scripts) | `data/hf` | env `MSR_DATA_PATH` |
| evaluation results | `results/` | |

Paths without a default that **must be set** where they are used: the stem restoration dataset of
stage 3b (`dataset_root`, set by the script) and the checkpoints passed to the evaluation and
dataset generation commands. Optional: real room impulse responses (`ir_paths`, default `null` =
disabled). The config keys can be changed in the yaml files in [configs/](configs) or passed on the
command line (`key=value`); see [SETUP.md](SETUP.md#3-paths) for details.

## Repository structure

```
configs/          yaml configs of the three models
preprocessing/    dataset conversion (MoisesDB, MUSDB25, RawStems, Slakh2100, MSRBench), mss8s, activity masks
scripts/          scripts of our experiments (training, dataset generation, evaluation)
resources/        fixed validation samples for the wandb audio uploads
src/
  train.py                 entry point (training, dataset generation, evaluation)
  config_updates.py        output paths, pretrained checkpoints, named configs
  data/                    mss8s dataset, MSRBench, MSR test set, stem restoration dataset
  datamodules/             Lightning datamodule incl. data scheduler
  dsp/                     augmentation, mastering and degradation pipelines
  models/                  BS-RoFormer, Apollo, discriminator, MultipleModelsWrapper (full system)
  training/                Lightning modules, losses, metrics, evaluation, helpers
tests/            unit tests (pytest tests)
```
