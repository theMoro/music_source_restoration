# Training and evaluation

Prerequisites: everything in [SETUP.md](SETUP.md) (environment, pretrained checkpoints, datasets,
activity masks, dataset paths).

The system consists of three stages that are trained separately:

```
degraded mixture ──► stage 1: mixture restoration ──► stage 2: separation ──► stage 3: stem-wise restoration ──► 8 stems
                        Apollo                           BS-RoFormer              one Apollo model per stem
```

| Stage | Config | Trained on | Initialized from |
|---|---|---|---|
| 1 mixture restoration | [configs/mixture_restoration/apollo_mixture_restoration.yaml](configs/mixture_restoration/apollo_mixture_restoration.yaml) | mss8s mixtures, mastered + degraded on the fly | pretrained Apollo |
| 2 separation | [configs/separation/bs_roformer_8stem.yaml](configs/separation/bs_roformer_8stem.yaml) | mss8s (random stem mixing, augmentation) | BS-RoFormer-SW |
| 3 stem-wise restoration | [configs/stem_restoration/apollo_stem_restoration.yaml](configs/stem_restoration/apollo_stem_restoration.yaml) | (separated stem, clean stem) pairs generated with stages 1+2 | stage 1 checkpoint |

Stages 1 and 2 are independent; stage 3 needs the checkpoints of both.

## General usage

Everything is started with [src/train.py](src/train.py) ([Sacred](https://github.com/IDSIA/sacred)):

```bash
python -m src.train with config_yaml_file=<config> [named configs] [key=value overrides]
```

- **Overrides**: every config entry can be overridden, e.g. `train.trainer.args.devices=2`,
  `datamodule.args.train_dataloader.batch_size=1`, `dataset_path_8_sources=/my/data/mss8s`.
  Print the resolved config with `python -m src.train print_config with config_yaml_file=<config> ...`.
- **Weights & Biases**: all metrics are logged to wandb; set `wandb.project=...`, `wandb.name=...`,
  optionally `wandb.entity=...` (default: `WANDB_ENTITY` or your default entity) and `wandb.tags="['a', 'b']"`.
  Without an account, use `export WANDB_MODE=offline`. At the end of a run, the script waits a few
  minutes for the wandb upload to finish.
- **Checkpoints** are written to `<CKPT_PATH>/<config file name>/<wandb run id>/` (`CKPT_PATH` =
  `outputs/checkpoints` or `$MSR_CKPT_PATH`, see [SETUP.md](SETUP.md#3-paths)), e.g.
  `outputs/checkpoints/bs_roformer_8stem/<run id>/last.ckpt`.
- **Resuming** an interrupted run: add `resume_wandb_id=<run id>` (continues the wandb run and
  loads `<CKPT_PATH>/<config file name>/<run id>/last.ckpt`) or `resume_ckpt_path=<path>`.
- **GPUs**: the configs use DDP with `train.trainer.args.devices` GPUs per node (stages 1 and 2: 4,
  stage 3: 1). Under SLURM, the number of nodes is taken from `SLURM_NNODES` (start the job with
  `srun`, one task per GPU). Our runs used 4 nodes x 4 GPUs for stages 1 and 2; with fewer GPUs,
  increase the gradient accumulation to keep the effective batch size (stage 2:
  `train.trainer.args.accumulate_grad_batches`; stage 1: `lightning_module.args.accumulate_grad_batches`).
- **Seeds / reproducibility**: `manual_seed` is the only seed to set. Sacred's `seed` and all seeds
  left `null` in the configs (dataloader shuffle / sampler, DataLoader workers, dataset chunk and
  mixing RNG, degradations, stem restoration dataset generation) are derived from it
  ([src/training/helpers/seeding.py](src/training/helpers/seeding.py)); the resolved values are logged
  to wandb. Seeds that select data (MSRBench validation subset, stem restoration train/val split)
  are fixed so all training seeds are validated on the same data. All random draws are keyed by
  (seed, epoch, batch / item, rank), so a resumed run continues exactly like an uninterrupted one.
  Remaining sources of non-determinism: CUDA ops without a deterministic implementation (they only
  warn, `torch.use_deterministic_algorithms(warn_only=True)`), and a different GPU count / type or
  software version.
- **Experiment scripts**: [scripts/](scripts) contains the scripts of our experiments; they are
  started from the repository root with `bash` and use the paths of [scripts/common.sh](scripts/common.sh)
  (see [SETUP.md](SETUP.md#3-paths)). For multi-GPU / multi-node training on a SLURM cluster, submit
  them with `sbatch` (add the `#SBATCH` options of your cluster) and set `MSR_LAUNCHER=srun`
  (e.g. `MSR_LAUNCHER="srun --gpu-bind=none" sbatch -N 4 --gres=gpu:4 --ntasks-per-node=4 scripts/separation/train_bs_roformer.sh 41`).
  The scripts train / evaluate one *final system* each, see [below](#final-systems-seeds).

## Final systems (seeds)

Every stage of the final system is trained 3 times independently. A system is named after its
system seed `S` ∈ {41, 42, 43}; stage `k` of system `S` uses `manual_seed = 100 * S + k`:

| Stage | `stage_idx` | S = 41 | S = 42 | S = 43 |
|---|---|---|---|---|
| 1 mixture restoration (Apollo) | 0 | 4100 | 4200 | 4300 |
| 2 separation (BS-RoFormer) | 1 | 4101 | 4201 | 4301 |
| 3a stem restoration dataset | 2 | 4102 | 4202 | 4302 |
| 3b stem-wise restoration (8 x Apollo) | 3 | 4103 | 4203 | 4303 |

The scripts compute the seed themselves (you only pass `S`), log `system_seed`, `stage_idx` and
`manual_seed` to wandb, name the runs `..._system<S>_seed<seed>_final` and tag them `final` +
`system<S>`. Dataset, checkpoint and result paths are derived from the system seed and the run ids.
Order of the commands for one system (`<MIXRES_ID>` / `<SEP_ID>` = wandb run ids of stages 1 / 2):

```bash
bash scripts/mixture_restoration/train_apollo.sh 41                 # stage 1  -> <MIXRES_ID>
bash scripts/separation/train_bs_roformer.sh 41                     # stage 2  -> <SEP_ID>
bash scripts/evaluation/evaluate.sh 41 2   msrbench <SEP_ID>
bash scripts/evaluation/evaluate.sh 41 1+2 msrbench <SEP_ID> <MIXRES_ID>
bash scripts/stem_restoration/generate_dataset.sh 41 <MIXRES_ID> <SEP_ID>         # stage 3a
bash scripts/stem_restoration/submit_all.sh 41 <MIXRES_ID> <SEP_ID> 3e-5 1e-5 3e-6  # stage 3b
# select the best run per stem (val/mel_snr) -> STEM_IDS="{'bass': '<id>', 'drums': '<id>', ...}"
bash scripts/evaluation/evaluate.sh 41 1+2+3 msrbench <SEP_ID> <MIXRES_ID> "${STEM_IDS}"
```

Use `test_set` instead of `msrbench` for the MSR Challenge test set. Resume an interrupted stage 1
or 2 run by appending `resume_wandb_id=<run id>` to the same command.

## Stage 1: mixture restoration (Apollo)

```bash
python -m src.train with config_yaml_file=configs/mixture_restoration/apollo_mixture_restoration.yaml \
    wandb.project=mixture_restoration wandb.name=apollo_restoration
```

- Fine-tunes the pretrained Apollo model (band-split module and first 3 of 6 BSNet blocks frozen)
  with a GAN loss for 220 epochs (generator lr 1.5e-6, discriminator lr 3e-7, no identity loss,
  learning rate decayed by 0.97 every 11 epochs). Training mixtures are built from mss8s stems, mastered with
  probability 0.5 and degraded with the MSRBench-like degradation pipeline
  ([src/dsp/enhancement_degradation.py](src/dsp/enhancement_degradation.py): bandwidth limitation,
  MP3, AAC, DAC, EnCodec, radio, cassette, vinyl, live recording, clean).
- Validation (every 11 epochs): real degraded MSRBench mixtures (10% of the clips, all 13 degradation
  types) vs. the original mixtures; `val/mel_snr` (checkpoint selection), `val/sdr`, `val/lsd`,
  `val/zimtohrli`, `val/mel_snr_by_family/DT*`.
- The checkpoint of every validation (20 in total) is kept, plus `last.ckpt` (used in our experiments).
- Script: `bash scripts/mixture_restoration/train_apollo.sh <system_seed>`.

## Stage 2: separation (BS-RoFormer)

```bash
python -m src.train with config_yaml_file=configs/separation/bs_roformer_8stem.yaml \
    manual_seed=44 \
    wandb.project=separation wandb.name=bs_separation_seed44
```

- Separates 9 stems: bass, drums, other, vocals, guitars, keyboards, synthesizers, percussions,
  orchestral (`other` is not trained). The model is initialized from BS-RoFormer-SW (6 stems); the
  mask estimators of synthesizers, percussions and orchestral are new.
- 200 epochs with random stem mixing (mixing probability decays linearly from 0.9 to 0 over 170
  epochs), stem augmentation (RawStems) and a data scheduler (Slakh2100 is decayed to 0 by epoch 170).
- Validation every 10 epochs on the original (DT0) MSRBench mixtures (SDR, SI-SNR, Multi-Mel-SNR,
  Zimtohrli, ... per stem). Only `last.ckpt` is kept.
- We trained 3 seeds (42, 43, 44); the system evaluated in the paper uses the model of seed 44.
- Script: `bash scripts/separation/train_bs_roformer.sh <system_seed>`.

## Stage 3: stem-wise restoration (Apollo)

### 3a. Generate the stem restoration dataset

The training data of stage 3 are pairs of (separated stem, clean stem): mss8s mixtures (RawStems
songs only, no stem mixing, with the stage-2 augmentations) are restored with the stage-1 model and
separated with the stage-2 model.

```bash
python -m src.train with config_yaml_file=configs/separation/bs_roformer_8stem.yaml generate_restoration_dataset \
    post_training_ckpt_path=<stage 2 checkpoint, e.g. outputs/checkpoints/bs_roformer_8stem/<run id>/last.ckpt> \
    lightning_module.args.model.args.apollo_model.args.ckpt_path=<stage 1 checkpoint (last.ckpt)> \
    generate_inference_dataset_config.output_dir=data/hf/stem_restoration \
    wandb.name=generate_stem_restoration_dataset
```

- Writes the audio files and the HF dataset `data/hf/stem_restoration/final_dataset` (up to
  4592 examples, `generate_inference_dataset_config.num_samples`; one example = one stem of one
  10 s segment). Runs on 1 GPU.
- Use an empty output folder (temporary batches of a previous run in the same folder would be merged).
- Script: `bash scripts/stem_restoration/generate_dataset.sh <system_seed> <mixres run id> <sep run id>`
  (the output folder is derived from the arguments).

### 3b. Train one restoration model per stem

```bash
python -m src.train with config_yaml_file=configs/stem_restoration/apollo_stem_restoration.yaml \
    stem=vocals \
    dataset_root=data/hf/stem_restoration/final_dataset \
    stem_restoration_ckpt_root=outputs/checkpoints/stem_restoration \
    stem_restoration_version=gan0.5_lr1e-5 \
    lightning_module.args.generator.args.ckpt_path=<stage 1 checkpoint (last.ckpt)> \
    lightning_module.args.optimizer_g.args.lr=1e-5 \
    lightning_module.args.optimizer_d.args.lr=6e-6 \
    wandb.project=stem_restoration wandb.name=vocals_lr1e-5
```

- One run per stem (`bass, drums, vocals, guitars, keyboards, synthesizers, percussions,
  orchestral`); `other` is not restored. Each run fine-tunes the stage-1 model for 40 epochs on 1 GPU
  (GAN loss, 20% of the RawStems examples of the stem are held out for validation).
- We trained every stem with 3 generator learning rates (3e-5, 1e-5, 3e-6; discriminator learning
  rate = 0.6 x generator learning rate) and selected the best model per stem on the validation split
  (`val/mel_snr`).
- Checkpoints: `<stem_restoration_ckpt_root>/<stem_restoration_version>/<stem>/<wandb run id>/last.ckpt`.
- Script: `bash scripts/stem_restoration/train_apollo.sh <system_seed> <mixres run id> <sep run id> <stem> <lr>`
  or `bash scripts/stem_restoration/submit_all.sh <system_seed> <mixres run id> <sep run id> <lr> [<lr> ...]`
  (8 stems x the given learning rates; there is no default learning rate).

## Evaluation

### Benchmarks

| Benchmark | What is measured | Output |
|---|---|---|
| MSR Challenge 2025 test set (`non-blind` + `streaming`) | per stem: SDR, SI-SNR, Multi-Mel-SNR, Zimtohrli, FAD-CLAP | printed table + wandb (`epoch_test_*`) |
| MSRBench (2000 clips x 13 degradation types) | per stem and degradation type: SDR, SNR, SI-SNR, Multi-Mel-SNR, Zimtohrli, Mel-SSIM, KL, PESQ, production quality, LogWMSE, FAD-CLAP, ... | wandb + CSV files (`per_sample_scores.csv`, `per_stem_metrics.csv`) |

See [METRICS.md](METRICS.md) for the metric definitions.

### Separation alone and combined systems (stages 2, 1+2, 2+3, 1+2+3)

The separation config is combined with the named configs `inference` (stage 2, no training),
`stage1` (adds the mixture restoration before the separation) and `stage3` (adds the stem-wise
restoration after the separation):

| Evaluated system | Named configs |
|---|---|
| stage 2 (separation only) | `inference` |
| stages 1+2 | `inference stage1` |
| stages 2+3 | `inference stage3` |
| stages 1+2+3 (full system) | `inference stage1 stage3` |

and with one benchmark:

| Benchmark | Arguments |
|---|---|
| MSR Challenge test set | `test_config.enabled=True` |
| MSRBench | `msrbench_inference_eval_config.enabled=True msrbench_inference_eval_config.per_sample_scores_output_path=results/<name>/per_sample_scores.csv msrbench_inference_eval_config.per_stem_metrics_output_path=results/<name>/per_stem_metrics.csv` |

Checkpoints:

| Stage | Arguments |
|---|---|
| 2 | `post_training_ckpt_path=<stage 2 last.ckpt>` |
| 1 | `lightning_module.args.model.args.apollo_model.args.ckpt_path=<stage 1 last.ckpt>` |
| 3 | `lightning_module.args.model.args.apollo_stemwise_model.args.ckpt_root=<stem_restoration_ckpt_root>` and `lightning_module.args.model.args.apollo_stemwise_model.args.ckpt_ids="{'bass': '<run id>', 'drums': '<run id>', ...}"` (the run id of the selected model per stem; searched in `<ckpt_root>/*/<stem>/<run id>/last.ckpt`) |

The examples below load the separation model by its wandb run id (`post_training_config_filename` =
checkpoint sub-folder, `post_training_wandb_id` = run id, i.e. `<CKPT_PATH>/bs_roformer_8stem/<stage 2 run id>/last.ckpt`);
`post_training_ckpt_path=<stage 2 last.ckpt>` can be given instead. Replace the checkpoints by your own.

Example: separation only (stage 2) on the MSR Challenge test set

```bash
python -m src.train with config_yaml_file=configs/separation/bs_roformer_8stem.yaml inference \
    test_config.enabled=True \
    post_training_config_filename=bs_roformer_8stem post_training_wandb_id=<stage 2 run id> \
    wandb.name=eval_stages_2_test_set
```

Example: stages 1+2 on the MSR Challenge test set

```bash
python -m src.train with config_yaml_file=configs/separation/bs_roformer_8stem.yaml inference stage1 \
    test_config.enabled=True \
    post_training_config_filename=bs_roformer_8stem post_training_wandb_id=<stage 2 run id> \
    lightning_module.args.model.args.apollo_model.args.ckpt_path=outputs/checkpoints/apollo_mixture_restoration/<stage 1 run id>/last.ckpt \
    wandb.name=eval_stages_1+2_test_set
```

Example: stages 2+3 on the MSR Challenge test set

```bash
python -m src.train with config_yaml_file=configs/separation/bs_roformer_8stem.yaml inference stage3 \
    test_config.enabled=True \
    post_training_config_filename=bs_roformer_8stem post_training_wandb_id=<stage 2 run id> \
    lightning_module.args.model.args.apollo_stemwise_model.args.ckpt_root=outputs/checkpoints/stem_restoration \
    lightning_module.args.model.args.apollo_stemwise_model.args.ckpt_ids="{'bass': '<id>', 'drums': '<id>', 'vocals': '<id>', 'guitars': '<id>', 'keyboards': '<id>', 'synthesizers': '<id>', 'percussions': '<id>', 'orchestral': '<id>'}" \
    lightning_module.args.model.args.apollo_stemwise_model.args.skip_stems="['other']" \
    wandb.name=eval_stages_2+3_test_set
```

Example: full system (stages 1+2+3) on the MSR Challenge test set

```bash
python -m src.train with config_yaml_file=configs/separation/bs_roformer_8stem.yaml inference stage1 stage3 \
    test_config.enabled=True \
    post_training_config_filename=bs_roformer_8stem post_training_wandb_id=<stage 2 run id> \
    lightning_module.args.model.args.apollo_model.args.ckpt_path=outputs/checkpoints/apollo_mixture_restoration/<stage 1 run id>/last.ckpt \
    lightning_module.args.model.args.apollo_stemwise_model.args.ckpt_root=outputs/checkpoints/stem_restoration \
    lightning_module.args.model.args.apollo_stemwise_model.args.ckpt_ids="{'bass': '<id>', 'drums': '<id>', 'vocals': '<id>', 'guitars': '<id>', 'keyboards': '<id>', 'synthesizers': '<id>', 'percussions': '<id>', 'orchestral': '<id>'}" \
    lightning_module.args.model.args.apollo_stemwise_model.args.skip_stems="['other']" \
    wandb.name=eval_stages_1+2+3_test_set
```

For MSRBench, replace `test_config.enabled=True` by the MSRBench arguments above.

Notes:
- A stem in `skip_stems` passes through without stem-wise restoration (`other` is never restored).
  For our test-set results, bass, drums and keyboards were not restored either
  (`skip_stems="['other', 'bass', 'drums', 'keyboards']"`); for MSRBench, all stems except `other` were restored.
- A stem without a resolvable checkpoint falls back to the pretrained Apollo model; run
  `python scripts/evaluation/check_stem_restoration_ckpts.py <ckpt_root> "<ckpt_ids>"` first to
  check that every stem resolves to exactly one checkpoint.
- The evaluation runs on the GPUs of `train.trainer.args.devices` (default 4).
- Script: `bash scripts/evaluation/evaluate.sh <system_seed> <2 | 1+2 | 2+3 | 1+2+3> <test_set | msrbench> <sep run id> [<mixres run id>] [<stem ckpt ids>]`
  (see the script header).
- Both benchmarks write `results/<label>/per_stem_metrics.csv` (test set: with
  `test_config.per_stem_metrics_output_path=...`; MSRBench additionally `per_sample_scores.csv`).
