# Setup: environment, checkpoints and datasets

This guide prepares everything needed for [TRAINING.md](TRAINING.md):

1. [Conda environment](#1-conda-environment)
2. [Pretrained checkpoints](#2-pretrained-checkpoints)
3. [Paths](#3-paths)
4. [Training data: the mss8s dataset](#4-training-data-the-mss8s-dataset) (download, conversion, activity masks)
5. [Evaluation data: MSRBench and the MSR Challenge test set](#5-evaluation-data-msrbench-and-the-msr-challenge-test-set)
6. [Checklist](#6-checklist)

All commands are run from the repository root.

## 1. Conda environment

```bash
# creates the env "music_source_restoration" (Python 3.11, ffmpeg, requirements.txt)
CFLAGS="-Wno-error=incompatible-pointer-types" conda env create -f environment.yml
conda activate music_source_restoration
```

- `CFLAGS` is needed to compile `pypesq` (PESQ metric) with GCC >= 14, which treats a pointer type
  mismatch in its C code as an error ("Failed to build pypesq"); with older compilers it is only a
  warning. If the environment was already created and only the pip step failed, finish it with
  `conda activate music_source_restoration && CFLAGS="-Wno-error=incompatible-pointer-types" pip install -r requirements.txt`.
- `requirements.txt` installs PyTorch 2.5.1 for CUDA 12.1. For another CUDA version, change the
  `--extra-index-url` in `requirements.txt` (see https://pytorch.org).
- `ffmpeg` must be on the `PATH` (installed by `environment.yml`); it is needed for the AAC
  degradation and by `museval`.
- Log in to Weights & Biases (`wandb login`), or run without an account via `export WANDB_MODE=offline`.
- Check the installation with `pytest tests`.

## 2. Pretrained checkpoints

| Model | Used by | How to get it |
|---|---|---|
| BS-RoFormer-SW (6 stems), `BS-Rofo-SW-Fixed.ckpt` | separation (initialization) | download (below) |
| Apollo (`JusperLee/Apollo`) | mixture restoration (initialization) | downloaded automatically from the Hugging Face Hub at the first run |

```bash
mkdir -p checkpoints/pretrained/6stem
hf download enerjazzer/BS-ROFO-SW-Fixed BS-Rofo-SW-Fixed.ckpt \
    --local-dir checkpoints/pretrained/6stem
sha256sum checkpoints/pretrained/6stem/BS-Rofo-SW-Fixed.ckpt
# 24e7d35ee9c64415673d3fd33e06a67cac2c103c5df6267ba1576459c775916e
```

The same file is also available at
[FractalSurfer/BS-RoFormer-SW-6stem](https://huggingface.co/FractalSurfer/BS-RoFormer-SW-6stem).
The folder of the pretrained checkpoints defaults to `checkpoints/pretrained` (relative to the
repository root); set `export MSR_RESOURCES_FOLDER=<folder>` to use another folder (the checkpoint
must be at `<folder>/6stem/BS-Rofo-SW-Fixed.ckpt`).

If your GPU machines have no internet access, call `hf download JusperLee/Apollo pytorch_model.bin`
once on a machine with internet access to fill the Hugging Face cache, and point `HF_HOME` to the
same cache on the GPU machines.

## 3. Paths

All paths are relative to the repository root (the scripts and `python -m src.train` are always
started from there) and can be replaced by absolute paths. Nothing has to be changed if you use the
default folder layout of this guide; `data/`, `checkpoints/`, `outputs/` and `results/` are ignored by git.

**Dataset paths** (top-level keys of the yaml configs in [configs/](configs); change them in the
configs or pass them on the command line, e.g. `python -m src.train with config_yaml_file=... dataset_path_8_sources=/my/data/mss8s`):

| Config key | Default | Needed for |
|---|---|---|
| `dataset_path_8_sources` | `data/hf/mss8s` | training of stages 1 and 2, stem restoration dataset generation |
| `dataset_path_msr_bench` | `data/hf/msrbench` | validation of stages 1 and 2, MSRBench evaluation |
| `dataset_path_msr_test_set_2025` | `data/raw/msr_test_set_2025` | MSR Challenge test set evaluation (only in `configs/separation/bs_roformer_8stem.yaml`) |
| `prefix_path_8_sources`, `prefix_path_msr_bench` | `null` | optional prefix for relative audio paths in the HF datasets (not needed: the conversion scripts store absolute paths) |
| `ir_paths` | `null` (disabled) | optional, **TODO** if wanted: folder of real room impulse responses (e.g. the [EchoThief library](http://www.echothief.com/)) used by the "live" degradation of stage 1 instead of synthetic reverb |
| `dataset_root` | `null` | **must be set** for stage 3b (stem restoration dataset, set by `scripts/stem_restoration/train_apollo.sh`) |

**Output / resource paths** (environment variables, read by [src/config_updates.py](src/config_updates.py)
and [scripts/common.sh](scripts/common.sh)):

| Environment variable | Default | What |
|---|---|---|
| `MSR_RESOURCES_FOLDER` | `checkpoints/pretrained` | pretrained checkpoints (section 2) |
| `MSR_CKPT_PATH` | `outputs/checkpoints` | checkpoints of all runs: `<MSR_CKPT_PATH>/<config file name>/<wandb run id>/` |
| `MSR_LOG_PATH` | `outputs/logs` | wandb run files (unless `WANDB_DIR` is set) |
| `MSR_DATA_PATH` | `data/hf` | output folder of the stem restoration datasets generated by the scripts (stage 3a) |
| `MSR_LAUNCHER` | empty | optional command prefix of the scripts' python call, e.g. `srun` (see [TRAINING.md](TRAINING.md)) |
| `WANDB_ENTITY` | your default wandb entity | wandb team / user the runs are logged to |

Evaluation results are written to `results/<label>/`.
Hugging Face, torch and wandb caches use their usual defaults (`HF_HOME`, `TORCH_HOME`,
`WANDB_CACHE_DIR`, ...); set these variables to move the caches, e.g. to a larger disk.

The conversion scripts below store absolute audio paths, so do not move the audio files after the
conversion (or set the prefix paths above).

## 4. Training data: the mss8s dataset

All three models are trained on **mss8s**, a combination of four multitrack datasets whose stems are
mapped to 8 instrument categories: **vocals, guitars, bass, keyboards, synthesizers, drums,
percussions, orchestral**. Building it takes four steps:

| Step | What it does | Script | Reads | Writes |
|---|---|---|---|---|
| [4.1](#41-download) | download the four datasets | (`hf download`, `wget`) | | `data/raw/<dataset>/` |
| [4.2](#42-convert-the-datasets) | convert every dataset separately: map its stems to the 8 categories, sum the stems of each category into one WAV file per song, store the WAV paths as a Hugging Face dataset | `preprocessing/convert_<dataset>.py` (one per dataset) | `data/raw/<dataset>/` | `data/hf/<dataset>/` |
| [4.3](#43-build-mss8s) | merge the four converted datasets into one dataset | `preprocessing/build_mss8s.py` | the four `data/hf/<dataset>/` | `data/hf/mss8s/` |
| [4.4](#44-activity-masks) | compute in which seconds each stem is active (used by the data loader to sample segments and mix stems) | `preprocessing/compute_activity_masks.py` | `data/hf/mss8s/` | `data/hf/mss8s_metadata/` |

The configs point to `data/hf/mss8s/`; the data loader expects `mss8s_metadata/` next to it. mss8s
does not copy the audio, it only stores the paths of the WAV files written in step 4.2, so keep the
four `data/hf/<dataset>/` folders as well.

`data/raw` and `data/hf` (relative to the repository root) are the default paths used throughout
this guide and in the configs. To store the data elsewhere (the datasets need about 1 TB), either
make `data` a symlink to a folder on a large disk (`ln -s /path/to/large/disk data`) or replace the
paths in the commands and set the dataset paths of section 3.

### 4.1 Download

| Dataset | Songs | Download size | Folder after this step (input of 4.2) |
|---|---|---|---|
| MoisesDB | 240 | see website | `data/raw/moisesdb/moisesdb_v0.1/<song uuid>/` |
| MUSDB25 (multitrack recreation of MUSDB18) | 150 | 240 GB | `data/raw/musdb25/` |
| Slakh2100 (FLAC redux) | 1710 | 104 GB | `data/raw/slakh2100_flac_redux/{train,validation,test}/Track*/` |
| RawStems | 576 | 187 GB | `data/raw/rawstems/<Artist\|Song>/<category>/...` |

**MoisesDB**: request access at https://music.ai/research/, download the archive and extract it
into `data/raw/moisesdb/` (via `unzip -q data/raw/moisesdb.zip -d data/raw`) so that the song folders (one folder per song uuid, each with a
`data.json`) are in `data/raw/moisesdb/moisesdb_v0.1/`.

**MUSDB25**:

```bash
hf download kwatcharasupat/musdb25 --repo-type dataset --local-dir data/raw/musdb25
```

**Slakh2100**: the archive contains the folder `slakh2100_flac_redux/`, so extract it with `-C` into
`data/raw`:

```bash
mkdir -p data/raw
wget -O data/raw/slakh2100_flac_redux.tar.gz "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1"
tar -xzf data/raw/slakh2100_flac_redux.tar.gz -C data/raw      # -> data/raw/slakh2100_flac_redux/
rm data/raw/slakh2100_flac_redux.tar.gz
```

To avoid storing the archive and the extracted files at the same time (2 x 104 GB), you can also
extract while downloading:
`wget -O - "https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1" | tar -xzf - -C data/raw`.
The folder `omitted/` in the archive is not used by the conversion.

**RawStems**: six zip files (~31 GB each). Each zip contains a part of the songs as top-level song
folders (`<Artist>|<Song>/`), so all six are extracted into the same folder:

```bash
hf download yongyizang/RawStems --repo-type dataset --include "RawStems_*.zip" --local-dir data/raw/rawstems_zips
mkdir -p data/raw/rawstems
for f in data/raw/rawstems_zips/RawStems_*.zip; do
    unzip -q -n "$f" -d data/raw/rawstems      # -n: never overwrite, no interactive prompts
done
ls data/raw/rawstems | wc -l                   # 576 song folders
rm -r data/raw/rawstems_zips                   # optional, after checking the extraction
```

The conversion (4.2) writes one summed WAV file per song and category (and, for MUSDB25, one WAV
file per stem), so plan for roughly the same amount of additional space under `data/hf`. The
MUSDB25 download (npz files) can be deleted after the conversion.

### 4.2 Convert the datasets

```bash
python preprocessing/convert_moisesdb.py  --dataset-root data/raw/moisesdb/moisesdb_v0.1 --save-dir data/hf/moisesdb
python preprocessing/convert_musdb25.py   --dataset-root data/raw/musdb25                  --save-dir data/hf/kwatcharasupat_musdb25
python preprocessing/convert_slakh2100.py --dataset-root data/raw/slakh2100_flac_redux    --save-dir data/hf/slakh2100 --num-workers 16
python preprocessing/convert_rawstems.py  --dataset-root data/raw/rawstems                --save-dir data/hf/raw_stems --num-workers 16
```

- Each script writes a Hugging Face dataset (`save_to_disk`) plus the rendered WAV files
  (`<save-dir>/additional_wavs/`) and needs the output folder to not exist (or `--overwrite`).
- `convert_rawstems.py` can optionally reject songs with inconsistent stems (`--quality-filter`,
  see `--help`); our mss8s dataset was built without this filter.
- These steps take several hours; `convert_musdb25.py` loads the raw tracks of a whole song (an npz file of up to ~1.5 GB) into memory at once.

**Mapping to the 8 categories.** Every stem of a song is assigned to one category; all stems of a
category are summed into one WAV file per song. Stems that cannot be mapped are summed into the
residual `other` file (empty for most songs). `other` is **not used for training**: it has no
activity masks and is never added to a training mixture.

| Dataset | Mapping (source label → category) | Script |
|---|---|---|
| MoisesDB (stem folders) | vocals→vocals; bass→bass; guitar, other_plucked→guitars; piano→keyboards; other_keys, other→synthesizers; drums→drums; percussion→percussions; bowed_strings, wind→orchestral | [convert_moisesdb.py](preprocessing/convert_moisesdb.py) |
| Slakh2100 (`inst_class` in `metadata.yaml`) | Bass→bass; Guitar→guitars; Piano, Organ→keyboards; Synth Lead, Synth Pad, Sound Effects→synthesizers; Drums→drums; Percussive, Chromatic Percussion→percussions; Strings, Strings (continued), Brass, Reed, Pipe, Ethnic→orchestral (no vocals); stems are resampled to 48 kHz | [convert_slakh2100.py](preprocessing/convert_slakh2100.py) |
| MUSDB25 (MedleyDB-style `instrument` names in `*_METADATA.yaml`) | lookup table of the instrument names (e.g. "female singer"→vocals, "distorted electric guitar"→guitars, "synthesizer"→synthesizers, "drum set"→drums, "tambourine"→percussions, "violin section"→orchestral, ...) with keyword fallbacks | [convert_musdb25.py](preprocessing/convert_musdb25.py) |
| RawStems (folder names) | Voc, VocLV→vocals; Gtr→guitars; Bass, Bss→bass; Kbs→keyboards; Synth, Misc→synthesizers; Rhy/DK→drums; Rhy/PERC→percussions; Orch→orchestral | [convert_rawstems.py](preprocessing/convert_rawstems.py) |

### 4.3 Build mss8s

```bash
python preprocessing/build_mss8s.py \
    --moisesdb-path data/hf/moisesdb \
    --raw-stems-path data/hf/raw_stems \
    --slakh2100-path data/hf/slakh2100 \
    --musdb25-path data/hf/kwatcharasupat_musdb25 \
    --output-path data/hf/mss8s
```

This merges the four datasets into one `train` split (the Slakh2100 validation/test splits are
used for training as well; the models are validated on MSRBench), adds the `dataset_name` and
`original_idx` columns and prints the number of songs per dataset (expected: about 1710 slakh2100,
576 raw_stems, 240 moisesdb, ~150 kwatcharasupat_musdb25).

### 4.4 Activity masks

```bash
python preprocessing/compute_activity_masks.py --dataset-root data/hf/mss8s --num-workers 16
```

The training data loader samples only segments in which the selected stems are active. This script
computes, per song and stem, which seconds are active (RMS > -40 dB, max over both channels) and
writes the metadata folder that the data loader expects **next to** the dataset:

```
data/hf/mss8s/                                  the HF dataset
data/hf/mss8s_metadata/activity_masks_v1.jsonl  activity mask per song and stem (keyed by original_idx)
data/hf/mss8s_metadata/stem_mapping_v1.jsonl    for every stem: the songs in which it is active (random mixing)
```

### 4.5 Use the dataset

`dataset_path_8_sources` defaults to `data/hf/mss8s`; if you used another folder, set it in the
configs or pass `dataset_path_8_sources=<folder>` on the command line.

## 5. Evaluation data: MSRBench and the MSR Challenge test set

**MSRBench** (validation of the separation and mixture restoration models, evaluation; 28 GB):

```bash
hf download yongyizang/MSRBench --repo-type dataset --local-dir data/raw/msrbench_zips
mkdir -p data/raw/msrbench && for f in data/raw/msrbench_zips/*.zip; do unzip -q "$f" -d data/raw/msrbench; done
# -> data/raw/msrbench/{Bass,Drums,Guitars,Keyboards,Orchestral Elements,Percussions,Synthesizers,Vocals}/{mixture,target}/
python preprocessing/convert_msrbench.py --dataset-root data/raw/msrbench --save-dir data/hf/msrbench
```

`dataset_path_msr_bench` defaults to `data/hf/msrbench` (all three configs).

**MSR Challenge 2025 test set** (final evaluation; 2.7 GB; only the `non-blind` and `streaming`
subsets have targets):

```bash
hf download MSRChallenge/2025TestSet --repo-type dataset --local-dir data/raw/msr_test_set_2025
```

`dataset_path_msr_test_set_2025` defaults to `data/raw/msr_test_set_2025` in
`configs/separation/bs_roformer_8stem.yaml` (the folder is read directly, no conversion needed).

## 6. Checklist

- [ ] `conda activate music_source_restoration`, `pytest tests` passes, `wandb login` (or `WANDB_MODE=offline`)
- [ ] `checkpoints/pretrained/6stem/BS-Rofo-SW-Fixed.ckpt` (or `MSR_RESOURCES_FOLDER`)
- [ ] `data/hf/mss8s` and `data/hf/mss8s_metadata/{activity_masks_v1,stem_mapping_v1}.jsonl`
- [ ] `data/hf/msrbench` and `data/raw/msr_test_set_2025`
- [ ] dataset paths: defaults kept, or set in the configs / on the command line (section 3)

Continue with [TRAINING.md](TRAINING.md).
