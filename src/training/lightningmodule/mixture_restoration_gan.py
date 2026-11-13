"""
GAN fine-tuning pipeline for the mixture restoration model (Apollo).

Follows the training recipe from https://github.com/linfeng-feng/Hachimi
(cfg.py, dataset.py::MRDataset, train.py): manual-optimization GAN training with a
multi-resolution STFT reconstruction loss, an LSGAN spectral discriminator, and a
feature-matching loss, training the restoration generator
(src/models/apollo/apollo_wrapper.py) to map degraded (codec-compressed /
bandwidth-limited / ...) mixtures back to clean mixtures.

This is a fresh pl.LightningModule, not a BaseLightningModule subclass:
BaseLightningModule assumes a single optimizer with automatic optimization, while
Hachimi's recipe needs two optimizers (generator/discriminator) with manual
backward passes, per-optimizer gradient clipping, and schedulers stepped manually
once per "epoch" (matching Hachimi's train.py::on_train_batch_end). It still
follows this codebase's module/main/args convention for its `generator` and
`discriminator` sub-configs (via initialize_config) for consistency.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import zimtohrli
from torchaudio.transforms import Spectrogram
from torchmetrics.functional.audio import signal_distortion_ratio

from src.utils import initialize_config
from src.dsp import mix_stems_to_snr_range
from src.dsp.enhancement_degradation import build_degradation_pipeline
from src.dsp.torch_mastering import ReferenceMasteringPipeline
from src.training.loss.gan_loss import (
    MultiResolutionSTFTLoss,
    discriminator_loss_with_stats,
    generator_gan_loss,
    feature_matching_loss,
)
from src.training.loss.mel_spectrogram_loss import MultiResolutionMelLoss, ScaleInvariantMultiMelSNRLoss
from src.training.metrics.multi_mel_snr import multi_mel_snr_stereo_ct
from src.training.metrics.lsd import lsd_metric
from src.training.metrics.zimtohrli_distance import zimtohrli_distance
from src.training.helpers.seeding import batch_rng

DEFAULT_COLUMNS = [
    "bass", "drums", "other", "vocals", "guitars",
    "keyboards", "synthesizers", "percussions", "orchestral",
]


class MixtureRestorationGANLightning(pl.LightningModule):
    """
    Args:
        generator: module/main/args config for the restoration generator
            (src.models.apollo.ApolloWrapper).
        discriminator: module/main/args config for the spectral discriminator
            (src.models.frequency_discriminator.MultiFrequencyDiscriminator).
        optimizer_g / optimizer_d: module/main/args config for the generator/discriminator
            optimizers (params filled in automatically). Hachimi defaults: AdamW lr=1e-4 / 1e-5.
        columns: stem column names present in the training batch (used to build the
            clean mixture via mix_stems_to_snr_range).
        training_chunk_size: expected training segment length in samples. Hachimi trains
            on 4s segments (192000 samples @ 48kHz) — this is informational only, the
            actual length comes from the Dataset/collate output.
        snr_range: dB range used to mix stems into the clean training mixture.
        peak_headroom: peak-normalization headroom applied to the clean mixture,
            matching Hachimi's `mixture / scale * 0.93`.
        degradation_config: kwargs forwarded to
            src.dsp.enhancement_degradation.build_degradation_pipeline. Set
            pipeline_version="legacy" (default) for the original Hachimi 50/50
            bandwidth-limit/MP3 scheme (bandwidth_limitation_prob,
            bandwidth_limitation_sample_rates, mp3_bitrate_kbps_range, mp3_quality),
            pipeline_version="v2" for the MSR-Bench-motivated category+severity scheme
            covering DT1-DT4 (category_weights, severe_prob), or pipeline_version="v3"
            for v2 plus MSR-Bench's remaining DT5-DT12 conditions (exact-bitrate AAC/
            MP3 + real DAC/EnCodec neural-codec round-trips; same category_weights/
            severe_prob kwargs, with 3 additional category names). See that module for details.
        apply_mastering_pipeline / mastering_probability / mastering_pipeline_config: when
            apply_mastering_pipeline=True (default: True -- see below), build_pair runs
            src.dsp.torch_mastering.ReferenceMasteringPipeline (broadband+multiband EQ -> multiband
            compression -> stereo width -> loudness normalize -> true-peak limiter) on the clean
            mixture -- matching MSR-Bench's DT0 reference-mastering condition (real-world degraded
            audio, e.g. radio rips or cassette transfers, actually originated from a mastered
            source) and its description of what mastering includes (broadband + multiband EQ,
            multiband compression, streaming-platform limiting, stereo width -- see
            ReferenceMasteringPipeline's docstring for the full rationale, including why it's a
            separate class from MasteringPipeline / multisource_separation.py's mastering
            augmentation, which mixes in bandwidth-reduction/MP3 stages that are degradations, not
            mastering). Every ReferenceMasteringPipeline stage always fires once selected for an
            item (unlike MasteringPipeline's apply_prob, there is no per-stage skip probability
            here) -- only its *internal* parameters (EQ gains, compressor thresholds/ratios, etc.)
            are randomized per item; mastering_probability (below) controls whether the chain runs
            on a given item at all.

            mastering_probability (default: None) selects between two mutually exclusive modes:
              - None (legacy, unpaired): the degradation pipeline's input is *always* run through
                the mastering chain, but `clean` (the generator's target / discriminator's real
                input) always stays the raw pre-mastering mixture -- the model is trained to undo
                mastering and degradation together on every example, landing back on the same
                unmastered-mixture distribution the separation model is trained on downstream.
              - a float in [0, 1] (probabilistic, 3-way): independently per item (same per-item
                numpy.random.Generator convention as build_degradation_pipeline's category/severity
                draws -- pass `rng` for reproducibility, e.g. from validation_step), two sequential
                per-item draws split every example into one of three regimes:
                  1. "clean" (probability 1 - mastering_probability): neither target nor the
                     degradation pipeline's input is mastered -- the original, pre-mastering
                     restoration task, untouched.
                  2. "paired" (probability mastering_probability * mastering_pairing_prob): BOTH
                     target and input are set to the *same* mastered signal -- restore through
                     mastering without being asked to remove it. Target and source are always in
                     the same mastering state here, so there's no fixed gain/EQ/compression jump
                     between them for the model to have to learn to undo on top of whatever
                     degradation did.
                  3. "de-master" (probability mastering_probability * (1 - mastering_pairing_prob)):
                     input is mastered, target stays the raw clean mixture -- the legacy task
                     (actually learn to undo mastering), but now only a *fraction* of examples
                     instead of every single one.
                mastering_pairing_prob (default: 1.0) is only consulted when mastering_probability
                is not None; it splits the "mastered" fraction between regimes 2 and 3.
                mastering_pairing_prob=1.0 (the default) means every mastered item is "paired" --
                zero de-mastering signal, purely distributional exposure to mastered material.
                mastering_pairing_prob<1.0 reintroduces the de-mastering objective at a controllable
                rate instead of unconditionally (legacy mode's rate=1.0 on every example, which is
                the suspected source of the fixed-gain-mismatch SDR cost this parameter exists to
                avoid) -- e.g. mastering_probability=0.667, mastering_pairing_prob=0.5 gives
                clean/paired/de-master each ~1/3 of examples.
            apply_mastering_pipeline=False disables mastering entirely (mastering_probability and
            mastering_pairing_prob are ignored): both target and source are always the raw clean
            mixture.
            mastering_pipeline_config accepts ReferenceMasteringPipeline's kwargs: width_db,
            target_lufs, tp_db, oversample, lookahead_ms, release_ms, run_on_gpu (plus eq_bands /
            multiband_compressor_config for Python callers -- not exposed in the yaml configs, since
            their nested list/dict shape doesn't play well with Sacred's `with` CLI overrides; the
            class's own tasteful defaults are used unless overridden in code).
            run_on_gpu=False (the shipped default) avoids cuFFT/OOM issues seen when the true-peak
            limiter's oversampled resampling ran on GPU (see torch_mastering.py).
        ir_paths: directory (or list) of real-room impulse-response files (e.g. the
            EchoThief library — see the top-level `ir_paths` config key, injected here by
            src.training.helpers.main_helpers.initialize_components). Only used by
            pipeline_version="v2"/"v3"'s "live" category, which convolves with a real IR
            instead of the synthetic NoiseShapedReverb when this is set (falls back to
            synthetic reverb if null/no files found, or when pipeline_version="legacy").
        stft_window_sizes: window sizes for both the reconstruction loss and the
            discriminator's per-resolution STFT (Hachimi default: 512, 1024, 2048).
        warmup_epochs: epochs during which only the reconstruction loss is used and
            the discriminator isn't trained (Hachimi's mechanism; shipped default 0).
        mel_loss_weight: weight on MultiResolutionMelLoss (log-mel L1, same resolutions as the
            Multi-Mel-SNR eval metric), added to the generator loss alongside rec_loss (the
            linear-frequency multi-res STFT loss) so training optimizes directly in the domain
            the top-priority eval metric is measured in, rather than relying on rec_loss to
            correlate with it indirectly. Included during warmup too (it's a reconstruction loss,
            not adversarial). Default of 1.0, like fm_loss_weight/gan_loss_weight below, is an
            unswept starting guess -- see StemRestorationGANLightning, where this was introduced.
        mel_snr_loss_weight: weight on ScaleInvariantMultiMelSNRLoss (src.training.loss.
            mel_spectrogram_loss), added to the generator loss alongside rec_loss/mel_loss.
            Unlike mel_loss_weight's L1 term, this optimizes the exact scale-invariant Mel-SNR
            formula val/mel_snr is scored with (see that class's docstring). Included during
            warmup too (a reconstruction-domain term, not adversarial). Default 0.0 -- this term is
            still computed and logged as train/mel_snr_loss even at weight 0 for visibility,
            matching mel_loss's own always-computed pattern above.
        identity_loss_weight: weight on an L1 penalty between the generator's output and its own
            input (the degraded/mastered mixture fed to it, before generation -- `degraded` in
            training_step), added to the generator loss. Anchors the output toward the input so
            the network is discouraged from making large changes on cases that don't need much
            correction. Still computed and logged as train/identity_loss at weight 0. Default 0.0.
        fm_loss_weight / gan_loss_weight: weights for the feature-matching / GAN terms
            in the generator loss (Hachimi: g_loss = rec_loss + fm_loss + gan_loss, i.e. 1.0 each).
        train_discriminator: default True (unaffected existing behavior). When False, the
            discriminator is never trained or run at all -- no d_loss backward pass, no opt_d
            step, and the generator loss skips gan_loss/fm_loss entirely (same code path as
            warmup). Requires gan_loss_weight=0 and fm_loss_weight=0 (raises ValueError otherwise,
            since those terms would silently read an untrained/frozen-at-init discriminator). `self.discriminator` is still constructed
            (cheap: hidden_channels=8) so configure_optimizers/self.optimizers() keep returning the
            same (opt_g, opt_d) pair regardless of this flag.
        grad_clip_g / grad_clip_d: gradient clipping norm for each optimizer (Hachimi: 5.0 both).
        lr_scheduler_gamma: per-"epoch" ExponentialLR decay (Hachimi: 0.95).
        val_check_interval: used to replicate Hachimi's scheduler-stepping cadence
            (steps every `total_batches * val_check_interval` training steps).
        lr_scheduler: unused here. src/train.py's default_conf() hardcodes an
            unconditional `lightning_module.args.lr_scheduler` default (meant for the
            separation-model lightning modules) that Sacred's config-merge re-injects
            into `lightning_module.args` as soon as any CLI `with` override touches a
            *nested* key under `lightning_module.args.*` (e.g. `degradation_config=...`
            or `generator.args.ckpt_path=...`) — even though config_yaml_file's own
            `lightning_module.args` never sets it. Accepting and discarding it here
            avoids a spurious `unexpected keyword argument 'lr_scheduler'` TypeError
            whenever a training script overrides anything under lightning_module.args.
        seed: seed of the training degradation / mastering draws (category, severity tier,
            mastered or not), drawn from a numpy Generator seeded per (seed, epoch, batch_idx, rank)
            in training_step. The transforms' internal torch draws come from the global torch RNG,
            which src.training.helpers.seeding.PerBatchSeedCallback reseeds every batch. null:
            filled with manual_seed by src.training.helpers.seeding.resolve_seeds.
        accumulate_grad_batches / accumulate_grad_batches_g / accumulate_grad_batches_d:
            accumulate_grad_batches_g/_d gate opt_g/opt_d's zero_grad/step independently across
            micro-batches (manual optimization bypasses Trainer(accumulate_grad_batches=...), so
            this must live here instead). null falls back to accumulate_grad_batches.
    """

    def __init__(
        self,
        generator: dict,
        discriminator: dict,
        optimizer_g: dict,
        optimizer_d: dict,
        columns: Optional[List[str]] = None,
        sampling_rate: int = 48000,
        training_chunk_size: int = 192000,
        snr_range: Tuple[float, float] = (-3.0, 3.0),
        peak_headroom: float = 0.93,
        degradation_config: Optional[dict] = None,
        apply_mastering_pipeline: bool = True,
        mastering_probability: Optional[float] = None,
        mastering_pairing_prob: float = 1.0,
        mastering_pipeline_config: Optional[dict] = None,
        stft_window_sizes: Tuple[int, ...] = (512, 1024, 2048),
        warmup_epochs: int = 0,
        mel_loss_weight: float = 1.0,
        mel_snr_loss_weight: float = 0.0,
        identity_loss_weight: float = 0.0,
        fm_loss_weight: float = 1.0,
        gan_loss_weight: float = 1.0,
        train_discriminator: bool = True,
        grad_clip_g: float = 5.0,
        grad_clip_d: float = 5.0,
        lr_scheduler_gamma: float = 0.95,
        val_check_interval: float = 1.0,
        accumulate_grad_batches: int = 1,
        accumulate_grad_batches_g: Optional[int] = None,
        accumulate_grad_batches_d: Optional[int] = None,
        n_sources: int = 9,
        ir_paths = None,
        lr_scheduler: Optional[dict] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.seed = int(seed) if seed is not None else 0

        self.columns = list(columns) if columns is not None else list(DEFAULT_COLUMNS)
        self.sampling_rate = int(sampling_rate)
        self.training_chunk_size = int(training_chunk_size)
        self.snr_range = tuple(snr_range)
        self.peak_headroom = float(peak_headroom)
        self.warmup_epochs = int(warmup_epochs)
        self.mel_loss_weight = float(mel_loss_weight)
        self.mel_snr_loss_weight = float(mel_snr_loss_weight)
        self.identity_loss_weight = float(identity_loss_weight)
        self.fm_loss_weight = float(fm_loss_weight)
        self.gan_loss_weight = float(gan_loss_weight)
        self.train_discriminator = bool(train_discriminator)
        if not self.train_discriminator and (self.gan_loss_weight != 0.0 or self.fm_loss_weight != 0.0):
            raise ValueError(
                "train_discriminator=False requires gan_loss_weight=0 and fm_loss_weight=0 -- both "
                "read the discriminator, which train_discriminator=False never runs."
            )
        self.grad_clip_g = float(grad_clip_g)
        self.grad_clip_d = float(grad_clip_d)
        self.lr_scheduler_gamma = float(lr_scheduler_gamma)
        self.val_check_interval = float(val_check_interval)
        self.accumulate_grad_batches = int(accumulate_grad_batches)
        self.accumulate_grad_batches_g = (
            int(accumulate_grad_batches_g) if accumulate_grad_batches_g is not None else self.accumulate_grad_batches
        )
        self.accumulate_grad_batches_d = (
            int(accumulate_grad_batches_d) if accumulate_grad_batches_d is not None else self.accumulate_grad_batches
        )
        # Counts opt_d.step() calls (not micro-batches); val/d_logits is only logged once D has
        # taken at least one step (see validation_step).
        self._d_opt_step_count = 0

        self.generator_config = generator
        self.generator = initialize_config(self.generator_config)

        self.discriminator_config = discriminator
        self.discriminator = initialize_config(self.discriminator_config)

        self.optimizer_g_config = optimizer_g
        self.optimizer_d_config = optimizer_d

        degradation_config = dict(degradation_config or {})
        # Seeded fallback RNG (only used if a caller passes no `rng`; training_step / validation_step
        # always pass one).
        degradation_config.setdefault('seed', self.seed)
        self.degradation_pipeline = build_degradation_pipeline(
            sample_rate=self.sampling_rate,
            ir_paths=ir_paths,
            **degradation_config,
        )

        # Reference mastering pre-step -- see apply_mastering_pipeline's docstring above. Always
        # constructed (like multisource_separation.py's self.mastering_pipeline) so the flag can be
        # toggled from a resumed run without changing __init__; only *applying* it is gated by
        # self.apply_mastering_pipeline. mastering_pipeline_config's keys map 1:1 onto
        # ReferenceMasteringPipeline's kwargs -- see torch_mastering.py for defaults/rationale.
        self.apply_mastering_pipeline = bool(apply_mastering_pipeline)
        if mastering_probability is not None and not (0.0 <= float(mastering_probability) <= 1.0):
            raise ValueError(f"mastering_probability must be in [0, 1] or None, got {mastering_probability!r}")
        self.mastering_probability = float(mastering_probability) if mastering_probability is not None else None
        if not (0.0 <= float(mastering_pairing_prob) <= 1.0):
            raise ValueError(f"mastering_pairing_prob must be in [0, 1], got {mastering_pairing_prob!r}")
        self.mastering_pairing_prob = float(mastering_pairing_prob)
        # Seeded fallback RNG for the probabilistic-mastering draws (mastering_probability is not
        # None) when build_pair is called without an explicit `rng`; training_step / validation_step
        # always pass one.
        self._mastering_rng = np.random.default_rng(self.seed)
        mastering_pipeline_config = mastering_pipeline_config or {}
        self.mastering_pipeline = ReferenceMasteringPipeline(
            sample_rate=self.sampling_rate,
            eq_bands=mastering_pipeline_config.get('eq_bands'),
            multiband_compressor_config=mastering_pipeline_config.get('multiband_compressor_config'),
            width_db=mastering_pipeline_config.get('width_db', 1.5),
            target_lufs=mastering_pipeline_config.get('target_lufs', -14.0),
            loudness_backend=mastering_pipeline_config.get('loudness_backend', 'auto'),
            tp_db=mastering_pipeline_config.get('tp_db', -1.0),
            oversample=mastering_pipeline_config.get('oversample', 4),
            lookahead_ms=mastering_pipeline_config.get('lookahead_ms', 1.0),
            release_ms=mastering_pipeline_config.get('release_ms', 50.0),
            run_on_gpu=mastering_pipeline_config.get('run_on_gpu', False),
        )

        self.rec_loss_fn = MultiResolutionSTFTLoss(window_sizes=stft_window_sizes)
        self.mel_loss_fn = MultiResolutionMelLoss(sample_rate=self.sampling_rate)
        self.mel_snr_loss_fn = ScaleInvariantMultiMelSNRLoss(sample_rate=self.sampling_rate)

        # Used only for val/lsd, which needs complex STFT tensors (mirrors Hachimi's self.spec).
        self.lsd_spec = Spectrogram(n_fft=1024, win_length=1024, hop_length=512, power=None)

        # Non-differentiable (CPU, numpy-backed) -- val/zimtohrli logging only, never a loss term.
        # See StemRestorationGANLightning, where this was introduced, for the reasoning.
        try:
            self._zimt_calculator = zimtohrli.Pyohrli()
        except Exception as e:
            print(f"Warning: failed to initialize Zimtohrli backend ({e}); val/zimtohrli will be skipped.")
            self._zimt_calculator = None

    # ------------------------------------------------------------------
    # clean / degraded pair construction
    # ------------------------------------------------------------------

    def build_clean_mixture(self, batch: Dict) -> torch.Tensor:
        """Sum the present clean stems into a peak-normalized mixture, mirroring
        Hachimi's MRDataset.__getitem__ train branch (`mixture / scale * 0.93`)."""
        batch_size = batch[self.columns[0]]['waveform'].shape[0]
        mixtures = []
        for idx in range(batch_size):
            present = {
                col: batch[col]['waveform'][idx]
                for col in self.columns
                if batch[col]['is_present'][idx]
            }
            if len(present) >= 2:
                reference = next(iter(present))
                mixture, _ = mix_stems_to_snr_range(
                    present, self.snr_range, reference_stem_key=reference, normalize_output=False,
                )
            elif len(present) == 1:
                mixture = next(iter(present.values()))
            else:
                mixture = batch[self.columns[0]]['waveform'][idx] * 0.0

            scale = mixture.abs().max().clamp_min(torch.finfo(mixture.dtype).eps)
            mixture = mixture / scale * self.peak_headroom
            mixtures.append(mixture)
        return torch.stack(mixtures, dim=0)

    def build_pair(self, batch: Dict, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (clean_mixture, degraded_mixture), both [B, C, T]."""
        if isinstance(batch.get('mixture'), dict) and 'waveform' in batch['mixture']:
            # Full-song validation dataset (MultiAudioFullSongDataset) already
            # provides the clean mixture directly.
            clean = batch['mixture']['waveform']
        else:
            clean = self.build_clean_mixture(batch)

        if isinstance(batch.get('degraded_mixture'), dict) and 'waveform' in batch['degraded_mixture']:
            # MSRBenchRestorationValDataset already provides a real (not synthetic)
            # MSRBench-family-degraded mixture paired with `clean` above -- use it directly
            # instead of degrading `clean` ourselves, so this validation measures against
            # real degradations rather than our own synthetic approximation of them.
            return clean, batch['degraded_mixture']['waveform']

        # Reference mastering pre-step -- see apply_mastering_pipeline/mastering_probability's
        # docstring for the legacy-vs-probabilistic-3-way mode distinction. Same
        # float32/autocast-disabled + dtype-restore rationale as the degradation_pipeline call
        # below: ReferenceMasteringPipeline's transforms aren't bf16-safe under precision="bf16-mixed".
        target = clean
        source = clean
        if self.apply_mastering_pipeline:
            original_dtype = clean.dtype
            with torch.autocast(device_type=clean.device.type, enabled=False):
                mastered = self.mastering_pipeline(clean.float(), self.sampling_rate)
            mastered = mastered.to(original_dtype)

            if self.mastering_probability is None:
                # Legacy, unpaired: the degradation input is always mastered, but the target
                # (returned as `clean` below) always stays the raw pre-mastering mixture.
                source = mastered
            else:
                # Probabilistic 3-way: two sequential per-item draws (is_mastered, then, only if
                # mastered, is_paired) split every item into "clean" / "paired" / "de-master" -- see
                # mastering_probability/mastering_pairing_prob's docstring for the three regimes.
                # Mirrors build_degradation_pipeline's own per-item rng convention (category draw
                # then severity-tier draw; fall back to our own seeded generator when no `rng` is
                # passed).
                item_rng = rng if rng is not None else self._mastering_rng
                is_mastered = torch.tensor(
                    [item_rng.random() < self.mastering_probability for _ in range(clean.shape[0])],
                    device=clean.device,
                )
                is_paired = torch.tensor(
                    [item_rng.random() < self.mastering_pairing_prob for _ in range(clean.shape[0])],
                    device=clean.device,
                )
                source_mask = is_mastered.view(-1, 1, 1)
                target_mask = (is_mastered & is_paired).view(-1, 1, 1)
                source = torch.where(source_mask, mastered, clean)
                target = torch.where(target_mask, mastered, clean)

        # degradation_pipeline internally runs in float32 with autocast disabled and
        # casts back to `clean`'s dtype (see _run_transform_fp32_safe in
        # src/dsp/enhancement_degradation.py) — required because under
        # precision="bf16-mixed", `clean` can arrive already cast to bf16 by
        # Lightning's ambient autocast, and some DSP transforms (dasp_pytorch,
        # librosa-based IR convolution) only compute correctly in float32.
        degraded = self.degradation_pipeline(source, rng=rng)

        # Aggregate-count visibility into which DAC checkpoint each item actually went through --
        # "severe" = the 24kHz-native model (this pipeline's stand-in for MSR-Bench's DT9 "22kHz"
        # condition, see src/dsp/enhancement_degradation.py's EnhancementDegradationPipelineV3
        # docstring for why no true 22.05kHz checkpoint exists), "mild" = the 44kHz-native model
        # (DT10). last_categories/last_tiers are set by every __call__ above regardless of
        # pipeline_version, so this is a no-op ([] counts) for pipeline_version="legacy"/"v2" or
        # any run with category_weights.dac=0.0 -- logged every step (summed per epoch) rather
        # than gated behind pipeline_version, so a config that unexpectedly zeroes out "dac" shows
        # up as an all-zero count instead of a missing metric.
        categories = getattr(self.degradation_pipeline, "last_categories", [])
        tiers = getattr(self.degradation_pipeline, "last_tiers", [])
        dac_24khz_proxy_count = sum(1 for c, t in zip(categories, tiers) if c == "dac" and t == "severe")
        dac_44khz_count = sum(1 for c, t in zip(categories, tiers) if c == "dac" and t == "mild")
        batch_size = degraded.shape[0]
        # sync_dist=True is safe here (unlike val/mel_snr_by_family's per-family logging below,
        # which warns against it): these two keys are logged unconditionally, every step, on
        # every rank, so the all_reduce is always symmetric across ranks -- no deadlock risk.
        self.log(
            "train/degradation_dac_24khz_severe_count", float(dac_24khz_proxy_count),
            on_step=False, on_epoch=True, reduce_fx="sum", sync_dist=True, batch_size=batch_size,
        )
        self.log(
            "train/degradation_dac_44khz_mild_count", float(dac_44khz_count),
            on_step=False, on_epoch=True, reduce_fx="sum", sync_dist=True, batch_size=batch_size,
        )

        return target, degraded

    # ------------------------------------------------------------------
    # Lightning hooks
    # ------------------------------------------------------------------

    def forward(self, x):
        return self.generator(x)

    def _log_d_stats(self, prefix: str, d_stats, batch_size: int, sync_dist: bool = False):
        """Logs discriminator_loss_with_stats' per-sub-discriminator stats dicts
        (train/d_logits/... and val/d_logits/...)."""
        for window_size, s in zip(self.discriminator.window, d_stats):
            win_prefix = f'{prefix}/win{window_size}'
            for key, value in s.items():
                if value is not None:
                    self.log(f'{win_prefix}/{key}', value, batch_size=batch_size, sync_dist=sync_dist)

    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        # Degradation / mastering draws seeded per (seed, epoch, batch, rank): reproducible, also
        # after resuming, and different on every rank.
        rng = batch_rng(self.seed, "degradation", self.current_epoch, batch_idx, self.global_rank)
        clean, degraded = self.build_pair(batch, rng=rng)
        enhanced = self.generator({'mixture': degraded})['mixture']

        length = min(enhanced.shape[-1], clean.shape[-1])
        enhanced = enhanced[..., :length]
        clean = clean[..., :length]
        # identity_loss_weight's anchor target -- the generator's own pre-generation input,
        # trimmed to match enhanced/clean's length (see identity_loss_weight's docstring).
        input_for_identity = degraded[..., :length]

        is_warmup = self.current_epoch < self.warmup_epochs
        # Decoupled per-optimizer accumulation -- see accumulate_grad_batches_g/_d docstring.
        # Each optimizer's zero_grad/step is gated independently so accumulate_grad_batches_d can
        # be swept without also changing how often opt_g steps.
        d_is_accum_start = batch_idx % self.accumulate_grad_batches_d == 0
        d_is_accum_end = ((batch_idx + 1) % self.accumulate_grad_batches_d == 0) or self.trainer.is_last_batch
        g_is_accum_start = batch_idx % self.accumulate_grad_batches_g == 0
        g_is_accum_end = ((batch_idx + 1) % self.accumulate_grad_batches_g == 0) or self.trainer.is_last_batch

        d_grad_norm = torch.tensor(0.0, device=self.device)
        d_grad_norm_postclip = torch.tensor(0.0, device=self.device)
        d_grad_clipped = None
        if not is_warmup and self.train_discriminator:
            # Sanity guardrail: a real/fake logit gap of ~0 is consistent both with "D hasn't
            # trained long enough" and with "D is literally seeing near-identical real/fake
            # inputs" (e.g. a mastering-pipeline or detach bug upstream). These are cheap enough
            # to always log; the assert only fires in the genuinely-broken case.
            fake_for_d = enhanced.detach()
            debug_batch_size = clean.shape[0]
            with torch.no_grad():
                real_fake_diff = (clean - fake_for_d).abs()
                real_fake_mae = real_fake_diff.mean()
                real_fake_max = real_fake_diff.max()
            self.log('debug/real_audio_rms', clean.detach().square().mean().sqrt(), batch_size=debug_batch_size)
            self.log('debug/fake_audio_rms', fake_for_d.square().mean().sqrt(), batch_size=debug_batch_size)
            self.log('debug/real_fake_mae', real_fake_mae, batch_size=debug_batch_size)
            self.log('debug/real_fake_max', real_fake_max, batch_size=debug_batch_size)
            assert real_fake_mae > 1e-6, (
                "Real and fake discriminator inputs are effectively identical "
                f"(mean abs difference={float(real_fake_mae):.3e}) -- check the mastering "
                "pipeline / build_pair and the enhanced.detach() path feeding the discriminator "
                "before blaming GAN/FM weights."
            )

            if d_is_accum_start:
                opt_d.zero_grad()
            d_loss, d_loss_real, d_loss_fake, d_stats = discriminator_loss_with_stats(
                self.discriminator, clean, fake_for_d
            )
            # D's own forward pass on the (already-detached) fake audio must still carry gradients
            # w.r.t. D's parameters, or opt_d.step() below would silently be a no-op.
            assert d_loss.requires_grad, "Discriminator loss has no gradient w.r.t. D's parameters"
            self.manual_backward(d_loss / self.accumulate_grad_batches_d)
            if d_is_accum_end:
                d_grad_norm = torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.grad_clip_d)
                d_grad_clipped = float(d_grad_norm > self.grad_clip_d)
                d_grad_norm_postclip = torch.sqrt(sum(
                    p.grad.detach().pow(2).sum() for p in self.discriminator.parameters() if p.grad is not None
                ))
                # Sanity check (see the real_fake_mae assert above): the actual parameter
                # movement opt_d.step() produces, distinct from the pre-step gradient norms above
                # -- a nonzero grad norm with a ~0 update norm would point at the optimizer step
                # itself (e.g. a zeroed/mismatched LR) rather than the loss/gradient path.
                d_params_before_step = [p.detach().clone() for p in self.discriminator.parameters()]
                opt_d.step()
                update_sq = sum(
                    (after.detach() - before).float().pow(2).sum()
                    for after, before in zip(self.discriminator.parameters(), d_params_before_step)
                )
                param_sq = sum(after.detach().float().pow(2).sum() for after in self.discriminator.parameters())
                d_param_update_norm = update_sq.sqrt()
                d_relative_update_norm = d_param_update_norm / (param_sq.sqrt() + 1e-12)
                self.log(
                    'debug/d_param_update_norm', d_param_update_norm,
                    batch_size=clean.shape[0], on_step=True, on_epoch=True,
                )
                self.log(
                    'debug/d_relative_update_norm', d_relative_update_norm,
                    batch_size=clean.shape[0], on_step=True, on_epoch=True,
                )
                self._d_opt_step_count += 1
        else:
            d_loss = torch.tensor(0.0, device=self.device)
            d_loss_real = torch.tensor(0.0, device=self.device)
            d_loss_fake = torch.tensor(0.0, device=self.device)
            d_stats = None

        g_grad_norm = torch.tensor(0.0, device=self.device)
        g_grad_norm_postclip = torch.tensor(0.0, device=self.device)
        g_grad_clipped = None
        if g_is_accum_start:
            opt_g.zero_grad()
        rec_loss = self.rec_loss_fn(enhanced, clean)
        mel_loss = self.mel_loss_fn(enhanced, clean)
        mel_snr_loss = self.mel_snr_loss_fn(enhanced, clean)
        identity_loss = F.l1_loss(enhanced, input_for_identity)
        base_g_loss = (
            rec_loss
            + self.mel_loss_weight * mel_loss
            + self.mel_snr_loss_weight * mel_snr_loss
            + self.identity_loss_weight * identity_loss
        )

        if is_warmup or not self.train_discriminator:
            g_loss = base_g_loss
            fm_loss = torch.tensor(0.0, device=self.device)
            gan_loss = torch.tensor(0.0, device=self.device)
        else:
            gan_loss, fake_features, real_features = generator_gan_loss(
                self.discriminator, enhanced, clean
            )
            fm_loss = feature_matching_loss(fake_features, real_features)
            g_loss = (
                base_g_loss
                + self.fm_loss_weight * fm_loss
                + self.gan_loss_weight * gan_loss
            )

        self.manual_backward(g_loss / self.accumulate_grad_batches_g)
        if g_is_accum_end:
            g_grad_norm = torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.grad_clip_g)
            g_grad_clipped = float(g_grad_norm > self.grad_clip_g)
            g_grad_norm_postclip = torch.sqrt(sum(
                p.grad.detach().pow(2).sum() for p in self.generator.parameters() if p.grad is not None
            ))
            opt_g.step()

        # Weighted counterparts of the raw per-term losses above (rec_loss's implicit weight is
        # 1.0, so it has no separate "weighted" value).
        mel_loss_weighted = self.mel_loss_weight * mel_loss
        mel_snr_loss_weighted = self.mel_snr_loss_weight * mel_snr_loss
        identity_loss_weighted = self.identity_loss_weight * identity_loss
        gan_loss_weighted = self.gan_loss_weight * gan_loss
        fm_loss_weighted = self.fm_loss_weight * fm_loss

        batch_size = clean.shape[0]
        self.log('train/g_loss', g_loss, prog_bar=True, batch_size=batch_size)
        self.log('train/d_loss', d_loss, prog_bar=True, batch_size=batch_size)
        # on_epoch=True (in addition to the default on_step=True) on the raw/weighted per-term
        # losses and both grad norms below, so they are also available as epoch aggregates.
        self.log('train/rec_loss', rec_loss, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/mel_loss', mel_loss, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/mel_loss_weighted', mel_loss_weighted, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/mel_snr_loss', mel_snr_loss, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/mel_snr_loss_weighted', mel_snr_loss_weighted, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/identity_loss', identity_loss, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/identity_loss_weighted', identity_loss_weighted, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/fm_loss', fm_loss, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/fm_loss_weighted', fm_loss_weighted, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/gan_loss', gan_loss, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/gan_loss_weighted', gan_loss_weighted, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/d_loss_real', d_loss_real, batch_size=batch_size)
        self.log('train/d_loss_fake', d_loss_fake, batch_size=batch_size)
        self.log('train/d_grad_norm', d_grad_norm, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/g_grad_norm', g_grad_norm, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/d_grad_norm_postclip', d_grad_norm_postclip, batch_size=batch_size, on_step=True, on_epoch=True)
        self.log('train/g_grad_norm_postclip', g_grad_norm_postclip, batch_size=batch_size, on_step=True, on_epoch=True)
        # Fraction-of-optimizer-steps-that-clipped -- only meaningful at accum-end steps where
        # opt_d.step()/opt_g.step() actually ran, so only logged then; Lightning's on_epoch mean
        # of these 0.0/1.0 values is the gradient clipping fraction per epoch.
        if d_grad_clipped is not None:
            self.log('train/d_grad_clip_fraction', d_grad_clipped, batch_size=batch_size, on_step=True, on_epoch=True)
        if g_grad_clipped is not None:
            self.log('train/g_grad_clip_fraction', g_grad_clipped, batch_size=batch_size, on_step=True, on_epoch=True)
        if d_stats is not None:
            self._log_d_stats('train/d_logits', d_stats, batch_size)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        total_batches = self.trainer.num_training_batches
        step_interval = max(1, int(total_batches * self.val_check_interval))
        if (batch_idx + 1) % step_interval == 0:
            sch_g, sch_d = self.lr_schedulers()
            sch_g.step()
            if self.current_epoch >= self.warmup_epochs:
                sch_d.step()

    def on_validation_epoch_start(self):
        # Per-DT-family mel_snr accumulator for this validation epoch (see validation_step /
        # on_validation_epoch_end). Reset here rather than logged incrementally per-step because
        # with batch_size=1 + a distributed sampler, different ranks see different (and generally
        # non-identical) sets of families on any given step -- self.log(sync_dist=True) requires
        # every rank to issue the same collective calls, so per-step family logging with
        # sync_dist=True can deadlock DDP. Accumulating locally and gathering once at epoch end
        # (on_validation_epoch_end) sidesteps that: it's a single collective call regardless of
        # which families each rank happened to see.
        self._val_family_mel_snr: Dict[str, List[float]] = {}

    def validation_step(self, batch, batch_idx):
        # Deterministic degradation RNG (both the numpy choice-of-degradation RNG
        # and the torch RNG any transform uses internally) so val/mel_snr is
        # reproducible run-to-run without perturbing the training RNG stream. Only used for
        # validation datasets without real degraded mixtures (MSRBenchRestorationValDataset
        # provides them, then build_pair does not degrade).
        seed = (self.current_epoch * 100_003 + batch_idx) % (2**31 - 1)
        fork_devices = [self.device] if self.device.type == 'cuda' else []
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            clean, degraded = self.build_pair(batch, rng=rng)
            with torch.no_grad():
                enhanced = self.generator({'mixture': degraded})['mixture']

        length = min(enhanced.shape[-1], clean.shape[-1])
        enhanced = enhanced[..., :length]
        clean = clean[..., :length]

        rec_loss = self.rec_loss_fn(enhanced, clean)
        mel_loss = self.mel_loss_fn(enhanced, clean)

        batch_size = clean.shape[0]
        mel_snrs, sdrs, lsds, zimts = [], [], [], []
        for b in range(batch_size):
            mel_snrs.append(multi_mel_snr_stereo_ct(enhanced[b], clean[b], self.sampling_rate))
            sdrs.append(signal_distortion_ratio(enhanced[b], clean[b]).mean().item())
            pred_stft = self.lsd_spec(enhanced[b])
            target_stft = self.lsd_spec(clean[b])
            lsds.append(lsd_metric(pred_stft, target_stft).item())
            zimts.append(zimtohrli_distance(self._zimt_calculator, enhanced[b], clean[b]))

        self.log('val/loss', rec_loss, prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('val/mel_loss', mel_loss, batch_size=batch_size, sync_dist=True)
        # Checkpoint selection (see the `checkpoint` callback config) still monitors val/mel_snr
        # only -- val/zimtohrli is logged for visibility, not wired into ModelCheckpoint's
        # `monitor`.
        self.log('val/mel_snr', float(np.mean(mel_snrs)), prog_bar=True, batch_size=batch_size, sync_dist=True)
        self.log('val/sdr', float(np.mean(sdrs)), batch_size=batch_size, sync_dist=True)
        self.log('val/lsd', float(np.mean(lsds)), batch_size=batch_size, sync_dist=True)
        finite_zimts = [z for z in zimts if math.isfinite(z)]
        if finite_zimts:
            self.log('val/zimtohrli', float(np.mean(finite_zimts)), prog_bar=True, batch_size=batch_size, sync_dist=True)

        # Accumulate per-family mel_snr locally (gathered/logged in on_validation_epoch_end) --
        # `family` is provided by MSRBenchRestorationValDataset (and MultiAudioFullSongDataset),
        # absent for other val_dataloader datasets.
        families = batch.get('family')
        if families is not None:
            for b, family in enumerate(families):
                self._val_family_mel_snr.setdefault(family, []).append(float(mel_snrs[b]))

        # Discriminator diagnostics on this held-out batch (calibration visibility). Skipped
        # entirely when train_discriminator=False -- D is never trained in that mode, so its
        # logits are just noise from an untrained network. Also skipped for src/train.py's
        # step-zero validation pass (before trainer.fit() starts, i.e. self._d_opt_step_count
        # still 0): D is randomly initialized and can produce wildly unstable logits on its very
        # first forward pass, which would otherwise corrupt that epoch's logged
        # gap_mean/gap_median/auc with a meaningless outlier.
        if self.train_discriminator and self._d_opt_step_count > 0:
            with torch.no_grad():
                _, _, _, d_stats = discriminator_loss_with_stats(self.discriminator, clean, enhanced)
            self._log_d_stats('val/d_logits', d_stats, batch_size, sync_dist=True)

    def on_validation_epoch_end(self):
        # Must always run the gather below when distributed (even with an empty local dict) --
        # all_gather_object is a collective call, so skipping it on some ranks would hang the
        # others waiting for their contribution.
        if not hasattr(self, '_val_family_mel_snr'):
            return

        local = self._val_family_mel_snr
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            gathered: List[Dict[str, List[float]]] = [None] * world_size  # type: ignore[list-item]
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered = [local]

        merged: Dict[str, List[float]] = {}
        for rank_values in gathered:
            for family, values in (rank_values or {}).items():
                merged.setdefault(family, []).extend(values)

        for family, values in sorted(merged.items()):
            self.log(f'val/mel_snr_by_family/{family}', float(np.mean(values)), rank_zero_only=True)

    @staticmethod
    def _submodule_freeze_status(module: nn.Module) -> str:
        """'frozen' (every param requires_grad=False), 'trainable' (every param
        requires_grad=True), or 'mixed' -- for a submodule with no parameters at all,
        reports 'trainable' (vacuously, since there's nothing frozen to flag)."""
        params = list(module.parameters())
        if not params:
            return "trainable"
        n_trainable = sum(1 for p in params if p.requires_grad)
        if n_trainable == 0:
            return "frozen"
        if n_trainable == len(params):
            return "trainable"
        return "mixed"

    def on_fit_start(self):
        """Print a startup summary of which generator submodules are frozen vs.
        trainable, and the resulting parameter counts -- run once, rank-0 only,
        before the first training step. Covers ApolloWrapper's freeze_band_split/
        num_frozen_bsnet_layers granularity by name (self.model.BN, each
        self.model.net[i] BSNet block, self.model.output) when the generator is an
        ApolloWrapper; falls back to a generic per-top-level-submodule summary for
        any other generator class."""
        if not self.trainer.is_global_zero:
            return

        gen = self.generator
        frozen_names: List[str] = []
        trainable_names: List[str] = []
        mixed_names: List[str] = []

        apollo_model = getattr(gen, "model", None)
        if apollo_model is not None and hasattr(apollo_model, "net") and hasattr(apollo_model, "BN") and hasattr(apollo_model, "output"):
            named_submodules = [("generator.model.BN (band-split front-end)", apollo_model.BN)]
            bsnet_blocks = list(apollo_model.net.children())
            named_submodules += [
                (f"generator.model.net[{i}] (BSNet block {i}/{len(bsnet_blocks)})", block)
                for i, block in enumerate(bsnet_blocks)
            ]
            named_submodules.append(("generator.model.output (output head)", apollo_model.output))
        else:
            named_submodules = [(f"generator.{name}", module) for name, module in gen.named_children()]

        for name, module in named_submodules:
            status = self._submodule_freeze_status(module)
            {"frozen": frozen_names, "trainable": trainable_names, "mixed": mixed_names}[status].append(name)

        total_params = sum(p.numel() for p in gen.parameters())
        trainable_params = sum(p.numel() for p in gen.parameters() if p.requires_grad)
        pct_trainable = 100.0 * trainable_params / total_params if total_params else 0.0

        lines = [
            "=" * 88,
            "[Generator freeze/trainable summary] (on_fit_start, rank 0)",
            f"Frozen submodules ({len(frozen_names)}): " + (", ".join(frozen_names) if frozen_names else "none"),
            f"Trainable submodules ({len(trainable_names)}): " + (", ".join(trainable_names) if trainable_names else "none"),
        ]
        if mixed_names:
            lines.append(
                f"Mixed submodules ({len(mixed_names)}, contain both frozen and trainable params): "
                + ", ".join(mixed_names)
            )
        lines += [
            f"Total generator parameters:     {total_params:,}",
            f"Trainable generator parameters: {trainable_params:,}",
            f"Trainable percentage:           {pct_trainable:.2f}%",
            "=" * 88,
        ]
        print("\n".join(lines))

    def configure_optimizers(self):
        opt_g_config = dict(self.optimizer_g_config)
        # Only trainable params -- some generators (e.g. src.models.apollo.apollo_wrapper.
        # ApolloWrapper's freeze_band_split/num_frozen_bsnet_layers) can leave part of
        # self.generator frozen.
        trainable_g_params = [p for p in self.generator.parameters() if p.requires_grad]
        opt_g_config['args'] = {**opt_g_config.get('args', {}), 'params': trainable_g_params}
        opt_g = initialize_config(opt_g_config)

        opt_d_config = dict(self.optimizer_d_config)
        opt_d_config['args'] = {**opt_d_config.get('args', {}), 'params': list(self.discriminator.parameters())}
        opt_d = initialize_config(opt_d_config)
        # Cheap, one-time sanity check: confirms opt_d holds every one of D's own parameters and
        # nothing else.
        opt_d_param_ids = {id(p) for group in opt_d.param_groups for p in group['params']}
        discriminator_param_ids = {id(p) for p in self.discriminator.parameters()}
        assert opt_d_param_ids == discriminator_param_ids, (
            "optimizer_d's parameters don't exactly match self.discriminator.parameters() -- "
            f"{len(opt_d_param_ids - discriminator_param_ids)} in opt_d only, "
            f"{len(discriminator_param_ids - opt_d_param_ids)} in discriminator only."
        )

        # Manual optimization: schedulers are not auto-stepped by Lightning, they're
        # stepped explicitly in on_train_batch_end (matching Hachimi's train.py).
        sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=self.lr_scheduler_gamma)
        sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=self.lr_scheduler_gamma)

        return [opt_g, opt_d], [sch_g, sch_d]
