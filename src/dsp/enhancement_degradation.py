"""
Degradation pipelines for training the mixture restoration model.

Two pipelines are available, selected via `build_degradation_pipeline(pipeline_version=...)`:

- `EnhancementDegradationPipelineV2` ("v2"): a richer, MSR-Bench-motivated scheme —
  per example, draws a degradation *category* (bandwidth/mp3/radio/cassette/vinyl/live/
  clean) and a severity *tier* (mild/severe), then applies that category's fixed chain
  of effects. Covers MSR-Bench's DT1-DT4 (analog/acoustic). See its docstring for details.
- `EnhancementDegradationPipelineV3` ("v3"): extends V2 with the remaining MSR-Bench
  conditions — exact-bitrate AAC/MP3 (DT5-DT8) and two real neural codecs, DAC
  (DT9/DT10) and EnCodec (DT11/DT12). Together with V2's categories, this covers all
  12 MSR-Bench degradation conditions (DT1-DT12) plus DT0 (clean). See its docstring
  for details, and src/dsp/neural_codec_effects.py for the new transforms.

Both are built on this repo's own batched `torch_audiomentations` transforms
(`DownUpSample`, `Mp3Compression`, etc.).

Note: all pipelines only randomize the *choice* of degradation (and category/tier
sub-parameters like bandwidth-limitation target rate) via an explicit
`numpy.random.Generator`. The underlying transforms are constructed with p=1.0 so they
always fire once chosen; any *internal* randomization they do (e.g. MP3 bitrate, noise
SNR draw) uses the global torch RNG. Callers that need reproducible degradation (e.g.
validation) should wrap calls in `torch.random.fork_rng()` with a deterministic seed.
"""

import inspect
import logging
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch_audiomentations import AddColoredNoise, Compose

from .music_effect_pipeline import (
    ApplyImpulseResponse,
    BiquadHighPassFilter,
    BiquadLowPassFilter,
    DownUpSample,
    NoiseShapedReverb,
)
from .neural_codec_effects import AacCompression, DacCompression, EncodecCompression
from .torch_mastering import Mp3Compression
from .vinyl_cassette_effects import ClickPopNoise, CompandClip

logger = logging.getLogger(__name__)


def _run_transform_fp32_safe(transform, batched: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Run a torch_audiomentations transform/Compose in true float32, regardless of the
    ambient autocast context, then cast the result back to `batched`'s original dtype.

    Under precision="bf16-mixed", Lightning's autocast wraps the whole training/
    validation step, so `batched` can arrive already cast to bf16. Several DSP
    transforms used here (NoiseShapedReverb/dasp_pytorch, ApplyImpulseResponse/
    librosa) always compute and return float32 internally regardless of input dtype,
    which then hits a hard dtype mismatch inside torch_audiomentations' base Compose
    forward() (its masked assignment `cloned_samples[idx] = ...` requires an exact
    dtype match, unlike a plain per-item `tensor[i] = ...` which silently casts).
    DSP math like reverb/codec simulation is also numerically unreliable in bf16's
    ~8-bit mantissa anyway, so this always computes in float32 with autocast disabled.
    """
    original_dtype = batched.dtype
    with torch.autocast(device_type=batched.device.type, enabled=False):
        out = transform(batched.float(), sample_rate=sample_rate)
    return out.to(original_dtype)


DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "bandwidth": 0.15,
    "mp3": 0.15,
    "radio": 0.15,
    "cassette": 0.15,
    "vinyl": 0.15,
    "live": 0.20,
    "clean": 0.05,
}


class EnhancementDegradationPipelineV2:
    """
    Richer degradation pipeline for the enhancement/restoration model, modeled on
    MSR-Bench's DT1-DT4 (analog/acoustic) family plus a broad-range MP3/bandwidth
    approximation of DT5-DT8 (traditional codec). AAC and the neural-codec conditions
    (DT9-DT12) are out of scope here; see `EnhancementDegradationPipelineV3`, which
    subclasses this and adds those (exact-bitrate AAC/MP3 + real DAC/EnCodec
    round-trips), covering all 12 MSR-Bench conditions.

    Unlike `EnhancementDegradationPipeline` (single degradation, no noise, uniform
    intensity), this pipeline independently, per example:
      1. draws a degradation *category* from `category_weights` (each category is a
         fixed chain of 1-4 effects, e.g. "live" = reverb -> phone-band filter ->
         additive noise, approximating a concert recording captured on a phone), and
      2. draws a severity *tier* ("mild" or "severe", via `severe_prob`) that scales
         that category's parameter ranges (noise SNR, filter cutoffs, bitrate, etc.).

    Per-category parameter ranges are intentionally not individually configurable
    (that's ~25 separate ranges); `category_weights` and `severe_prob` are the two
    knobs worth tuning — they control *how often* each failure mode is seen and *how
    harsh* it gets, which is where most of the training-distribution leverage is.

    ir_paths (optional): a directory/list of real room impulse-response files (e.g.
    the EchoThief library, already used elsewhere in this repo by
    `ApplyImpulseResponse`). When set and at least one IR file is found, the "live"
    category convolves with a real IR instead of the synthetic `NoiseShapedReverb` —
    more realistic room acoustics for the category modeled on MSR-Bench's DT4 ("live",
    a concert-style recording captured on a phone). The same real-IR stage is used at
    both severity tiers (severity is still carried by the noise/filter stages); falls
    back to synthetic reverb (both tiers) if ir_paths is None or no files are found.
    """

    CATEGORIES: Tuple[str, ...] = ("bandwidth", "mp3", "radio", "cassette", "vinyl", "live", "clean")
    # Class attribute (not the bare module global) so subclasses -- e.g. V3, which adds
    # categories the base DEFAULT_CATEGORY_WEIGHTS dict knows nothing about -- can
    # override it wholesale and have __init__ below pick up the right defaults.
    DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = DEFAULT_CATEGORY_WEIGHTS
    # Per-category override of the mild/severe tier draw, keyed by category name --
    # falls back to `severe_prob` for any category not listed here. Empty for V2;
    # V3 overrides this for "dac" (see its docstring/CATEGORY_SEVERE_PROB_OVERRIDES).
    CATEGORY_SEVERE_PROB_OVERRIDES: Dict[str, float] = {}

    def __init__(
        self,
        sample_rate: int = 48000,
        category_weights: Optional[Dict[str, float]] = None,
        severe_prob: float = 0.25,
        ir_paths: Optional[Union[str, Sequence[str]]] = None,
        seed: Optional[int] = None,
    ):
        self.sample_rate = int(sample_rate)
        self.severe_prob = float(severe_prob)
        self.ir_paths = ir_paths

        weights = dict(self.DEFAULT_CATEGORY_WEIGHTS)
        if category_weights:
            unknown = set(category_weights) - set(self.CATEGORIES)
            if unknown:
                raise ValueError(f"Unknown degradation category/categories: {sorted(unknown)}")
            weights.update({k: float(v) for k, v in category_weights.items()})

        # Fail fast, before any training step, rather than silently renormalizing a
        # miskeyed/mistyped category_weights override. `weights` is guaranteed to already
        # contain every name in self.CATEGORIES by construction (it starts from a full copy of
        # self.DEFAULT_CATEGORY_WEIGHTS, which -- like every DEFAULT_CATEGORY_WEIGHTS(_V3)
        # dict/CATEGORIES pair in this module -- declares all of them); the `missing` check below
        # is a defensive assertion against that invariant, not a reachable user-facing error from
        # a partial category_weights override (those fall back to the class default per-key, they
        # don't raise).
        negative = {name: w for name, w in weights.items() if w < 0}
        if negative:
            raise ValueError(f"category_weights must all be nonnegative, got negative values for: {negative}")
        missing = set(self.CATEGORIES) - set(weights)
        if missing:
            raise ValueError(f"category_weights is missing expected categories: {sorted(missing)}")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("category_weights must sum to a positive value")
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"category_weights must sum to 1.0 (within floating-point tolerance), got "
                f"{total!r}: {weights!r}"
            )
        self._category_names = list(self.CATEGORIES)
        self._category_probs = [weights[name] / total for name in self._category_names]

        self._chains = self._build_chains()
        self._default_rng = np.random.default_rng(seed)

        # Populated by the most recent __call__, for debugging/logging only.
        self.last_categories: list[str] = []
        self.last_tiers: list[str] = []

    def _make_live_reverb_stage(self, room_size: Tuple[float, float], wet_level: Tuple[float, float]):
        """Real-IR convolution if ir_paths resolves to at least one file, else the
        synthetic NoiseShapedReverb fallback with the given (mild/severe) parameters."""
        if self.ir_paths:
            ir_transform = ApplyImpulseResponse(
                self.ir_paths,
                p=1.0,
                compensate_for_propagation_delay=True,
                sample_rate=self.sample_rate,
            )
            if ir_transform.ir_paths:
                return ir_transform
            logger.warning(
                "EnhancementDegradationPipelineV2: no IR files found under ir_paths=%r; "
                "falling back to synthetic NoiseShapedReverb for the 'live' category.",
                self.ir_paths,
            )
        return NoiseShapedReverb(sample_rate=self.sample_rate, room_size=room_size, wet_level=wet_level, p=1.0)

    def _build_chains(self) -> Dict[str, Dict[str, Compose]]:
        sr = self.sample_rate

        def compose(*transforms) -> Compose:
            return Compose(transforms=list(transforms), shuffle=False, output_type="tensor")

        return {
            "bandwidth": {
                "mild": compose(DownUpSample(source_sample_rate=sr, target_sample_rate=24000, p=1.0)),
                "severe": compose(DownUpSample(source_sample_rate=sr, target_sample_rate=8000, p=1.0)),
            },
            "mp3": {
                "mild": compose(Mp3Compression(bitrate_kbps_range=(128, 320), quality=7, p=1.0)),
                "severe": compose(Mp3Compression(bitrate_kbps_range=(32, 96), quality=7, p=1.0)),
            },
            "radio": {
                "mild": compose(
                    BiquadHighPassFilter(min_cutoff_freq=150.0, max_cutoff_freq=250.0, p=1.0),
                    BiquadLowPassFilter(min_cutoff_freq=6000.0, max_cutoff_freq=8000.0, p=1.0),
                    AddColoredNoise(min_snr_in_db=18.0, max_snr_in_db=30.0, p=1.0),
                ),
                "severe": compose(
                    BiquadHighPassFilter(min_cutoff_freq=250.0, max_cutoff_freq=400.0, p=1.0),
                    BiquadLowPassFilter(min_cutoff_freq=3000.0, max_cutoff_freq=5000.0, p=1.0),
                    AddColoredNoise(min_snr_in_db=0.0, max_snr_in_db=10.0, p=1.0),
                ),
            },
            "cassette": {
                "mild": compose(
                    BiquadHighPassFilter(min_cutoff_freq=60.0, max_cutoff_freq=100.0, p=1.0),
                    BiquadLowPassFilter(min_cutoff_freq=10000.0, max_cutoff_freq=14000.0, p=1.0),
                    CompandClip(sample_rate=sr, compand_drive_db=(2.0, 5.0), quant_bits=(10, 12), p=1.0),
                    AddColoredNoise(min_snr_in_db=25.0, max_snr_in_db=40.0, p=1.0),
                ),
                "severe": compose(
                    BiquadHighPassFilter(min_cutoff_freq=80.0, max_cutoff_freq=150.0, p=1.0),
                    BiquadLowPassFilter(min_cutoff_freq=6000.0, max_cutoff_freq=9000.0, p=1.0),
                    CompandClip(sample_rate=sr, compand_drive_db=(6.0, 12.0), quant_bits=(6, 8), p=1.0),
                    AddColoredNoise(min_snr_in_db=8.0, max_snr_in_db=20.0, p=1.0),
                ),
            },
            "vinyl": {
                "mild": compose(
                    BiquadHighPassFilter(min_cutoff_freq=40.0, max_cutoff_freq=70.0, p=1.0),
                    BiquadLowPassFilter(min_cutoff_freq=12000.0, max_cutoff_freq=16000.0, p=1.0),
                    ClickPopNoise(
                        sample_rate=sr, clicks_per_second=(0.5, 2.0), click_level_db=(-30.0, -18.0), p=1.0,
                    ),
                    AddColoredNoise(min_snr_in_db=30.0, max_snr_in_db=45.0, p=1.0),
                ),
                "severe": compose(
                    BiquadHighPassFilter(min_cutoff_freq=60.0, max_cutoff_freq=100.0, p=1.0),
                    BiquadLowPassFilter(min_cutoff_freq=7000.0, max_cutoff_freq=10000.0, p=1.0),
                    ClickPopNoise(
                        sample_rate=sr, clicks_per_second=(3.0, 8.0), click_level_db=(-15.0, -6.0), p=1.0,
                    ),
                    AddColoredNoise(min_snr_in_db=15.0, max_snr_in_db=28.0, p=1.0),
                ),
            },
            "live": {
                "mild": compose(
                    self._make_live_reverb_stage(room_size=(0.2, 0.4), wet_level=(0.05, 0.15)),
                    BiquadHighPassFilter(min_cutoff_freq=250.0, max_cutoff_freq=350.0, p=1.0),
                    BiquadLowPassFilter(min_cutoff_freq=3000.0, max_cutoff_freq=3600.0, p=1.0),
                    AddColoredNoise(min_snr_in_db=15.0, max_snr_in_db=28.0, p=1.0),
                ),
                "severe": compose(
                    self._make_live_reverb_stage(room_size=(0.4, 0.7), wet_level=(0.15, 0.35)),
                    BiquadHighPassFilter(min_cutoff_freq=250.0, max_cutoff_freq=350.0, p=1.0),
                    BiquadLowPassFilter(min_cutoff_freq=3000.0, max_cutoff_freq=3600.0, p=1.0),
                    AddColoredNoise(min_snr_in_db=-5.0, max_snr_in_db=10.0, p=1.0),
                ),
            },
        }

    def degrade_one(self, waveform: torch.Tensor, rng: Optional[np.random.Generator] = None) -> Tuple[torch.Tensor, str, str]:
        """Apply one randomly-chosen category+tier to a single item, shape [C, T]."""
        rng = rng if rng is not None else self._default_rng
        category = str(rng.choice(self._category_names, p=self._category_probs))
        if category == "clean":
            return waveform, category, "n/a"

        severe_prob = self.CATEGORY_SEVERE_PROB_OVERRIDES.get(category, self.severe_prob)
        tier = "severe" if rng.random() < severe_prob else "mild"
        transform = self._chains[category][tier]
        batched = waveform.unsqueeze(0)
        degraded = _run_transform_fp32_safe(transform, batched, self.sample_rate)
        return degraded.squeeze(0), category, tier

    def __call__(self, waveforms: torch.Tensor, rng: Optional[np.random.Generator] = None) -> torch.Tensor:
        """Apply independently-chosen degradations across a batch, shape [B, C, T]."""
        outputs = []
        self.last_categories = []
        self.last_tiers = []
        for i in range(waveforms.shape[0]):
            degraded, category, tier = self.degrade_one(waveforms[i], rng=rng)
            outputs.append(degraded)
            self.last_categories.append(category)
            self.last_tiers.append(tier)
        return torch.stack(outputs, dim=0)


# Extends DEFAULT_CATEGORY_WEIGHTS (v2's 7 categories) with the 3 new codec categories.
# The analog/acoustic categories (radio/cassette/vinyl/live) and clean keep the same
# relative weight as the "noise-heavy" v2 category weighting -- the
# best-performing v2 configuration on the full MSR-Bench validation set, evaluated
# together with the separation module -- with its combined bandwidth+mp3 budget (0.16)
# now split five ways across bandwidth/mp3/aac/dac/encodec so DT5-DT12 get meaningful
# training exposure without diluting the analog/acoustic categories that config tuned.
DEFAULT_CATEGORY_WEIGHTS_V3: Dict[str, float] = {
    "bandwidth": 0.02,
    "mp3": 0.04,
    "aac": 0.04,
    "dac": 0.03,
    "encodec": 0.03,
    "radio": 0.19,
    "cassette": 0.19,
    "vinyl": 0.19,
    "live": 0.24,
    "clean": 0.03,
}


class EnhancementDegradationPipelineV3(EnhancementDegradationPipelineV2):
    """
    Extends `EnhancementDegradationPipelineV2` with MSR-Bench's remaining degradation
    conditions -- DT5-DT12 (traditional + neural codecs) -- so that, combined with V2's
    radio/cassette/vinyl/live (DT1-DT4), all 12 MSR-Bench conditions plus DT0 (clean)
    are covered:

    - "mp3" is overridden here (vs. V2's broader bitrate ranges) to use MSR-Bench's
      exact bitrates: severe=64kbps (DT6), mild=128kbps (DT8).
    - "aac" (new): AAC encode/decode round-trip via `AacCompression` (ffmpeg subprocess,
      no batched AAC codec library is available). severe=64kbps (DT5), mild=128kbps (DT7).
    - "dac" (new): Descript Audio Codec round-trip via `DacCompression`. No 22kHz
      checkpoint is released, so severe uses the 24kHz-native model as a stand-in for
      MSR-Bench's DT9 "22kHz" condition (input is downsampled to the model's native
      24kHz first, so the combined effect is bandwidth-limiting *and* codec
      quantization, similar in spirit to the paper's condition); mild uses the
      44kHz-native model (DT10), closer to a pure codec-quantization-only condition.
      Unlike every other category, the mild/severe draw for "dac" is a fixed 50/50
      (`CATEGORY_SEVERE_PROB_OVERRIDES = {"dac": 0.5}` below) instead of inheriting the
      shared `severe_prob` -- the two DAC tiers are two different checkpoints (22kHz-proxy
      vs. 44kHz), not a "how harsh" dial like every other category's tiers, so they should
      get roughly equal training exposure rather than being skewed toward "mild" by
      `severe_prob`'s usual low default (0.1-0.25 in this repo's configs).
    - "encodec" (new): Meta EnCodec (48kHz-native stereo model, no resampling needed)
      round-trip via `EncodecCompression`. mild=6kbps (DT11), severe=3kbps (DT12).

    "dac"/"encodec" lazily import their optional dependencies (descript-audio-codec,
    encodec) and download pretrained weights on first use; if unavailable (package
    missing, or no network access to fetch weights), they log a warning once and pass
    audio through unchanged for that category for the rest of the run -- same fallback
    convention as V2's dasp_pytorch-based transforms -- so a run without these installed
    doesn't crash, it just silently under-trains on DT9-DT12. See
    src/dsp/neural_codec_effects.py for the transform implementations.
    """

    CATEGORIES: Tuple[str, ...] = (
        "bandwidth", "mp3", "radio", "cassette", "vinyl", "live",
        "aac", "dac", "encodec", "clean",
    )
    DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = DEFAULT_CATEGORY_WEIGHTS_V3
    # See "dac" bullet in this class's docstring: the two DAC tiers are a checkpoint
    # choice (22kHz-proxy vs. 44kHz), not a severity dial, so they get an even 50/50
    # draw regardless of the shared severe_prob used by every other category.
    CATEGORY_SEVERE_PROB_OVERRIDES: Dict[str, float] = {"dac": 0.5}

    def _build_chains(self) -> Dict[str, Dict[str, Compose]]:
        sr = self.sample_rate

        def compose(*transforms) -> Compose:
            return Compose(transforms=list(transforms), shuffle=False, output_type="tensor")

        chains = super()._build_chains()
        chains["mp3"] = {
            "severe": compose(Mp3Compression(bitrate_kbps_range=(64, 64), quality=7, p=1.0)),  # DT6
            "mild": compose(Mp3Compression(bitrate_kbps_range=(128, 128), quality=7, p=1.0)),  # DT8
        }
        chains["aac"] = {
            "severe": compose(AacCompression(bitrate_kbps_range=(64, 64), p=1.0)),  # DT5
            "mild": compose(AacCompression(bitrate_kbps_range=(128, 128), p=1.0)),  # DT7
        }
        chains["dac"] = {
            "mild": compose(DacCompression(sample_rate=sr, model_type="44khz", p=1.0)),  # DT10
            "severe": compose(DacCompression(sample_rate=sr, model_type="24khz", p=1.0)),  # DT9 (22kHz proxy)
        }
        chains["encodec"] = {
            "mild": compose(EncodecCompression(sample_rate=sr, target_bandwidth=6.0, p=1.0)),  # DT11
            "severe": compose(EncodecCompression(sample_rate=sr, target_bandwidth=3.0, p=1.0)),  # DT12
        }
        return chains


def build_degradation_pipeline(sample_rate: int = 48000, pipeline_version: str = "v3", **kwargs):
    """
    Factory for the mixture restoration model's degradation pipeline.

    pipeline_version:
        "v2": the richer, MSR-Bench-motivated category+severity scheme
            (`EnhancementDegradationPipelineV2`), covering DT1-DT4. Accepts:
            category_weights, severe_prob, ir_paths (real room IRs for the "live"
            category, e.g. EchoThief; falls back to synthetic reverb if unset/empty), seed.
        "v3" (default): V2 plus MSR-Bench's remaining conditions, DT5-DT12 (traditional + neural
            codecs) (`EnhancementDegradationPipelineV3`). Accepts the same kwargs as "v2".

    Kwargs not accepted by the selected pipeline_version's class are silently ignored here
    rather than raising a TypeError.

    "v2" and "v3" share the category_weights kwarg name but validate different sets
    of valid category names (V2: 7 categories; V3: 10, +aac/dac/encodec). Since the
    yaml's declared category_weights default has to include all 10 keys (so Sacred's
    `with` CLI recognizes aac/dac/encodec as pre-existing entries rather than raising
    ConfigAddedError when a v3 sweep script overrides them), a category_weights dict
    containing v3-only keys can reach here even when pipeline_version="v2" (e.g. via
    Sacred's config merge, if a script overrides pipeline_version alone without also
    overriding category_weights down to v2's 7 keys). Rather than let
    EnhancementDegradationPipelineV2 raise on those extra keys, silently drop any
    category_weights entries the selected class doesn't recognize.
    """
    if pipeline_version == "v2":
        cls = EnhancementDegradationPipelineV2
    elif pipeline_version == "v3":
        cls = EnhancementDegradationPipelineV3
    else:
        raise ValueError(
            f"Unknown degradation pipeline_version: {pipeline_version!r} (expected 'v2' or 'v3')"
        )

    accepted = set(inspect.signature(cls.__init__).parameters) - {"self"}
    ignored = sorted(set(kwargs) - accepted)
    if ignored:
        logger.debug(
            "build_degradation_pipeline(pipeline_version=%r): ignoring kwargs not used by %s: %s",
            pipeline_version, cls.__name__, ignored,
        )
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}

    category_weights = filtered_kwargs.get("category_weights")
    valid_categories = getattr(cls, "CATEGORIES", None)
    if category_weights and valid_categories is not None:
        unsupported = sorted(set(category_weights) - set(valid_categories))
        if unsupported:
            logger.debug(
                "build_degradation_pipeline(pipeline_version=%r): dropping category_weights "
                "entries not valid for %s: %s",
                pipeline_version, cls.__name__, unsupported,
            )
            filtered_kwargs["category_weights"] = {
                k: v for k, v in category_weights.items() if k in valid_categories
            }

    return cls(sample_rate=sample_rate, **filtered_kwargs)
