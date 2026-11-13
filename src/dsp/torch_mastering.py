"""
Torch-based offline mastering chain built from torch_audiomentations transforms.

Pipeline stages:
1) Linear-phase high-pass to clean sub frequencies.
2) Loudness normalization to a target LUFS (multiple backends supported).
3) True-peak limiting with oversampling and lookahead.
4) Optional mid/side width adjustment.

All transforms operate on tensors shaped (batch, channels, time).
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch import Tensor
from .music_effect_pipeline import Compressor, DownUpSample
from .utils import fix_length_to_duration_torch
from torch_audiomentations import Compose
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict


class LoudnessNormalize(nn.Module):
    """
    Normalize integrated loudness to a target LUFS value.

    backends:
        - 'torchaudio': uses torchaudio.functional.loudness (GPU-capable).
        - 'pyloudnorm': converts to numpy and uses pyloudnorm.Meter.
        - 'loudness_py': delegates to the Brecht De Man reference implementation.
        - 'auto': pick the first available backend in the order above.
    """

    def __init__(self, target_lufs: float = -14.0, backend: str = "auto", sample_rate: int = 48_000) -> None:
        super().__init__()
        self.target_lufs = float(target_lufs)
        self.backend = backend
        self.sample_rate = int(sample_rate)

    @torch.no_grad()
    def _lufs_torchaudio(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        import torchaudio.functional as taf

        # original_device = samples.device
        # original_dtype = samples.dtype
        # loudness_input = samples.detach().to(device="cpu", dtype=torch.float32)
        # loudness = taf.loudness(loudness_input, sample_rate=sample_rate)
        # if loudness.ndim == 0:
        #     loudness = loudness.unsqueeze(0)
        # return loudness.to(device=original_device, dtype=original_dtype)

        return taf.loudness(samples, sample_rate=sample_rate)

    @torch.no_grad()
    def _lufs_pyloudnorm(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        import pyloudnorm as pyln

        batch, _, _ = samples.shape
        values = []
        meter = pyln.Meter(sample_rate)
        for idx in range(batch):
            data = samples[idx].T.detach().cpu().numpy()
            values.append(float(meter.integrated_loudness(data)))
        return torch.tensor(values, dtype=samples.dtype, device=samples.device)

    @torch.no_grad()
    def _lufs_loudness_py(self, samples: torch.Tensor, sample_rate: int) -> torch.Tensor:
        try:
            import loudness as ldn  # type: ignore[import]
        except Exception:  # pragma: no cover - best effort import name
            from loudness import loudness as ldn  # type: ignore[import]

        batch, _, _ = samples.shape
        values = []
        for idx in range(batch):
            data = samples[idx].T.detach().cpu().numpy()
            values.append(float(ldn.loudness(data, sample_rate)))
        return torch.tensor(values, dtype=samples.dtype, device=samples.device)

    def _select_backend(self) -> str:
        backend = self.backend
        if backend != "auto":
            return backend

        try:
            import torchaudio.functional  # noqa: F401

            return "torchaudio"
        except Exception:  # pragma: no cover - optional dependency
            pass

        try:
            import pyloudnorm  # noqa: F401

            return "pyloudnorm"
        except Exception:  # pragma: no cover - optional dependency
            pass

        return "loudness_py"

    @torch.no_grad()
    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        if samples.ndim != 3:
            raise ValueError(f"Expected (batch, channels, time) audio, received shape {samples.shape}")

        backend = self._select_backend()
        if backend == "torchaudio":
            lufs = self._lufs_torchaudio(samples, self.sample_rate)
        elif backend == "pyloudnorm":
            lufs = self._lufs_pyloudnorm(samples, self.sample_rate)
        else:
            lufs = self._lufs_loudness_py(samples, self.sample_rate)

        # Handle non-finite loudness values (NaN, inf, -inf) e.g. silent inputs
        mask = ~torch.isfinite(lufs)
        if mask.any():
            lufs = lufs.clone()
            lufs[mask] = float(self.target_lufs)

        gain_db = self.target_lufs - lufs
        gain = (10.0 ** (gain_db / 20.0)).view(-1, 1, 1)
        return samples * gain


class MidSideWidth(BaseWaveformTransform):
    """Mid/side stereo width adjustment. Mono inputs are returned unchanged."""

    supports_multichannel = True
    requires_sample_rate = False

    def __init__(self, width_db: float = 0.0, p: float = 0.1) -> None:
        super().__init__(p=p)
        self.width_db = float(width_db)

    def _apply_width(self, samples: Tensor) -> Tensor:
        if samples.ndim != 3:
            raise ValueError(f"Expected (batch, channels, time) audio, received shape {samples.shape}")
        if samples.size(1) < 2:
            return samples

        gain = 10.0 ** (self.width_db / 20.0)
        left = samples[:, 0:1, :]
        right = samples[:, 1:2, :]
        mid = 0.5 * (left + right)
        side = 0.5 * (left - right) * gain
        left_out = mid + side
        right_out = mid - side
        if samples.size(1) > 2:
            return torch.cat([left_out, right_out, samples[:, 2:, :]], dim=1)
        return torch.cat([left_out, right_out], dim=1)

    def apply_transform(
        self,
        samples: torch.Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        if samples is None:
            raise ValueError("samples parameter is required and cannot be None")

        return ObjectDict(
            samples=self._apply_width(samples),
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class TruePeakLimiter(BaseWaveformTransform):
    """Lookahead limiter that respects a true-peak ceiling via oversampling."""

    supports_multichannel = True
    requires_sample_rate = False

    def __init__(
        self,
        tp_db: float = -1.0,
        oversample: int = 4,
        sample_rate: int = 48_000,
        lookahead_ms: float = 1.0,
        release_ms: float = 50.0,
        eps: float = 1e-8,
        p: float = 0.1,
    ) -> None:
        super().__init__(p=p)
        if oversample < 2:
            raise ValueError("oversample must be >= 2")
        self.tp_db = float(tp_db)
        self.oversample = int(oversample)
        self.sample_rate = int(sample_rate)
        self.lookahead_ms = float(lookahead_ms)
        self.release_ms = float(release_ms)
        self.eps = float(eps)
        self._resampler_cache: dict = {}

    def _resampler(self, src: int, dst: int, device: torch.device, dtype: torch.dtype) -> torchaudio.transforms.Resample:
        key = (src, dst, device, dtype)
        if key not in self._resampler_cache:
            self._resampler_cache[key] = torchaudio.transforms.Resample(src, dst).to(device=device, dtype=dtype)
        return self._resampler_cache[key]

    def _limit(self, samples: Tensor, sample_rate: int) -> Tensor:
        if samples.ndim != 3:
            raise ValueError(f"Expected (batch, channels, time) audio, received shape {samples.shape}")

        batch, channels, frames = samples.shape
        device, dtype = samples.device, samples.dtype
        up_rate = sample_rate * self.oversample

        up = self._resampler(sample_rate, up_rate, device, dtype)
        down = self._resampler(up_rate, sample_rate, device, dtype)

        reshaped = samples.reshape(batch * channels, 1, frames)
        upsampled = up(reshaped)
        up_frames = upsampled.size(-1)

        magnitude = upsampled.abs()
        lookahead = max(1, int(self.lookahead_ms * 1e-3 * up_rate))
        envelope = F.max_pool1d(magnitude, kernel_size=lookahead, stride=1, padding=lookahead // 2)

        threshold = 10.0 ** (self.tp_db / 20.0)
        gain = torch.clamp(threshold / (envelope + self.eps), max=1.0)

        # Causal moving-average release smoothing. A brute-force conv1d with a uniform
        # `release`-tap kernel is O(up_frames * release) -- at oversample=4/48kHz and a typical
        # release_ms=50 (release ~= 9600 taps), that's ~10 seconds of direct convolution per batch
        # item on CPU, dwarfing every other stage in this pipeline (measured: >90% of
        # ReferenceMasteringPipeline's total cost, and the reason apply_prob=0.1 in the old
        # MasteringPipeline usage mostly hid it -- see apply_mastering_pipeline's docstring on
        # MixtureRestorationGANLightning, which now runs this at p=1.0 on every step). A prefix-sum
        # moving average is mathematically identical (uniform-kernel convolution == a sliding-window
        # mean) but O(up_frames) regardless of release width -- verified numerically equivalent
        # (max abs diff ~6e-5 in float32, from cumsum accumulation error over ~10^5-10^6 samples).
        release = max(1, int(self.release_ms * 1e-3 * up_rate))
        gain = F.pad(gain, (release - 1, 0), mode="replicate")
        cumulative = F.pad(torch.cumsum(gain, dim=-1), (1, 0), mode="constant", value=0.0)
        out_len = gain.size(-1) - release + 1
        gain = (cumulative[..., release:] - cumulative[..., :out_len]) / release

        if gain.size(-1) != up_frames:
            gain = F.pad(gain, (0, up_frames - gain.size(-1))) if gain.size(-1) < up_frames else gain[..., :up_frames]
        limited = upsampled * gain

        restored = down(limited)
        restored = restored.view(batch, channels, -1)
        if restored.size(-1) != frames:
            restored = (
                F.pad(restored, (0, frames - restored.size(-1)))
                if restored.size(-1) < frames
                else restored[..., :frames]
            )
        return restored

    def apply_transform(
        self,
        samples: torch.Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        if samples is None:
            raise ValueError("samples parameter is required and cannot be None")

        effective_sample_rate = int(sample_rate) if sample_rate is not None else self.sample_rate
        return ObjectDict(
            samples=self._limit(samples, sample_rate=effective_sample_rate),
            sample_rate=effective_sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


class Mp3Compression(BaseWaveformTransform):
    """
    Apply MP3 encode/decode round-trip augmentation using fast_mp3_augment.

    Simulates MP3 compression artefacts at a randomly chosen bitrate, which is
    a common degradation encountered in real-world music production pipelines.
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        bitrate_kbps_range: tuple[int, int] = (64, 320),
        quality: int = 7,
        preserve_delay: bool = False,
        p: float = 0.1,
    ) -> None:
        super().__init__(p=p)
        self.bitrate_kbps_range = bitrate_kbps_range
        self.quality = int(quality)
        self.preserve_delay = bool(preserve_delay)

    def randomize_parameters(
        self,
        samples: torch.Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        batch_size = samples.shape[0]
        lo, hi = self.bitrate_kbps_range
        self.transform_parameters["bitrates"] = [
            int(torch.randint(lo, hi + 1, (1,)).item()) for _ in range(batch_size)
        ]

    def apply_transform(
        self,
        samples: torch.Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        try:
            import fast_mp3_augment  # pragma: no cover
        except ImportError:  # pragma: no cover
            return ObjectDict(
                samples=samples,
                sample_rate=sample_rate,
                targets=targets,
                target_rate=target_rate,
            )

        bitrates = self.transform_parameters.get("bitrates", [])
        device, dtype = samples.device, samples.dtype
        original_length = samples.shape[-1]
        output_list = []

        for b in range(samples.shape[0]):
            audio_np = samples[b].detach().cpu().float().numpy()  # (C, T)
            bitrate = bitrates[b] if b < len(bitrates) else self.bitrate_kbps_range[0]
            try:
                augmented_np = fast_mp3_augment.compress_roundtrip(  # pragma: no cover
                    audio_np,
                    sample_rate=sample_rate,
                    bitrate_kbps=bitrate,
                    preserve_delay=self.preserve_delay,
                    quality=self.quality,
                )
            except Exception:  # pragma: no cover
                augmented_np = audio_np
            output_list.append(torch.from_numpy(augmented_np))

        output = torch.stack(output_list, dim=0).to(device=device, dtype=dtype)
        # Restore length in case the codec shifted the length slightly
        output = fix_length_to_duration_torch(output, original_length)

        return ObjectDict(
            samples=output,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


def build_mastering_compose(
    sample_rate: int,
    apply_prob=0.1,
    tp_db: float = -1.0,
    width_db: float = 0.0,
    oversample: int = 4,
    lookahead_ms: float = 1.0,
    release_ms: float = 50.0,
    downsample_sample_rate: Optional[int] = 16000,
    enable_mp3: bool = True,
) -> Compose:
    """Create a torch_audiomentations.Compose mastering chain."""

    if Compose is None:  # pragma: no cover - runtime guard
        raise RuntimeError("torch_audiomentations is required for build_mastering_compose but is not installed")

    transforms = [
        TruePeakLimiter(
            tp_db=tp_db,
            oversample=oversample,
            sample_rate=sample_rate,
            lookahead_ms=lookahead_ms,
            release_ms=release_ms,
            p=apply_prob,
        ),
        DownUpSample(target_sample_rate=downsample_sample_rate, p=apply_prob),
        Compressor(sample_rate=sample_rate, p=apply_prob),
    ]
    if enable_mp3:
        transforms.append(Mp3Compression(p=apply_prob))
    if abs(width_db) > 1e-9:
        transforms.append(MidSideWidth(width_db=width_db, p=apply_prob))

    return Compose(transforms=transforms, shuffle=False, output_type="tensor")


class MasteringPipeline:
    """
    Wraps a mastering compose chain with per-item RMS preservation.

    Pipeline:
        1. Record per-item RMS of the input.
        2. Peak-normalize each item to [-1, 1].
        3. Apply the mastering chain.
        4. Restore original length (in case any stage changed it).
        5. Compute new per-item RMS and scale back to the original level.
    """

    def __init__(
        self,
        sample_rate: int,
        apply_prob=0.1,
        tp_db: float = -1.0,
        width_db: float = 0.0,
        oversample: int = 4,
        lookahead_ms: float = 1.0,
        release_ms: float = 50.0,
        run_on_gpu: bool = False,
        enable_mp3: bool = True,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.run_on_gpu = run_on_gpu
        self.chain = build_mastering_compose(
            sample_rate=sample_rate,
            apply_prob=apply_prob,
            tp_db=tp_db,
            width_db=width_db,
            oversample=oversample,
            lookahead_ms=lookahead_ms,
            release_ms=release_ms,
            enable_mp3=enable_mp3,
        )

    def __call__(self, samples: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        """
        Apply the mastering chain with per-item RMS preservation.

        Parameters
        ----------
        samples:
            Tensor shaped (batch, channels, time) in the range [-1, 1].
        sample_rate:
            Optionally override the stored sample rate.
        """
        sr = int(sample_rate) if sample_rate is not None else self.sample_rate
        original_length = samples.shape[-1]
        device = samples.device

        # Per-item RMS: mean over channels + time → shape [B, 1, 1]
        rms_origin = torch.sqrt(torch.mean(samples ** 2, dim=(-2, -1), keepdim=True))

        # Peak-normalize each item to [-1, 1] before processing
        normalized = samples / (torch.amax(torch.abs(samples), dim=(-2, -1), keepdim=True) + 1e-8)

        # Run chain on CPU when input is on CUDA unless run_on_gpu is True (avoids cuFFT/OOM issues)
        shift_to_cpu = device.type == "cuda" and not self.run_on_gpu
        if shift_to_cpu:
            normalized = normalized.cpu()
        mastered = self.chain(normalized, sample_rate=sr)
        if shift_to_cpu:
            mastered = mastered.to(device)

        # Restore original length in case any stage altered it
        mastered = fix_length_to_duration_torch(mastered, original_length)

        # New per-item RMS
        rms_new = torch.sqrt(torch.mean(mastered ** 2, dim=(-2, -1), keepdim=True))

        # Project back to the original RMS level
        return mastered * (rms_origin / (rms_new + 1e-8))


class ParametricEQ(BaseWaveformTransform):
    """
    Cascaded shelving + peaking EQ -- covers both "broadband" (low-shelf/high-shelf, a coarse
    tonal tilt) and "multiband" (peaking bands at a few fixed center frequencies, more surgical
    correction) EQ in one class, since both are just a cascade of biquad filters and splitting
    them into two transform classes would duplicate the same per-item-loop machinery for no
    behavioral difference. Each band's gain is drawn independently per item from its
    `gain_db` range on every call (frequency/Q/type are fixed hyperparameters, not randomized --
    only *how much* EQ is applied varies, not *where*), so the reference mastering condition
    still gets per-track tonal variation rather than one identical curve stamped on every example.

    bands: sequence of {"type": "low_shelf"|"high_shelf"|"peak", "freq": float,
        "gain_db": (lo, hi), "q": float}. Shelf bands use torchaudio.functional.bass_biquad/
        treble_biquad; peak bands use equalizer_biquad -- torchaudio's standard biquad EQ
        implementations (same family already used by BiquadHighPassFilter/BiquadLowPassFilter in
        music_effect_pipeline.py, including their per-item-loop pattern for per-item-randomized
        filter parameters).
    """

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(self, bands: Sequence[Dict], p: float = 0.1) -> None:
        super().__init__(p=p)
        if not bands:
            raise ValueError("ParametricEQ requires at least one band")
        for band in bands:
            if band.get("type") not in ("low_shelf", "high_shelf", "peak"):
                raise ValueError(f"Unknown EQ band type: {band.get('type')!r}")
        self.bands = [dict(b) for b in bands]

    def randomize_parameters(
        self,
        samples: torch.Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        if samples is None:
            raise ValueError("samples parameter is required and cannot be None")
        batch_size = samples.shape[0]
        gains = torch.empty(batch_size, len(self.bands), device=samples.device)
        for band_idx, band in enumerate(self.bands):
            lo, hi = band["gain_db"]
            gains[:, band_idx].uniform_(float(lo), float(hi))
        self.transform_parameters["band_gains_db"] = gains

    def apply_transform(
        self,
        samples: torch.Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        if samples is None:
            raise ValueError("samples parameter is required and cannot be None")
        if sample_rate is None:
            raise ValueError("sample_rate parameter is required and cannot be None")

        output = samples.clone()
        gains = self.transform_parameters.get("band_gains_db")
        sr = int(sample_rate)
        for i in range(output.shape[0]):
            item = output[i]
            for band_idx, band in enumerate(self.bands):
                gain = float(gains[i, band_idx].item()) if gains is not None else 0.0
                if abs(gain) < 1e-6:
                    continue
                freq = float(band["freq"])
                q = float(band.get("q", 0.707))
                if band["type"] == "low_shelf":
                    item = torchaudio.functional.bass_biquad(item, sample_rate=sr, gain=gain, central_freq=freq, Q=q)
                elif band["type"] == "high_shelf":
                    item = torchaudio.functional.treble_biquad(item, sample_rate=sr, gain=gain, central_freq=freq, Q=q)
                else:
                    item = torchaudio.functional.equalizer_biquad(item, sample_rate=sr, center_freq=freq, gain=gain, Q=q)
            output[i] = item

        return ObjectDict(
            samples=output,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate,
        )


# Gentle, "mastering-grade" ranges (a couple dB at most) -- corrective tonal balance, not creative/
# aggressive EQ. Low/high shelf = broadband; the three peak bands = multiband.
DEFAULT_REFERENCE_EQ_BANDS = [
    {"type": "low_shelf", "freq": 100.0, "gain_db": (-2.0, 2.0), "q": 0.707},    # broadband: low-end warmth/thinness
    {"type": "peak", "freq": 250.0, "gain_db": (-2.0, 2.0), "q": 1.0},           # multiband: low-mid mud control
    {"type": "peak", "freq": 1000.0, "gain_db": (-1.5, 1.5), "q": 1.0},          # multiband: presence
    {"type": "peak", "freq": 4000.0, "gain_db": (-2.0, 2.0), "q": 1.0},          # multiband: clarity/harshness
    {"type": "high_shelf", "freq": 9000.0, "gain_db": (-2.0, 2.0), "q": 0.707},  # broadband: air/brightness
]


class MultibandCompressor(nn.Module):
    """
    3-band (low/mid/high) downward compression -- the "multiband compression" stage of the
    reference mastering chain. Splits with a simple complementary lowpass/highpass crossover
    (low = lowpass(x), high = highpass(x), mid = x - low - high, so low+mid+high reconstructs x
    exactly before any band-specific gain change) -- not a phase-perfect Linkwitz-Riley crossover,
    but sufficient for training-data augmentation rather than release-quality mastering. Compresses
    each band independently with this repo's existing dasp_pytorch-backed `Compressor` (the same
    transform class already used -- and proven in production -- by MasteringPipeline /
    multisource_separation.py), then sums the bands back together.
    """

    def __init__(
        self,
        sample_rate: int,
        low_crossover_hz: float = 200.0,
        high_crossover_hz: float = 2000.0,
        crossover_q: float = 0.707,
        low_band_config: Optional[dict] = None,
        mid_band_config: Optional[dict] = None,
        high_band_config: Optional[dict] = None,
    ) -> None:
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.low_crossover_hz = float(low_crossover_hz)
        self.high_crossover_hz = float(high_crossover_hz)
        self.crossover_q = float(crossover_q)

        # Gentle, mastering-style ranges (low ratios, moderate knees, band-appropriate timing) --
        # multiband mastering compression corrects balance between bands, it doesn't squash the
        # mix like a limiter. Faster attack/release and lower threshold/ratio as frequency rises,
        # matching how mastering engineers typically compress bass more conservatively than highs.
        low_band_config = low_band_config or dict(
            threshold_db=(-30.0, -20.0), ratio=(1.5, 2.5), attack_ms=(10.0, 30.0),
            release_ms=(100.0, 300.0), knee_db=(3.0, 6.0), makeup_gain_db=(0.0, 2.0),
        )
        mid_band_config = mid_band_config or dict(
            threshold_db=(-24.0, -16.0), ratio=(1.5, 2.5), attack_ms=(5.0, 15.0),
            release_ms=(80.0, 200.0), knee_db=(3.0, 6.0), makeup_gain_db=(0.0, 2.0),
        )
        high_band_config = high_band_config or dict(
            threshold_db=(-20.0, -12.0), ratio=(1.3, 2.0), attack_ms=(1.0, 5.0),
            release_ms=(50.0, 150.0), knee_db=(3.0, 6.0), makeup_gain_db=(0.0, 1.5),
        )

        def _band_chain(config: dict) -> Compose:
            return Compose(
                transforms=[Compressor(sample_rate=sample_rate, p=1.0, **config)],
                shuffle=False,
                output_type="tensor",
            )

        self._low_chain = _band_chain(low_band_config)
        self._mid_chain = _band_chain(mid_band_config)
        self._high_chain = _band_chain(high_band_config)

    def forward(self, samples: Tensor, sample_rate: Optional[int] = None) -> Tensor:
        sr = int(sample_rate) if sample_rate is not None else self.sample_rate
        low = torchaudio.functional.lowpass_biquad(
            samples, sample_rate=sr, cutoff_freq=self.low_crossover_hz, Q=self.crossover_q
        )
        high = torchaudio.functional.highpass_biquad(
            samples, sample_rate=sr, cutoff_freq=self.high_crossover_hz, Q=self.crossover_q
        )
        mid = samples - low - high

        low_out = self._low_chain(low, sample_rate=sr)
        mid_out = self._mid_chain(mid, sample_rate=sr)
        high_out = self._high_chain(high, sample_rate=sr)
        return low_out + mid_out + high_out


class ReferenceMasteringPipeline:
    """
    Complete "mastering" chain matching MSR-Bench's DT0 reference-mastering description:
    broadband + multiband EQ, multiband compression, stereo width adjustment, and
    streaming-platform loudness normalization + true-peak limiting. The mastered mixture this
    produces is the reference condition that degradation is applied to (and that a de-mastering
    restoration model must reconstruct) -- see apply_mastering_pipeline's docstring on
    MixtureRestorationGANLightning (src/training/lightningmodule/mixture_restoration_gan.py).

    Unlike `MasteringPipeline` above (used by the separation module's mixture augmentation via
    multisource_separation.py), this pipeline intentionally excludes bandwidth-reduction/
    MP3-round-trip stages: those are degradations, not mastering, and a DT0 reference condition
    must stay a clean, undegraded signal. Every stage always fires (every sub-transform is
    constructed with p=1.0) -- the reference mastering condition applies to every example, not
    probabilistically. Only each stage's *internal* parameters (EQ band gains, compressor
    thresholds/ratios/etc.) are randomized per item, for per-track variation.

    Pipeline order: ParametricEQ (tone) -> MultibandCompressor (dynamics) -> MidSideWidth (space)
    -> LoudnessNormalize -> TruePeakLimiter (final loudness/safety ceiling, applied last so the
    true-peak ceiling still holds regardless of what EQ/compression/width did to peak level).

    No RMS-preservation wrapper (unlike MasteringPipeline.__call__): the entire point of the final
    loudness-normalize + true-peak-limit stage is to *set* the output level deliberately --
    restoring the pre-mastering RMS afterward would undo exactly the loudness change mastering is
    supposed to make.
    """

    def __init__(
        self,
        sample_rate: int,
        eq_bands: Optional[Sequence[Dict]] = None,
        multiband_compressor_config: Optional[dict] = None,
        width_db: float = 1.5,
        target_lufs: float = -14.0,
        loudness_backend: str = "auto",
        tp_db: float = -1.0,
        oversample: int = 4,
        lookahead_ms: float = 1.0,
        release_ms: float = 50.0,
        run_on_gpu: bool = False,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.run_on_gpu = run_on_gpu

        eq_bands = eq_bands if eq_bands is not None else DEFAULT_REFERENCE_EQ_BANDS
        multiband_compressor_config = multiband_compressor_config or {}

        self._eq_chain = Compose(
            transforms=[ParametricEQ(bands=eq_bands, p=1.0)],
            shuffle=False,
            output_type="tensor",
        )
        self._multiband_compressor = MultibandCompressor(sample_rate=sample_rate, **multiband_compressor_config)
        self._width_chain = Compose(
            transforms=[MidSideWidth(width_db=width_db, p=1.0)],
            shuffle=False,
            output_type="tensor",
        )
        self._loudness_normalize = LoudnessNormalize(
            target_lufs=target_lufs, backend=loudness_backend, sample_rate=sample_rate
        )
        self._limiter_chain = Compose(
            transforms=[
                TruePeakLimiter(
                    tp_db=tp_db,
                    oversample=oversample,
                    sample_rate=sample_rate,
                    lookahead_ms=lookahead_ms,
                    release_ms=release_ms,
                    p=1.0,
                )
            ],
            shuffle=False,
            output_type="tensor",
        )

    def __call__(self, samples: torch.Tensor, sample_rate: Optional[int] = None) -> torch.Tensor:
        sr = int(sample_rate) if sample_rate is not None else self.sample_rate
        original_length = samples.shape[-1]
        device = samples.device

        # Run on CPU when input is on CUDA unless run_on_gpu is True -- same cuFFT/OOM-avoidance
        # rationale as MasteringPipeline.__call__ above.
        shift_to_cpu = device.type == "cuda" and not self.run_on_gpu
        x = samples.cpu() if shift_to_cpu else samples

        x = self._eq_chain(x, sample_rate=sr)
        x = self._multiband_compressor(x, sample_rate=sr)
        x = self._width_chain(x, sample_rate=sr)
        x = self._loudness_normalize(x)
        x = self._limiter_chain(x, sample_rate=sr)

        x = fix_length_to_duration_torch(x, original_length)
        if shift_to_cpu:
            x = x.to(device)
        return x


__all__ = [
    "LoudnessNormalize",
    "MidSideWidth",
    "TruePeakLimiter",
    "build_mastering_compose",
    "MasteringPipeline",
    "Mp3Compression",
    "ParametricEQ",
    "DEFAULT_REFERENCE_EQ_BANDS",
    "MultibandCompressor",
    "ReferenceMasteringPipeline",
]
