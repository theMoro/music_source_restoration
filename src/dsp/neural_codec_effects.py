"""
Codec round-trip transforms for MSR-Bench's DT5-DT12 degradation conditions
(traditional lossy codecs + neural codecs), used by
`EnhancementDegradationPipelineV3` (see src/dsp/enhancement_degradation.py).

- `AacCompression`: AAC encode/decode round-trip. No batched/native AAC codec
  library is available in this repo's dependency set (unlike MP3, which has
  `fast_mp3_augment`), so this shells out to the system `ffmpeg` binary per
  item (two subprocess calls: encode to ADTS AAC, then decode back to PCM).
- `DacCompression`: Descript Audio Codec (https://github.com/descriptinc/descript-audio-codec,
  pip package `descript-audio-codec`, imported as `dac`) encode/decode
  round-trip, using a pretrained checkpoint downloaded on first use (cached at
  ~/.cache/descript/dac/, hardcoded by the `dac` package itself). Officially
  released checkpoints are mono-only at 16/24/44 kHz (no 22 kHz checkpoint, so
  the 24kHz model is used as a stand-in for MSR-Bench's DT9 "22kHz" condition
  -- see EnhancementDegradationPipelineV3's docstring). Stereo input is
  processed as two independent mono channels.
- `EncodecCompression`: Meta's EnCodec (https://github.com/facebookresearch/encodec,
  pip package `encodec`) 48kHz stereo model, encode/decode round-trip at a
  configurable target bandwidth (weights cached via torch.hub, i.e. respect
  TORCH_HOME same as this repo's other torch.hub-cached models).

`DacCompression` and `EncodecCompression` lazily import their backend and
download pretrained weights on first use; if the optional dependency is
missing (or the download fails, e.g. no network access from a training node),
they log a warning once and pass the audio through unchanged for that
category -- same fallback convention as the dasp_pytorch-based transforms in
music_effect_pipeline.py (Compressor/NoiseShapedReverb). `AacCompression`
falls back the same way if the `ffmpeg` binary isn't on PATH.
"""

from __future__ import annotations

import io
import logging
import subprocess
from typing import Any, Dict, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict

from .utils import fix_length_to_duration_torch

logger = logging.getLogger(__name__)

# Module-level caches so repeatedly constructing pipelines (e.g. in tests, or
# scripts that build several EnhancementDegradationPipelineV3 instances) don't
# re-download/re-load the same pretrained weights every time.
_DAC_MODEL_CACHE: Dict[str, Any] = {}
_ENCODEC_MODEL_CACHE: Dict[str, Any] = {}


def _aac_roundtrip_numpy(audio: np.ndarray, sample_rate: int, bitrate_kbps: int) -> np.ndarray:
    """Encode (C, T) float32 PCM to AAC at bitrate_kbps and decode back to (C, T) float32 PCM."""
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, audio.T, sample_rate, format="WAV", subtype="PCM_16")

    encode = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "wav", "-i", "pipe:0",
            "-c:a", "aac", "-b:a", f"{bitrate_kbps}k",
            "-f", "adts", "pipe:1",
        ],
        input=wav_buffer.getvalue(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    decode = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "aac", "-i", "pipe:0",  # demuxer name for raw ADTS AAC (the muxer name, used above, is "adts")
            "-f", "wav", "-ar", str(sample_rate), "pipe:1",
        ],
        input=encode.stdout,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    decoded, _ = sf.read(io.BytesIO(decode.stdout), dtype="float32", always_2d=True)
    return decoded.T  # (channels, frames)


class AacCompression(BaseWaveformTransform):
    """AAC encode/decode round-trip augmentation via an ffmpeg subprocess (mirrors
    torch_mastering.Mp3Compression's fast_mp3_augment round-trip, minus batching)."""

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        bitrate_kbps_range: tuple[int, int] = (64, 320),
        p: float = 0.1,
    ) -> None:
        super().__init__(p=p)
        self.bitrate_kbps_range = tuple(bitrate_kbps_range)
        self._ffmpeg_unavailable = False

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
        sr = int(sample_rate)
        if self._ffmpeg_unavailable:
            return ObjectDict(samples=samples, sample_rate=sr, targets=targets, target_rate=target_rate)

        bitrates = self.transform_parameters.get("bitrates", [])
        device, dtype = samples.device, samples.dtype
        original_length = samples.shape[-1]
        output_list = []

        for b in range(samples.shape[0]):
            audio_np = samples[b].detach().cpu().float().numpy()
            bitrate = bitrates[b] if b < len(bitrates) else self.bitrate_kbps_range[0]
            try:
                augmented_np = _aac_roundtrip_numpy(audio_np, sr, bitrate)
            except FileNotFoundError:
                logger.warning(
                    "AacCompression: `ffmpeg` binary not found on PATH; passing audio "
                    "through unchanged for the 'aac' category for the rest of this run."
                )
                self._ffmpeg_unavailable = True
                augmented_np = audio_np
            except Exception:
                logger.exception("AacCompression: ffmpeg round-trip failed, passing this item through unchanged.")
                augmented_np = audio_np
            output_list.append(torch.from_numpy(augmented_np))

        output = torch.stack(output_list, dim=0).to(device=device, dtype=dtype)
        output = fix_length_to_duration_torch(output, original_length)

        return ObjectDict(samples=output, sample_rate=sr, targets=targets, target_rate=target_rate)


class DacCompression(BaseWaveformTransform):
    """Descript Audio Codec encode/decode round-trip (mono model, applied per-channel)."""

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        sample_rate: int = 48_000,
        model_type: str = "44khz",
        p: float = 0.1,
    ) -> None:
        super().__init__(p=p)
        self.sample_rate = int(sample_rate)
        self.model_type = model_type
        self._unavailable = False

    def _load_model(self):
        if self.model_type in _DAC_MODEL_CACHE or self._unavailable:
            return
        try:
            import dac
        except ImportError:
            logger.warning(
                "DacCompression: `dac` package (descript-audio-codec) not installed; "
                "passing audio through unchanged for the 'dac' category for the rest "
                "of this run. `pip install descript-audio-codec` to enable it."
            )
            self._unavailable = True
            return
        try:
            model_path = dac.utils.download(model_type=self.model_type)
            model = dac.DAC.load(model_path)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            _DAC_MODEL_CACHE[self.model_type] = model
        except Exception:
            logger.exception(
                "DacCompression: failed to load pretrained DAC model_type=%r; passing "
                "audio through unchanged for the 'dac' category for the rest of this run.",
                self.model_type,
            )
            self._unavailable = True

    def apply_transform(
        self,
        samples: torch.Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        sr = int(sample_rate or self.sample_rate)
        self._load_model()
        if self._unavailable:
            return ObjectDict(samples=samples, sample_rate=sr, targets=targets, target_rate=target_rate)

        device, dtype = samples.device, samples.dtype
        model = _DAC_MODEL_CACHE[self.model_type].to(device)
        model_sr = int(model.sample_rate)

        batch, channels, frames = samples.shape
        flat = samples.reshape(batch * channels, 1, frames).float()

        if model_sr != sr:
            flat = torchaudio.transforms.Resample(sr, model_sr).to(device)(flat)

        with torch.no_grad():
            out = model(flat, model_sr)["audio"]

        if model_sr != sr:
            out = torchaudio.transforms.Resample(model_sr, sr).to(device)(out)

        out = out.reshape(batch, channels, -1).to(dtype=dtype)
        out = fix_length_to_duration_torch(out, frames)

        return ObjectDict(samples=out, sample_rate=sr, targets=targets, target_rate=target_rate)


class EncodecCompression(BaseWaveformTransform):
    """Meta EnCodec (48kHz stereo model) encode/decode round-trip at a target bandwidth."""

    supports_multichannel = True
    requires_sample_rate = True

    MODEL_SAMPLE_RATE = 48_000

    def __init__(
        self,
        sample_rate: int = 48_000,
        target_bandwidth: float = 6.0,
        p: float = 0.1,
    ) -> None:
        super().__init__(p=p)
        self.sample_rate = int(sample_rate)
        self.target_bandwidth = float(target_bandwidth)
        self._unavailable = False

    def _load_model(self):
        if "model" in _ENCODEC_MODEL_CACHE or self._unavailable:
            return
        try:
            from encodec import EncodecModel
        except ImportError:
            logger.warning(
                "EncodecCompression: `encodec` package not installed; passing audio "
                "through unchanged for the 'encodec' category for the rest of this "
                "run. `pip install encodec` to enable it."
            )
            self._unavailable = True
            return
        try:
            model = EncodecModel.encodec_model_48khz()
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            _ENCODEC_MODEL_CACHE["model"] = model
        except Exception:
            logger.exception(
                "EncodecCompression: failed to load the pretrained 48kHz EnCodec model; "
                "passing audio through unchanged for the 'encodec' category for the rest of this run."
            )
            self._unavailable = True

    def apply_transform(
        self,
        samples: torch.Tensor = None,
        sample_rate: Optional[int] = None,
        targets: Optional[torch.Tensor] = None,
        target_rate: Optional[int] = None,
    ):
        sr = int(sample_rate or self.sample_rate)
        self._load_model()
        if self._unavailable:
            return ObjectDict(samples=samples, sample_rate=sr, targets=targets, target_rate=target_rate)

        device, dtype = samples.device, samples.dtype
        model = _ENCODEC_MODEL_CACHE["model"].to(device)
        model.set_target_bandwidth(self.target_bandwidth)
        model_sr = self.MODEL_SAMPLE_RATE

        x = samples.float()
        needs_resample = model_sr != sr
        if needs_resample:
            x = torchaudio.transforms.Resample(sr, model_sr).to(device)(x)

        with torch.no_grad():
            out = model(x)

        if needs_resample:
            out = torchaudio.transforms.Resample(model_sr, sr).to(device)(out)

        out = fix_length_to_duration_torch(out, samples.shape[-1]).to(dtype=dtype)

        return ObjectDict(samples=out, sample_rate=sr, targets=targets, target_rate=target_rate)


__all__ = ["AacCompression", "DacCompression", "EncodecCompression"]
