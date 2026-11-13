"""
Unit tests for MusicEffectPipeline and MasteringPipeline.

Covers:
- Output shape preservation for batched stereo audio at 48 kHz and 44.1 kHz.
- No NaN/Inf values in output.
- Per-item RMS projection (output RMS closely tracks input RMS).
- dtype and device preservation (CPU float32).
"""

from __future__ import annotations

import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rand_audio(batch: int, channels: int, seconds: float, sr: int, *, seed: int = 0) -> torch.Tensor:
    """Return a random (batch, channels, time) float32 tensor in [-0.5, 0.5]."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    frames = int(seconds * sr)
    return torch.rand(batch, channels, frames, generator=gen).float() - 0.5


def _per_item_rms(audio: torch.Tensor) -> torch.Tensor:
    """RMS per item, shape (B,)."""
    return torch.sqrt(torch.mean(audio ** 2, dim=(-2, -1)))


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

SAMPLE_RATES = [48_000, 44_100]
BATCH_SIZES   = [1, 4]
CHANNELS      = 2   # stereo
DURATION_S    = 1.0


# ---------------------------------------------------------------------------
# MusicEffectPipeline
# ---------------------------------------------------------------------------

class TestMusicEffectPipeline:

    @pytest.fixture(params=SAMPLE_RATES, ids=lambda sr: f"sr{sr}")
    def sr(self, request):
        return request.param

    @pytest.fixture(params=BATCH_SIZES, ids=lambda b: f"B{b}")
    def batch(self, request):
        return request.param

    @pytest.fixture
    def pipeline(self, sr):
        from src.dsp.music_effect_pipeline import MusicEffectPipeline
        return MusicEffectPipeline(sample_rate=sr)

    @pytest.fixture
    def audio(self, batch, sr):
        return _rand_audio(batch, CHANNELS, DURATION_S, sr)

    # --- shape ---

    def test_output_shape(self, pipeline, audio, sr):
        out = pipeline(audio, sample_rate=sr)
        assert out.shape == audio.shape, f"Shape mismatch: {out.shape} != {audio.shape}"

    # --- numerical health ---

    def test_no_nan(self, pipeline, audio, sr):
        out = pipeline(audio, sample_rate=sr)
        assert not torch.isnan(out).any(), "NaN values in output"

    def test_no_inf(self, pipeline, audio, sr):
        out = pipeline(audio, sample_rate=sr)
        assert not torch.isinf(out).any(), "Inf values in output"

    # --- RMS preservation (per item within 12 dB) ---

    def test_rms_preserved_per_item(self, pipeline, audio, sr):
        rms_in  = _per_item_rms(audio)
        rms_out = _per_item_rms(pipeline(audio, sample_rate=sr))
        ratio_db = 20 * torch.log10((rms_out + 1e-8) / (rms_in + 1e-8))
        assert (ratio_db.abs() < 12.0).all(), (
            f"RMS shift too large (>12 dB) for some items: {ratio_db}"
        )

    # --- dtype / device ---

    def test_dtype_preserved(self, pipeline, audio, sr):
        out = pipeline(audio, sample_rate=sr)
        assert out.dtype == audio.dtype

    def test_device_preserved(self, pipeline, audio, sr):
        out = pipeline(audio, sample_rate=sr)
        assert out.device == audio.device

    # --- silent input guard ---

    def test_silent_input(self, pipeline, sr):
        silent = torch.zeros(2, CHANNELS, int(DURATION_S * sr))
        out = pipeline(silent, sample_rate=sr)
        assert out.shape == silent.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


# ---------------------------------------------------------------------------
# MasteringPipeline
# ---------------------------------------------------------------------------

class TestMasteringPipeline:

    @pytest.fixture(params=SAMPLE_RATES, ids=lambda sr: f"sr{sr}")
    def sr(self, request):
        return request.param

    @pytest.fixture(params=BATCH_SIZES, ids=lambda b: f"B{b}")
    def batch(self, request):
        return request.param

    @pytest.fixture
    def pipeline(self, sr):
        from src.dsp.torch_mastering import MasteringPipeline
        return MasteringPipeline(sample_rate=sr, apply_prob=1.0)

    @pytest.fixture
    def audio(self, batch, sr):
        return _rand_audio(batch, CHANNELS, DURATION_S, sr)

    # --- shape ---

    def test_output_shape(self, pipeline, audio, sr):
        out = pipeline(audio, sample_rate=sr)
        assert out.shape == audio.shape, f"Shape mismatch: {out.shape} != {audio.shape}"

    # --- numerical health ---

    def test_no_nan(self, pipeline, audio, sr):
        out = pipeline(audio, sample_rate=sr)
        assert not torch.isnan(out).any(), "NaN values in output"

    def test_no_inf(self, pipeline, audio, sr):
        out = pipeline(audio, sample_rate=sr)
        assert not torch.isinf(out).any(), "Inf values in output"

    # --- RMS preservation (per item within 12 dB) ---

    def test_rms_preserved_per_item(self, pipeline, audio, sr):
        rms_in  = _per_item_rms(audio)
        rms_out = _per_item_rms(pipeline(audio, sample_rate=sr))
        ratio_db = 20 * torch.log10((rms_out + 1e-8) / (rms_in + 1e-8))
        assert (ratio_db.abs() < 12.0).all(), (
            f"RMS shift too large (>12 dB) for some items: {ratio_db}"
        )

    # --- dtype / device ---

    def test_dtype_preserved(self, pipeline, audio, sr):
        out = pipeline(audio, sample_rate=sr)
        assert out.dtype == audio.dtype

    def test_device_preserved(self, pipeline, audio, sr):
        out = pipeline(audio, sample_rate=sr)
        assert out.device == audio.device

    # --- silent input guard ---

    def test_silent_input(self, pipeline, sr):
        silent = torch.zeros(2, CHANNELS, int(DURATION_S * sr))
        out = pipeline(silent, sample_rate=sr)
        assert out.shape == silent.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    # --- implicit sample_rate (uses stored sr) ---

    def test_implicit_sample_rate(self, pipeline, audio):
        out = pipeline(audio)
        assert out.shape == audio.shape
