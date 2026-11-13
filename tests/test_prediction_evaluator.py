from __future__ import annotations

import math

import pytest
import torch

from src.training.evaluation import prediction_evaluator as pe_module
from src.training.evaluation.prediction_evaluator import PredictionEvaluator


def _make_eval_batch(source_label: str = "vocals", family: str = "DT0", samples: int = 48_000):
    gt = torch.randn(1, 2, samples, dtype=torch.float32)
    pred = gt.clone()
    return {
        "ground_truth": {
            "waveform": gt,
            "lengths": torch.tensor([samples], dtype=torch.long),
        },
        "prediction": {
            "waveform": pred,
            "lengths": torch.tensor([samples], dtype=torch.long),
        },
        "source_label": [source_label],
        "family": [family],
        "clip_id": ["clip_0"],
    }


def test_mel_metrics_identical_signal_behaviour():
    if pe_module.torchaudio is None:
        pytest.skip("torchaudio is required for mel-domain metric tests")
    if pe_module.skimage_ssim is None:
        pytest.skip("scikit-image is required for MEL_SSIM test")

    evaluator = PredictionEvaluator(
        columns=["vocals"],
        sampling_rate=16_000,
        prediction_metrics=["mel_ssim", "mel_kl_div", "kl"],
    )

    t = torch.linspace(0, 1.0, 16_000, dtype=torch.float32)
    wave = torch.stack([torch.sin(2 * torch.pi * 220 * t), torch.sin(2 * torch.pi * 330 * t)], dim=0)
    estimated = wave.unsqueeze(0)  # [1,2,T]
    target = wave.unsqueeze(0)

    metrics = evaluator._compute_metrics(estimated, target)

    mel_ssim = metrics["mel_ssim"][0]
    mel_kl = metrics["mel_kl_div"][0]
    kl = metrics["kl"][0]

    assert math.isfinite(mel_ssim)
    assert math.isfinite(mel_kl)
    assert math.isfinite(kl)
    assert mel_ssim > 0.95
    assert mel_kl < 1e-4
    assert abs(kl - mel_kl) < 1e-9


def test_pesq_uses_resample_to_configured_sample_rate(monkeypatch):
    if pe_module.torchaudio is None:
        pytest.skip("torchaudio is required for PESQ resampling test")

    captured: dict[str, int] = {}

    def _fake_pesq(ref, deg, sr):
        captured["sr"] = int(sr)
        captured["ref_len"] = int(ref.shape[0])
        captured["deg_len"] = int(deg.shape[0])
        return 3.2

    monkeypatch.setattr(pe_module, "pypesq_pesq", _fake_pesq)

    evaluator = PredictionEvaluator(
        columns=["vocals"],
        sampling_rate=48_000,
        prediction_metrics=["pesq"],
        prediction_pesq_sample_rate=16_000,
    )

    estimated = torch.randn(1, 2, 48_000, dtype=torch.float32)
    target = torch.randn(1, 2, 48_000, dtype=torch.float32)
    metrics = evaluator._compute_metrics(estimated, target)

    assert captured["sr"] == 16_000
    assert captured["ref_len"] == 16_000
    assert captured["deg_len"] == 16_000
    assert abs(metrics["pesq"][0] - 3.2) < 1e-9


def test_missing_backend_warns_and_returns_nan(monkeypatch, capsys):
    monkeypatch.setattr(pe_module, "pypesq_pesq", None)

    evaluator = PredictionEvaluator(
        columns=["vocals"],
        sampling_rate=16_000,
        prediction_metrics=["pesq"],
        prediction_pesq_sample_rate=16_000,
    )

    estimated = torch.randn(1, 2, 16_000, dtype=torch.float32)
    target = torch.randn(1, 2, 16_000, dtype=torch.float32)

    metrics = evaluator._compute_metrics(estimated, target)
    out = capsys.readouterr().out

    assert "pypesq unavailable" in out
    assert math.isnan(metrics["pesq"][0])


def test_integration_accumulates_sources_and_family(monkeypatch):
    evaluator = PredictionEvaluator(
        columns=["vocals", "bass"],
        sampling_rate=48_000,
        prediction_metrics=["sdr", "snr"],
    )

    def _fake_compute(estimated, target, mixture=None):
        return {
            "sdr": {0: 1.0, 1: float("nan")},
            "snr": {0: 2.0, 1: float("nan")},
        }

    monkeypatch.setattr(evaluator, "_compute_metrics", _fake_compute)

    batch = _make_eval_batch(source_label="vocals", family="DT1", samples=48_000)
    evaluator.test_step(batch, 0)

    assert evaluator.per_source_metric_values["sdr"][0] == [1.0]
    assert evaluator.per_source_metric_values["snr"][0] == [2.0]
    # NaN values of the other (silent) sources are kept; the summaries use nanmean / nanmedian
    assert [v for v in evaluator.per_family_metric_values["sdr"]["DT1"] if math.isfinite(v)] == [1.0]
    assert evaluator.per_family_and_source_metric_values["sdr"]["DT1"][0] == [1.0]
    assert evaluator.per_sample_scores == [
        {"dataset_idx": None, "clip_id": "clip_0", "source_label": "vocals", "sdr": 1.0, "snr": 2.0}
    ]

    logged = []

    def _fake_log(name: str, value: float, **kwargs):
        logged.append(name)

    monkeypatch.setattr(evaluator, "_safe_log", _fake_log)
    evaluator.on_test_epoch_end()

    assert any(name.startswith("eval_prediction_sources_median/vocals/sdr") for name in logged)
    assert any(name.startswith("eval_prediction_sources_mean/vocals/snr") for name in logged)
    assert any(name.startswith("eval_prediction_families_median/DT1/sdr") for name in logged)


def test_state_roundtrip_merge():
    ev_a = PredictionEvaluator(
        columns=["vocals"],
        sampling_rate=16_000,
        prediction_metrics=["sdr", "mel_ssim", "kl"],
    )
    ev_b = PredictionEvaluator(
        columns=["vocals"],
        sampling_rate=16_000,
        prediction_metrics=["sdr", "mel_ssim", "kl"],
    )

    ev_a.per_source_metric_values["sdr"][0].extend([1.0])
    ev_a.per_family_metric_values["mel_ssim"]["DTA"] = [0.9]
    ev_a.per_family_and_source_metric_values["sdr"]["DTA"] = {0: [1.2]}
    ev_a.total_samples_processed = 3

    ev_b.per_source_metric_values["sdr"][0].extend([2.0])
    ev_b.per_family_metric_values["mel_ssim"]["DTA"] = [0.8]
    ev_b.per_family_and_source_metric_values["sdr"]["DTA"] = {0: [2.2]}
    ev_b.total_samples_processed = 5

    ev_a.merge_state_from_other(ev_b.get_state_for_gather())

    assert ev_a.total_samples_processed == 8
    assert ev_a.per_source_metric_values["sdr"][0] == [1.0, 2.0]
    assert ev_a.per_family_metric_values["mel_ssim"]["DTA"] == [0.9, 0.8]
    assert ev_a.per_family_and_source_metric_values["sdr"]["DTA"][0] == [1.2, 2.2]
