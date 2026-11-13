"""
Perceptual Zimtohrli distance metric.

Non-differentiable (CPU, numpy-backed) -- for validation-time monitoring only
(src.training.lightningmodule.stem_restoration_gan.StemRestorationGANLightning.validation_step),
never as a training loss term. Lower is better. Mirrors
src.training.evaluation.prediction_evaluator.PredictionEvaluator._zimtohrli_distance, duplicated
here (rather than shared) to keep the two call sites independent.
"""

import math

import torch


def zimtohrli_distance(calculator, estimated: torch.Tensor, target: torch.Tensor) -> float:
    """Average per-channel Zimtohrli distance between two [C, T] tensors. `calculator` is a
    zimtohrli.Pyohrli() instance, or None (in which case NaN is returned)."""
    if calculator is None:
        return float("nan")
    est = estimated.detach().cpu().float()
    tgt = target.detach().cpu().float()
    scores = []
    for ch in range(est.shape[0]):
        try:
            d = float(calculator.distance(tgt[ch].numpy(), est[ch].numpy()))
        except Exception:
            continue
        if math.isfinite(d):
            scores.append(d)
    return float(sum(scores) / len(scores)) if scores else float("nan")
