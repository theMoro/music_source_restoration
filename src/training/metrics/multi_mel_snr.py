"""
Multi-Mel-SNR (Multi-Mel Spectrogram Signal-to-Noise Ratio) metric implementation.

Computes SNR in the mel-spectrogram domain across three resolutions (n_fft 512/1024/2048)
and averages the results. Before computing, the prediction is scale-invariant-normalised
by projecting it onto the reference. Formula per resolution:
  alpha = dot(ref, pred) / (dot(pred, pred) + eps)
  SNR_k = 10 * log10(sum(mel(ref)^2) / (sum(mel(ref) - mel(alpha*pred))^2 + eps))
  Multi-Mel-SNR = mean(SNR_0, SNR_1, SNR_2)  [dB]

Higher is better. Captures both temporal and spectral reconstruction quality in a
perceptually-motivated frequency scale.
"""

import torch
import torchaudio
import math


def multi_mel_snr_mel_f_max(sample_rate: float) -> float:
    """Mel ``f_max`` for Multi-Mel-SNR: cap at 24 kHz (48 kHz audio) and below Nyquist."""
    nyq = float(sample_rate) * 0.5
    return min(24000.0, max(100.0, nyq - 1.0))


def multi_mel_snr(reference, prediction, sr=48000):
    """Compute Multi-Mel-SNR between reference and prediction (mono waveforms).

    Returns a dB value. Higher is better.
    """
    if not isinstance(reference, torch.Tensor):
        reference = torch.from_numpy(reference).float()

    if not isinstance(prediction, torch.Tensor):
        prediction = torch.from_numpy(prediction).float()

    # Check for NaN/inf in inputs and replace with zeros
    reference = torch.where(torch.isfinite(reference), reference, torch.zeros_like(reference))
    prediction = torch.where(torch.isfinite(prediction), prediction, torch.zeros_like(prediction))

    # Scale-invariant normalization
    dot_ref_pred = torch.dot(reference, prediction)
    dot_pred_pred = torch.dot(prediction, prediction) + 1e-8
    alpha = dot_ref_pred / dot_pred_pred
    # Check for NaN/inf in alpha and handle
    if not torch.isfinite(alpha):
        alpha = torch.tensor(0.0)
    prediction = alpha * prediction

    # Three mel configurations
    configs = [
        (512, 256, 80),  # (n_fft, hop_length, n_mels)
        (1024, 512, 128),
        (2048, 1024, 192)
    ]

    f_max = multi_mel_snr_mel_f_max(sr)
    snrs = []
    for n_fft, hop, n_mels in configs:
        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr, n_fft=n_fft, hop_length=hop,
            n_mels=n_mels, f_min=0, f_max=f_max, power=2.0
        ).to(reference.device)
        M_ref = mel(reference)
        M_pred = mel(prediction)
        
        # Check for NaN/inf in mel spectrograms and replace with zeros
        M_ref = torch.where(torch.isfinite(M_ref), M_ref, torch.zeros_like(M_ref))
        M_pred = torch.where(torch.isfinite(M_pred), M_pred, torch.zeros_like(M_pred))
        
        ref_power = M_ref.pow(2).sum()
        diff_power = (M_ref - M_pred).pow(2).sum() + 1e-8
        
        # Check for NaN/inf in power values
        if not torch.isfinite(ref_power):
            ref_power = torch.tensor(0.0)
        if not torch.isfinite(diff_power):
            diff_power = torch.tensor(1e-8)
        
        snr_ratio = (ref_power / diff_power).clamp_min(1e-8)
        snr = 10 * torch.log10(snr_ratio)
        
        # Check for NaN/inf in SNR and replace with -80 dB
        if not torch.isfinite(snr):
            # snr = torch.tensor(-80.0)
            print("-" * 100)
            print("ALARM: assigning -80 value to SNR entry in multi-mel-snr!")
            raise Exception("ALARM: returned NaN/inf SNR value in multi-mel-snr!")

        snrs.append(snr.item())

    # Filter out any NaN/inf values from snrs list
    finite_snrs = [s for s in snrs if math.isfinite(s)]
    if finite_snrs:
        return sum(finite_snrs) / len(finite_snrs)
    else:
        print("-" * 100)
        print("ALARM: returned -80 value in multi-mel-snr!")
        raise Exception("ALARM: returned NaN/inf SNR value in multi-mel-snr!")
        # return -80.0  # Default to very negative SNR if all values are NaN/inf


def _align_stereo_ct(est: torch.Tensor, tgt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = min(est.shape[-1], tgt.shape[-1])
    return est[..., :n], tgt[..., :n]


def multi_mel_snr_stereo_ct(
    estimated_ct: torch.Tensor,
    target_ct: torch.Tensor,
    sample_rate: int,
) -> float:
    """Multi-Mel-SNR for stereo (or multi-channel) waveforms ``[C, T]``.

    Applies the same scale-invariant mel SNR logic as ``multi_mel_snr`` per channel
    and averages across channels. Returns dB. Higher is better.
    """
    est, tgt = _align_stereo_ct(estimated_ct.float(), target_ct.float())
    if est.shape[-1] == 0 or est.shape[0] == 0:
        return float("nan")
    f_max = multi_mel_snr_mel_f_max(float(sample_rate))
    configs = [(512, 256, 80), (1024, 512, 128), (2048, 1024, 192)]
    channel_snrs: list[float] = []
    for c in range(est.shape[0]):
        ref = torch.where(torch.isfinite(tgt[c]), tgt[c], torch.zeros_like(tgt[c]))
        pred = torch.where(torch.isfinite(est[c]), est[c], torch.zeros_like(est[c]))
        dot_rp = torch.dot(ref, pred)
        dot_pp = torch.dot(pred, pred) + 1e-8
        alpha = dot_rp / dot_pp if torch.isfinite(dot_pp) else torch.tensor(0.0)
        if not torch.isfinite(alpha):
            alpha = torch.tensor(0.0)
        pred = alpha * pred
        snrs: list[float] = []
        for n_fft, hop, n_mels in configs:
            mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                hop_length=hop,
                n_mels=n_mels,
                f_min=0,
                f_max=f_max,
                power=2.0,
            ).to(ref.device)
            M_ref = mel(ref.unsqueeze(0)).squeeze(0)
            M_pred = mel(pred.unsqueeze(0)).squeeze(0)
            M_ref = torch.where(torch.isfinite(M_ref), M_ref, torch.zeros_like(M_ref))
            M_pred = torch.where(torch.isfinite(M_pred), M_pred, torch.zeros_like(M_pred))
            ref_power = M_ref.pow(2).sum()
            diff_power = (M_ref - M_pred).pow(2).sum() + 1e-8
            if not torch.isfinite(ref_power):
                ref_power = torch.tensor(0.0)
            if not torch.isfinite(diff_power):
                diff_power = torch.tensor(1e-8)
            snr = (10 * torch.log10((ref_power / diff_power).clamp_min(1e-8))).item()
            if math.isfinite(snr):
                snrs.append(snr)
        if snrs:
            channel_snrs.append(sum(snrs) / len(snrs))
    if not channel_snrs:
        return float("nan")
    return float(sum(channel_snrs) / len(channel_snrs))