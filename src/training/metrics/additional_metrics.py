import numpy as np
import torch
import librosa
import torch.nn.functional as F
from typing import Tuple


def L1Freq_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        fft_size: int = 2048,
        hop_size: int = 1024,
        device: str = 'cpu'
) -> float:
    """
    Compute the L1 Frequency Metric between the reference and estimated audio signals.

    This metric compares the magnitude spectrograms of the reference and estimated audio signals
    using the Short-Time Fourier Transform (STFT) and calculates the L1 loss between them. The result
    is scaled to the range [0, 100] where a higher value indicates better performance.

    Parameters:
    ----------
    reference : np.ndarray
        A 2D numpy array of shape (num_channels, num_samples) representing the reference (ground truth) audio signal.

    estimate : np.ndarray
        A 2D numpy array of shape (num_channels, num_samples) representing the estimated (predicted) audio signal.

    fft_size : int, optional
        The size of the FFT (Short-Time Fourier Transform). Default is 2048.

    hop_size : int, optional
        The hop size between STFT frames. Default is 1024.

    device : str, optional
        The device to run the computation on ('cpu' or 'cuda'). Default is 'cpu'.

    Returns:
    -------
    float
        L1Freq score in (0, 100]. **Higher is better** — 100 means perfect spectral
        magnitude match; the score approaches 0 as the spectral L1 error grows.
        Formula: 100 / (1 + 10 * L1_loss(|STFT(estimate)|, |STFT(reference)|))
    """

    reference = torch.from_numpy(reference).to(device)
    estimate = torch.from_numpy(estimate).to(device)

    reference_stft = torch.stft(reference, fft_size, hop_size, return_complex=True)
    estimated_stft = torch.stft(estimate, fft_size, hop_size, return_complex=True)

    reference_mag = torch.abs(reference_stft)
    estimate_mag = torch.abs(estimated_stft)

    loss = 10 * F.l1_loss(estimate_mag, reference_mag)

    ret = 100 / (1. + float(loss.cpu().numpy()))

    return ret

def AuraSTFT_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        device: str = 'cpu',
) -> float:
    """
    Calculate the AuraSTFT metric, which evaluates the spectral difference between the reference and estimated
    audio signals using Short-Time Fourier Transform (STFT) loss.

    The AuraSTFT metric computes the STFT loss in both logarithmic and linear magnitudes, and it is commonly used
    to assess the quality of audio separation tasks. The result is returned as a value scaled to the range [0, 100].

    Parameters:
    ----------
    reference : np.ndarray
        The ground truth audio signal of shape (channels, time), where channels is the number of audio channels
        (e.g., 1 for mono, 2 for stereo) and time is the length of the audio in samples.

    estimate : np.ndarray
        The estimated audio signal of shape (channels, time).

    device : str, optional
        The device to run the computation on, either 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
    -------
    float
        AuraSTFT score in (0, 100]. **Higher is better** — 100 means no spectral
        difference. Combines log-magnitude STFT loss and spectral centroid loss.
        Formula: 100 / (1 + 10 * STFTLoss(reference, estimate))
    """

    from auraloss.freq import STFTLoss

    stft_loss = STFTLoss(
        w_log_mag=1.0,  # weight for log magnitude
        w_lin_mag=0.0,  # weight for linear magnitude
        w_sc=1.0,       # weight for spectral centroid
        device=device,
    )

    reference = torch.from_numpy(reference).unsqueeze(0).to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).to(device)

    res = 100 / (1. + 10 * stft_loss(reference, estimate))
    return float(res.cpu().numpy())


def AuraMRSTFT_metric(
        reference: np.ndarray,
        estimate: np.ndarray,
        device: str = 'cpu',
        sample_rate: int = 44100,
) -> float:
    """
    Calculate the AuraMRSTFT metric, which evaluates the spectral difference between the reference and estimated
    audio signals using Multi-Resolution Short-Time Fourier Transform (STFT) loss.

    The AuraMRSTFT metric uses multi-resolution STFT analysis, which allows better representation of both
    low- and high-frequency components in the audio signals. The result is returned as a value scaled to the range [0, 100].

    Parameters:
    ----------
    reference : np.ndarray
        The ground truth audio signal of shape (channels, time), where channels is the number of audio channels
        (e.g., 1 for mono, 2 for stereo) and time is the length of the audio in samples.

    estimate : np.ndarray
        The estimated audio signal of shape (channels, time).

    device : str, optional
        The device to run the computation on, either 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
    -------
    float
        AuraMRSTFT score in (0, 100]. **Higher is better** — 100 means no spectral
        difference. Uses multi-resolution mel-scaled STFT loss with perceptual weighting
        (FFT sizes 1024/2048/4096). Formula: 100 / (1 + 10 * MRSTFTLoss(reference, estimate))
    """

    from auraloss.freq import MultiResolutionSTFTLoss

    mrstft_loss = MultiResolutionSTFTLoss(
        fft_sizes=[1024, 2048, 4096],
        hop_sizes=[256, 512, 1024],
        win_lengths=[1024, 2048, 4096],
        scale="mel",  # mel scale for frequency resolution
        n_bins=128,   # number of bins for mel scale
        sample_rate=sample_rate,
        perceptual_weighting=True,  # apply perceptual weighting
        device=device
    )

    reference = torch.from_numpy(reference).unsqueeze(0).float().to(device)
    estimate = torch.from_numpy(estimate).unsqueeze(0).float().to(device)

    res = 100 / (1. + 10 * mrstft_loss(reference, estimate))
    return float(res.cpu().numpy())


def bleed_full(
        reference: np.ndarray,
        estimate: np.ndarray,
        sr: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 1024,
        n_mels: int = 512,
        device: str = 'cpu',
) -> Tuple[float, float]:
    """
    Calculate the 'bleed' and 'fullness' metrics between a reference and an estimated audio signal.

    The 'bleed' metric measures how much the estimated signal bleeds into the reference signal,
    while the 'fullness' metric measures how much the estimated signal retains its distinctiveness
    in relation to the reference signal, both using mel spectrograms and decibel scaling.

    Parameters:
    ----------
    reference : np.ndarray
        The reference audio signal, shape (channels, time), where channels is the number of audio channels
        (e.g., 1 for mono, 2 for stereo) and time is the length of the audio in samples.

    estimate : np.ndarray
        The estimated audio signal, shape (channels, time).

    sr : int, optional
        The sample rate of the audio signals. Default is 44100 Hz.

    n_fft : int, optional
        The FFT size used to compute the STFT. Default is 4096.

    hop_length : int, optional
        The hop length for STFT computation. Default is 1024.

    n_mels : int, optional
        The number of mel frequency bins. Default is 512.

    device : str, optional
        The device for computation, either 'cpu' or 'cuda'. Default is 'cpu'.

    Returns:
    -------
    tuple
        A tuple (bleedless, fullness), each a float in (0, 100]. **Higher is better for both.**
        - ``bleedless``: measures absence of spectral bleed — how much the estimate's mel
          spectrogram (dB) *exceeds* the reference (positive diff). Score of 100 means the
          estimate never has more energy than the reference. Low values indicate artefacts or
          bleed-through from other sources.
        - ``fullness``: measures content retention — how much the estimate's mel spectrogram
          *falls below* the reference (negative diff). Score of 100 means no missing content.
          Low values indicate an overly attenuated or incomplete separation.
    """

    from torchaudio.transforms import AmplitudeToDB

    reference = torch.from_numpy(reference).float().to(device)
    estimate = torch.from_numpy(estimate).float().to(device)

    window = torch.hann_window(n_fft).to(device)

    # Compute STFTs with the Hann window
    D1 = torch.abs(torch.stft(reference, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
                              pad_mode="constant"))
    D2 = torch.abs(torch.stft(estimate, n_fft=n_fft, hop_length=hop_length, window=window, return_complex=True,
                              pad_mode="constant"))

    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_filter_bank = torch.from_numpy(mel_basis).to(device)

    S1_mel = torch.matmul(mel_filter_bank, D1)
    S2_mel = torch.matmul(mel_filter_bank, D2)

    S1_db = AmplitudeToDB(stype="magnitude", top_db=80)(S1_mel)
    S2_db = AmplitudeToDB(stype="magnitude", top_db=80)(S2_mel)

    diff = S2_db - S1_db

    positive_diff = diff[diff > 0]
    negative_diff = diff[diff < 0]

    average_positive = torch.mean(positive_diff) if positive_diff.numel() > 0 else torch.tensor(0.0).to(device)
    average_negative = torch.mean(negative_diff) if negative_diff.numel() > 0 else torch.tensor(0.0).to(device)

    bleedless = 100 * 1 / (average_positive + 1)
    fullness = 100 * 1 / (-average_negative + 1)

    return bleedless.cpu().numpy(), fullness.cpu().numpy()
