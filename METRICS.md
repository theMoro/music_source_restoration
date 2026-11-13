# Metrics Reference

All metrics used in this codebase for evaluating source separation / restoration quality.

---

## SNR — Signal-to-Noise Ratio
- **Better:** Higher
- **Range:** dB (−∞ to +∞)
- **What it measures:** Ratio of target signal power to reconstruction error power, computed entirely in the waveform domain. Formula: `10 * log10(sum(target²) / sum((estimate − target)²))`. A score of 0 dB means the error is as powerful as the signal itself; good separations typically land in the 10–30 dB range.

---

## SI-SNR — Scale-Invariant Signal-to-Noise Ratio
- **Better:** Higher
- **Range:** dB (−∞ to +∞)
- **What it measures:** Like SNR, but both signals are zero-meaned and the estimate is projected onto the reference before measuring the residual. This makes the score independent of amplitude scaling — only directional reconstruction errors are penalised. Widely used in speech and music separation benchmarks.

---

## SDR — Signal-to-Distortion Ratio
- **Better:** Higher
- **Range:** dB (−∞ to +∞)
- **What it measures:** The standard BSS Eval metric (via museval), as used on the MUSDB18 benchmark. Decomposes the output into target, interference, and artefact components and measures how much is attributable to the target. Typical good values for music source separation are 8–20+ dB.

---

## Multi-Mel-SNR
- **Better:** Higher
- **Range:** dB (−∞ to +∞)
- **What it measures:** SNR computed in the mel-spectrogram domain at three resolutions (n\_fft 512 / 1024 / 2048), then averaged. Before scoring, the prediction is scale-invariant-normalised by projecting it onto the reference. Captures spectral reconstruction quality on a perceptually motivated frequency scale rather than the raw waveform.

---

## LogWMSE — Log-Weighted Mean Square Error
- **Better:** Higher
- **Range:** Positive float
- **What it measures:** Reconstruction error in log-frequency scale, using the mixture as a noise-floor reference so that errors in louder, more salient regions are weighted more heavily. Computed with `return_as_loss=False` so a higher score means less error. Evaluated over 10-second chunks; silent target chunks are skipped.

---

## Zimtohrli
- **Better:** Lower
- **Range:** ≥ 0
- **What it measures:** A psychoacoustic perceptual distance developed by Google. Computes a perceptual distance per channel based on auditory models and averages across channels. A distance of 0 means the audio is perceptually identical; larger values indicate more audible differences.

---

## MEL-SSIM — Mel Spectrogram Structural Similarity
- **Better:** Higher
- **Range:** [−1, 1] (practically [0, 1] for audio)
- **What it measures:** Structural Similarity Index (scikit-image SSIM) applied to log-mel spectrograms (dB scale). Measures luminance, contrast and structural similarity of the spectral representation. A score of 1.0 means the spectrograms are identical. Averaged over channels.

---

## Mel-KL-Div / KL — KL Divergence on Mel Spectrograms
- **Better:** Lower
- **Range:** ≥ 0
- **What it measures:** KL divergence KL(target ‖ estimate) computed on the softmax-normalised, flattened log-mel spectrograms. Measures how different the predicted spectral energy distribution is from the reference distribution. A score of 0 means the distributions match perfectly. `mel_kl_div` and `kl` are aliases for the same computation.

---

## PESQ — Perceptual Evaluation of Speech Quality
- **Better:** Higher
- **Range:** ~−0.5 to 4.5 (MOS-LQO scale)
- **What it measures:** ITU-T P.862 standardised speech quality metric. Scores ≥ 4 indicate excellent quality; scores ≤ 1 indicate poor quality. Audio is mixed to mono and resampled to 16 kHz before scoring. Most meaningful when the separated source contains speech-like content.

---

## Production Quality
- **Better:** Higher
- **Range:** Model-dependent (AudioBox Aesthetics `PQ` output)
- **What it measures:** A no-reference production quality predictor from AudioBox Aesthetics. Estimates perceived audio production quality (e.g. clarity, mix quality) without needing a reference signal — only the predicted output is evaluated.

---

## L1Freq
- **Better:** Higher
- **Range:** (0, 100]
- **What it measures:** L1 loss between STFT magnitude spectrograms, converted to a 0–100 score via `100 / (1 + 10 * L1_loss(|STFT(estimate)|, |STFT(reference)|))`. A score of 100 means perfect spectral magnitude match; the score approaches 0 as spectral differences grow.

---

## AuraSTFT
- **Better:** Higher
- **Range:** (0, 100]
- **What it measures:** Single-resolution STFT loss (log-magnitude + spectral centroid, from auraloss) converted to a 0–100 score via `100 / (1 + 10 * loss)`. Captures both the shape of the magnitude spectrum and the timbral centre of mass. A score of 100 means no spectral difference.

---

## AuraMRSTFT — Multi-Resolution STFT
- **Better:** Higher
- **Range:** (0, 100]
- **What it measures:** Multi-resolution mel-scaled STFT loss (FFT sizes 1024 / 2048 / 4096, perceptual weighting, auraloss), converted to a 0–100 score. By using multiple resolutions it provides better time-frequency coverage than a single-resolution STFT — capturing both fine spectral detail and broader temporal structure.

---

## Bleedless
- **Better:** Higher
- **Range:** (0, 100]
- **What it measures:** Measures the absence of spectral bleed. Computed from the average of mel-dB bins where the estimate *exceeds* the reference energy. A score of 100 means the estimate never has more energy than the reference anywhere in the mel spectrum. Low values indicate artefacts or bleed-through from other sources leaking into the separated stem.

---

## Fullness
- **Better:** Higher
- **Range:** (0, 100]
- **What it measures:** Measures content retention. Computed from the average of mel-dB bins where the estimate *falls below* the reference energy. A score of 100 means no content is missing. Low values indicate that the model over-attenuates the target, producing a thin or incomplete separation.

---

## FAD-CLAP — Fréchet Audio Distance with CLAP Embeddings
- **Better:** Lower
- **Range:** ≥ 0
- **What it measures:** Fréchet Distance between the distributions of reference and estimated audio in CLAP (laion/clap-htsat-unfused) embedding space. Formula: `||mu_ref − mu_est||² + Tr(sigma_ref + sigma_est − 2 * sqrt(sigma_ref @ sigma_est))`. A score of 0 means the estimated audio has the same perceptual distribution as the reference. Computed at epoch level over 10-second segments, not per batch.