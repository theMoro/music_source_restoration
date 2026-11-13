"""
Multi-resolution STFT-domain discriminator, ported from Hachimi
(https://github.com/linfeng-feng/Hachimi/blob/main/frequencydis.py).

Used by the Apollo GAN fine-tuning pipelines
(src/training/lightningmodule/mixture_restoration_gan.py and stem_restoration_gan.py). Architecture and defaults are
unchanged from the original.
"""

import torch
import torch.nn as nn


class FrequencyDiscriminator(nn.Module):
    def __init__(self, in_channels, hidden_channels=512, use_spectral_norm: bool = True):
        super().__init__()

        # spectral_norm caps each conv layer's operator norm at ~1, which bounds how far logits
        # can move from their init in a given number of optimizer steps (set False to disable).
        sn = nn.utils.spectral_norm if use_spectral_norm else (lambda module: module)

        self.eps = torch.finfo(torch.float32).eps
        self.discriminator = nn.ModuleList()
        self.discriminator += [
            nn.Sequential(
                sn(nn.Conv2d(in_channels, hidden_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                sn(nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                sn(nn.Conv2d(hidden_channels*2, hidden_channels*4, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                sn(nn.Conv2d(hidden_channels*4, hidden_channels*8, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                sn(nn.Conv2d(hidden_channels*8, hidden_channels*16, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Sequential(
                sn(nn.Conv2d(hidden_channels*16, hidden_channels*32, kernel_size=(3, 3), padding=(1, 1), stride=(2, 2))),
                nn.LeakyReLU(0.2, True)
            ),
            nn.Conv2d(hidden_channels*32, 1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1))
        ]

    def forward(self, x):
        hiddens = []
        for layer in self.discriminator:
            x = layer(x)
            hiddens.append(x)
        return x, hiddens[:-1]


class MultiFrequencyDiscriminator(nn.Module):
    """
    Ensemble of FrequencyDiscriminator instances, one per STFT window size.

    Args:
        nch: number of (interleaved-mic) input channels of the waveform.
        window: list of STFT window sizes (also used as hop = window // 2).
        hidden_channels: base hidden width for each FrequencyDiscriminator
            (doubles at every strided layer). Hachimi hardcodes this to 8.
        normalize_power: divide the input waveform by its own RMS before the STFT (Hachimi's
            original behavior). This equalizes real/fake loudness before the discriminator ever sees them, which
            throws away any level/gain mismatch as a discrimination cue -- plausible for
            separator-artifact stems, where a gain/bleed mismatch vs. the clean target may be one
            of the most obvious differences. Set False to let the discriminator see raw levels.
        use_spectral_norm: forwarded to each FrequencyDiscriminator, see its docstring.
    """

    def __init__(self, nch, window, hidden_channels: int = 8, normalize_power: bool = True, use_spectral_norm: bool = True):
        super().__init__()

        self.nch = nch
        self.window = window
        self.hidden_channels = hidden_channels
        self.normalize_power = normalize_power
        self.eps = torch.finfo(torch.float32).eps
        self.discriminators = nn.ModuleList([
            FrequencyDiscriminator(2*nch, self.hidden_channels, use_spectral_norm=use_spectral_norm)
            for _ in range(len(self.window))
        ])

    def forward(self, est, sample_rate=48000):

        B, nch, _ = est.shape
        assert nch == self.nch

        if self.normalize_power:
            est = est / (est.pow(2).sum((1, 2)) + self.eps).sqrt().reshape(B, 1, 1)
        est = est.view(-1, est.shape[-1])

        est_outputs = []
        est_feature_maps = []

        for i in range(len(self.discriminators)):
            est_spec = torch.stft(est.float(), self.window[i], self.window[i]//2,
                                   window=torch.hann_window(self.window[i]).to(est.device).float(),
                                   return_complex=True)
            est_RI = torch.stack([est_spec.real, est_spec.imag], dim=1)
            est_RI = est_RI.view(B, nch*2, est_RI.shape[-2], est_RI.shape[-1]).type(est.type())

            valid_enc = int(est_RI.shape[2] * sample_rate / 48000)
            est_out, est_feat_map = self.discriminators[i](est_RI[:, :, :valid_enc].contiguous())
            est_outputs.append(est_out)
            est_feature_maps.append(est_feat_map)

        return est_outputs, est_feature_maps
