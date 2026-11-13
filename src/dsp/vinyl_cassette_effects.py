"""
Small torch_audiomentations transforms for analog-media degradation simulation
(vinyl surface noise, tape-style saturation), ported from
restoration/finally_gan/effects.py::ClickPopNoise / CompandClip.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform
from torch_audiomentations.utils.object_dict import ObjectDict


class ClickPopNoise(BaseWaveformTransform):
    """Inject sparse impulsive clicks/pops (vinyl surface-noise/crackle simulation)."""

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        sample_rate: int,
        clicks_per_second: tuple[float, float] = (0.2, 3.0),
        click_width_samples: tuple[int, int] = (1, 32),
        click_level_db: tuple[float, float] = (-24.0, -6.0),
        p: float = 0.0,
    ) -> None:
        super().__init__(p=p)
        self.sample_rate = int(sample_rate)
        self.clicks_per_second = clicks_per_second
        self.click_width_samples = click_width_samples
        self.click_level_db = click_level_db

    def randomize_parameters(
        self,
        samples: torch.Tensor,
        sample_rate: int,
        targets=None,
        target_rate: Optional[int] = None,
    ) -> None:
        super().randomize_parameters(samples, sample_rate, targets, target_rate)
        batch_size = samples.shape[0]
        device = samples.device
        cps = torch.empty(batch_size, device=device).uniform_(
            float(self.clicks_per_second[0]),
            float(self.clicks_per_second[1]),
        )
        click_level_db = torch.empty(batch_size, device=device).uniform_(
            float(self.click_level_db[0]),
            float(self.click_level_db[1]),
        )
        self.transform_parameters["clicks_per_second"] = cps
        self.transform_parameters["click_level_db"] = click_level_db

    def apply_transform(
        self,
        samples: torch.Tensor,
        sample_rate: Optional[int] = None,
        targets=None,
        target_rate: Optional[int] = None,
    ):
        sr = int(sample_rate or self.sample_rate)
        out = samples.clone()
        batch_size, _, frames = out.shape
        device = out.device

        cps = self.transform_parameters["clicks_per_second"].to(device)
        click_level_db = self.transform_parameters["click_level_db"].to(device)
        min_width = max(1, int(self.click_width_samples[0]))
        max_width = max(min_width, int(self.click_width_samples[1]))
        duration_seconds = float(frames) / float(sr)

        for b in range(batch_size):
            expected_events = float(cps[b].item()) * duration_seconds
            event_count = int(torch.poisson(torch.tensor(expected_events, device=device)).item())
            if event_count <= 0:
                continue

            click_amp = float(torch.pow(torch.tensor(10.0, device=device), click_level_db[b] / 20.0).item())
            for _ in range(event_count):
                width = int(torch.randint(min_width, max_width + 1, (1,), device=device).item())
                width = min(width, frames)
                if width <= 0:
                    continue
                start_max = max(0, frames - width)
                start = int(torch.randint(0, start_max + 1, (1,), device=device).item())
                end = start + width
                sign = -1.0 if torch.rand(1, device=device).item() < 0.5 else 1.0

                if width == 1:
                    pulse = torch.tensor([sign * click_amp], device=device, dtype=out.dtype)
                else:
                    env = torch.exp(torch.linspace(0.0, -6.0, width, device=device, dtype=out.dtype))
                    pulse = env * (sign * click_amp)
                out[b, :, start:end] = out[b, :, start:end] + pulse.view(1, -1)

        out = torch.clamp(out, min=-1.0, max=1.0)
        return ObjectDict(samples=out, sample_rate=sr, targets=targets, target_rate=target_rate)


class CompandClip(BaseWaveformTransform):
    """Apply mild nonlinear companding + quantization artifacts (tape-style saturation)."""

    supports_multichannel = True
    requires_sample_rate = True

    def __init__(
        self,
        sample_rate: int,
        compand_drive_db: tuple[float, float] = (2.0, 10.0),
        compand_mix: tuple[float, float] = (0.3, 0.9),
        quant_bits: tuple[int, int] = (8, 12),
        p: float = 0.0,
    ) -> None:
        super().__init__(p=p)
        self.sample_rate = int(sample_rate)
        self.compand_drive_db = compand_drive_db
        self.compand_mix = compand_mix
        self.quant_bits = quant_bits

    def randomize_parameters(
        self,
        samples: torch.Tensor,
        sample_rate: int,
        targets=None,
        target_rate: Optional[int] = None,
    ) -> None:
        super().randomize_parameters(samples, sample_rate, targets, target_rate)
        batch_size = samples.shape[0]
        device = samples.device

        drive_db = torch.empty(batch_size, device=device).uniform_(
            float(self.compand_drive_db[0]),
            float(self.compand_drive_db[1]),
        )
        mix = torch.empty(batch_size, device=device).uniform_(
            float(self.compand_mix[0]),
            float(self.compand_mix[1]),
        )
        clip_threshold = torch.empty(batch_size, device=device).uniform_(0.80, 0.98)
        bits = torch.randint(
            int(self.quant_bits[0]),
            int(self.quant_bits[1]) + 1,
            (batch_size,),
            device=device,
        )

        self.transform_parameters["drive"] = torch.pow(10.0, drive_db / 20.0).view(-1, 1, 1)
        self.transform_parameters["mix"] = mix.view(-1, 1, 1)
        self.transform_parameters["clip_threshold"] = clip_threshold.view(-1, 1, 1)
        self.transform_parameters["bits"] = bits

    def apply_transform(
        self,
        samples: torch.Tensor,
        sample_rate: Optional[int] = None,
        targets=None,
        target_rate: Optional[int] = None,
    ):
        sr = int(sample_rate or self.sample_rate)
        dry = samples
        drive = self.transform_parameters["drive"].to(samples.device)
        mix = self.transform_parameters["mix"].to(samples.device)
        clip_threshold = self.transform_parameters["clip_threshold"].to(samples.device)
        bits = self.transform_parameters["bits"].to(samples.device)

        shaped = torch.tanh(dry * drive)
        clipped = torch.clamp(shaped, min=-clip_threshold, max=clip_threshold) / clip_threshold

        levels = torch.pow(2.0, bits.to(samples.dtype)).view(-1, 1, 1) - 1.0
        quantized = torch.round((clipped + 1.0) * 0.5 * levels) / levels
        quantized = quantized * 2.0 - 1.0

        wet = 0.65 * clipped + 0.35 * quantized
        out = (1.0 - mix) * dry + mix * wet
        out = torch.clamp(out, min=-1.0, max=1.0)
        return ObjectDict(samples=out, sample_rate=sr, targets=targets, target_rate=target_rate)
