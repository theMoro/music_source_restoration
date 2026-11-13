import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from .model import Apollo


class ApolloWrapper(nn.Module):
    """
    Wrapper around JusperLee/Apollo (https://github.com/JusperLee/Apollo) — a deterministic
    band-sequence regression network (see the vendored architecture in .model), not a generative
    model in the sampling sense (no noise input, no stochastic latent); upstream trains it
    adversarially (GAN + spectral discriminator) purely to sharpen its deterministic output, the
    same way this project fine-tunes it (src/training/lightningmodule/mixture_restoration_gan.py::
    MixtureRestorationGANLightning and stem_restoration_gan.py::StemRestorationGANLightning, both
    with generator=ApolloWrapper and trainable=True — see those modules).

    By default (trainable=False) this wrapper is inference-only — frozen, permanently eval() —
    for its two roles in MultipleModelsWrapper, selected by which key is in forward()'s
    input_dict (see that method):

      - `apollo` flag: pre-separation mixture restoration.
      - `apollo_stemwise` flag: post-separation, per-stem restoration — every separated stem
        restored independently (see ApolloStemwiseRestorationWrapper for per-stem checkpoints).

    Either way, operates at its native 44.1kHz (like separation_model), so it should run inside
    MultipleModelsWrapper's resample_to_44100_for_model=True block (the default in
    configs/separation/bs_roformer_8stem.yaml) — for
    `apollo` that means after the mixture has been down-sampled from 48kHz; for `apollo_stemwise`
    that means before the final up-sample back to 48kHz (see MultipleModelsWrapper.forward).

    With trainable=True (fine-tuning use), the freeze/eval-forcing above is skipped entirely —
    params keep requires_grad=True and train()/eval() mode is left to the caller (a Lightning
    module).

    Set `resample_from_sr` when used standalone (e.g. as a GAN generator — see
    StemRestorationGANLightning) outside MultipleModelsWrapper, which does not pre-resample:
    audio is then resampled resample_from_sr -> sr around every call, transparently. Leave it
    None (default) when MultipleModelsWrapper already handles the 48kHz<->44.1kHz round trip
    externally (both `apollo` and `apollo_stemwise` roles) — the default preserves that existing
    behavior unchanged.

    freeze_band_split / num_frozen_bsnet_layers (only meaningful with trainable=True): optional
    partial fine-tuning. self.model.BN is Apollo's per-band input projection (band-split
    front-end); self.model.net is a stack of `layer` BSNet blocks (the Roformer+conv backbone);
    self.model.output is the per-band output head. Freezing the front-end and/or the first
    `num_frozen_bsnet_layers` BSNet blocks leaves only the later blocks and the output head
    trainable, as a regularizer against full-parameter fine-tuning drifting Apollo's pretrained
    "real audio" prior on a dataset far narrower than what it was pretrained on.
    """

    OFFICIAL_REPO_ID = "JusperLee/Apollo"
    OFFICIAL_FILENAME = "pytorch_model.bin"

    def __init__(
        self,
        ckpt_path: Optional[str] = None,
        sr: int = 44100,
        win: int = 20,
        feature_dim: int = 256,
        layer: int = 6,
        inference_chunk_size: Optional[int] = None,
        inference_chunk_overlap_ratio: float = 0.25,
        trainable: bool = False,
        resample_from_sr: Optional[int] = None,
        freeze_band_split: bool = False,
        num_frozen_bsnet_layers: int = 0,
    ):
        super().__init__()

        self.model = Apollo(sr=sr, win=win, feature_dim=feature_dim, layer=layer)

        self.inference_chunk_size = inference_chunk_size
        self.inference_chunk_overlap_ratio = float(inference_chunk_overlap_ratio)
        self.trainable = bool(trainable)

        self.resample_from_sr = int(resample_from_sr) if resample_from_sr is not None else None
        if self.resample_from_sr is not None and self.resample_from_sr != sr:
            self.downsample_resampler = torchaudio.transforms.Resample(
                orig_freq=self.resample_from_sr, new_freq=sr,
                resampling_method='kaiser_window', lowpass_filter_width=64, rolloff=0.99, beta=14.0,
            )
            self.upsample_resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.resample_from_sr,
                resampling_method='kaiser_window', lowpass_filter_width=64, rolloff=0.99, beta=14.0,
            )
        else:
            self.downsample_resampler = None
            self.upsample_resampler = None

        resolved_ckpt_path = self._resolve_checkpoint(ckpt_path)
        self._load_checkpoint(resolved_ckpt_path)

        if not self.trainable:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self._freeze_early_layers(freeze_band_split, int(num_frozen_bsnet_layers))

    def _freeze_early_layers(self, freeze_band_split: bool, num_frozen_bsnet_layers: int) -> None:
        """See freeze_band_split/num_frozen_bsnet_layers in the class docstring."""
        if freeze_band_split:
            for param in self.model.BN.parameters():
                param.requires_grad = False
        if num_frozen_bsnet_layers > 0:
            bsnet_blocks = list(self.model.net.children())
            for bsnet_block in bsnet_blocks[:num_frozen_bsnet_layers]:
                for param in bsnet_block.parameters():
                    param.requires_grad = False

    def _resolve_checkpoint(self, ckpt_path: Optional[str]) -> str:
        """
        Use a local checkpoint file if given and present; otherwise auto-download the official
        pretrained weights from the Hugging Face Hub. The download is cached under
        HF_HUB_CACHE / HUGGINGFACE_HUB_CACHE (or HF_HOME), so it only actually hits the network
        once per cache location — but that first download still requires outbound internet
        access, which HPC compute nodes often don't have. If you're running on such nodes,
        either pre-download once from a login/head node (so the cache is warm before the job
        submits), or resolve the file yourself and pass its local path as ckpt_path to skip the
        Hub entirely.
        """
        if ckpt_path is not None:
            if os.path.isfile(ckpt_path):
                return ckpt_path
            print(
                f"Apollo checkpoint not found at '{ckpt_path}', falling back to auto-download "
                f"from the Hugging Face Hub ({self.OFFICIAL_REPO_ID})."
            )

        from huggingface_hub import hf_hub_download

        print(f"Downloading (or loading cached) Apollo checkpoint from {self.OFFICIAL_REPO_ID}...")
        return hf_hub_download(repo_id=self.OFFICIAL_REPO_ID, filename=self.OFFICIAL_FILENAME)

    def _load_checkpoint(self, ckpt_path: str):
        print("-" * 50)
        print(f"Loading Apollo restoration model from: {ckpt_path}")
        # The official checkpoint (pytorch_model.bin from the Hub, or a local copy of it) is a
        # dict with 'model_name'/'state_dict'/'model_args' keys (from upstream
        # look2hear.models.base_model.BaseModel.serialize()), not a bare state_dict — mirrors
        # BaseModel.from_pretrain's torch.load(..., weights_only=False).
        conf = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = conf['state_dict'] if isinstance(conf, dict) and 'state_dict' in conf else conf

        # A fine-tuned checkpoint (MixtureRestorationGANLightning's / StemRestorationGANLightning's
        # last.ckpt) is a full Lightning checkpoint
        # whose 'state_dict' is keyed by attribute path off the LightningModule
        # (self.generator = ApolloWrapper(...), self.discriminator = ...) rather than this bare
        # Apollo module's own keys -- unwrap it here so both checkpoint formats load the same way.
        generator_prefix = "generator.model."
        if any(k.startswith(generator_prefix) for k in state_dict.keys()):
            state_dict = {
                k[len(generator_prefix):]: v
                for k, v in state_dict.items() if k.startswith(generator_prefix)
            }

        self.model.load_state_dict(state_dict, strict=True)
        print("Successfully loaded Apollo checkpoint.")
        print("-" * 50)

    def _restore_chunk(self, audio: torch.Tensor) -> torch.Tensor:
        """Restore a single audio chunk. audio: [B, C, T]"""
        T = audio.shape[-1]
        original_dtype = audio.dtype
        # Apollo's STFT/ISTFT use a 20ms window (882 samples at 44.1kHz) -- not a power of two,
        # which cuFFT only supports in float32/float64, not float16. Under a trainer configured
        # for mixed precision (e.g. Lightning's precision="16-mixed"), the mixture reaching here
        # is already fp16, so force float32 for the whole forward pass (autocast disabled, so
        # nothing downstream gets silently re-cast to fp16 either) and cast back afterward for
        # compatibility with the rest of the (possibly fp16) pipeline.
        with torch.autocast(device_type=audio.device.type, enabled=False):
            restored = self.model(audio.float()).to(original_dtype)
        if restored.shape[-1] > T:
            restored = restored[..., :T]
        elif restored.shape[-1] < T:
            restored = F.pad(restored, (0, T - restored.shape[-1]))
        return restored

    def _restore(self, audio: torch.Tensor) -> torch.Tensor:
        """Restore audio [B, C, T], transparently resampling around the native-rate model call
        when resample_from_sr is set (see class docstring)."""
        if self.downsample_resampler is None:
            return self._restore_native(audio)

        original_length = audio.shape[-1]
        original_dtype = audio.dtype
        audio_native = self.downsample_resampler(audio.float())
        restored_native = self._restore_native(audio_native)
        restored = self.upsample_resampler(restored_native.float())
        if restored.shape[-1] > original_length:
            restored = restored[..., :original_length]
        elif restored.shape[-1] < original_length:
            restored = F.pad(restored, (0, original_length - restored.shape[-1]))
        return restored.to(original_dtype)

    def _restore_native(self, audio: torch.Tensor) -> torch.Tensor:
        """Restore audio [B, C, T] already at Apollo's native sample rate, with optional
        overlap-add chunking."""
        total_length = audio.shape[-1]
        use_chunking = self.inference_chunk_size is not None and total_length > self.inference_chunk_size

        if not use_chunking:
            return self._restore_chunk(audio)

        chunk_length = self.inference_chunk_size
        overlap_samples = int(chunk_length * self.inference_chunk_overlap_ratio)
        hop_length = chunk_length - overlap_samples
        n_chunks = max(1, (total_length - overlap_samples + hop_length - 1) // hop_length)

        output_full = torch.zeros_like(audio)

        if overlap_samples > 0:
            window = torch.hann_window(2 * overlap_samples, device=audio.device, dtype=audio.dtype)
            chunk_window = torch.ones(chunk_length, device=audio.device, dtype=audio.dtype)
            chunk_window[:overlap_samples] = window[:overlap_samples]
            chunk_window[-overlap_samples:] = window[overlap_samples:]
            weight_acc = torch.zeros(total_length, device=audio.device, dtype=audio.dtype)
        else:
            chunk_window = None

        for i in range(n_chunks):
            start = i * hop_length
            end = min(start + chunk_length, total_length)
            chunk = audio[..., start:end]
            if chunk.shape[-1] < chunk_length:
                chunk = F.pad(chunk, (0, chunk_length - chunk.shape[-1]))

            out_chunk = self._restore_chunk(chunk)

            actual_len = end - start
            out_chunk = out_chunk[..., :actual_len]

            if chunk_window is not None:
                w = chunk_window[:actual_len]
                out_chunk = out_chunk * w
                weight_acc[start:start + actual_len] += w
                output_full[..., start:start + actual_len] += out_chunk
            else:
                output_full[..., start:start + actual_len] = out_chunk

        if overlap_samples > 0 and weight_acc.max() > 0:
            weight_acc = weight_acc.clamp(min=1e-8)
            output_full = output_full / weight_acc

        return output_full

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Restore audio in input_dict. Serves two roles in MultipleModelsWrapper depending on
        which key is present:

          - 'mixture' ([B, C, T]): pre-separation mixture restoration (the `apollo` flag).
          - 'waveform' ([B, S, C, T] or [B, C, T]): post-separation per-stem restoration (the
            `apollo_stemwise` flag) — every stem is restored independently and uniformly.

        Checked in that order (waveform first) since after separation both keys are typically
        present in input_dict simultaneously.
        """
        if 'waveform' in input_dict:
            waveform = input_dict['waveform']
            if waveform.dim() == 4:  # [B, S, C, T] — restore each stem independently
                B, S, C, T = waveform.shape
                flat = waveform.reshape(B * S, C, T)
                restored_flat = self._restore(flat)
                restored = restored_flat.reshape(B, S, C, -1)
            else:  # [B, C, T]
                restored = self._restore(waveform)
            return {**input_dict, 'waveform': restored}
        elif 'mixture' in input_dict:
            mixture = input_dict['mixture']
            restored = self._restore(mixture)
            return {**input_dict, 'mixture': restored}
        else:
            raise ValueError("input_dict must contain a 'waveform' or 'mixture' key")
