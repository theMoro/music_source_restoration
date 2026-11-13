"""
Applies one fine-tuned Apollo restoration model per stem to a separator's [B, S, C, T] output.

Companion to src.training.lightningmodule.stem_restoration_gan.StemRestorationGANLightning,
which (via configs/stem_restoration/apollo_stem_restoration.yaml and
scripts/stem_restoration/train_apollo.sh) fine-tunes a separate copy of
src.models.apollo.apollo_wrapper.ApolloWrapper per stem. Per-stem checkpoints are resolved as
<ckpt_root>/<version>/<stem>/<wandb_id>/<ckpt_filename> (globbing over <version> unless
ckpt_version is set). A stem with no resolved per-stem checkpoint falls back to a single *shared*
Apollo instance (the official pretrained checkpoint, or an explicit shared_ckpt_path) rather than
being left unrestored, so `apollo_stemwise` works with zero fine-tuned checkpoints (every stem uses
the shared model), a full set (every stem uses its own fine-tuned model), or anything in between.
ckpt_version pins the <version> glob segment to one training run instead of searching every
version folder under ckpt_root (useful if several runs wrote a checkpoint for the same stem and
wandb_id-less auto-selection would be ambiguous). ckpt_filenames overrides ckpt_filename (default
"last.ckpt") per stem — e.g. to pick an earlier/best checkpoint for a specific stem.

Meant to be plugged in as MultipleModelsWrapper.apollo_stemwise_model, running after separation,
before the final 44.1kHz -> 48kHz up-sample — i.e. at Apollo's native rate,
same placement as the `apollo` (mixture-restoration) role, see MultipleModelsWrapper.forward.
"""

import glob
import os
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .apollo_wrapper import ApolloWrapper

DEFAULT_GENERATOR_ARGS = dict(
    sr=44100,
    win=20,
    feature_dim=256,
    layer=6,
)


class ApolloStemwiseRestorationWrapper(nn.Module):
    def __init__(
        self,
        columns: List[str],
        ckpt_root: Optional[str] = None,
        ckpt_version: Optional[str] = None,
        ckpt_ids: Optional[Dict[str, str]] = None,
        ckpt_filename: str = "last.ckpt",
        ckpt_filenames: Optional[Dict[str, str]] = None,
        skip_stems: Optional[List[str]] = None,
        shared_ckpt_path: Optional[str] = None,
        generator_args: Optional[dict] = None,
        inference_chunk_size: Optional[int] = None,
        inference_chunk_overlap_ratio: float = 0.25,
    ):
        super().__init__()

        self.columns = list(columns)
        self.skip_stems = set(skip_stems or [])
        ckpt_ids = ckpt_ids or {}
        ckpt_filenames = ckpt_filenames or {}

        generator_args = {**DEFAULT_GENERATOR_ARGS, **(generator_args or {})}
        generator_args.pop('ckpt_path', None)
        generator_args.pop('trainable', None)

        def build_apollo(ckpt_path: Optional[str]) -> ApolloWrapper:
            return ApolloWrapper(
                ckpt_path=ckpt_path,
                inference_chunk_size=inference_chunk_size,
                inference_chunk_overlap_ratio=inference_chunk_overlap_ratio,
                trainable=False,
                **generator_args,
            )

        # Per-stem fine-tuned checkpoints (only for stems that resolve to exactly one).
        self.stem_models = nn.ModuleDict()
        stems_needing_shared = []
        for stem in self.columns:
            if stem in self.skip_stems:
                continue
            stem_ckpt = self._resolve_checkpoint(
                ckpt_root, ckpt_version, stem, ckpt_ids.get(stem),
                ckpt_filenames.get(stem) or ckpt_filename,
            )
            if stem_ckpt is not None:
                model = build_apollo(stem_ckpt)
                model.eval()
                for p in model.parameters():
                    p.requires_grad = False
                self.stem_models[stem] = model
            else:
                stems_needing_shared.append(stem)

        # One shared instance (the official pretrained checkpoint, or shared_ckpt_path) covers
        # every stem without its own fine-tuned checkpoint — built once, referenced by every such
        # stem in forward() (not duplicated per stem: same nn.Module object, so its parameters
        # aren't duplicated in memory either).
        self.shared_model: Optional[ApolloWrapper] = None
        if stems_needing_shared:
            print(f"[ApolloStemwiseRestorationWrapper] Using a shared Apollo checkpoint for "
                  f"stems without their own fine-tuned one: {stems_needing_shared}")
            self.shared_model = build_apollo(shared_ckpt_path)
            self.shared_model.eval()
            for p in self.shared_model.parameters():
                p.requires_grad = False

        if self.stem_models:
            print(f"[ApolloStemwiseRestorationWrapper] Per-stem fine-tuned checkpoints loaded "
                  f"for: {list(self.stem_models.keys())}")

    @staticmethod
    def _resolve_checkpoint(
        ckpt_root: Optional[str],
        ckpt_version: Optional[str],
        stem: str,
        wandb_id: Optional[str],
        ckpt_filename: str,
    ) -> Optional[str]:
        """Locates <ckpt_root>/<version>/<stem>/<wandb_id>/<ckpt_filename>. <version> defaults to
        a wildcard (a wandb_id is unique regardless of which version folder it landed in), but can
        be pinned via ckpt_version — e.g. to restrict resolution to a specific training run's
        version folder and ignore checkpoints from other runs that would otherwise make the match
        ambiguous. Returns None (falls back to the shared model) if ckpt_root is unset, no
        checkpoint is found, or the match is ambiguous."""
        if not ckpt_root:
            return None

        version = ckpt_version or "*"

        if wandb_id:
            matches = glob.glob(os.path.join(ckpt_root, version, stem, wandb_id, ckpt_filename))
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                print(f"[ApolloStemwiseRestorationWrapper] WARNING: wandb_id '{wandb_id}' for "
                      f"stem '{stem}' matched multiple checkpoints under {ckpt_root} "
                      f"(version={version!r}): {matches} — falling back to the shared model for "
                      f"this stem.")
                return None
            print(f"[ApolloStemwiseRestorationWrapper] WARNING: no checkpoint found for stem "
                  f"'{stem}' with wandb_id '{wandb_id}' under {ckpt_root} (version={version!r}) "
                  f"— falling back to the shared model for this stem.")
            return None

        # No explicit wandb_id given: only auto-select if the stem is unambiguous (exactly one
        # checkpoint under ckpt_root/version), otherwise fall back rather than silently guessing.
        matches = glob.glob(os.path.join(ckpt_root, version, stem, "*", ckpt_filename))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            print(f"[ApolloStemwiseRestorationWrapper] {len(matches)} candidate checkpoints "
                  f"found for stem '{stem}' under {ckpt_root} (version={version!r}) and no "
                  f"wandb_id was given via ckpt_ids to disambiguate — falling back to the shared "
                  f"model for this stem. Candidates: {matches}")
            return None
        return None

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'waveform' not in input_dict:
            raise ValueError("input_dict must contain a 'waveform' key of shape [B, S, C, T]")

        waveform = input_dict['waveform']
        if waveform.dim() != 4:
            raise ValueError(f"Expected waveform of shape [B, S, C, T], got shape {tuple(waveform.shape)}")

        _, n_sources, _, T = waveform.shape
        outputs = []
        for idx in range(n_sources):
            # .contiguous(): a single-index select out of the S dim leaves the [B, C, T] slice's
            # B-stride spanning all S stems (non-contiguous) -- the vendored Apollo model's
            # spec_band_split does input.view(B*nch, nsample), which needs the B/C dims mergeable
            # and raises "view size is not compatible..." without this (ApolloWrapper.forward's
            # own 4D-waveform branch avoids the same issue via .reshape(B*S, C, T) instead).
            stem_wave = waveform[:, idx].contiguous()  # [B, C, T]
            stem = self.columns[idx] if idx < len(self.columns) else None
            model = None
            if stem is not None and stem not in self.skip_stems:
                model = self.stem_models[stem] if stem in self.stem_models else self.shared_model
            if model is not None:
                with torch.no_grad():
                    restored = model({'mixture': stem_wave})['mixture']
                if restored.shape[-1] > T:
                    restored = restored[..., :T]
                elif restored.shape[-1] < T:
                    restored = F.pad(restored, (0, T - restored.shape[-1]))
                stem_wave = restored
            outputs.append(stem_wave)

        return {**input_dict, 'waveform': torch.stack(outputs, dim=1)}
