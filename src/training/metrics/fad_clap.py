"""
FAD-CLAP metric implementation for source separation evaluation.

FAD-CLAP (Fréchet Audio Distance with CLAP embeddings) measures the distributional
distance between target and estimated audio in CLAP embedding space, using the
Fréchet Distance formula:
  FAD = ||mu_ref - mu_est||^2 + Tr(sigma_ref + sigma_est - 2*sqrt(sigma_ref @ sigma_est))

Lower is better. A score of 0 means the estimated audio has the same statistical
distribution as the reference in CLAP's perceptual embedding space. Computed at
epoch level (not per batch) over 10-second segments.
"""

import warnings
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
from scipy.linalg import sqrtm

warnings.filterwarnings("ignore")

try:
    from transformers import ClapModel, ClapProcessor
except ImportError:
    print("Warning: transformers library not installed. FAD-CLAP will not be available.")
    ClapModel = None
    ClapProcessor = None


class FADCLAPMetric(nn.Module):
    """FAD-CLAP metric for evaluating source separation quality.

    Computes Fréchet Audio Distance using CLAP (laion/clap-htsat-unfused) embeddings.
    Lower is better — 0 means the distributions of estimates and targets are identical
    in perceptual embedding space.
    """
    
    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        batch_size: int = 2,  # Further reduced for stability
        device: Optional[str] = None,
        cache_embeddings: bool = True,
        max_samples_per_epoch: int = 0,  # Max clips per source per epoch; 0 = no limit (all clips)
        force_cpu: bool = False,  # Force CPU usage to avoid CUDA issues
        max_audio_seconds: float = 10.0,  # Max supported audio duration (raises if longer)
        max_audio_overage_tolerance_seconds: float = 0.5,  # Silently trim overages up to this
                                                             # much past max_audio_seconds (e.g.
                                                             # resample-rounding drift); raise if
                                                             # audio exceeds max_audio_seconds by
                                                             # more than this, since that signals
                                                             # a real data/config bug rather than
                                                             # rounding noise.
    ):
        super().__init__()
        
        if ClapModel is None or ClapProcessor is None:
            raise ImportError(
                "transformers library is required for FAD-CLAP. "
                "Install with: pip install transformers"
            )
        
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = "cpu" if force_cpu else (device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cache_embeddings = cache_embeddings
        self.max_samples_per_epoch = max_samples_per_epoch
        self.force_cpu = force_cpu
        self.max_audio_seconds = float(max_audio_seconds)
        self._max_audio_samples = int(round(48000 * self.max_audio_seconds))
        self.max_audio_overage_tolerance_seconds = float(max_audio_overage_tolerance_seconds)
        self._max_audio_overage_samples = int(round(48000 * self.max_audio_overage_tolerance_seconds))
        
        # Initialize CLAP model
        self._load_clap_model()
        
        # Storage for embeddings during epoch
        # We store per-source embeddings so we can compute per-source FAD-CLAP.
        # Each entry is a list of numpy arrays [n_examples, embed_dim] (batched over time).
        self.target_embeddings_by_source: Dict[int, List[np.ndarray]] = {}
        self.output_embeddings_by_source: Dict[int, List[np.ndarray]] = {}
        # Count how many *tracks* (batch items) we processed this epoch (not multiplied by sources)
        self.sample_count = 0
        
    def _load_clap_model(self):
        """Load the CLAP model and processor."""
        try:
            # Force CPU if CUDA issues detected
            if self.force_cpu or self.device == "cpu":
                self.clap_model = ClapModel.from_pretrained(self.model_name, torch_dtype=torch.float32)
                self.clap_processor = ClapProcessor.from_pretrained(self.model_name)
                self.clap_model.eval()
                self.clap_model.to("cpu")
                print(f"CLAP model loaded successfully on CPU (forced)")
            else:
                self.clap_model = ClapModel.from_pretrained(self.model_name)
                self.clap_processor = ClapProcessor.from_pretrained(self.model_name)
                self.clap_model.eval()
                self.clap_model.to(self.device)
                print(f"CLAP model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Warning: Failed to load CLAP model on {self.device}, falling back to CPU: {e}")
            try:
                self.clap_model = ClapModel.from_pretrained(self.model_name, torch_dtype=torch.float32)
                self.clap_processor = ClapProcessor.from_pretrained(self.model_name)
                self.clap_model.eval()
                self.clap_model.to("cpu")
                self.device = "cpu"
                print("CLAP model loaded successfully on CPU (fallback)")
            except Exception as e2:
                raise RuntimeError(f"Failed to load CLAP model on both {self.device} and CPU: {e2}")

    def _get_clap_device(self) -> torch.device:
        """Return the actual device where the CLAP model currently lives."""
        return next(self.clap_model.parameters()).device
    
    def _audio_to_embeddings(self, audio_tensor: torch.Tensor) -> np.ndarray:
        """
        Convert audio tensor to CLAP embeddings.
        
        Args:
            audio_tensor: [batch_size, n_sources, n_channels, n_samples] or [batch_size, n_channels, n_samples]
        
        Returns:
            embeddings: [batch_size * n_sources, embedding_dim] or [batch_size, embedding_dim]
        """
        # Handle different input shapes
        if audio_tensor.dim() == 4:  # [batch_size, n_sources, n_channels, n_samples]
            batch_size, n_sources, n_channels, n_samples = audio_tensor.shape
            # Reshape to [batch_size * n_sources, n_channels, n_samples]
            audio_reshaped = audio_tensor.view(batch_size * n_sources, n_channels, n_samples)
        elif audio_tensor.dim() == 3:  # [batch_size, n_channels, n_samples]
            audio_reshaped = audio_tensor
            batch_size = audio_tensor.shape[0]
            n_sources = 1
        else:
            raise ValueError(f"Unexpected audio tensor shape: {audio_tensor.shape}")
        
        # Convert to numpy and prepare for CLAP
        audio_np = audio_reshaped.detach().cpu().numpy()
        
        # Process each audio sample
        embeddings = []
        for i in range(0, len(audio_np), self.batch_size):
            batch_audio = audio_np[i:i + self.batch_size]
            batch_embeddings = self._process_audio_batch(batch_audio)
            if batch_embeddings is not None:
                embeddings.append(batch_embeddings)
        
        if not embeddings:
            return np.array([])
        
        return np.concatenate(embeddings, axis=0)
    
    def _process_audio_batch(self, audio_batch: np.ndarray) -> Optional[np.ndarray]:
        """Process a batch of audio samples through CLAP."""
        try:
            # Prepare audio for CLAP (expects mono audio)
            audio_list = []
            for audio in audio_batch:
                # audio shape: [n_channels, n_samples]
                if audio.ndim == 2 and audio.shape[0] == 2:  # Stereo [2, n_samples]
                    # Use left channel for CLAP
                    audio_list.append(audio[0])  # [n_samples]
                elif audio.ndim == 1:  # Mono [n_samples]
                    audio_list.append(audio)
                else:
                    # Handle other cases - flatten if needed
                    audio_list.append(audio.flatten())
            
            if not audio_list:
                return None
            
            # Process through CLAP. The feature extractor is called directly because the
            # ClapProcessor keyword changed across transformers versions (audios= vs audio=);
            # with the wrong one the audio is silently dropped.
            inputs = self.clap_processor.feature_extractor(
                audio_list,
                sampling_rate=48000,
                return_tensors="pt",
            )
            # Always place processor outputs on the model's actual device
            # to avoid CPU/GPU mismatch errors.
            clap_device = self._get_clap_device()
            inputs = {key: val.to(clap_device) for key, val in inputs.items()}
            
            with torch.no_grad():
                audio_features = self.clap_model.get_audio_features(**inputs)

            return audio_features.cpu().numpy()
            
        except Exception as e:
            print(f"Warning: Failed to process audio batch: {e}")
            return None
    
    def _calculate_frechet_distance(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> Optional[float]:
        """Calculate Fréchet Distance between two sets of embeddings."""
        if embeddings1.shape[0] < 2 or embeddings2.shape[0] < 2:
            return None
        
        mu1, mu2 = np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0)
        sigma1, sigma2 = np.cov(embeddings1, rowvar=False), np.cov(embeddings2, rowvar=False)
        
        ssdiff = np.sum((mu1 - mu2) ** 2.0)
        
        try:
            covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
        except Exception:
            return None
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fad_score = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
        return fad_score
    
    def update(self, output_dict: Dict[str, torch.Tensor], target_dict: Dict[str, torch.Tensor]):
        """
        Update metric with a batch of predictions and targets.

        Args:
            output_dict: Dictionary containing 'waveform' key with separated sources
                        [batch_size, n_sources, n_channels, n_samples]
            target_dict: Dictionary containing 'waveform' key with target sources
                        [batch_size, n_sources, n_channels, n_samples]
        """
        # Extract waveforms
        output_wav = output_dict['waveform']  # [batch_size, n_sources, n_channels, n_samples]
        target_wav = target_dict['waveform']  # [batch_size, n_sources, n_channels, n_samples]
        source_mask = target_dict.get("source_mask", None)  # [batch_size, n_sources] (optional)

        # Enforce max duration (your policy: only ~10-second audios). A small overage (e.g. from
        # resample rounding through enhancement/separation) is trimmed silently; anything past
        # the tolerance raises, since that signals a real data/config bug.
        if self._max_audio_samples > 0 and output_wav.shape[-1] > self._max_audio_samples:
            overage_samples = output_wav.shape[-1] - self._max_audio_samples
            if overage_samples > self._max_audio_overage_samples:
                got_seconds = float(output_wav.shape[-1]) / 48000.0
                raise ValueError(
                    f"FAD-CLAP only supports audio <= {self.max_audio_seconds:.1f}s "
                    f"(got {got_seconds:.2f}s, exceeding the "
                    f"{self.max_audio_overage_tolerance_seconds:.1f}s trim tolerance)."
                )
            output_wav = output_wav[..., :self._max_audio_samples]
            target_wav = target_wav[..., :self._max_audio_samples]

        try:
            # Compute per-source embeddings (respecting source_mask when available)
            batch_size, n_sources, _, _ = output_wav.shape
            if source_mask is not None:
                source_mask_bool = (source_mask.detach() > 0.5).to(output_wav.device)
            else:
                source_mask_bool = None

            for s in range(n_sources):
                # Per-source cap: each source is limited independently so that stems
                # processed later in the dataset are not starved when shuffle=False.
                if (self.max_samples_per_epoch > 0
                        and len(self.output_embeddings_by_source.get(s, [])) >= self.max_samples_per_epoch):
                    continue

                if source_mask_bool is not None:
                    keep_idx = torch.nonzero(source_mask_bool[:, s], as_tuple=False).squeeze(-1)
                    if keep_idx.numel() == 0:
                        continue
                    out_s = output_wav.index_select(0, keep_idx)[:, s, ...]  # [b_s, C, T]
                    tgt_s = target_wav.index_select(0, keep_idx)[:, s, ...]  # [b_s, C, T]
                else:
                    out_s = output_wav[:, s, ...]  # [B, C, T]
                    tgt_s = target_wav[:, s, ...]  # [B, C, T]

                out_emb = self._audio_to_embeddings(out_s)
                tgt_emb = self._audio_to_embeddings(tgt_s)
                if out_emb.size == 0 or tgt_emb.size == 0:
                    continue

                self.output_embeddings_by_source.setdefault(s, []).append(out_emb)
                self.target_embeddings_by_source.setdefault(s, []).append(tgt_emb)

            self.sample_count += batch_size
        except Exception as e:
            print(f"Warning: Failed to update FAD-CLAP metric: {e}")
            return
    
    def compute(self) -> Dict[str, float]:
        """
        Compute the FAD-CLAP score.
        
        Returns:
            Dictionary containing FAD-CLAP score
        """
        if not self.target_embeddings_by_source or not self.output_embeddings_by_source:
            return {"fad_clap": 0.0, "fad_clap_overall": 0.0}

        fad_per_source: Dict[int, float] = {}
        all_target_parts: List[np.ndarray] = []
        all_output_parts: List[np.ndarray] = []

        all_sources = sorted(set(self.target_embeddings_by_source.keys()) & set(self.output_embeddings_by_source.keys()))
        for s in all_sources:
            tgt_list = self.target_embeddings_by_source.get(s, [])
            out_list = self.output_embeddings_by_source.get(s, [])
            if not tgt_list or not out_list:
                continue
            tgt = np.concatenate(tgt_list, axis=0)
            out = np.concatenate(out_list, axis=0)

            score = self._calculate_frechet_distance(tgt, out)
            fad_per_source[s] = float(score) if score is not None else 0.0

            all_target_parts.append(tgt)
            all_output_parts.append(out)

        if not all_target_parts or not all_output_parts:
            return {"fad_clap": 0.0, "fad_clap_overall": 0.0, "fad_clap_per_source": fad_per_source}

        all_target = np.concatenate(all_target_parts, axis=0)
        all_output = np.concatenate(all_output_parts, axis=0)
        overall = self._calculate_frechet_distance(all_target, all_output)
        overall_value = float(overall) if overall is not None else 0.0

        # Keep "fad_clap" as overall for backward compatibility
        return {
            "fad_clap": overall_value,
            "fad_clap_overall": overall_value,
            "fad_clap_per_source": fad_per_source,
        }

    def compute_for_sources(self, source_indices: List[int]) -> Optional[float]:
        """Compute FAD-CLAP pooling embeddings from only the given source indices.

        Useful for an aggregate score restricted to a subset of stems (e.g. a
        "5-stem" or "overall excluding other" summary) without re-running inference.
        Returns None if no embeddings are available for any of the requested sources.
        """
        target_parts: List[np.ndarray] = []
        output_parts: List[np.ndarray] = []
        for s in source_indices:
            tgt_list = self.target_embeddings_by_source.get(s, [])
            out_list = self.output_embeddings_by_source.get(s, [])
            if not tgt_list or not out_list:
                continue
            target_parts.append(np.concatenate(tgt_list, axis=0))
            output_parts.append(np.concatenate(out_list, axis=0))

        if not target_parts or not output_parts:
            return None

        all_target = np.concatenate(target_parts, axis=0)
        all_output = np.concatenate(output_parts, axis=0)
        score = self._calculate_frechet_distance(all_target, all_output)
        return float(score) if score is not None else None

    def get_concatenated_embeddings_state(self) -> Dict[str, Dict[int, np.ndarray]]:
        """
        Export the current metric state as concatenated embedding matrices.

        This is intended for distributed aggregation where we want to merge embeddings
        across ranks before computing the Fréchet distances.
        """
        target_state: Dict[int, np.ndarray] = {}
        output_state: Dict[int, np.ndarray] = {}

        for s, tgt_list in self.target_embeddings_by_source.items():
            if not tgt_list:
                continue
            target_state[s] = np.concatenate(tgt_list, axis=0).astype(np.float32, copy=False)

        for s, out_list in self.output_embeddings_by_source.items():
            if not out_list:
                continue
            output_state[s] = np.concatenate(out_list, axis=0).astype(np.float32, copy=False)

        return {"target": target_state, "output": output_state}
    
    def reset(self):
        """Reset the metric state."""
        self.target_embeddings_by_source = {}
        self.output_embeddings_by_source = {}
        self.sample_count = 0


# For backward compatibility and easy integration