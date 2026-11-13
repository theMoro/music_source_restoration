"""
Wrapper for BS-Roformer to make it compatible with the training pipeline.
"""

import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

from .bs_roformer import BSRoformer


class RoformerWrapper(nn.Module):
    """
    Wrapper for BS-Roformer (music source separation) to make it compatible with the training pipeline.
    """
    
    def __init__(
        self,
        dim: int = 384,
        depth: int = 12,
        heads: int = 8,
        dim_head: int = 64,
        stereo: bool = True,
        num_stems: int = 4,
        mlp_expansion_factor: int = 4,
        load_pretrained: bool = False,
        model_type: Optional[str] = None,
        pretrained_model_type: Optional[str] = None,
        n_pretrained_stems: int = None,

        # default configuration
        time_transformer_depth: int = 1,
        freq_transformer_depth: int = 1,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        flash_attn: bool = True,
        num_residual_streams: int = 1,
        num_residual_fracs: int = 1,
        stft_n_fft: int = 2048,
        stft_hop_length: int = 512,  # 512 for the BS-RoFormer-SW checkpoint
        stft_win_length: int = 2048,
        stft_normalized: bool = False,
        zero_dc: bool = False,
        mask_estimator_depth: int = 2,

        # fine-tuning configuration
        finetune_attention_layers: bool = True,
        finetune_feedforward_layers: bool = False,
        finetune_bandsplit_module: bool = False,
        finetune_pretrained_mask_estimators: bool = False,
        finetune_new_mask_estimators: bool = False,
        inference_chunk_size: Optional[int] = None,
        inference_chunk_overlap_ratio: float = 0.5,
        **kwargs
    ):
        """
        Initialize Roformer wrapper.

        Args:
            dim: Dimension of the transformer model. Default: 384.
            depth: Number of transformer layers. Default: 12.
            heads: Number of attention heads. Default: 8.
            dim_head: Dimension per attention head. Default: 64.
            stereo: Whether to process stereo audio (2 channels) or mono (1 channel). Default: True.
            num_stems: Number of output stems/sources to separate. Default: 4.
            mlp_expansion_factor: Expansion factor for MLP layers in transformers. Default: 4.
            load_pretrained: Whether to load pretrained weights. Default: False.
            model_type: Key of the pretrained checkpoint in CKPT_DICT (src/config_updates.py), e.g. "8s_separation". Default: None.
            pretrained_model_type: Optional override of model_type for selecting the pretrained checkpoint.
            n_pretrained_stems: Number of pretrained stems (6 for BS-RoFormer-SW). Used to determine which mask estimators are pretrained. Default: None.
            time_transformer_depth: Depth of time transformer layers. Default: 1.
            freq_transformer_depth: Depth of frequency transformer layers. Default: 1.
            attn_dropout: Dropout rate for attention layers. Default: 0.1.
            ff_dropout: Dropout rate for feedforward layers. Default: 0.1.
            flash_attn: Whether to use flash attention for efficiency. Default: True.
            num_residual_streams: Number of residual streams in the model. Default: 1.
            num_residual_fracs: Number of residual fractions. Default: 1.
            stft_n_fft: FFT window size for STFT. Default: 2048.
            stft_hop_length: Hop length for STFT in samples. Default: 441.
            stft_win_length: Window length for STFT in samples. Default: 2048.
            stft_normalized: Whether to use normalized STFT. Default: False.
            zero_dc: Whether to zero the DC component. Default: False.
            mask_estimator_depth: Depth of mask estimator layers. Default: 2.
            finetune_attention_layers: Whether to fine-tune attention layers. Default: True.
            finetune_feedforward_layers: Whether to fine-tune feedforward layers. Default: False.
            finetune_bandsplit_module: Whether to fine-tune the bandsplit module. Default: False.
            finetune_pretrained_mask_estimators: Whether to fine-tune pretrained mask estimators. Default: False.
            finetune_new_mask_estimators: Whether to fine-tune new mask estimators. Default: False.
            inference_chunk_size: If set, inputs longer than this (in samples) are processed in chunks
                and overlap-added (e.g. use 48000 for 1s-trained models when inferring on longer audio).
                None disables chunking (process full length; use when model supports arbitrary length).
            inference_chunk_overlap_ratio: Overlap between chunks when chunking (0.0 = no overlap, 0.5 = 50%),
                cross-faded with a Hann window. Only used when inference_chunk_size is set.
            **kwargs: Additional keyword arguments passed to the underlying model.
        """
        super().__init__()

        # Initialize Roformer model
        roformer_kwargs = {
            'dim': dim,
            'depth': depth,
            'stereo': stereo,
            'num_stems': num_stems,
            'time_transformer_depth': time_transformer_depth,
            'freq_transformer_depth': freq_transformer_depth,
            'dim_head': dim_head,
            'heads': heads,
            'attn_dropout': attn_dropout,
            'ff_dropout': ff_dropout,
            'flash_attn': flash_attn,
            'num_residual_streams': num_residual_streams,
            'num_residual_fracs': num_residual_fracs,
            'stft_n_fft': stft_n_fft,
            'stft_hop_length': stft_hop_length,
            'stft_win_length': stft_win_length,
            'stft_normalized': stft_normalized,
            'zero_dc': zero_dc,
            'mask_estimator_depth': mask_estimator_depth,
            'mlp_expansion_factor': mlp_expansion_factor,
            'load_pretrained': load_pretrained,
            'model_type': model_type,
            'pretrained_model_type': pretrained_model_type,
            'n_pretrained_stems': n_pretrained_stems
        }

        roformer_kwargs['mask_max_value'] = kwargs.get('mask_max_value', 1.0)
        self.roformer = BSRoformer(**roformer_kwargs)

        # Store fine-tuning configuration
        self.finetune_attention_layers = finetune_attention_layers
        self.finetune_feedforward_layers = finetune_feedforward_layers
        self.finetune_bandsplit_module = finetune_bandsplit_module
        self.finetune_pretrained_mask_estimators = finetune_pretrained_mask_estimators
        self.finetune_new_mask_estimators = finetune_new_mask_estimators

        self.n_pretrained_stems = n_pretrained_stems if n_pretrained_stems is not None else 0

        if self.n_pretrained_stems == 6:
            # Pretrained mask estimators: 0=bass, 1=drums, 2=other, 3=vocals, 4=guitars, 5=keyboards
            # New mask estimators: 6=synthesizers, 7=percussions, 8=orchestral
            self.pretrained_mask_indices = {0, 1, 2, 3, 4, 5}
            self.new_mask_indices = {6, 7, 8}
        elif self.n_pretrained_stems == 0:
            # No pretrained mask estimators - all are new
            self.pretrained_mask_indices = set()
            self.new_mask_indices = set(range(9))  # All indices from 0 to 8 are new
        else:
            raise ValueError(f"Unsupported n_pretrained_stems value: {self.n_pretrained_stems}. Expected 0 or 6.")

        # Configure parameter trainability based on fine-tuning flags
        self._configure_parameter_trainability()

        # Store configuration
        self.model_type = model_type
        self.num_stems = num_stems
        self.stereo = stereo
        self.input_channels = 2 if stereo else 1
        self.output_channels = 2 if stereo else 1
        if inference_chunk_size is None and "max_length_samples" in kwargs:
            inference_chunk_size = kwargs["max_length_samples"]
        self.inference_chunk_size = inference_chunk_size
        self.inference_chunk_overlap_ratio = float(inference_chunk_overlap_ratio)

        if self.inference_chunk_size is not None and not self.training:
            if self.inference_chunk_overlap_ratio > 0:
                print(
                    f"RoformerWrapper: chunked inference enabled (inference_chunk_size={self.inference_chunk_size}, "
                    f"overlap={self.inference_chunk_overlap_ratio * 100:.1f}%)"
                )
    
    def _configure_parameter_trainability(self):
        """
        Configure which parameters are trainable based on fine-tuning flags.
        
        This function sets requires_grad for different module types:
        - Attention layers: Controlled by finetune_attention_layers flag (to_qkv, to_out, etc.)
        - FeedForward layers: Controlled by finetune_feedforward_layers flag (default: False)
        - BandSplit module: Controlled by finetune_bandsplit_module flag
        - MaskEstimator modules: Controlled by finetune_pretrained_mask_estimators and finetune_new_mask_estimators flags
        - All other parameters: Frozen by default (for memory efficiency)
        """
        frozen_count = 0
        attention_count = 0
        feedforward_count = 0
        bandsplit_count = 0
        mask_estimator_pretrained_count = 0
        mask_estimator_new_count = 0
        
        attention_mode = "full weights" if self.finetune_attention_layers else "frozen"
        
        print("Parameter trainability configuration:")
        print(f"  - Attention layers: {attention_mode}")
        print(f"  - FeedForward layers: {'trainable' if self.finetune_feedforward_layers else 'frozen'}")
        print(f"  - BandSplit: {'trainable' if self.finetune_bandsplit_module else 'frozen'}")
        print(f"  - MaskEstimator (pretrained {', '.join([str(idx) for idx in self.pretrained_mask_indices])}): {'trainable' if self.finetune_pretrained_mask_estimators else 'frozen'}")
        print(f"  - MaskEstimator (new {', '.join([str(idx) for idx in self.new_mask_indices])}): {'trainable' if self.finetune_new_mask_estimators else 'frozen'}")
        
        for name, param in self.named_parameters():
            if 'band_split' in name.lower():
                param.requires_grad = self.finetune_bandsplit_module
                if self.finetune_bandsplit_module:
                    bandsplit_count += 1
                else:
                    frozen_count += 1
            elif 'mask_estimator' in name.lower():
                # Extract mask estimator index
                match = re.search(r'mask_estimators\.(\d+)', name)
                if match:
                    estimator_idx = int(match.group(1))
                    if estimator_idx in self.pretrained_mask_indices:
                        param.requires_grad = self.finetune_pretrained_mask_estimators
                        if self.finetune_pretrained_mask_estimators:
                            mask_estimator_pretrained_count += 1
                        else:
                            frozen_count += 1
                    elif estimator_idx in self.new_mask_indices:
                        param.requires_grad = self.finetune_new_mask_estimators
                        if self.finetune_new_mask_estimators:
                            mask_estimator_new_count += 1
                        else:
                            frozen_count += 1
                    else:
                        # Unknown mask estimator index - freeze it to be safe
                        param.requires_grad = False
                        frozen_count += 1
                else:
                    # Could not extract index - freeze to be safe
                    param.requires_grad = False
                    frozen_count += 1
            else:
                # Check if this is an attention layer parameter
                # Attention layers contain: to_qkv, to_out, to_gates, to_value_residual_mix, norm
                # Structure: roformer.layers.X.Y.branch.to_qkv.weight (where X is layer, Y is 0 for attention)
                # or roformer.layers.X.Y.branch.norm.gamma
                is_attention_layer = (
                    'layers' in name and (
                        'to_qkv' in name or 
                        'to_out' in name or 
                        'to_gates' in name or 
                        'to_value_residual_mix' in name or
                        ('norm' in name and 'branch' in name and 'layers' in name and 'final_norm' not in name)
                    )
                )
                
                # Check if this is a feed-forward layer (should be frozen when not explicitly enabled)
                # Feed-forward layers are in the same structure but Y=1 (second element in ModuleList)
                # They contain: net.0 (norm), net.1 (linear), net.3 (linear), etc.
                is_feedforward = (
                    'layers' in name and 
                    'net' in name and 
                    'branch' in name and
                    'to_qkv' not in name and
                    'to_out' not in name and
                    'to_gates' not in name
                )
                
                if is_attention_layer:
                    param.requires_grad = self.finetune_attention_layers
                    if self.finetune_attention_layers:
                        attention_count += 1
                    else:
                        frozen_count += 1
                elif is_feedforward:
                    # Feed-forward layers are controlled by finetune_feedforward_layers flag
                    param.requires_grad = self.finetune_feedforward_layers
                    if self.finetune_feedforward_layers:
                        feedforward_count += 1
                    else:
                        frozen_count += 1
                else:
                    # Freeze all other parameters (final_norm, etc.)
                    param.requires_grad = False
                    frozen_count += 1
        
        # Print summary
        trainable_desc = []
        if attention_count > 0:
            trainable_desc.append(f"{attention_count} attention layer weights")
        if feedforward_count > 0:
            trainable_desc.append(f"{feedforward_count} FeedForward layer weights")
        if bandsplit_count > 0:
            trainable_desc.append(f"{bandsplit_count} BandSplit")
        if mask_estimator_pretrained_count > 0:
            trainable_desc.append(f"{mask_estimator_pretrained_count} MaskEstimator (pretrained)")
        if mask_estimator_new_count > 0:
            trainable_desc.append(f"{mask_estimator_new_count} MaskEstimator (new)")
        
        if trainable_desc:
            print(f"Frozen {frozen_count} parameters, {' + '.join(trainable_desc)} parameters remain trainable")
        else:
            print(f"All {frozen_count} parameters are frozen")
        
        # Verify that optimizer will only receive intended trainable parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        print(f"Total trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
    def forward_separation(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Separation Roformer wrapper.
        
        Args:
            input_dict: Dictionary containing:
                       - 'mixture': input audio [batch_size, n_channels, n_samples]
        
        Returns:
            Dictionary containing 'waveform' key with separated sources
            [batch_size, n_sources, n_channels, n_samples]
        """
        # Extract mixture from input dictionary
        mixture = input_dict['mixture']  # [B, C, T]
        
        # Ensure correct input format for BS-Roformer
        if mixture.dim() == 2:
            # Add channel dimension if mono
            mixture = mixture.unsqueeze(1)  # [B, 1, T]
        
        # Roformer expects [B, C, T] format
        # Forward through Roformer
        separated_audio = self.roformer(mixture)  # [B, S, C, T] or [B, C, T] if num_stems=1

        # Ensure output has correct shape
        if self.num_stems == 1:
            # Single source case - add source dimension
            separated_audio = separated_audio.unsqueeze(1)  # [B, 1, C, T]
        
        # Create output dictionary
        output_dict = {
            'waveform': separated_audio,  # multiple sources --> waveforms
            'source_mask': torch.ones(separated_audio.shape[0], self.num_stems, 
                                    device=separated_audio.device, dtype=separated_audio.dtype)
        }
        
        return output_dict

    def _process_single_in_chunks(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Process a single-sample input (batch size 1) in chunks and overlap-add.
        Used when input length exceeds inference_chunk_size.
        """
        input_key = "mixture"
        forward_fn = self.forward_separation
        output_key = "waveform"

        waveform = input_dict[input_key]  # [1, C, T]
        _, n_channels, total_length = waveform.shape
        device = waveform.device
        dtype = waveform.dtype
        chunk_length = self.inference_chunk_size
        overlap_samples = int(chunk_length * self.inference_chunk_overlap_ratio)
        hop_length = chunk_length - overlap_samples

        if overlap_samples == 0:
            n_chunks = (total_length + chunk_length - 1) // chunk_length
        else:
            n_chunks = max(1, (total_length - overlap_samples + hop_length - 1) // hop_length)

        # Output shape: [1, S, C, T]
        output_full = torch.zeros((1, self.num_stems, n_channels, total_length), dtype=dtype, device=device)

        if overlap_samples > 0:
            overlap_window = torch.hann_window(2 * overlap_samples, device=device)
            fade_in = overlap_window[:overlap_samples]
            fade_out = overlap_window[overlap_samples:]
            chunk_window = torch.ones(chunk_length, device=device)
            chunk_window[:overlap_samples] = fade_in
            chunk_window[-overlap_samples:] = fade_out
            chunk_window = chunk_window.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, chunk_length]
            weight_accumulator = torch.zeros(total_length, device=device)
        else:
            chunk_window = None

        for i in range(n_chunks):
            if overlap_samples == 0:
                start_idx = i * chunk_length
                end_idx = min(start_idx + chunk_length, total_length)
            else:
                start_idx = i * hop_length
                end_idx = min(start_idx + chunk_length, total_length)

            chunk = waveform[:, :, start_idx:end_idx]
            if chunk.shape[-1] < chunk_length:
                chunk = F.pad(chunk, (0, chunk_length - chunk.shape[-1]), mode='constant', value=0.0)
            chunk_dict = {input_key: chunk}
            with torch.no_grad():
                out_dict = forward_fn(chunk_dict)
            out_chunk = out_dict[output_key]
            actual_length = end_idx - start_idx
            # Model may return different length (e.g. STFT/ISTFT rounding); use min so window and write length match
            actual_out_length = min(out_chunk.shape[-1], actual_length)
            out_chunk = out_chunk[..., :actual_out_length]

            if chunk_window is not None:
                w = chunk_window[..., :actual_out_length]
                out_chunk = out_chunk * w
                if out_chunk.dim() == 4:
                    weight_accumulator[start_idx:start_idx + actual_out_length] += w[0, 0, 0, :]
                else:
                    weight_accumulator[start_idx:start_idx + actual_out_length] += w[0, 0, :]
            if overlap_samples > 0:
                if out_chunk.dim() == 4:
                    output_full[:, :, :, start_idx:start_idx + actual_out_length] += out_chunk
                else:
                    output_full[:, :, start_idx:start_idx + actual_out_length] += out_chunk
            else:
                if out_chunk.dim() == 4:
                    output_full[:, :, :, start_idx:start_idx + actual_out_length] = out_chunk
                else:
                    output_full[:, :, start_idx:start_idx + actual_out_length] = out_chunk

        if overlap_samples > 0 and weight_accumulator.max() > 0:
            weight_accumulator = torch.clamp(weight_accumulator, min=1e-8)
            if output_full.dim() == 4:
                output_full = output_full / weight_accumulator.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            else:
                output_full = output_full / weight_accumulator.unsqueeze(0).unsqueeze(0)

        result = {output_key: output_full}
        result["source_mask"] = torch.ones(output_full.shape[0], self.num_stems, device=device, dtype=dtype)
        return result

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Roformer wrapper. When inference_chunk_size is set and input
        exceeds it, processes in chunks with overlap-add.
        """
        input_key = "mixture"

        if input_key not in input_dict:
            raise ValueError(f"input_dict must contain key '{input_key}'")
        x = input_dict[input_key]
        if x.dim() == 2:
            x = x.unsqueeze(0)
        batch_size, _, total_length = x.shape

        use_chunking = self.inference_chunk_size is not None and total_length > self.inference_chunk_size and not self.training
        if not use_chunking:
            return self.forward_separation(input_dict)

        # Chunked path: process each batch item
        out_key = "waveform"
        outputs = []
        for b in range(batch_size):
            single_dict = {input_key: x[b : b + 1]}
            out_dict = self._process_single_in_chunks(single_dict)
            outputs.append(out_dict[out_key])
        out_tensor = torch.cat(outputs, dim=0)
        result = {out_key: out_tensor}
        result["source_mask"] = torch.ones(batch_size, self.num_stems, device=out_tensor.device, dtype=out_tensor.dtype)
        return result
    
    def parameter_groups(self, lr: float, lr_pretrained_masks: float, lr_new_masks: float, lr_decay_factor: float, weight_decay: float):
        """
        Parameter groups for optimizer.
        
        Args:
            lr: Learning rate for roformer parameters
            lr_pretrained_masks: Learning rate for pretrained mask estimator parameters
            lr_new_masks: Learning rate for new mask estimator parameters
            lr_decay_factor: Layer-wise learning rate decay factor for the transformer layers, in (0, 1].
                            If < 1.0, earlier layers (closer to the input) get lower learning rates:
                            layer i gets lr * lr_decay_factor ** ((max_layer_idx - i) / max_layer_idx),
                            i.e. layer 0 gets lr * lr_decay_factor and the last layer gets lr.
                            If 1.0, all layers use the same learning rate.
            weight_decay: Weight decay to apply to all parameter groups
        
        Returns:
            List of parameter groups
        """
        # Normalize weight_decay (handle None values)
        if weight_decay is None:
            weight_decay = 0.0

        # Values > 1.0 would give earlier layers a higher LR (reversed layer-wise decay)
        if not 0.0 < lr_decay_factor <= 1.0:
            raise ValueError(f"lr_decay_factor must be in (0, 1], got {lr_decay_factor}")

        # Group parameters by layer depth for lr_decay_factor
        roformer_params_by_layer = {}  # layer_idx -> list of params
        roformer_params_other = []  # band_split, final_norm, etc.
        mask_pretrained_params = []
        mask_new_params = []

        for name, p in self.named_parameters():
            # Only include parameters that require gradients (are trainable)
            if not p.requires_grad:
                continue
                
            if name.startswith('roformer.layers'):
                # Extract layer index from name (e.g., "roformer.layers.5.branch.to_qkv.weight" -> layer 5)
                match = re.search(r'roformer\.layers\.(\d+)', name)
                if match:
                    layer_idx = int(match.group(1))
                    if layer_idx not in roformer_params_by_layer:
                        roformer_params_by_layer[layer_idx] = []
                    roformer_params_by_layer[layer_idx].append(p)
                else:
                    # Fallback: add to other params if we can't extract layer index
                    roformer_params_other.append(p)
            elif name.startswith('roformer.band_split') or name.startswith('roformer.final_norm'):
                roformer_params_other.append(p)
            elif name.startswith('roformer.mask_estimators'):
                # Extract mask estimator index using regex to handle multi-digit indices
                match = re.search(r'roformer.mask_estimators\.(\d+)', name)
                if match:
                    estimator_idx = int(match.group(1))
                    if estimator_idx in self.pretrained_mask_indices:
                        mask_pretrained_params.append(p)
                    else:
                        mask_new_params.append(p)
                else:
                    raise ValueError(f"Could not extract mask_estimator index from parameter name: {name}")
            else:
                raise ValueError(f"Unexpected key in model: {name}")

        # Build parameter groups
        param_groups = []
        
        # Add roformer layers with decay factor applied to deeper layers
        if roformer_params_by_layer:
            max_layer_idx = max(roformer_params_by_layer.keys())
            for layer_idx in sorted(roformer_params_by_layer.keys()):
                layer_params = roformer_params_by_layer[layer_idx]
                if layer_params:  # Only add if there are parameters
                    # Apply decay factor: earlier layers (lower indices) get lower learning rates
                    # This preserves pretrained features in early layers while allowing later layers to adapt
                    # With lr_decay_factor < 1.0, earlier layers get lr * (lr_decay_factor ** depth_factor)
                    # where depth_factor decreases with layer depth (earlier = higher depth_factor)
                    if lr_decay_factor != 1.0 and max_layer_idx > 0:
                        # Normalize layer depth to [0, 1] range, then invert so earlier layers have higher factor
                        # Layer 0: depth_factor = 1.0 (full decay), Layer max: depth_factor = 0.0 (no decay)
                        depth_factor = (max_layer_idx - layer_idx) / max_layer_idx
                        # Apply decay: earlier layers get lr * (lr_decay_factor ** depth_factor)
                        layer_lr = lr * (lr_decay_factor ** depth_factor)
                    else:
                        layer_lr = lr
                    
                    param_groups.append({
                        'params': layer_params,
                        'lr': layer_lr,
                        'weight_decay': weight_decay,
                        'name': f'roformer_layer_{layer_idx}'
                    })
        
        # Add other roformer params (band_split, final_norm, etc.) - use base lr
        if roformer_params_other:
            param_groups.append({
                'params': roformer_params_other,
                'lr': lr,
                'weight_decay': weight_decay,
                'name': 'roformer_other'
            })
        
        # Add mask estimator groups with weight_decay
        if mask_pretrained_params:
            param_groups.append({
                'params': mask_pretrained_params,
                'lr': lr_pretrained_masks,
                'weight_decay': weight_decay,
                'name': 'mask_pretrained'
            })
        
        if mask_new_params:
            param_groups.append({
                'params': mask_new_params,
                'lr': lr_new_masks,
                'weight_decay': weight_decay,
                'name': 'mask_new'
            })

        # Print summary
        lr_info = []
        for group in param_groups:
            group_name = group.get('name', 'unknown')
            group_lr = group.get('lr')
            group_wd = group.get('weight_decay')
            # Handle None values and convert to float
            if group_lr is None:
                group_lr = 0.0
            else:
                group_lr = float(group_lr)
            if group_wd is None:
                group_wd = 0.0
            else:
                group_wd = float(group_wd)
            lr_info.append(f"{group_name}: lr={group_lr:.2e}, wd={group_wd:.2e}")
        
        print(f"Parameter groups: {', '.join(lr_info)}")
        if lr_decay_factor != 1.0:
            print(f"Applied lr_decay_factor={lr_decay_factor} to roformer layers (earlier layers have lower LR)")
        
        return param_groups

