from __future__ import annotations

import os
from functools import partial

import torch
from torch import nn, tensor
from torch.nn import Module, ModuleList
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .attend import Attend

from beartype.typing import Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, pack, unpack

from hyper_connections import get_init_and_expand_reduce_stream_functions

from src.config_updates import RESOURCES_FOLDER, CKPT_DICT

CKPT_NAME = "model_bs_roformer.ckpt"

# helper functions

def exists(val):
    return val is not None

def default(v, d):
    return v if exists(v) else d

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# attention

class FeedForward(Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        rotary_embed = None,
        flash = True,
        learned_value_residual_mix = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head **-0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash = flash, dropout = dropout)

        self.norm = RMSNorm(dim)
        
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.to_value_residual_mix = nn.Linear(dim, heads) if learned_value_residual_mix else None

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias = False),
            nn.Dropout(dropout)
        )

    def forward(self, x, value_residual = None):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv = 3, h = self.heads)

        orig_v = v

        if exists(self.to_value_residual_mix):
            mix = self.to_value_residual_mix(x)
            mix = rearrange(mix, 'b n h -> b h n 1').sigmoid()

            assert exists(value_residual)
            v = v.lerp(value_residual, mix)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), orig_v

class Transformer(Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        ff_mult = 4,
        norm_output = True,
        rotary_embed = None,
        flash_attn = True,
        add_value_residual = False,
        num_residual_streams = 1,
        num_residual_fracs = 1
    ):
        super().__init__()
        self.layers = ModuleList([])

        init_hyper_conn, *_ = get_init_and_expand_reduce_stream_functions(num_residual_streams, num_fracs = num_residual_fracs)

        for _ in range(depth):
            self.layers.append(ModuleList([
                init_hyper_conn(dim = dim, branch = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_embed = rotary_embed, flash = flash_attn, learned_value_residual_mix = add_value_residual)),
                init_hyper_conn(dim = dim, branch = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout))
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x, value_residual = None):

        first_values = None

        for attn, ff in self.layers:
            x, next_values = attn(x, value_residual = value_residual)

            first_values = default(first_values, next_values)

            x = ff(x)

        return self.norm(x), first_values

# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim = -1) # the first 8 frequencies go to the first band, next 8 to the second, etc.

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim = -2)

def MLP(
    dim_in,
    dim_out,
    dim_hidden = None,
    depth = 1,
    activation = nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)

class MaskEstimator(Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: tuple[int, ...],
        depth,
        mlp_expansion_factor = 2
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden = dim_hidden, depth = depth),
                nn.GLU(dim = -1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim = -2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim = -1)

# main class

DEFAULT_FREQS_PER_BANDS = (
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
  2, 2, 2, 2,
  4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
  12, 12, 12, 12, 12, 12, 12, 12,
  24, 24, 24, 24, 24, 24, 24, 24,
  48, 48, 48, 48, 48, 48, 48, 48,
  128, 129,
)

class BSRoformer(Module):

    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        stereo = False,
        num_stems = 1,
        time_transformer_depth = 2,
        freq_transformer_depth = 2,
        freqs_per_bands: tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,  # in the paper, they divide into ~60 bands, test with 1 for starters
        freq_range: tuple[int, int] | None = (0, 1025), # specifying a frequency range, with (<min freq>, <max freq). `-1` implies 0 and inf
        dim_head = 64,
        heads = 8,
        attn_dropout = 0.,
        ff_dropout = 0.,
        flash_attn = True,
        num_residual_streams = 1, # set to 1. to disable hyper connections
        num_residual_fracs = 1,   # can be used as an alternative to residual streams for memory efficiency while retaining benefits of hyper connections
        stft_n_fft = 2048,
        stft_hop_length = 441, # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
        stft_win_length = 2048,
        stft_normalized = False,
        zero_dc = False, # @firebirdblue23 in https://github.com/lucidrains/BS-RoFormer/issues/47
        stft_window_fn: Callable | None = None,
        mask_estimator_depth = 2,
        mlp_expansion_factor = 4,
        load_pretrained: bool = False,
        model_type=None,
        pretrained_model_type=None,
        n_pretrained_stems=None,
        sr_pretrained=44100,  # Sample rate for pretrained model STFT
        mask_max_value: float = 1.0,  # Maximum value for the mask. Values > 1.0 allow amplification.
                                     # Uses identity-centered residual formulation: real part = 1.0 + alpha * tanh(logits)
                                     # When mask_max_value = 1.0: alpha = 0, mask is identity (backward compatible).
                                     # When mask_max_value > 1.0: real part range is [2.0 - mask_max_value, mask_max_value],
                                     #   centered at 1.0 (identity), allowing both attenuation and amplification.
                                     #   E.g., mask_max_value=1.5 gives range [0.5, 1.5], mask_max_value=2.0 gives [0.0, 2.0]
    ):
        super().__init__()

        self.model_type = model_type
        self.pretrained_model_type = pretrained_model_type
        self.n_pretrained_stems = n_pretrained_stems

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        
        # Mask configuration: allow values > 1.0 for amplification using identity-centered residual
        self.mask_max_value = mask_max_value
        # alpha controls the range: real part will be in [2.0 - mask_max_value, mask_max_value]
        # When mask_max_value = 1.0: alpha = 0, mask is identity (backward compatible)
        # When mask_max_value = 1.5: alpha = 0.5, range is [0.5, 1.5]
        # When mask_max_value = 2.0: alpha = 1.0, range is [0.0, 2.0]
        self.mask_alpha = mask_max_value - 1.0

        if self.mask_max_value < 1.0: 
            raise ValueError("mask_max_value must be greater or equal to 1.0!")

        print(f"Mask scaling: mask_max_value: {self.mask_max_value}, mask_alpha: {self.mask_alpha}, mask_range: [{2.0 - self.mask_max_value}, {self.mask_max_value}]")


        _, self.expand_stream, self.reduce_stream = get_init_and_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        self.layers = ModuleList([])

        transformer_kwargs = dict(
            dim = dim,
            heads = heads,
            dim_head = dim_head,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            flash_attn = flash_attn,
            num_residual_streams = num_residual_streams,
            num_residual_fracs = num_residual_fracs,
            norm_output = False,
        )

        time_rotary_embed = RotaryEmbedding(dim = dim_head)
        freq_rotary_embed = RotaryEmbedding(dim = dim_head)

        for layer_index in range(depth):
            is_first = layer_index == 0

            self.layers.append(nn.ModuleList([
                Transformer(depth = time_transformer_depth, rotary_embed = time_rotary_embed, add_value_residual = False, **transformer_kwargs),  # not is_first, Maybe change that here to always be False
                Transformer(depth = freq_transformer_depth, rotary_embed = freq_rotary_embed, add_value_residual = False, **transformer_kwargs)
            ]))

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs_pretrained = dict(
            n_fft = stft_n_fft,
            hop_length = stft_hop_length,
            win_length = stft_win_length,
            normalized = stft_normalized
        )

        self.stft_window_fn_pretrained = partial(default(stft_window_fn, torch.hann_window), stft_win_length)

        # Calculate frequency bins based on the actual STFT we'll be using
        freq_bins_pretrained = \
        torch.stft(torch.randn(1, 4096), **self.stft_kwargs_pretrained, return_complex=True).shape[1]

        # Use pretrained STFT settings (44.1kHz)
        freqs = freq_bins_pretrained

        # enforcing a frequency range

        freq_range = default(freq_range, (-1, -1))
        min_freq, max_freq = freq_range

        min_freq = 0 if min_freq == -1 else min_freq
        max_freq = freqs if max_freq == -1 else max_freq

        assert min_freq >= 0 and max_freq <= freqs and min_freq < max_freq

        self.min_freq = min_freq
        self.max_freq = max_freq

        self.freq_slice = slice(min_freq, max_freq)  # for slicing out the frequency for training
        self.freq_pad = (min_freq, freqs - max_freq) # for reconstruction during istft

        assert len(freqs_per_bands) > 1
        # assert sum(freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim = dim,
            dim_inputs = freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim = dim,
                dim_inputs = freqs_per_bands_with_complex,
                depth = mask_estimator_depth,
                mlp_expansion_factor = mlp_expansion_factor
            )

            self.mask_estimators.append(mask_estimator)

        # whether to zero out dc
        self.zero_dc = min_freq == 0 and zero_dc
        
        self.sr_pretrained = sr_pretrained
        self.n_fft_pretrained = stft_n_fft

        self.num_residual_streams = num_residual_streams

        self.bins_pretrained = stft_n_fft // 2 + 1  # 1025 bins (44.1kHz)
        print(f"Processing at {sr_pretrained}Hz (pretrained model): {self.bins_pretrained} bins (0-{self.bins_pretrained-1})")

        if load_pretrained:
            pretrained_model_type_ = self.pretrained_model_type if self.pretrained_model_type else self.model_type
            ckpt_name = CKPT_DICT[pretrained_model_type_]
            self.load_pretrained_checkpoint(ckpt_name)


    def load_pretrained_checkpoint(self, ckpt_name):
        ckpt_file = os.path.join(RESOURCES_FOLDER, ckpt_name)

        print("-" * 50)
        print("Loading checkpoint of pretrained model from:", ckpt_file)

        # ----------------------------------- Load checkpoint state-dict -----------------------------------
        ckpt = torch.load(ckpt_file, map_location='cpu')

        # Handle both checkpoint formats: direct state_dict or wrapped in 'state_dict' key
        if isinstance(ckpt, dict) and 'state_dict' in ckpt:
            ckpt_state_dict = ckpt['state_dict']
        else:
            ckpt_state_dict = ckpt

        # ----------------------------------- Map checkpoint keys to current model structure -----------------------------------
        # Checkpoint may not have 'branch' prefix (from hyper-connections)
        mapped_state_dict = {}
        model_state_dict = self.state_dict()

        def try_map_key(ckpt_key):
            """Try to map a checkpoint key to a model key."""
            # Try direct match first
            if ckpt_key in model_state_dict:
                return ckpt_key

            # Pattern: layers.X.Y.layers.Z.W.module.param -> layers.X.Y.layers.Z.W.branch.module.param
            # Need to insert 'branch' before module names (to_qkv, to_gates, net, norm, rotary_embed, etc.)

            # List of module names that come after 'branch' in the hyper-connection wrapper
            module_names = ['to_qkv', 'to_gates', 'to_out', 'to_value_residual_mix', 'net', 'norm', 'rotary_embed']

            # Try adding 'branch' prefix before module names
            for module_name in module_names:
                # Pattern: ...layers.X.Y.module_name... -> ...layers.X.Y.branch.module_name...
                # Use rsplit to only replace the last occurrence (in case module name appears multiple times)
                if ckpt_key.endswith(f'.{module_name}'):
                    # Handle case where module_name is at the end
                    potential_key = ckpt_key.rsplit(f'.{module_name}', 1)[0] + f'.branch.{module_name}'
                    if potential_key in model_state_dict:
                        return potential_key
                elif f'.{module_name}.' in ckpt_key:
                    # Handle case where module_name is in the middle (e.g., to_qkv.weight)
                    # Replace only the first occurrence after layers pattern
                    # Find the position after layers.X.Y pattern
                    potential_key = ckpt_key.replace(f'.{module_name}.', f'.branch.{module_name}.', 1)
                    if potential_key in model_state_dict:
                        return potential_key

            return None

        for ckpt_key, ckpt_value in ckpt_state_dict.items():
            model_key = try_map_key(ckpt_key)
            if model_key is not None:
                # Check for shape mismatches that could indicate configuration differences
                if model_key in model_state_dict:
                    ckpt_shape = ckpt_value.shape
                    model_shape = model_state_dict[model_key].shape
                    if ckpt_shape != model_shape:
                        # print(f"Warning: Shape mismatch for {model_key}: checkpoint {ckpt_shape} vs model {model_shape} - skipping")
                        raise Exception(
                            f"Shape mismatch for {model_key}: checkpoint {ckpt_shape} vs model {model_shape}")
                mapped_state_dict[model_key] = ckpt_value
            # If no mapping found, we'll skip it (may be hyper-connection params not in checkpoint)

        # ----------------------------------- load the state-dict --------------------------------
        # Load with strict=False to allow missing keys (hyper-connection params, etc.)
        missing_keys, unexpected_keys = self.load_state_dict(mapped_state_dict, strict=False)

        # Initialize value_residual_mix layers for NO mixing (they're not in reference checkpoints)
        # sigmoid(negative_bias) → 0 means v.lerp(value_residual, 0) = v (no mixing, uses original v)
        # This matches reference behavior (which doesn't have value residual mixing)
        for name, param in self.named_parameters():
            if 'value_residual_mix' in name:
                if 'weight' in name:
                    nn.init.zeros_(param)  # Zero weights
                elif 'bias' in name:
                    # Initialize bias to large negative value so sigmoid(bias) ≈ 0
                    # This ensures no mixing: v.lerp(value_residual, 0) = v
                    nn.init.constant_(param, -10.0)

        # Print informative messages
        if missing_keys:
            hyper_keys = [k for k in missing_keys if
                          any(x in k for x in ['static_alpha', 'dynamic_alpha', 'static_beta', 'dynamic_beta'])]
            value_residual_keys = [k for k in missing_keys if 'value_residual_mix' in k]
            other_keys = [k for k in missing_keys if
                          k not in hyper_keys and k not in value_residual_keys]

            if hyper_keys:
                print(
                    f"Info: {len(hyper_keys)} hyper-connection parameters not in checkpoint (will use default initialization)")

                print(hyper_keys[:3], "...")
                print("-" * 20)

            if value_residual_keys:
                print(
                    f"Info: {len(value_residual_keys)} value_residual_mix parameters not in checkpoint (initialized to zero for neutral behavior)")

                print(value_residual_keys[:3], "...")
                print("-" * 20)

            if other_keys:
                print(f"Warning: {len(other_keys)} other parameters not in checkpoint: {other_keys[:5]}...")

                print(other_keys[:3], "...")
                print("-" * 20)

        if unexpected_keys:
            print(f"Info: {len(unexpected_keys)} unexpected keys in checkpoint (ignored)")

        # Check for potential configuration mismatches
        if self.num_residual_streams > 1:
            print(
                f"Warning: Using hyper-connections (num_residual_streams={self.num_residual_streams}) but checkpoint likely trained without them.")
            print(f"         Hyper-connection parameters use default initialization and may affect performance.")
            print(f"         Consider setting num_residual_streams=1 for best checkpoint compatibility.")

        print("Pretrained model loaded successfully!")
        print("-" * 50)


    def forward(
        self,
        raw_audio,
        target = None,
        return_loss_breakdown = False
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = rearrange(raw_audio, 'b t -> b 1 t')

        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # Store original length for proper reconstruction
        original_length = raw_audio.shape[-1]
        
        # Pad input to ensure compatibility with STFT parameters
        # The input length should be compatible with hop_length for perfect reconstruction
        # hop_length_48k = self.stft_kwargs_pretrained['hop_length']
        # pad_length = (hop_length_48k - (original_length % hop_length_48k)) % hop_length_48k
        # if pad_length > 0:
        #     raw_audio = F.pad(raw_audio, (0, pad_length))

        # to stft

        stft_window_pretrained = self.stft_window_fn_pretrained(device = device)
        
        # Ensure STFT runs in fp32 precision for numerical stability
        with autocast(enabled=False):
            raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

            # Compute STFT at pretrained sample rate
            stft_repr_complex = torch.stft(
                raw_audio,
                **self.stft_kwargs_pretrained,
                window=stft_window_pretrained,
                return_complex=True
            )  # [..., 1025, time_pretrained] - bins 0-1024, covering 0-22050 Hz
            
            stft_repr = torch.view_as_real(stft_repr_complex)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        
        stft_repr = stft_repr[:, :, self.freq_slice] # slice out frequency range

        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c') # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting  # 1025 --> 2050 (stereo)

        x = rearrange(stft_repr, 'b f t c -> b t (f c)')

        x = self.band_split(x)  # input: bs x 1001 x 4100, output: bs x 1001 x 62 x 384

        # value residuals

        time_v_residual = None
        freq_v_residual = None

        # maybe expand residual streams

        x = self.expand_stream(x)

        # axial / hierarchical attention

        for time_transformer, freq_transformer in self.layers:  # per layer we have a time and freq transformer

            x = rearrange(x, 'b t f d -> b f t d')
            x, ps = pack([x], '* t d')

            x, next_time_v_residual = time_transformer(x, value_residual = time_v_residual)

            time_v_residual = default(time_v_residual, next_time_v_residual)

            x, = unpack(x, ps, '* t d')
            x = rearrange(x, 'b f t d -> b t f d')
            x, ps = pack([x], '* f d')

            x, next_freq_v_residual = freq_transformer(x, value_residual = freq_v_residual)

            freq_v_residual = default(freq_v_residual, next_freq_v_residual)

            x, = unpack(x, ps, '* f d')

        # maybe reduce residual streams

        x = self.reduce_stream(x)

        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        mask = torch.stack([fn(x) for fn in self.mask_estimators], dim = 1)
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c = 2)

        # Apply identity-centered residual complex ratio mask transformation (similar to CRN)
        # This allows mask values > 1.0 for amplification, while maintaining identity-centered initialization
        # Real part: centered at 1.0, range [2.0 - mask_max_value, mask_max_value]
        # Imaginary part: centered at 0.0, range [-mask_alpha, mask_alpha]
        # When mask_max_value = 1.0: alpha = 0, no transformation applied (backward compatible)
        if self.mask_max_value > 1.0:
            mask_real = mask[..., 0]  # Extract real part
            mask_imag = mask[..., 1]  # Extract imaginary part
            
            # Choose transformation method (can be made configurable)
            # Option 1: Sigmoid-based (RECOMMENDED - better gradient flow, less saturation)
            # Maps unbounded logits to [2.0 - mask_max_value, mask_max_value] via sigmoid
            # sigmoid(x) ∈ [0, 1], so: output = (2-mask_max_value) + (2*mask_max_value-2) * sigmoid(x)
            # This gives better gradient flow than tanh and is more natural for bounded outputs
            mask_real_transformed = (2.0 - self.mask_max_value) + (2.0 * self.mask_max_value - 2.0) * torch.sigmoid(mask_real)  # sigmoid term has mean 0.5
            # Imaginary part: maps to [-mask_alpha, mask_alpha] centered at 0.0
            # mask_alpha * (2 * sigmoid(x) - 1) maps sigmoid(x) ∈ [0,1] to [-mask_alpha, mask_alpha]
            mask_imag_transformed = self.mask_alpha * (2.0 * torch.sigmoid(mask_imag) - 1.0)
            
            # Option 2: Temperature-scaled tanh (alternative - less saturation than original)
            # mask_real_transformed = 1.0 + self.mask_alpha * torch.tanh(mask_real / 2.0)  # Divide by 2 to reduce saturation
            # mask_imag_transformed = self.mask_alpha * torch.tanh(mask_imag / 2.0)
            
            # Option 3: Original tanh (can saturate, but simple)
            # mask_real_transformed = 1.0 + self.mask_alpha * torch.tanh(mask_real)
            # mask_imag_transformed = self.mask_alpha * torch.tanh(mask_imag)
            
            # Reconstruct mask with transformed values
            mask = torch.stack([mask_real_transformed, mask_imag_transformed], dim = -1)

        # modulate frequency representation

        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)

        stft_repr = stft_repr * mask

        # istft

        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s = self.audio_channels)

        if self.zero_dc:
            # whether to dc filter
            stft_repr = stft_repr.index_fill(1, tensor(0, device = device), 0.)

        # Ensure iSTFT runs in fp32 precision for numerical stability
        # Use length parameter for perfect reconstruction (especially important for non-power-of-2 n_fft)
        with autocast(enabled=False):
            # stft_repr is already complex at this point (converted at line 921)
            # Check if it's already complex or needs conversion
            if torch.is_complex(stft_repr):
                stft_repr_complex = stft_repr
            else:
                # Convert from real representation to complex
                stft_repr_complex = torch.view_as_complex(stft_repr)

            recon_audio = torch.istft(
                stft_repr_complex,  # [..., 1025, time_pretrained]
                **self.stft_kwargs_pretrained,
                window=stft_window_pretrained,
                length=None,
                return_complex=False
            )

        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        # Trim back to original length
        recon_audio = recon_audio[..., :original_length]

        if num_stems == 1:
            recon_audio = rearrange(recon_audio, 'b 1 s t -> b s t')

        return recon_audio