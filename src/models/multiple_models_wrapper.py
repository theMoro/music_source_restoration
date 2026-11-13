import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from typing import Dict, List, Optional
from src.utils import initialize_config


class MultipleModelsWrapper(nn.Module):
    """
    Chains the models of the full music source restoration system:

        mixture -> [apollo: mixture restoration] -> [separate: BS-RoFormer separation]
                -> [apollo_stemwise: per-stem restoration] -> stems

    Only the separation model can be trained through this wrapper (separation training);
    both Apollo models are inference-only here (they are fine-tuned with their own
    lightning modules, see configs/mixture_restoration and configs/stem_restoration).
    """

    def __init__(
        self,
        separate: bool,
        train_separation_model: bool,
        separation_model: dict,
        resample_to_44100_for_model: bool,
        apollo: bool = False,
        apollo_stemwise: bool = False,
        apollo_model: Optional[dict] = None,
        apollo_stemwise_model: Optional[dict] = None,
    ):
        super().__init__()

        self.separate = separate
        self.apollo = apollo
        self.apollo_stemwise = apollo_stemwise

        self.train_separation_model = train_separation_model

        if self.separate:
            self.separation_model_config = separation_model
            self.separation_model = initialize_config(self.separation_model_config)

        if (self.apollo or self.apollo_stemwise) and not resample_to_44100_for_model:
            raise ValueError(
                "apollo=True/apollo_stemwise=True requires resample_to_44100_for_model=True — "
                "Apollo only operates correctly on 44.1kHz-native audio (see ApolloWrapper)."
            )

        if self.apollo:
            # JusperLee/Apollo (https://github.com/JusperLee/Apollo) mixture restoration (inference-only).
            self.apollo_model_config = apollo_model
            self.apollo_model = initialize_config(self.apollo_model_config)
            print("Apollo restoration model created!")

        if self.apollo_stemwise:
            # Per-stem Apollo restoration (src.models.apollo.ApolloStemwiseRestorationWrapper):
            # one fine-tuned checkpoint per stem, falling back to a shared checkpoint for any stem
            # without its own (see that wrapper's docstring). Inference-only.
            self.apollo_stemwise_model_config = apollo_stemwise_model
            self.apollo_stemwise_model = initialize_config(self.apollo_stemwise_model_config)
            print("Apollo stem-wise restoration model created!")

        self.resample_to_44100_for_model = resample_to_44100_for_model

        # Initialize resamplers if resampling is enabled
        if self.resample_to_44100_for_model:
            # Down-sample: 48kHz -> 44.1kHz
            self.downsample_resampler = torchaudio.transforms.Resample(
                orig_freq=48000,
                new_freq=44100,
                resampling_method='kaiser_window',
                lowpass_filter_width=64,
                rolloff=0.99,
                beta=14.0
            )
            # Up-sample: 44.1kHz -> 48kHz
            self.upsample_resampler = torchaudio.transforms.Resample(
                orig_freq=44100,
                new_freq=48000,
                resampling_method='kaiser_window',
                lowpass_filter_width=64,
                rolloff=0.99,
                beta=14.0
            )
            print("Resampling enabled in MultipleModelsWrapper: mixture will be down-sampled to 44.1kHz for model processing, then up-sampled back to 48kHz")

        # Set non-trainable models to eval mode and disable gradients
        self._configure_model_training_states()

    def _configure_model_training_states(self):
        """Configure training states for all sub-models based on train flags."""

        # Separation model
        if self.separate and hasattr(self, 'separation_model'):
            if not self.train_separation_model:
                self.separation_model.eval()
                # Disable gradients
                for param in self.separation_model.parameters():
                    param.requires_grad = False

        # Apollo restoration models (always inference-only)
        if self.apollo and hasattr(self, 'apollo_model'):
            self.apollo_model.eval()
            for param in self.apollo_model.parameters():
                param.requires_grad = False

        if self.apollo_stemwise and hasattr(self, 'apollo_stemwise_model'):
            self.apollo_stemwise_model.eval()
            for param in self.apollo_stemwise_model.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        """
        Set the module in training mode.

        Only models that are being trained will be set to train mode.
        Non-trainable models remain in eval mode.
        """
        super().train(mode)
        # Reconfigure training states based on train flags
        # This ensures non-trainable models stay in eval mode even when wrapper is in train mode
        self._configure_model_training_states()
        return self

    def eval(self):
        """
        Set the module in evaluation mode.

        All sub-models are set to eval mode.
        """
        super().eval()
        if self.separate and hasattr(self, 'separation_model'):
            self.separation_model.eval()
        if self.apollo and hasattr(self, 'apollo_model'):
            self.apollo_model.eval()
        if self.apollo_stemwise and hasattr(self, 'apollo_stemwise_model'):
            self.apollo_stemwise_model.eval()
        return self

    def parameter_groups(
        self,
        lr: float,
        lr_pretrained_masks: float,
        lr_new_masks: float,
        lr_decay_factor: float,
        weight_decay: float,
        lr_separation: Optional[float] = None,
    ) -> List[Dict]:
        """
        Parameter groups of the (trainable) separation model.

        Args:
            lr: Base learning rate
            lr_pretrained_masks: Learning rate for pretrained mask estimators
            lr_new_masks: Learning rate for new mask estimators
            lr_decay_factor: Layer-wise learning rate decay factor in (0, 1]; values < 1.0 give
                earlier transformer layers a lower learning rate (see RoformerWrapper.parameter_groups)
            weight_decay: Weight decay to apply to all parameter groups
            lr_separation: Optional learning rate override for the separation model

        Returns:
            List of parameter groups
        """
        param_groups = []

        if self.separate and self.train_separation_model and hasattr(self, 'separation_model'):
            model_lr = lr_separation if lr_separation is not None else lr
            sub_groups = self.separation_model.parameter_groups(
                lr=model_lr,
                lr_pretrained_masks=lr_pretrained_masks,
                lr_new_masks=lr_new_masks,
                lr_decay_factor=lr_decay_factor,
                weight_decay=weight_decay,
            )
            # Prefix group names
            for group in sub_groups:
                prefixed_group = {
                    'params': group['params'],
                    'lr': group.get('lr', model_lr),
                    'weight_decay': group.get('weight_decay', weight_decay),
                    'name': f"separation_model.{group.get('name', 'params')}",
                }
                param_groups.append(prefixed_group)

        return param_groups

    def forward(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict = input_dict

        # Down-sample input if resampling is enabled
        original_length = None
        if self.resample_to_44100_for_model:
            # Extract mixture from output_dict
            if 'mixture' in output_dict:
                original_dtype = output_dict['mixture'].dtype
                original_length = output_dict['mixture'].shape[-1]
                # Ensure mixture is on same device as resampler (avoids CUDA illegal memory access
                # when resampler kernel and input were on different devices, e.g. after pretrained load)
                resampler_device = next(self.downsample_resampler.buffers()).device
                mixture = output_dict['mixture'].float().to(resampler_device, non_blocking=False)
                output_dict['mixture'] = self.downsample_resampler(mixture).to(dtype=original_dtype)

        if self.apollo:
            # Mixture restoration at Apollo's native 44.1kHz, before separation.
            with torch.no_grad():
                output_dict = self.apollo_model(output_dict)

        if self.separate:
            # Run separation model
            if self.train_separation_model:
                # Model is being trained - allow gradients
                output_dict = self.separation_model(output_dict)
            else:
                # Model is not being trained - disable gradients
                with torch.no_grad():
                    output_dict = self.separation_model(output_dict)

        if self.apollo_stemwise:
            # Per-stem restoration of every separated stem, still at Apollo's native 44.1kHz
            # (before the up-sample below).
            with torch.no_grad():
                output_dict = self.apollo_stemwise_model(output_dict)

        # Up-sample output if resampling is enabled
        if self.resample_to_44100_for_model and original_length is not None:
            original_dtype = output_dict['waveform'].dtype
            waveform = output_dict['waveform']
            # Use stored original_length for proper reconstruction
            if waveform.dim() == 4:  # [B, S, C, T]
                bs, n_sources, nch, wlen = waveform.shape
                waveform_reshaped = waveform.reshape(bs * n_sources, nch, wlen)
                waveform_upsampled = self.upsample_resampler(waveform_reshaped.float())
                waveform_upsampled = waveform_upsampled.view(bs, n_sources, nch, -1)
                current_length = waveform_upsampled.shape[-1]
                if current_length > original_length:
                    waveform_upsampled = waveform_upsampled[..., :original_length]
                elif current_length < original_length:
                    pad_length = original_length - current_length
                    waveform_upsampled = F.pad(waveform_upsampled, (0, pad_length), mode='constant', value=0.0)
                output_dict['waveform'] = waveform_upsampled.to(dtype=original_dtype)
            else:  # [B, C, T]
                waveform_upsampled = self.upsample_resampler(waveform.float())
                current_length = waveform_upsampled.shape[-1]
                if current_length > original_length:
                    waveform_upsampled = waveform_upsampled[..., :original_length]
                elif current_length < original_length:
                    pad_length = original_length - current_length
                    waveform_upsampled = F.pad(waveform_upsampled, (0, pad_length), mode='constant', value=0.0)
                output_dict['waveform'] = waveform_upsampled.to(dtype=original_dtype)

        return output_dict
