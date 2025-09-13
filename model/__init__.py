# AudioClassifier Class
## Imports
#basics
from typing import Any, Optional, Sequence, Tuple

#torch
import torch
import torch.nn as nn

class AudioClassifier(torch.nn.Module):
    """Neural network architecture to train an audio classifier from waveforms."""
    def __init__(self,
                 num_outputs: int,
                 frontend: Optional[torch.nn.Module] = None,
                 encoder: Optional[torch.nn.Module] = None):
        super(AudioClassifier, self).__init__()
        self._frontend = frontend
        self._encoder = encoder
        self._pool = torch.nn.Sequential(
            torch.nn.AdaptiveMaxPool2d(1),
            torch.nn.Flatten()
        )
        self._head = torch.nn.Linear(in_features=1280, out_features=num_outputs)
        #self._head = torch.nn.LazyLinear(out_features=num_outputs)

    def forward(self, inputs: torch.Tensor):
        output = inputs
        if self._frontend is not None:
            output = self._frontend(output)  # pylint: disable=not-callable
            if output.ndim == 3:
                output = output[:,None,:,:] #add 1 "color" channel
        if self._encoder:
            output = self._encoder(output)
        output = self._pool(output)
        return self._head(output)

class Ada_AudioClassifier(torch.nn.Module):
    """Neural network architecture to train an audio classifier from waveforms."""
    def __init__(self,
                 num_outputs: int,
                 n_filters: int,
                 frontend: Optional[torch.nn.Module] = None,
                 encoder: Optional[torch.nn.Module] = None):
        super(Ada_AudioClassifier, self).__init__()
        self._frontend = frontend
        self._encoder = encoder
        self._pool = torch.nn.Sequential(
            torch.nn.AdaptiveMaxPool2d(1),
            torch.nn.Flatten()
        )
        self.n_filters = n_filters
        self._head = torch.nn.Linear(in_features=1280, out_features=num_outputs)
        #self._head = torch.nn.LazyLinear(out_features=num_outputs)

    def forward(self, inputs: torch.Tensor):
        """
        inputs: [B, T, L]  (T frames, each of length L)
        Frontend returns per-frame features of shape [B, n_filters, L’] (example).
        We concatenate along time.
        """
        B, T, L = inputs.shape
        device = inputs.device

        if self._frontend is None:
            # Fall back: raw waveform path (unlikely what you want)
            x_feat = inputs[:, None, :, :]  # [B,1,T,L]
        else:
            # We’ll accumulate along time on feature dim 2
            # First forward pass to infer feature length
            with torch.no_grad():
                probe = self._frontend(inputs[:, 0, :].unsqueeze(1), None)  # [B, F, W]
            F, W = probe[0].shape[1], probe[0].shape[2]

            # Pre-allocate concatenated feature: [B, F, T*W]
            filtered = torch.zeros((B, F, T * W), device=device)

            pre_frame = None  # previous frontend output (feature domain)
            M_prev = None    # previous smoother state (feature domain)
            for i in range(T):
                cur_wave = inputs[:, i, :].unsqueeze(1)      # [B,1,L]
                cur_feat, M_prev = self._frontend(cur_wave, pre_frame, M_prev)  # [B,F,W]
                # DO NOT detach here during training; you want gradients
                filtered[:, :, i * W : (i + 1) * W] = cur_feat
                pre_frame = cur_feat  # carry feature from previous step

            # Encoder expects 4D: [B, C, F, T’]
            x_feat = filtered.unsqueeze(1)  # [B,1,F,T*W]

        # Encoder + head
        if self._encoder is not None:
            x_feat = self._encoder(x_feat)     # e.g., EfficientNet expects [B,1,F,T’]
        x_feat = self._pool(x_feat)            # AdaptiveMaxPool2d(1) then Flatten
        return self._head(x_feat)
