# models/adaleaf.py --> AdaDRC
# Author: Hanyu Meng
# Date: 2025-05-30
# Description: Adaptive LEAF frontend with PCEN using dynamic per-frame parameter prediction from PCEN controller.
#              Based on the original LEAF implementation:
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .leaf import gabor_filters, gauss_windows, mel_filter_params
from typing import Optional
import math

def freq_normalize_with_E(E: torch.Tensor, eps: float = 1e-6):
    # shape-massage Eprev to [B, 1, 3] for broadcasting over the 40 bands
    if E.dim() == 2:         # [B, 3]
        E_sum = E.unsqueeze(1)   # -> [B, 1, 3]
    elif E.dim() == 1:       # [B]
        E_sum= E.view(-1, 1, 1) # -> [B, 1, 1]
    else:
        E_sum = E
    # if Eprev accidentally has a frequency dimension, collapse it
    if E_sum.shape[1] != 1:
        E_sum = E_sum.sum(dim=1, keepdim=True)

    E_norm = E / (E_sum + eps)   # [B, 40, 3]
    return E_norm

class FixedGaborFilterbank(nn.Module):
    """
    Gabor filterbank with fixed (non‐learnable) parameters.
    All parameters are initialized from the mel scale and then held constant.
    """
    def __init__(self,
                 n_filters: int,
                 min_freq: float,
                 max_freq: float,
                 sample_rate: int,
                 filter_size: int,
                 pool_size: int,
                 pool_stride: int,
                 pool_init: float = 0.4):
        super().__init__()
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride

        # Compute initial center frequencies and bandwidths
        center_freqs, bandwidths = mel_filter_params(n_filters,
                                                     min_freq,
                                                     max_freq,
                                                     sample_rate)
        # Register as buffers so they’re moved with .to(device) but not trained
        self.register_buffer('center_freqs', center_freqs)
        self.register_buffer('bandwidths', bandwidths)

        # Pooling widths fixed at pool_init
        init_pool = torch.full((n_filters,), float(pool_init))
        self.register_buffer('pooling_widths', init_pool)

    def forward(self, x):
        # Clamp to valid ranges
        cfs = self.center_freqs.clamp(min=0., max=np.pi)
        z = np.sqrt(2 * np.log(2)) / np.pi
        bws = self.bandwidths.clamp(min=4 * z, max=self.filter_size * z)

        # Build Gabor filters (complex) then split into real+imag channels
        filters = gabor_filters(self.filter_size, cfs, bws)
        filters = torch.cat((filters.real, filters.imag), dim=0).unsqueeze(1)  # [2F,1,K]

        # 1D convolution & squared modulus
        x = F.conv1d(x, filters, padding=self.filter_size // 2)
        x = x**2
        x = x[:, :self.n_filters] + x[:, self.n_filters:]  # sum real+imag

        # Gaussian pooling
        pw = self.pooling_widths.clamp(min=2. / self.pool_size, max=0.5)
        windows = gauss_windows(self.pool_size, pw).unsqueeze(1)  # [F,1,K]
        x = F.conv1d(x, windows, stride=self.pool_stride,
                     padding=self.filter_size // 2,
                     groups=self.n_filters)
        return x
    
class GaborFilterbank(nn.Module):
    """
    Torch module that functions as a gabor filterbank. Initializes n_filters center frequencies
    and bandwidths that are based on a mel filterbank. The parameters are used to calculate Gabor filters
    for a 1D convolution over the input signal. The squared modulus is taken from the results.
    To reduce the temporal resolution a gaussian lowpass filter is calculated from pooling_widths,
    which are used to perform a pooling operation.
    The center frequencies, bandwidths and pooling_widths are learnable parameters.
    :param n_filters: number of filters
    :param min_freq: minimum frequency (used for the mel filterbank initialization)
    :param max_freq: maximum frequency (used for the mel filterbank initialization)
    :param sample_rate: sample rate (used for the mel filterbank initialization)
    :param filter_size: size of the kernels/filters for gabor convolution
    :param pool_size: size of the kernels/filters for pooling convolution
    :param pool_stride: stride of the pooling convolution
    :param pool_init: initial value for the gaussian lowpass function
    """
    def __init__(self, n_filters: int, min_freq: float, max_freq: float,
                 sample_rate: int, filter_size: int, pool_size: int,
                 pool_stride: int, pool_init: float=0.4):
        super(GaborFilterbank, self).__init__()
        self.n_filters = n_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        center_freqs, bandwidths = mel_filter_params(n_filters, min_freq,
                                                     max_freq, sample_rate)
        self.center_freqs = nn.Parameter(center_freqs)
        self.bandwidths = nn.Parameter(bandwidths)
        self.pooling_widths = nn.Parameter(torch.full((n_filters,),
                                                      float(pool_init)))

    def forward(self, x):
        # compute filters
        center_freqs = self.center_freqs.clamp(min=0., max=np.pi)
        z = np.sqrt(2 * np.log(2)) / np.pi
        bandwidths = self.bandwidths.clamp(min=4 * z, max=self.filter_size * z)
        filters = gabor_filters(self.filter_size, center_freqs, bandwidths)
        filters = torch.cat((filters.real, filters.imag), dim=0).unsqueeze(1)
        # convolve with filters
        x = F.conv1d(x, filters, padding=self.filter_size // 2)
        # compute squared modulus
        x = x ** 2
        x = x[:, :self.n_filters] + x[:, self.n_filters:]
        # compute pooling windows
        pooling_widths = self.pooling_widths.clamp(min=2. / self.pool_size,
                                                   max=0.5)
        windows = gauss_windows(self.pool_size, pooling_widths).unsqueeze(1)
        # apply temporal pooling
        x = F.conv1d(x, windows, stride=self.pool_stride,
                     padding=self.filter_size//2, groups=self.n_filters)
        return x

class PCENController(nn.Module):
    """
    Inputs:
      feats: [B, F, T]           (one feature per step)      OR
             [B, F, T, C]        (C features per step)
    Outputs:
      alpha: [B, F, 1]  in (0,1)
      r:     [B, F, 1]  in [r_min, r_max]
    """
    def __init__(self, input_size=1, hidden=32, r_min=0.2, r_max=1.0,
                 num_layers=1, bidirectional=False):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 32), nn.ReLU(), nn.Linear(32, 2)
        )
        self.r_min, self.r_max = r_min, r_max

    def forward(self, feats: torch.Tensor, h0: torch.Tensor=None):
        # ---- normalize shapes to [B*F, T, C] ----
        if feats.dim() == 3:                      # [B, F, T]
            B, F, T = feats.shape
            C = 1
            x = feats.view(B * F, T, 1)
        elif feats.dim() == 4:                    # [B, F, T, C]
            B, F, T, C = feats.shape
            x = feats.view(B * F, T, C)
        else:
            raise ValueError("feats must be [B,F,T] or [B,F,T,C]")

        assert x.size(-1) == self.gru.input_size, \
            f"GRU input_size={self.gru.input_size}, got {x.size(-1)}"

        # ---- GRU ----
        if h0 is None:
            num_dirs = 2 if self.gru.bidirectional else 1
            h0 = x.new_zeros(self.gru.num_layers * num_dirs, B * F, self.gru.hidden_size)
        out, h = self.gru(x, h0)                  # out: [B*F, T, H*D]

        # use last time step to produce one param per band
        out_last = out[:, -1, :]                  # [B*F, H*D]
        logits = self.head(out_last)              # [B*F, 2]
        a_hat, r_hat = torch.chunk(logits, 2, dim=-1)  # each [B*F, 1]

        # ---- bound to valid ranges & reshape to [B, F, 1] ----
        alpha = torch.sigmoid(a_hat).view(B, F, 1)  # (0,1)
        r = (self.r_min + (self.r_max - self.r_min) * torch.sigmoid(r_hat)).view(B, F, 1)
        return alpha, r, h

class AdaptivePCEN(nn.Module):
    """
    Adaptive PCEN using dynamic per-frame parameter prediction from PPN.
    Follows the formulation:
        PCEN_t = ((X_t / (M_t + eps)^alpha + delta)^r) - delta^r
        M_t = (1 - s) * M_{t-1} + s * X_t

    Args:
        eps: Small constant for numerical stability
        clamp: Optional minimum clamp value on input
    """
    def __init__(self, eps: float = 1e-6, clamp: Optional[float] = None):
        super().__init__()
        self.eps = eps
        self.clamp = clamp

    def forward(self, X: torch.Tensor, ppn):
        """
        Args:
            X: [B, F, T] input energy (STFT magnitude squared or similar)
            ppn: module that takes X[..., t-1], X[..., t] and returns (s, alpha, delta, r)

        Returns:
            PCEN-transformed tensor of shape [B, F, T]
        """
        B, F, T = X.shape
        M_prev = torch.zeros(B, F, device=X.device)
        outputs = []
        for t in range(T):
            X_t = X[:, :, t]
            X_prev = X[:, :, t - 1] if t > 0 else X_t
            if self.clamp is not None:
                X_t = X_t.clamp(min=self.clamp)
                X_prev = X_prev.clamp(min=self.clamp)

            s, alpha, delta, r = ppn((X_prev, X_t), dim=-1)
            M_t = (1 - s) * M_prev + s * X_t

            # === Apply PCEN ===
            norm = (X_t / (M_t + self.eps).pow(alpha) + delta).pow(r)
            PCEN_t = norm - delta.pow(r)
            
            outputs.append(PCEN_t)
            M_prev = M_t
        return torch.stack(outputs, dim=-1)  # [B, F, T]

class Simplify_AdaptivePCEN(nn.Module):
    """
    Adaptive PCEN using dynamic per-frame parameter prediction from PPN.
    Follows the formulation:
        PCEN_t = ((X_t / (M_t + eps)^alpha + delta)^r) - delta^r
        M_t = (1 - s) * M_{t-1} + s * X_t
    Args:
        eps: Small constant for numerical stability
        clamp: Optional minimum clamp value on input
    """
    def __init__(self, eps: float = 1e-6, init_s: float = 0.04, init_alpha: float = 0.48, init_r: float = 0.5, clamp: Optional[float] = None):
        super().__init__()
        self.eps = eps
        self.clamp = clamp
        self.init_s = init_s  # Initial value for s, used in the first frame
        self.init_alpha = init_alpha
        self.init_r = init_r

    def forward(self, X_t: torch.Tensor, X_prev: torch.Tensor, M_prev:torch.Tensor, ppn):
        """
        Args:
            X: [B, F, T] input energy (STFT magnitude squared or similar)
            ppn: module that takes X[..., t-1], X[..., t] and returns (s, alpha, delta, r)

        Returns:
            PCEN-transformed tensor of shape [B, F, T]
        """
        B, F, T = X_t.shape
        s = self.init_s
        if M_prev is None:
            M_prev =  X_t 

        if self.clamp is not None:
            X_t = X_t.clamp(min=self.clamp)
            if X_prev is None:
                X_prev = X_t
            
            X_prev = X_prev.clamp(min=self.clamp)
            alpha, r, _ = ppn(torch.cat([X_prev, X_t], dim=-1))  # [B, F]
            M_t = (1 - s) * M_prev + s * X_t
            norm = (X_t**r) / ((M_t+self.eps)**alpha)
            PCEN_t = norm

        return PCEN_t, M_prev, alpha, r

class AdaptiveLeaf(nn.Module):
    """
    LEAF frontend, a learnable front-end that takes an audio waveform as input
    and outputs a learnable spectral representation. Initially approximates the
    computation of standard mel-filterbanks.

    A detailed technical description is presented in Section 3 of
    https://arxiv.org/abs/2101.08596 .
    :param n_filters: number of filters
    :param min_freq: minimum frequency (used for the mel filterbank initialization)
    :param max_freq: maximum frequency (used for the mel filterbank initialization)
    :param sample_rate: sample rate (used for the mel filterbank initialization)
    :param window_len: kernel/filter size of the convolutions in ms
    :param window_stride: stride used for the pooling convolution in ms
    :param compression: compression function used (default: PCEN)
    """
    def __init__(self,
                 n_filters: int=40,
                 min_freq: float=60.0,
                 max_freq: float=7800.0,
                 sample_rate: int=16000,
                 window_len: float=25.,
                 window_stride: float=10.,
                 compression: Optional[torch.nn.Module]=None,
                 ):
        super(AdaptiveLeaf, self).__init__()

        window_size = int(sample_rate * window_len / 1000)
        window_size += 1 - (window_size % 2)  # make odd
        window_stride = int(sample_rate * window_stride / 1000)
        self.ppn = PCENController(hidden=32, r_min=0.2, r_max=1.0)
        self.filterbank = GaborFilterbank(
            n_filters, min_freq, max_freq, sample_rate,
            filter_size=window_size, pool_size=window_size,
            pool_stride=window_stride)

        self.compression = compression if compression else AdaptivePCEN()

    def forward(self, x: torch.tensor, x_prev: torch.tensor, M_prev: torch.tensor = None):
        while x.ndim < 3:
            x = x[:, np.newaxis]
        x = self.filterbank(x)
        x,M_prev, alpha, r = self.compression(x, x_prev, M_prev,self.ppn)
        return x, M_prev, alpha, r