# file: models/time_encoders/sm_kernel.py (Final GPU-Compatible Version)

import torch
import torch.nn as nn
import gpytorch
import math
import numpy as np

class SMKernelLayer(nn.Module):
    """
    A learnable layer that uses the Spectral Mixture (SM) Kernel to encode
    relative time differences (delta_t).
    """
    def __init__(self, num_mixtures: int, input_dim: int = 1, use_layernorm: bool = True):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.input_dim = input_dim
        self.kernel = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=num_mixtures, 
            ard_num_dims=input_dim
        )
        self.layer_norm = nn.LayerNorm(num_mixtures) if use_layernorm else nn.Identity()
        print(f"Initialized SMKernelLayer with {num_mixtures} mixtures (output dimension).")

    def initialize_from_data(self, delta_t_sample: torch.Tensor):
        """
        Initializes the kernel's parameters based on the frequency spectrum of sample data.
        This method is now device-agnostic and will run on the same device as the input tensor.
        """
        print("Initializing SM-Kernel from data spectrum...")
        if not isinstance(delta_t_sample, torch.Tensor):
            raise TypeError("delta_t_sample must be a PyTorch tensor.")
        if delta_t_sample.dim() != 3 or delta_t_sample.shape[-1] != 1:
            raise ValueError("delta_t_sample must have shape (batch_size, seq_len, 1).")

        # Get the device from the input tensor
        device = delta_t_sample.device
        
        # --- START OF CORRECTION ---
        # Perform all operations on the correct device, removing .cpu() and .numpy()
        delta_t_flat = delta_t_sample.reshape(-1)
        
        # The signal is a discrete sequence. The sample spacing (d) is 1.
        freqs = torch.fft.fftfreq(len(delta_t_flat), d=1.0, device=device)
        # --- END OF CORRECTION ---

        fft_vals = torch.fft.fft(delta_t_flat.to(dtype=torch.float32))
        power_spectrum = torch.abs(fft_vals)**2

        positive_freq_indices = freqs > 0
        if not torch.any(positive_freq_indices):
             print("Warning: No positive frequencies found in data. SM-Kernel will be randomly initialized.")
             return

        positive_freqs = freqs[positive_freq_indices]
        positive_power = power_spectrum[positive_freq_indices]
        
        num_peaks = min(self.num_mixtures, len(positive_power))
        if num_peaks == 0:
            print("Warning: No frequency peaks found. SM-Kernel will be randomly initialized.")
            return

        peak_indices = torch.topk(positive_power, k=num_peaks).indices
        top_freqs = positive_freqs[peak_indices]

        with torch.no_grad():
            self.kernel.raw_mixture_means.zero_()
            target_shape = self.kernel.raw_mixture_means[:num_peaks].shape
            self.kernel.raw_mixture_means[:num_peaks] = top_freqs.reshape(target_shape)
            self.kernel.raw_mixture_scales.fill_(-1.0) 
            self.kernel.raw_mixture_weights.fill_(1.0 / self.num_mixtures)

        print(f"SM-Kernel initialized with top frequencies: {top_freqs.tolist()}")
    
    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """ The forward pass (already correct). """
        if delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)

        weights = self.kernel.mixture_weights
        means = self.kernel.mixture_means.squeeze(-1)
        scales = self.kernel.mixture_scales.squeeze(-1)

        weights = weights.view(1, 1, -1)
        means = means.view(1, 1, -1)
        scales = scales.view(1, 1, -1)

        dist_sq = delta_t.pow(2)
        exp_term = torch.exp(-2 * (math.pi**2) * dist_sq * scales)
        cos_term = torch.cos(2 * math.pi * delta_t * means)
        embedding = weights * exp_term * cos_term
        
        embedding = self.layer_norm(embedding)
        return embedding