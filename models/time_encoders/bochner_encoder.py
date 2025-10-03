"""
Bochner Time Encoder

Implementation of Bochner's theorem-based time encoding using random Fourier features.
This matches the original implementation from the temporal encoding paper.
Based on gaussian_time_encode() from the original TensorFlow code.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from .base_encoder import BaseTimeEncoder


class BochnerTimeEncoder(BaseTimeEncoder):
    """
    Bochner time encoding using learnable Gaussian-sampled Fourier features.
    
    Based on Bochner's theorem, uses frequencies sampled from a learnable
    Gaussian distribution to approximate shift-invariant kernels.
    
    Original implementation: gaussian_time_encode() in TensorFlow
    """
    
    def __init__(self, time_dim: int, device: str = 'cpu'):
        """
        Initialize Bochner time encoder.
        
        Args:
            time_dim: Output dimension of time encoding (must be even)
            device: Device to place encoder on
        """
        super().__init__(time_dim, device)
        
        if time_dim % 2 != 0:
            raise ValueError("time_dim must be even for Bochner encoding")
        
        self.half_dim = time_dim // 2
        
        # Initialize mean frequencies with log-spaced values (as in original)
        # Original: init_freq_base = np.linspace(0, 8, effe_numits) / np.pi / 2
        #           init_freq = 1 / 10 ** init_freq_base
        init_freq_base = np.linspace(0, 8, self.half_dim) / np.pi / 2
        init_freq = 1.0 / (10.0 ** init_freq_base)
        
        # Learnable mean vector for Gaussian sampling
        self.freq_mean = nn.Parameter(
            torch.tensor(init_freq, dtype=torch.float32)
        )
        
        # Learnable std vector (initialized to ones)
        self.freq_std = nn.Parameter(
            torch.ones(self.half_dim, dtype=torch.float32)
        )
        
        # FIX: Sample frequencies ONCE at initialization for deterministic encoding
        # (Original paper samples each forward for Monte Carlo, but this breaks supervised learning)
        # We keep them as buffers so they're consistent across forward passes
        self.register_buffer('_sampled_frequencies', None)
    
    def forward(self, timestamps: torch.Tensor = None, time_deltas: torch.Tensor = None,
                t_abs: torch.Tensor = None, t_rel: torch.Tensor = None) -> torch.Tensor:
        """
        Encode timestamps using Bochner's learnable Gaussian Fourier features.
        Supports both single-stream (timestamps) and dual-stream (t_abs, t_rel) interfaces.
        
        Args:
            timestamps: Tensor of shape (batch_size,) or (batch_size, seq_len, 1)
            time_deltas: Not used, kept for interface compatibility
            t_abs: Absolute timestamps (dual-stream interface)
            t_rel: Relative timestamps (dual-stream interface). If provided, this is preferred.
        
        Returns:
            Time encodings of shape (batch_size, time_dim) or (batch_size, seq_len, time_dim)
        """
        # Handle different input interfaces (prefer relative time if available)
        if t_rel is not None:
            input_tensor = t_rel.squeeze(-1) if t_rel.dim() == 3 else t_rel
        elif timestamps is not None:
            input_tensor = timestamps.squeeze(-1) if timestamps.dim() == 3 else timestamps
        elif t_abs is not None:
            input_tensor = t_abs.squeeze(-1) if t_abs.dim() == 3 else t_abs
        else:
            raise ValueError("One of 't_rel', 'timestamps', or 't_abs' must be provided")
        
        # FIX: Use deterministic frequencies for supervised learning
        # Sample once and cache (instead of sampling every forward pass)
        if self._sampled_frequencies is None:
            with torch.no_grad():
                eps = torch.randn(self.half_dim, device=input_tensor.device, dtype=input_tensor.dtype)
                sampled_frequencies = self.freq_mean + self.freq_std * eps
                self.register_buffer('_sampled_frequencies', sampled_frequencies)
        
        sampled_frequencies = self._sampled_frequencies
        
        # Project timestamps onto sampled frequencies
        # input_tensor: (B, S) or (B,)
        # sampled_frequencies: (half_dim,)
        if input_tensor.dim() == 1:
            # (B,) -> (B, 1, half_dim)
            projected = input_tensor.unsqueeze(-1).unsqueeze(-1) * sampled_frequencies
        else:
            # (B, S) -> (B, S, half_dim)
            projected = input_tensor.unsqueeze(-1) * sampled_frequencies
        
        # Apply sin and cos (note: original uses sin for cos_feat, cos for sin_feat)
        # Original:
        #   cos_feat = tf.sin(expand_input * sampled_freq)
        #   sin_feat = tf.cos(expand_input * sampled_freq)
        sin_features = torch.sin(projected)  # (B, [S], half_dim)
        cos_features = torch.cos(projected)  # (B, [S], half_dim)
        
        # Concatenate sin and cos features (sin first, like original)
        output = torch.cat([sin_features, cos_features], dim=-1)  # (B, [S], time_dim)
        
        # Normalize for stability
        output = output / math.sqrt(self.time_dim)
        
        return output
    
    def get_config(self) -> dict:
        """Return configuration for reproducibility."""
        config = super().get_config()
        config.update({
            'half_dim': self.half_dim,
            'freq_mean': self.freq_mean.detach().cpu().numpy().tolist(),
            'freq_std': self.freq_std.detach().cpu().numpy().tolist()
        })
        return config
