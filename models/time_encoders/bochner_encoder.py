"""
Bochner Time Encoder

Implementation of Bochner's theorem-based time encoding using random Fourier features.
This provides a baseline encoding method for comparison with KAN-MAMMOTE.
"""

import torch
import torch.nn as nn
import math
from .base_encoder import BaseTimeEncoder


class BochnerTimeEncoder(BaseTimeEncoder):
    """
    Bochner time encoding using random Fourier features.
    
    Based on Bochner's theorem, uses random frequencies to approximate
    kernel-based time embeddings through Monte Carlo sampling.
    """
    
    def __init__(self, time_dim: int, sigma: float = 1.0, device: str = 'cpu'):
        """
        Initialize Bochner time encoder.
        
        Args:
            time_dim: Output dimension of time encoding (must be even)
            sigma: Standard deviation for random frequency sampling
            device: Device to place encoder on
        """
        super().__init__(time_dim, device)
        
        if time_dim % 2 != 0:
            raise ValueError("time_dim must be even for Bochner encoding")
        
        self.sigma = sigma
        self.half_dim = time_dim // 2
        
        # Random frequencies sampled from Gaussian distribution
        # These are fixed (not learnable) as per Bochner's theorem
        self.register_buffer(
            'frequencies', 
            torch.randn(self.half_dim) * sigma
        )
        
        # Optional learnable scaling
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, timestamps: torch.Tensor, time_deltas: torch.Tensor = None) -> torch.Tensor:
        """
        Encode timestamps using Bochner's random Fourier features.
        
        Args:
            timestamps: Tensor of shape (batch_size,) or (batch_size, seq_len)
            time_deltas: Not used, kept for interface compatibility
        
        Returns:
            Time encodings of shape (batch_size, time_dim) or (batch_size, seq_len, time_dim)
        """
        # Handle input shapes
        original_shape = timestamps.shape
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1)  # (batch_size, 1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Project onto random frequencies: timestamps * frequencies
        # timestamps: (batch_size, [seq_len])
        # frequencies: (half_dim,)
        projected = timestamps.unsqueeze(-1) * self.frequencies  # (batch_size, [seq_len], half_dim)
        
        # Apply cosine and sine to get Fourier features
        cos_features = torch.cos(projected)  # (batch_size, [seq_len], half_dim)
        sin_features = torch.sin(projected)  # (batch_size, [seq_len], half_dim)
        
        # Concatenate cos and sin features
        features = torch.cat([cos_features, sin_features], dim=-1)  # (batch_size, [seq_len], time_dim)
        
        # Apply learnable scaling
        features = features * self.scale
        
        # Normalize to unit norm (optional, helps with training stability)
        features = features / math.sqrt(self.time_dim)
        
        # Restore original batch dimension if needed
        if squeeze_output and len(original_shape) == 1:
            features = features.squeeze(1)  # (batch_size, time_dim)
        
        return features
    
    def get_config(self) -> dict:
        """Return configuration for reproducibility."""
        config = super().get_config()
        config.update({
            'sigma': self.sigma,
            'half_dim': self.half_dim
        })
        return config
