"""
Mercer Time Encoder

Implementation of Mercer's theorem-based time encoding using basis function expansion.
This provides a baseline encoding method for comparison with KAN-MAMMOTE.
"""

import torch
import torch.nn as nn
import math
from .base_encoder import BaseTimeEncoder


class MercerTimeEncoder(BaseTimeEncoder):
    """
    Mercer time encoding using eigenbasis expansion.
    
    Based on Mercer's theorem, approximates kernel functions through
    eigenfunction decomposition of the time domain.
    """
    
    def __init__(self, time_dim: int, expand_dim: int = 8, time_range: float = 10.0, device: str = 'cpu'):
        """
        Initialize Mercer time encoder.
        
        Args:
            time_dim: Output dimension of time encoding
            expand_dim: Dimension for intermediate expansion
            time_range: Expected range of input timestamps for normalization
            device: Device to place encoder on
        """
        super().__init__(time_dim, device)
        
        self.expand_dim = expand_dim
        self.time_range = time_range
        
        # Learnable eigenvalue-like parameters
        self.eigenvalues = nn.Parameter(torch.randn(expand_dim) * 0.1)
        
        # Learnable frequency parameters for eigenfunctions
        self.frequencies = nn.Parameter(torch.randn(expand_dim) * 0.1)
        
        # Phase parameters for eigenfunctions
        self.phases = nn.Parameter(torch.randn(expand_dim) * 0.1)
        
        # Projection layer to final dimension
        self.projection = nn.Linear(expand_dim * 2, time_dim)  # *2 for cos+sin
        
        # Optional layer normalization
        self.layer_norm = nn.LayerNorm(time_dim)
    
    def forward(self, timestamps: torch.Tensor, time_deltas: torch.Tensor = None) -> torch.Tensor:
        """
        Encode timestamps using Mercer eigenfunction expansion.
        
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
        
        # Normalize timestamps to reasonable range
        normalized_time = timestamps / self.time_range
        
        # Compute eigenfunction values
        # normalized_time: (batch_size, [seq_len])
        # frequencies, phases: (expand_dim,)
        freq_proj = normalized_time.unsqueeze(-1) * self.frequencies  # (batch_size, [seq_len], expand_dim)
        phase_proj = freq_proj + self.phases  # (batch_size, [seq_len], expand_dim)
        
        # Apply trigonometric eigenfunctions
        cos_features = torch.cos(phase_proj)  # (batch_size, [seq_len], expand_dim)
        sin_features = torch.sin(phase_proj)  # (batch_size, [seq_len], expand_dim)
        
        # Weight by eigenvalues (importance of each eigenfunction)
        eigenweight = torch.softmax(self.eigenvalues, dim=0)  # Normalized importance
        cos_features = cos_features * eigenweight
        sin_features = sin_features * eigenweight
        
        # Concatenate trigonometric features
        features = torch.cat([cos_features, sin_features], dim=-1)  # (batch_size, [seq_len], expand_dim*2)
        
        # Project to final dimension
        output = self.projection(features)  # (batch_size, [seq_len], time_dim)
        
        # Apply layer normalization
        output = self.layer_norm(output)
        
        # Restore original batch dimension if needed
        if squeeze_output and len(original_shape) == 1:
            output = output.squeeze(1)  # (batch_size, time_dim)
        
        return output
    
    def get_config(self) -> dict:
        """Return configuration for reproducibility."""
        config = super().get_config()
        config.update({
            'expand_dim': self.expand_dim,
            'time_range': self.time_range
        })
        return config
