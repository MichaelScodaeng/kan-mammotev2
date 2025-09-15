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
    
    def forward(self, timestamps: torch.Tensor = None, time_deltas: torch.Tensor = None,
                t_abs: torch.Tensor = None, t_rel: torch.Tensor = None) -> torch.Tensor:
        """
        Encode timestamps using Mercer eigenfunction expansion.
        Supports both single-stream (timestamps) and dual-stream (t_abs, t_rel) interfaces.
        
        Args:
            timestamps: Tensor of shape (batch_size,) or (batch_size, seq_len)
            time_deltas: Not used, kept for interface compatibility
            t_abs: Absolute timestamps (dual-stream interface)
            t_rel: Relative timestamps (dual-stream interface, not used in Mercer)
        
        Returns:
            Time encodings of shape (batch_size, time_dim) or (batch_size, seq_len, time_dim)
        """
        # Handle different input interfaces
        if timestamps is not None:
            input_tensor = timestamps
        elif t_abs is not None:
            input_tensor = t_abs.squeeze(-1) if t_abs.dim() > 2 else t_abs
        else:
            raise ValueError("Either 'timestamps' or 't_abs' must be provided")
        
        # Handle input shapes
        original_shape = input_tensor.shape
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(-1)  # (batch_size, 1)
            squeeze_output = True
        else:
            squeeze_output = False
        
        # Normalize timestamps to reasonable range
        normalized_time = input_tensor / self.time_range
        
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
