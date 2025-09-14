"""
Original Time Encoder

Traditional cosine-based time encoding from the original DyGMamba implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from .base_encoder import BaseTimeEncoder


class OriginalTimeEncoder(BaseTimeEncoder):
    """
    Original time encoder using cosine basis functions.
    
    This encoder transforms time differences into embeddings using trainable
    frequency parameters and cosine activations, similar to positional encoding.
    """
    
    def __init__(self, time_dim: int, parameter_requires_grad: bool = True, device: str = 'cpu'):
        """
        Initialize original time encoder.
        
        Args:
            time_dim: Dimension of output time encoding
            parameter_requires_grad: Whether parameters should be trainable
            device: Device to place encoder on
        """
        super().__init__(time_dim, device)
        
        # Trainable linear layer for time encoding
        self.w = nn.Linear(1, time_dim)
        
        # Initialize with geometric progression frequencies
        freq_init = (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))
                    .reshape(time_dim, -1))
        self.w.weight = nn.Parameter(freq_init)
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        # Control gradient computation
        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    def forward(self, timestamps: torch.Tensor, time_deltas: torch.Tensor = None) -> torch.Tensor:
        """
        Compute time encodings from timestamps.
        
        Args:
            timestamps: Tensor of shape (batch_size,) or (batch_size, seq_len) 
            time_deltas: Not used in original encoder, kept for interface compatibility
        
        Returns:
            Time encodings of shape (batch_size, time_dim) or (batch_size, seq_len, time_dim)
        """
        # Handle different input shapes
        original_shape = timestamps.shape
        
        if timestamps.dim() == 1:
            # Single timestamps: (batch_size,) -> (batch_size, 1, 1)
            timestamps = timestamps.unsqueeze(-1).unsqueeze(-1)
            single_batch = True
        elif timestamps.dim() == 2:
            # Sequence timestamps: (batch_size, seq_len) -> (batch_size, seq_len, 1)
            timestamps = timestamps.unsqueeze(-1)
            single_batch = False
        else:
            single_batch = False
        
        # Apply cosine transformation: cos(W * t + b)
        output = torch.cos(self.w(timestamps))  # Shape: (batch_size, [seq_len,] time_dim)
        
        # Restore original shape if needed
        if single_batch and len(original_shape) == 1:
            output = output.squeeze(1)  # Remove sequence dimension
        
        return output
    
    def get_config(self) -> dict:
        """Return configuration for reproducibility."""
        config = super().get_config()
        config.update({
            'parameter_requires_grad': self.w.weight.requires_grad
        })
        return config
