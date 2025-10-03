"""
Mercer Time Encoder

Implementation of Mercer's theorem-based time encoding using eigenfunction expansion.
This matches the original implementation from the temporal encoding paper.
Based on basis_time_encode() from the original TensorFlow code.
"""

import torch
import torch.nn as nn
import numpy as np
import math
from .base_encoder import BaseTimeEncoder


class MercerTimeEncoder(BaseTimeEncoder):
    """
    Mercer time encoding using eigenfunction expansion with harmonic basis.
    
    Based on Mercer's theorem, decomposes time kernel into eigenfunctions
    using learnable periods and harmonic expansion (1x, 2x, 3x, ..., Nx).
    
    Original implementation: basis_time_encode() in TensorFlow
    Key feature: Harmonic expansion for capturing multi-scale periodicity
    """
    
    def __init__(self, time_dim: int, expand_dim: int = 8, device: str = 'cpu'):
        """
        Initialize Mercer time encoder.
        
        Args:
            time_dim: Output dimension of time encoding
            expand_dim: Number of harmonic expansions (1x, 2x, ..., Nx frequency)
            device: Device to place encoder on
        """
        super().__init__(time_dim, device)
        
        self.expand_dim = expand_dim
        
        # Initialize base periods with log spacing (as in original)
        # Original: init_period_base = np.linspace(0, 8, time_dim)
        #           period_var = 10.0 ** init_period_base
        init_period_base = np.linspace(0, 8, time_dim)
        init_periods = 10.0 ** init_period_base  # Periods: [1, 10, 100, ..., 10^8]
        
        # Learnable base periods (one per output dimension)
        self.periods = nn.Parameter(
            torch.tensor(init_periods, dtype=torch.float32)
        )
        
        # Harmonic expansion coefficients [1, 2, 3, ..., expand_dim]
        # These multiply the base frequencies to create harmonics
        self.register_buffer(
            'harmonic_coef',
            torch.arange(1, expand_dim + 1, dtype=torch.float32)
        )
        
        # Learnable basis expansion weights
        # Original: basis_expan_var with shape [time_dim, 2*expand_dim]
        # Weights for combining sin/cos features
        self.basis_weights = nn.Parameter(
            torch.randn(time_dim, 2 * expand_dim) * 0.1
        )
        
        # Learnable bias for each output dimension
        self.basis_bias = nn.Parameter(
            torch.zeros(time_dim)
        )
    
    def forward(self, timestamps: torch.Tensor = None, time_deltas: torch.Tensor = None,
                t_abs: torch.Tensor = None, t_rel: torch.Tensor = None) -> torch.Tensor:
        """
        Encode timestamps using Mercer eigenfunction expansion with harmonics.
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
        
        # Compute base frequencies from periods
        # periods: (time_dim,)
        # harmonic_coef: (expand_dim,) = [1, 2, 3, ..., expand_dim]
        base_freqs = 1.0 / self.periods.unsqueeze(-1)  # (time_dim, 1)
        
        # Create harmonic frequencies: freq[i,j] = (1/period[i]) * j
        # This is the KEY feature missing from the simplified version!
        frequencies = base_freqs * self.harmonic_coef.unsqueeze(0)  # (time_dim, expand_dim)
        
        # Expand input for broadcasting
        # input_tensor: (B, S) or (B,)
        # frequencies: (time_dim, expand_dim)
        if input_tensor.dim() == 1:
            # (B,) -> (B, 1, 1, 1)
            expand_input = input_tensor.view(-1, 1, 1, 1)
        else:
            # (B, S) -> (B, S, 1, 1)
            expand_input = input_tensor.unsqueeze(-1).unsqueeze(-1)
        
        # Broadcast multiplication: (B, [S], 1, 1) * (1, 1, time_dim, expand_dim)
        # Result: (B, [S], time_dim, expand_dim)
        phase = expand_input * frequencies.unsqueeze(0).unsqueeze(0)
        
        # Apply sin and cos eigenfunctions
        sin_enc = torch.sin(phase)  # (B, [S], time_dim, expand_dim)
        cos_enc = torch.cos(phase)  # (B, [S], time_dim, expand_dim)
        
        # Concatenate sin and cos along last dimension
        # Original: tf.concat([sin_enc, cos_enc], axis=-1)
        fourier_features = torch.cat([sin_enc, cos_enc], dim=-1)  # (B, [S], time_dim, 2*expand_dim)
        
        # Weight by learned basis expansion
        # fourier_features: (B, [S], time_dim, 2*expand_dim)
        # basis_weights: (time_dim, 2*expand_dim)
        # Element-wise multiply then sum over last dimension
        weighted = (fourier_features * self.basis_weights.unsqueeze(0).unsqueeze(0)).sum(dim=-1)
        # Result: (B, [S], time_dim)
        
        # Add learnable bias
        output = weighted + self.basis_bias.unsqueeze(0).unsqueeze(0)  # (B, [S], time_dim)
        
        # Remove extra dimensions if input was 1D
        if input_tensor.dim() == 1:
            output = output.squeeze(1)  # (B, time_dim)
        
        return output
    
    def get_config(self) -> dict:
        """Return configuration for reproducibility."""
        config = super().get_config()
        config.update({
            'expand_dim': self.expand_dim,
            'periods': self.periods.detach().cpu().numpy().tolist(),
            'harmonic_coef': self.harmonic_coef.detach().cpu().numpy().tolist()
        })
        return config
