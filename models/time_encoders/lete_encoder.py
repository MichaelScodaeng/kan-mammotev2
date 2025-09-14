"""
LeTE (Learnable Time Encoding)

Advanced time encoding using mixture of Fourier and spline components.
Based on the LeTE paper implementation with adaptations for the KAN-MAMMOTE framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .base_encoder import BaseTimeEncoder


class FourierSeries(nn.Module):
    """Fourier series component for periodic patterns."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.randn(dim) * 0.1)
        self.b = nn.Parameter(torch.randn(dim) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, [seq_len,] 1)
        # Output: (batch_size, [seq_len,] dim)
        return torch.sin(self.w * x + self.b)


class Spline(nn.Module):
    """B-spline component for local patterns."""
    
    def __init__(self, dim: int, num_knots: int = 16):
        super().__init__()
        self.dim = dim
        self.num_knots = num_knots
        
        # Learnable knot positions
        self.knots = nn.Parameter(torch.linspace(0, 1, num_knots))
        self.coeffs = nn.Parameter(torch.randn(dim, num_knots) * 0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize x to [0, 1] range using sigmoid
        x_norm = torch.sigmoid(x)  # (batch_size, [seq_len,] 1)
        
        # Compute B-spline basis functions (simplified RBF approach)
        knots = self.knots.view(1, 1, -1)  # (1, 1, num_knots)
        distances = torch.abs(x_norm - knots)  # (batch_size, [seq_len,] num_knots)
        basis = torch.exp(-distances * 5.0)  # RBF kernel
        
        # Apply learnable coefficients
        output = torch.matmul(basis, self.coeffs.T)  # (batch_size, [seq_len,] dim)
        return output


class LeTE(BaseTimeEncoder):
    """
    Learnable Time Encoding with Fourier and spline components.
    
    Combines periodic (Fourier) and local (spline) representations to capture
    diverse temporal patterns in absolute time.
    """
    
    def __init__(self, time_dim: int, p: float = 0.5, layer_norm: bool = True, 
                 scale: bool = True, parameter_requires_grad: bool = True, device: str = 'cpu'):
        """
        Initialize LeTE encoder.
        
        Args:
            time_dim: Total dimension of time encodings
            p: Fraction of dimensions allocated to Fourier vs spline (0.5 = equal split)
            layer_norm: Whether to apply layer normalization
            scale: Whether to apply learnable scaling
            parameter_requires_grad: Whether parameters should be trainable
            device: Device to place encoder on
        """
        super().__init__(time_dim, device)
        
        self.dim_fourier = math.floor(time_dim * p)
        self.dim_spline = time_dim - self.dim_fourier
        self.layer_norm = layer_norm
        self.scale = scale
        
        # Fourier component
        if self.dim_fourier > 0:
            self.w1_fourier = nn.Linear(1, self.dim_fourier)
            # Initialize with geometric progression
            fourier_vals = 1.0 / (10 ** np.linspace(0, 9, self.dim_fourier, dtype=np.float32))
            self.w1_fourier.weight = nn.Parameter(torch.from_numpy(fourier_vals).reshape(self.dim_fourier, -1))
            self.w1_fourier.bias = nn.Parameter(torch.zeros(self.dim_fourier))
            self.w2_fourier = FourierSeries(self.dim_fourier)
        
        # Spline component  
        if self.dim_spline > 0:
            self.w1_spline = nn.Linear(1, self.dim_spline)
            self.w2_spline = Spline(self.dim_spline)
        
        # Optional layer normalization
        if self.layer_norm:
            self.layernorm = nn.LayerNorm(time_dim)
        
        # Optional learnable scaling
        if self.scale:
            self.scale_weight = nn.Parameter(torch.ones(time_dim))
        
        # Mixture weights for combining components
        self.fourier_weight = nn.Parameter(torch.tensor(0.5))
        self.spline_weight = nn.Parameter(torch.tensor(0.5))
        
        # Control gradient computation
        if not parameter_requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, timestamps: torch.Tensor, time_deltas: torch.Tensor = None) -> torch.Tensor:
        """
        Encode timestamps using learnable Fourier and spline components.
        
        Args:
            timestamps: Tensor of shape (batch_size,) or (batch_size, seq_len)
            time_deltas: Not used in LeTE, kept for interface compatibility
        
        Returns:
            Time encodings of shape (batch_size, time_dim) or (batch_size, seq_len, time_dim)
        """
        # Handle input shapes
        original_shape = timestamps.shape
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1).unsqueeze(-1)  # (batch_size, 1, 1)
            single_batch = True
        elif timestamps.dim() == 2:
            timestamps = timestamps.unsqueeze(-1)  # (batch_size, seq_len, 1)
            single_batch = False
        else:
            single_batch = False
        
        components = []
        
        # Fourier component
        if self.dim_fourier > 0:
            fourier_proj = self.w1_fourier(timestamps)  # (batch_size, [seq_len,] dim_fourier)
            fourier_out = self.w2_fourier(fourier_proj)  # (batch_size, [seq_len,] dim_fourier)
            fourier_out = torch.sigmoid(self.fourier_weight) * fourier_out
            components.append(fourier_out)
        
        # Spline component
        if self.dim_spline > 0:
            spline_proj = self.w1_spline(timestamps)  # (batch_size, [seq_len,] dim_spline) 
            spline_out = self.w2_spline(spline_proj)  # (batch_size, [seq_len,] dim_spline)
            spline_out = torch.sigmoid(self.spline_weight) * spline_out
            components.append(spline_out)
        
        # Combine components
        if len(components) > 1:
            output = torch.cat(components, dim=-1)  # (batch_size, [seq_len,] time_dim)
        else:
            output = components[0]
        
        # Apply optional transformations
        if self.layer_norm:
            output = self.layernorm(output)
        
        if self.scale:
            output = output * self.scale_weight
        
        # Restore original shape if needed
        if single_batch and len(original_shape) == 1:
            output = output.squeeze(-2)  # Remove sequence dimension
        
        return output
    
    def get_config(self) -> dict:
        """Return configuration for reproducibility."""
        config = super().get_config()
        config.update({
            'p': self.dim_fourier / self.time_dim,
            'layer_norm': self.layer_norm,
            'scale': self.scale,
            'parameter_requires_grad': next(self.parameters()).requires_grad
        })
        return config
