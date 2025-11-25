"""
LeTE (Learnable Time Encoding)
The implementation should follow models/time_encoders/LeTE_original.py

Advanced time encoding using mixture of Fourier and spline components.
Based on the LeTE paper implementation with adaptations for the KMM framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from .base_encoder import BaseTimeEncoder


class FourierSeries(nn.Module):
    """
    Fourier series component following LeTE paper implementation.
    
    Args:
        dim_fourier (int): Dimension of both the input and output features.
        grid_size_fourier (int): Number of frequency components (excluding constant term). Defaults to 5.
    """
    
    def __init__(self, dim_fourier: int, grid_size_fourier: int = 5):
        super().__init__()
        self.dim_fourier = dim_fourier
        self.grid_size_fourier = grid_size_fourier

        # fourier_weight shape: (2, dim_fourier, dim_fourier, grid_size_fourier)
        #   - 2 corresponds to cosine and sine parts, respectively
        #   - The layer outputs dim_fourier features given dim_fourier inputs
        self.fourier_weight = torch.nn.Parameter(
            torch.randn(2, self.dim_fourier, self.dim_fourier, grid_size_fourier) /
            (np.sqrt(self.dim_fourier) * np.sqrt(self.grid_size_fourier))
        )

        # Bias term, one per output dimension
        self.bias = nn.Parameter(torch.zeros(self.dim_fourier))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        out_shape = original_shape[0:-1] + (self.dim_fourier,)

        # Flatten everything except the last dim
        x = x.reshape(-1, self.dim_fourier)  # (N, dim_fourier)

        # Frequency indices k = 1..grid_size_fourier
        k = torch.arange(1, self.grid_size_fourier + 1, device=x.device)
        k = k.reshape(1, 1, 1, self.grid_size_fourier)  # shape: (1,1,1,K)

        # Reshape x to broadcast with k
        x_reshaped = x.reshape(x.shape[0], 1, x.shape[1], 1)  # (N,1,dim_fourier,1)

        # Compute cos(k * x) and sin(k * x)
        c = torch.cos(k * x_reshaped)  # (N,1,dim_fourier,K)
        s = torch.sin(k * x_reshaped)  # (N,1,dim_fourier,K)

        # Sum up the contributions with the learned Fourier coefficients
        # fourier_weight[0:1] -> cosine part, fourier_weight[1:2] -> sine part
        y = torch.sum(c * self.fourier_weight[0:1], dim=(-2, -1))
        y += torch.sum(s * self.fourier_weight[1:2], dim=(-2, -1))

        # Add bias
        y += self.bias

        # Reshape back to original shape
        y = y.reshape(out_shape)
        return y


class Spline(nn.Module):
    """
    B-spline implementation following LeTE paper.
    
    Args:
        dim_spline (int): Dimension of the spline input (and typically output).
        grid_size_spline (int): Number of spline knots (excluding boundary). Defaults to 5.
        order_spline (int): Spline order (degree). Defaults to 3.
        grid_range (list): Value range of the grid. Defaults to [-1, 1].
    """
    def __init__(self, dim_spline: int, grid_size_spline: int = 5, order_spline: int = 3, grid_range: list = [-1, 1]):
        super().__init__()
        self.dim_spline = dim_spline
        self.grid_size_spline = grid_size_spline
        self.order_spline = order_spline

        # Compute grid spacing
        h = (grid_range[1] - grid_range[0]) / float(self.grid_size_spline)

        # Build the grid for each dimension
        grid = torch.arange(-self.order_spline, self.grid_size_spline + self.order_spline + 1)
        grid = grid * h + grid_range[0]
        grid = grid.expand(self.dim_spline, -1).contiguous()
        self.register_buffer("grid", grid)


        #This follows original LeTE implementation but it causes bug
        '''
        # Base weight for the linear+activation branch
        self.base_weight = nn.Parameter(torch.Tensor(self.dim_spline, self.dim_spline))

        # Spline coefficients
        self.spline_weight = nn.Parameter(torch.Tensor(self.dim_spline, self.dim_spline, self.grid_size_spline + self.order_spline))
        '''
        self.base_weight = nn.Parameter(
            torch.randn(self.dim_spline, self.dim_spline) / np.sqrt(self.dim_spline)
        )

        self.spline_weight = nn.Parameter(
            torch.randn(self.dim_spline, self.dim_spline, self.grid_size_spline + self.order_spline) /
            (np.sqrt(self.dim_spline) * np.sqrt(self.grid_size_spline + self.order_spline))
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.dim_spline)  # flatten all but the last dimension

        # Base branch: tanh + linear
        base_output = nn.functional.linear(torch.tanh(x), self.base_weight)  # shape: (N, dim_spline)

        # If the input batch is empty, return a zero-like tensor
        if x.size(0) == 0:
            spline_output = torch.zeros_like(base_output)
        else:
            # Evaluate B-spline basis
            # b_splines(x).shape -> (N, dim_spline, grid_size_spline + order_spline)
            # Flatten for linear
            b_splines_val = self.b_splines(x).view(x.size(0), -1)

            # Reshape spline_weight for linear operation
            w = self.spline_weight.view(self.dim_spline, -1)
            spline_output = nn.functional.linear(b_splines_val, w)

        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.dim_spline)
        return output
    
    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # A: (dim_spline, batch_size, grid_size_spline + order_spline)
        A = self.b_splines(x).transpose(0, 1)
        # B: (dim_spline, batch_size, dim_spline)
        B = y.transpose(0, 1)

        # Compute pseudo-inverse of A
        A_pinv = torch.pinverse(A)  # shape: (dim_spline, grid_size_spline+order_spline, batch_size)

        # Solve the least squares problem using the pseudo-inverse
        # shape: (dim_spline, grid_size_spline+order_spline, batch_size)
        solution = torch.bmm(A_pinv, B)

        # Permute to (batch_size, dim_spline, grid_size_spline+order_spline) -> (dim_spline, dim_spline, grid_size_spline+order_spline)
        result = solution.permute(2, 0, 1).contiguous()
        return result
    
    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        # grid: (dim_spline, grid_size_spline + 2*order_spline + 1)
        grid = self.grid
        x = x.unsqueeze(-1)  # shape: (N, dim_spline, 1)

        # bases will mark where x lies between [grid_i, grid_{i+1})
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # shape: (N, dim_spline, ?)

        # Recursively elevate the basis to higher spline orders
        for k in range(1, self.order_spline + 1):
            # Segment lengths in the left and right directions
            left_num = (x - grid[:, :-(k + 1)])
            left_den = (grid[:, k:-1] - grid[:, :-(k + 1)])

            right_num = (grid[:, k + 1:] - x)
            right_den = (grid[:, k + 1:] - grid[:, 1:-k])

            bases = (left_num / left_den) * bases[:, :, :-1] + (right_num / right_den) * bases[:, :, 1:]

        return bases.contiguous()


class LeTE(BaseTimeEncoder):
    """
    Learnable Time Encoding with Fourier and spline components.
    
    Combines periodic (Fourier) and local (spline) representations to capture
    diverse temporal patterns in absolute time.
    """
    
    def __init__(self, time_dim: int, p: float = 0.5, layer_norm: bool = True, 
                 scale: bool = True, parameter_requires_grad: bool = True, device: str = 'cuda'):
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
        #print("lete kuay")
        # Fourier component
        if self.dim_fourier > 0:
            self.w1_fourier = nn.Linear(1, self.dim_fourier)
            # Initialize with geometric progression
            fourier_vals = 1.0 / (10 ** np.linspace(0, 9, self.dim_fourier, dtype=np.float32))
            self.w1_fourier.weight = nn.Parameter(torch.from_numpy(fourier_vals).reshape(self.dim_fourier, -1))
            self.w1_fourier.bias = nn.Parameter(torch.zeros(self.dim_fourier))
            self.w2_fourier = FourierSeries(dim_fourier=self.dim_fourier, grid_size_fourier=5)
        
        # Spline component  
        if self.dim_spline > 0:
            self.w1_spline = nn.Linear(1, self.dim_spline)
            spline_vals = 1.0 / (10 ** np.linspace(0, 9, self.dim_spline, dtype=np.float32))
            self.w1_spline.weight = nn.Parameter(torch.from_numpy(spline_vals).reshape(self.dim_spline, -1))
            self.w1_spline.bias = nn.Parameter(torch.zeros(self.dim_spline))
            self.w2_spline = Spline(dim_spline=self.dim_spline, grid_size_spline=5)
        
        # Optional layer normalization
        if self.layer_norm:
            self.layernorm = nn.LayerNorm(time_dim)
        
        # Optional learnable scaling
        if self.scale:
            self.scale_weight = nn.Parameter(torch.ones(time_dim))
        
        # Control gradient computation
        if not parameter_requires_grad:
            if self.dim_fourier > 0:
                self.w1_fourier.weight.requires_grad = False
                self.w1_fourier.bias.requires_grad = False
                self.w2_fourier.requires_grad = False
            if self.dim_spline > 0:
                self.w1_spline.weight.requires_grad = False
                self.w1_spline.bias.requires_grad = False
                self.w2_spline.requires_grad = False
            if self.scale:
                self.scale_weight.requires_grad = False

    def forward(self, timestamps: torch.Tensor = None, time_deltas: torch.Tensor = None, 
                t_abs: torch.Tensor = None, t_rel: torch.Tensor = None, debug: bool = False) -> torch.Tensor:
        """
        Encode timestamps using learnable Fourier and spline components.
        Supports both single-stream (timestamps) and dual-stream (t_abs, t_rel) interfaces.
        
        Args:
            timestamps: Legacy interface - tensor of shape (batch_size,) or (batch_size, seq_len)
            time_deltas: Not used in LeTE, kept for interface compatibility
            t_abs: Absolute timestamps (dual-stream interface)
            t_rel: Relative timestamps (dual-stream interface). If provided, this is preferred.
        
        Returns:
            Time encodings of shape (batch_size, time_dim) or (batch_size, seq_len, time_dim)
        """
        # Handle different input interfaces (prefer relative time if available)
        #print("lete kuay forward")
        if t_rel is not None:
            input_tensor = t_rel.squeeze(-1) if t_rel.dim() > 2 else t_rel
            #print("lete kuay t_rel")
        elif timestamps is not None:
            input_tensor = timestamps
            #print("lete kuay timestamps")
        elif t_abs is not None:
            raise ValueError("Got t_abs not t_rel")
            input_tensor = t_abs.squeeze(-1) if t_abs.dim() > 2 else t_abs
        else:
            raise ValueError("One of 't_rel', 'timestamps', or 't_abs' must be provided")
        
        # Handle input shapes
        original_shape = input_tensor.shape
        
        # Ensure proper dimensions for processing
        if input_tensor.dim() == 1:
            input_tensor = input_tensor.unsqueeze(-1)  # (batch_size, 1)
            single_batch = True
        elif input_tensor.dim() == 2 and input_tensor.shape[-1] != 1:
            input_tensor = input_tensor.unsqueeze(-1)  # (batch_size, seq_len, 1)
            single_batch = False
        elif input_tensor.dim() == 2 and input_tensor.shape[-1] == 1:
            single_batch = False
        elif input_tensor.dim() == 3:
            single_batch = False
        else:
            raise ValueError(f"Unexpected input tensor shape: {input_tensor.shape}")
        
        # Fourier component
        if self.dim_fourier > 0:
            proj_fourier = self.w1_fourier(input_tensor)  # (batch_size, [seq_len,] dim_fourier)
            output_fourier = self.w2_fourier(proj_fourier)  # (batch_size, [seq_len,] dim_fourier)
        else:
            # If p=1.0, dim_spline=0 => no Spline part, so set an empty tensor
            output_fourier = torch.zeros(*input_tensor.shape[:-1], 0, device=input_tensor.device)

        # Spline component
        if self.dim_spline > 0:
            proj_spline = self.w1_spline(input_tensor)  # (batch_size, [seq_len,] dim_spline)
            output_spline = self.w2_spline(proj_spline)  # (batch_size, [seq_len,] dim_spline)
        else:
            # If p=0.0, dim_fourier=0 => no Fourier part, so set an empty tensor
            output_spline = torch.zeros(*input_tensor.shape[:-1], 0, device=input_tensor.device)

        # Concatenate Fourier and Spline encodings along the last dimension
        output = torch.cat((output_fourier, output_spline), dim=-1)  # (batch_size, [seq_len,] time_dim)
        
        # Apply optional transformations
        if self.layer_norm:
            output = self.layernorm(output)
        
        if self.scale:
            output = output * self.scale_weight
        
        # Handle output shape for compatibility
        if single_batch and len(original_shape) == 1:
            output = output.squeeze(-2) if output.dim() > 2 else output  # Remove sequence dimension if added
        
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
