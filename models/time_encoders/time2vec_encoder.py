import torch
import torch.nn as nn
import numpy as np
import math
from .base_encoder import BaseTimeEncoder
from typing import Optional

class Time2VecEncoder(BaseTimeEncoder):
    """
    Time2Vec: Learning a Vector Representation of Time
    
    Based on the Time2Vec paper: https://arxiv.org/abs/1907.05321
    Compatible with the KAN-MAMMOTE framework.
    """
    
    def __init__(self, time_dim: int, device: str = 'cpu', activation: str = 'sin'):
        """
        Initialize Time2Vec encoder.
        
        Args:
            time_dim: Output dimension of time encoding
            device: Device to place the encoder on
            activation: Activation function ('sin' or 'cos')
        """
        super().__init__(time_dim, device)
        
        self.activation = activation
        
        if activation == "sin":
            self.time2vec_layer = SineActivation(1, time_dim)
        elif activation == "cos":
            self.time2vec_layer = CosineActivation(1, time_dim)
        else:
            raise ValueError(f"Unsupported activation: {activation}. Use 'sin' or 'cos'")
        
        # Move to device
        self.to_device(device)
    
    def forward(self, timestamps: torch.Tensor = None, time_deltas: Optional[torch.Tensor] = None,
                t_abs: torch.Tensor = None, t_rel: torch.Tensor = None) -> torch.Tensor:
        """
        Encode timestamps using Time2Vec.
        Supports both single-stream (timestamps) and dual-stream (t_abs, t_rel) interfaces.
        
        Args:
            timestamps: Tensor of shape (batch_size,) or (batch_size, seq_len) or (batch_size, seq_len, 1)
            time_deltas: Optional, not used in Time2Vec but kept for interface compatibility
            t_abs: Absolute timestamps (dual-stream interface)
            t_rel: Relative timestamps (dual-stream interface). If provided, this is preferred.
            
        Returns:
            Time embeddings of shape (batch_size, time_dim) or (batch_size, seq_len, time_dim)
        """
        # Handle different input interfaces (prefer relative time if available)
        if t_rel is not None:
            input_tensor = t_rel.squeeze(-1) if t_rel.dim() > 2 else t_rel
        elif timestamps is not None:
            input_tensor = timestamps
        elif t_abs is not None:
            raise ValueError("Got t_abs not t_rel")
            input_tensor = t_abs.squeeze(-1) if t_abs.dim() > 2 else t_abs
        else:
            raise ValueError("One of 't_rel', 'timestamps', or 't_abs' must be provided")
        
        # Handle input shapes
        original_shape = input_tensor.shape
        
        if input_tensor.dim() == 1:
            # (batch_size,) -> (batch_size, 1)
            input_tensor = input_tensor.unsqueeze(-1)
        elif input_tensor.dim() == 2:
            # (batch_size, seq_len) -> (batch_size, seq_len, 1)
            input_tensor = input_tensor.unsqueeze(-1)
        elif input_tensor.dim() == 3 and input_tensor.shape[-1] == 1:
            # Already correct shape (batch_size, seq_len, 1)
            pass
        else:
            raise ValueError(f"Invalid timestamp shape: {input_tensor.shape}")
        
        # Apply Time2Vec encoding
        encoded = self.time2vec_layer(input_tensor)
        
        return encoded
    
    def get_config(self) -> dict:
        """Return configuration for reproducibility."""
        config = super().get_config()
        config.update({
            'activation': self.activation
        })
        return config

class Model(nn.Module):
    """Legacy Model class - deprecated, use Time2VecEncoder instead."""
    
    def __init__(self, activation, hidden_dim):
        super(Model, self).__init__()
        print("Warning: Using deprecated Model class. Use Time2VecEncoder instead.")
        
        if activation == "sin":
            self.l1 = SineActivation(1, hidden_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, hidden_dim)
        
        self.fc1 = nn.Linear(hidden_dim, 2)
    
    def forward(self, x):
        x = self.l1(x)
        x = self.fc1(x)
        return x
    
def t2v(tau, f, out_features, w, b, w0, b0, arg=None,dygmamba=True):
    """
    Core Time2Vec transformation function.
    """
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    output = torch.cat([v1, v2], -1)
    if dygmamba:
        return torch.nn.functional.layer_norm(output, output.shape[-1:])
    else:
        return output

class SineActivation(nn.Module):
    """Time2Vec with sine activation."""
    
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    """Time2Vec with cosine activation."""
    
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)