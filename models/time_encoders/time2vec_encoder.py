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
            # Use absolute time as fallback, but normalize it
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
        
        # Safety check: ensure input values are reasonable
        if torch.any(torch.isnan(input_tensor)) or torch.any(torch.isinf(input_tensor)):
            print("Warning: NaN or Inf detected in time input. Replacing with zeros.")
            input_tensor = torch.nan_to_num(input_tensor, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Clip extreme values to prevent numerical instability
        input_tensor = torch.clamp(input_tensor, min=-1e6, max=1e6)
        
        # Apply Time2Vec encoding
        encoded = self.time2vec_layer(input_tensor)
        
        # Safety check on output
        if torch.any(torch.isnan(encoded)) or torch.any(torch.isinf(encoded)):
            print("Warning: NaN or Inf detected in Time2Vec output. Applying layer norm.")
            encoded = torch.nn.functional.layer_norm(encoded, encoded.shape[-1:])
            encoded = torch.nan_to_num(encoded, nan=0.0, posinf=1.0, neginf=-1.0)
        
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
    
def t2v(tau, f, out_features, w, b, w0, b0, arg=None, dygmamba=True):
    """
    Core Time2Vec transformation function - FIXED VERSION.
    """
    # Input safety check
    if torch.any(torch.isnan(tau)) or torch.any(torch.isinf(tau)):
        print("Warning: NaN or Inf detected in tau input. Replacing with zeros.")
        tau = torch.nan_to_num(tau, nan=0.0, posinf=1e3, neginf=-1e3)
    
    # Clamp inputs to prevent extreme trigonometric values
    tau_safe = torch.clamp(tau, min=-100, max=100)
    
    # Compute periodic input
    periodic_input = torch.matmul(tau_safe, w) + b
    
    # FIX: Check for NaN/Inf BEFORE using periodic_input
    if torch.any(torch.isnan(periodic_input)) or torch.any(torch.isinf(periodic_input)):
        print("Warning: NaN or Inf detected in periodic input. Clamping values.")
        periodic_input = torch.nan_to_num(periodic_input, nan=0.0, posinf=10.0, neginf=-10.0)
    
    # Clamp to prevent extreme trigonometric inputs
    periodic_input = torch.clamp(periodic_input, min=-10, max=10)
    
    # Apply trigonometric function (now with clean input)
    if arg:
        v1 = f(periodic_input, arg)
    else:
        v1 = f(periodic_input)
    
    # Check v1 for any remaining NaN (shouldn't happen now)
    if torch.any(torch.isnan(v1)) or torch.any(torch.isinf(v1)):
        print("Warning: NaN in v1 after trigonometric function. This shouldn't happen!")
        v1 = torch.nan_to_num(v1, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Linear component
    v2 = torch.matmul(tau_safe, w0) + b0
    
    # Check v2 for NaN
    if torch.any(torch.isnan(v2)) or torch.any(torch.isinf(v2)):
        print("Warning: NaN or Inf detected in v2 (linear component).")
        v2 = torch.nan_to_num(v2, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Concatenate
    output = torch.cat([v1, v2], -1)
    
    # Final safety check before normalization
    if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
        print("Warning: NaN or Inf detected in concatenated output.")
        output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Apply normalization for DyGMamba with enhanced safety
    if dygmamba:
        # Check if output has any zero variance (which causes NaN in layer norm)
        output_std = torch.std(output, dim=-1, keepdim=True)
        if torch.any(output_std < 1e-8):
            print("Warning: Near-zero variance detected. Adding small noise for stability.")
            output = output + torch.randn_like(output) * 1e-6
        
        # Apply layer normalization with epsilon for numerical stability
        try:
            output = torch.nn.functional.layer_norm(output, output.shape[-1:], eps=1e-6)
        except Exception as e:
            print(f"Warning: Layer norm failed: {e}. Applying manual normalization.")
            mean = torch.mean(output, dim=-1, keepdim=True)
            std = torch.std(output, dim=-1, keepdim=True) + 1e-6
            output = (output - mean) / std
        
        # Final NaN check after normalization
        if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
            print("Warning: NaN or Inf detected after normalization.")
            output = torch.nan_to_num(output, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return output

class SineActivation(nn.Module):
    """Time2Vec with sine activation - BACKWARD COMPATIBLE."""
    
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        # KEEP original initialization for backward compatibility with trained models
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    """Time2Vec with cosine activation - BACKWARD COMPATIBLE."""
    
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        # KEEP original initialization for backward compatibility with trained models
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(1))
        self.w = nn.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)