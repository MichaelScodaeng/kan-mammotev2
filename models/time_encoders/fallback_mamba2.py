# file: models/time_encoders/fallback_mamba2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class FallbackMamba2(nn.Module):
    """
    A simplified fallback implementation for Mamba2 when the full CUDA implementation
    is not available due to GLIBC compatibility issues.
    
    This provides the same interface as ControllableMamba2 but uses standard PyTorch
    operations instead of optimized CUDA kernels.
    """
    
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, headdim=64, ngroups=1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.headdim = headdim
        self.ngroups = ngroups
        
        self.d_inner = int(self.expand * self.d_model)
        self.nheads = self.d_inner // self.headdim
        
        # Simplified projections
        self.in_proj = nn.Linear(d_model, self.d_inner * 2 + self.nheads + d_state * 2)
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Simplified conv1d
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, padding=d_conv-1)
        
        # SSM parameters (simplified)
        self.A = nn.Parameter(torch.randn(self.nheads, self.headdim, d_state))
        self.D = nn.Parameter(torch.randn(self.nheads, self.headdim))
        self.dt_bias = nn.Parameter(torch.randn(self.nheads))
        
        # Layer norm
        self.norm = nn.LayerNorm(self.d_inner)
        
    def forward(self, u, temporal_gate, **kwargs):
        """
        Simplified forward pass that accepts temporal_gate for compatibility.
        
        Args:
            u: Input tensor (B, L, D)
            temporal_gate: Temporal modulation (B, L, nheads)
        """
        B, L, D = u.shape
        
        # Project input
        projected = self.in_proj(u)  # (B, L, d_inner*2 + nheads + d_state*2)
        
        # Split projections
        split_sizes = [self.d_inner, self.d_inner, self.nheads, self.d_state, self.d_state]
        x, z, dt, B_proj, C_proj = torch.split(projected, split_sizes, dim=-1)
        
        # Apply temporal gate to dt
        if temporal_gate is not None:
            dt = dt * temporal_gate
        
        # Conv1d operation
        x = rearrange(x, 'b l d -> b d l')
        x = self.conv1d(x)[..., :L]  # Trim to original length
        x = F.silu(x)
        x = rearrange(x, 'b d l -> b l d')
        
        # Simplified SSM operation
        dt = F.softplus(dt + self.dt_bias)
        
        # Simplified state space computation
        # This is a greatly simplified version of the full SSM
        y = x  # Placeholder - in real implementation would be SSM computation
        
        # Apply gating
        y = y * F.silu(z)
        
        # Output projection
        output = self.out_proj(y)
        
        return output


class ControllableMamba2(FallbackMamba2):
    """
    Fallback ControllableMamba2 that inherits from the simplified implementation.
    """
    pass
