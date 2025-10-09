"""
KAN-MAMMOTE Lite: Simplified version for stateless temporal encoding

This version removes Mamba2 sequence modeling and focuses on dual-stream
encoding with KMOTE and SM-Kernel, making it suitable for attention-based
models like TGAT where temporal encodings are computed independently per
timestamp rather than as sequences.

Key differences from full KAN-MAMMOTE:
- No Mamba2 sequence modeling (stateless)
- Simpler fusion mechanism
- Lower parameter count (~10K vs ~50K)
- Faster inference
- Better for TGAT, JODIE, TGN (attention-based models)

When to use:
- TGAT, JODIE, TGN: Use KAN-MAMMOTE Lite ✓
- DyGFormer, GraphMixer: Use full KAN-MAMMOTE ✓
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .k_mote import KMOTE
from .sm_kernel import SMKernelLayer


class KAN_MAMMOTE_Lite(nn.Module):
    """
    Lightweight KAN-MAMMOTE without Mamba for stateless temporal encoding.
    
    Architecture:
    1. K-MOTE: Kolmogorov-Arnold encoding with wavelets (for absolute time)
    2. SM-Kernel: Spectral mixture kernel (for relative time / delta_t)
    3. Simple fusion: Linear projection + GELU
    
    Args:
        embedding_dim: Output dimension
        num_mixtures: Number of Gaussian mixtures in SM-Kernel
        wavelet_type: Type of wavelet ('shock', 'haar', 'db4', etc.)
        use_dual_stream: If True, use both t_abs and t_rel; if False, only t_rel
    """
    
    def __init__(
        self, 
        embedding_dim: int,
        num_mixtures: int = 12,
        wavelet_type: str = 'shock',
        use_dual_stream: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_mixtures = num_mixtures
        self.wavelet_type = wavelet_type
        self.use_dual_stream = use_dual_stream
        
        # K-MOTE encoder for absolute time (if dual-stream enabled)
        if use_dual_stream:
            self.k_mote = KMOTE(
                input_dim=1, 
                output_dim=embedding_dim // 2,  # Half dimension for each stream
                wavelet_type=wavelet_type
            )
        
        # SM-Kernel for relative time (delta_t) - always used
        self.sm_kernel = SMKernelLayer(
            num_mixtures=num_mixtures, 
            input_dim=1
        )
        
        # Fusion layer
        if use_dual_stream:
            # Combine K-MOTE output (embedding_dim//2) and SM-Kernel (num_mixtures)
            fusion_input_dim = (embedding_dim // 2) + num_mixtures
        else:
            # Only SM-Kernel
            fusion_input_dim = num_mixtures
        
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        print(f"Initialized KAN-MAMMOTE Lite:")
        print(f"  embedding_dim: {embedding_dim}")
        print(f"  num_mixtures: {num_mixtures}")
        print(f"  wavelet_type: {wavelet_type}")
        print(f"  use_dual_stream: {use_dual_stream}")
        print(f"  parameters: ~{self.count_parameters():,}")
    
    def initialize_sm_kernel(self, delta_t_sample: torch.Tensor):
        """Initialize SM-Kernel from data statistics"""
        self.sm_kernel.initialize_from_data(delta_t_sample)
    
    def forward(self, timestamps: torch.Tensor = None, t_abs: torch.Tensor = None, t_rel: torch.Tensor = None, debug: bool = False) -> torch.Tensor:
        """
        Forward pass for stateless temporal encoding with backward compatibility.
        
        Supports two calling conventions:
        1. Legacy interface: forward(timestamps=...) - used by DyGFormer/other models
        2. New interface: forward(t_abs=..., t_rel=...) - used by TGAT with wrapper
        
        Args:
            timestamps: Legacy interface - typically relative time (delta_t)
            t_abs: Absolute timestamps, shape (B, S, 1) or (B, 1)
            t_rel: Relative timestamps (delta_t), shape (B, S, 1) or (B, 1)
            debug: Enable detailed debugging output
        
        Returns:
            Temporal embeddings, shape (B, S, embedding_dim) or (B, embedding_dim)
        """
        if debug or hasattr(self, '_debug_mode'):
            print(f"\n{'='*50}")
            print(f"🔍 KAN-MAMMOTE Lite DEBUG")
            print(f"{'='*50}")
        
        # Handle different calling conventions
        if timestamps is not None:
            # Legacy interface: Use timestamps as relative time
            t_rel = timestamps
            # Create dummy absolute time if needed
            if self.use_dual_stream:
                t_abs = torch.zeros_like(timestamps) + 1e-6
        elif t_rel is not None:
            # New interface: t_rel already provided
            pass
        else:
            raise ValueError("Either 'timestamps' or 't_rel' must be provided")
        
        if t_rel is None:
            raise ValueError("KAN-MAMMOTE Lite requires t_rel (relative time / delta_t)")
        
        if debug or hasattr(self, '_debug_mode'):
            print(f"📊 INPUT SHAPES:")
            if t_abs is not None:
                print(f"   t_abs shape: {t_abs.shape}")
            print(f"   t_rel shape: {t_rel.shape}")
        
        # Encode relative time with SM-Kernel (always used)
        v_k = self.sm_kernel(t_rel)  # (B, [S], num_mixtures)
        
        if debug or hasattr(self, '_debug_mode'):
            print(f"🎯 SM-KERNEL OUTPUT: {v_k.shape}")
        
        if self.use_dual_stream and t_abs is not None:
            # Encode absolute time with K-MOTE
            u_k = self.k_mote(t_abs)  # (B, [S], embedding_dim//2)
            
            if debug or hasattr(self, '_debug_mode'):
                print(f"🎯 K-MOTE OUTPUT: {u_k.shape}")
            
            # Concatenate both streams
            combined = torch.cat([u_k, v_k], dim=-1)  # (B, [S], fusion_input_dim)
        else:
            # Use only relative time encoding
            combined = v_k
        
        # Fuse through MLP
        output = self.fusion(combined)  # (B, [S], embedding_dim)
        
        if debug or hasattr(self, '_debug_mode'):
            print(f"🎯 FINAL OUTPUT: {output.shape}")
            print(f"{'='*50}\n")
        
        return output
    
    def enable_debug_mode(self):
        """Enable persistent debug mode"""
        self._debug_mode = True
        print("🔍 KAN-MAMMOTE Lite Debug Mode ENABLED")
    
    def disable_debug_mode(self):
        """Disable persistent debug mode"""
        if hasattr(self, '_debug_mode'):
            delattr(self, '_debug_mode')
        print("🔍 KAN-MAMMOTE Lite Debug Mode DISABLED")
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_config(self):
        """Get configuration for reproducibility"""
        return {
            'embedding_dim': self.embedding_dim,
            'num_mixtures': self.num_mixtures,
            'wavelet_type': self.wavelet_type,
            'use_dual_stream': self.use_dual_stream,
            'parameters': self.count_parameters()
        }


def create_kan_mammote_lite(
    embedding_dim: int,
    num_mixtures: int = 12,
    wavelet_type: str = 'shock',
    use_dual_stream: bool = True,
    device: str = 'cpu'
):
    """
    Factory function to create KAN-MAMMOTE Lite.
    
    Args:
        embedding_dim: Output dimension
        num_mixtures: Number of Gaussian mixtures
        wavelet_type: Wavelet type for K-MOTE
        use_dual_stream: Whether to use both t_abs and t_rel
        device: Device to place model on
    
    Returns:
        KAN_MAMMOTE_Lite model
    """
    model = KAN_MAMMOTE_Lite(
        embedding_dim=embedding_dim,
        num_mixtures=num_mixtures,
        wavelet_type=wavelet_type,
        use_dual_stream=use_dual_stream
    )
    return model.to(device)


# Alias for backward compatibility
KANMAMMOTELite = KAN_MAMMOTE_Lite
