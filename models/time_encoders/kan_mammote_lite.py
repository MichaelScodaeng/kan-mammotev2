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


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion between absolute and relative time features"""
    
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        
        # Query from absolute time features, Key+Value from relative time features
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, abs_features: torch.Tensor, rel_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            abs_features: Absolute time features (B, S, dim)
            rel_features: Relative time features (B, S, dim)
        Returns:
            fused_features: Cross-attended features (B, S, dim)
        """
        B, S, _ = abs_features.shape
        
        # Project to Q, K, V
        Q = self.q_proj(abs_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, S, d)
        K = self.k_proj(rel_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, S, d)
        V = self.v_proj(rel_features).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)   # (B, H, S, d)
        
        # Scaled dot-product attention
        attn_scores = (Q @ K.transpose(-2, -1)) * self.scale  # (B, H, S, S)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        attended = attn_weights @ V  # (B, H, S, d)
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous().view(B, S, self.dim)  # (B, S, dim)
        output = self.out_proj(attended)
        
        return output


class WeightedSumFusion(nn.Module):
    """Learnable weighted sum fusion between absolute and relative time features"""
    
    def __init__(self, abs_dim: int, rel_dim: int, output_dim: int):
        super().__init__()
        self.abs_dim = abs_dim
        self.rel_dim = rel_dim
        self.output_dim = output_dim
        
        # Project both features to the same dimension if needed
        self.abs_proj = nn.Linear(abs_dim, output_dim) if abs_dim != output_dim else nn.Identity()
        self.rel_proj = nn.Linear(rel_dim, output_dim) if rel_dim != output_dim else nn.Identity()
        
        # Learnable gating weights
        self.gate = nn.Sequential(
            nn.Linear(abs_dim + rel_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),  # Two weights: one for abs, one for rel
            nn.Softmax(dim=-1)
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, abs_features: torch.Tensor, rel_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            abs_features: Absolute time features (B, S, abs_dim)
            rel_features: Relative time features (B, S, rel_dim)
        Returns:
            fused_features: Weighted sum features (B, S, output_dim)
        """
        # Project to common dimension
        abs_proj = self.abs_proj(abs_features)  # (B, S, output_dim)
        rel_proj = self.rel_proj(rel_features)  # (B, S, output_dim)
        
        # Compute gating weights based on concatenated input features
        gate_input = torch.cat([abs_features, rel_features], dim=-1)  # (B, S, abs_dim + rel_dim)
        weights = self.gate(gate_input)  # (B, S, 2)
        
        # Apply weighted sum
        w_abs = weights[..., 0:1]  # (B, S, 1)
        w_rel = weights[..., 1:2]  # (B, S, 1)
        
        weighted_sum = w_abs * abs_proj + w_rel * rel_proj  # (B, S, output_dim)
        
        # Final projection
        output = self.output_proj(weighted_sum)
        
        return output


class KAN_MAMMOTE_Lite(nn.Module):
    """
    Enhanced KAN-MAMMOTE Lite with dual K-MOTE support and configurable fusion.
    
    Architecture:
    1. K-MOTE (Absolute): Kolmogorov-Arnold encoding with wavelets for absolute time
    2. K-MOTE (Relative) OR SM-Kernel: For relative time / delta_t encoding
    3. Configurable fusion: Linear+MLP, Cross-Attention, or Weighted Sum
    
    Args:
        embedding_dim: Output dimension
        expert_dim: Dimension for each K-MOTE expert (should be multiple of 16)
        num_mixtures: Number of Gaussian mixtures in SM-Kernel (if using SM-Kernel)
        wavelet_type: Type of wavelet ('shock', 'haar', 'db4', etc.)
        use_dual_kmote: If True, use K-MOTE for both abs and rel; if False, use SM-Kernel for rel
        fusion_type: 'linear_mlp', 'cross_attention', or 'weighted_sum'
        use_dual_stream: If True, use both t_abs and t_rel; if False, only t_rel
    """
    
    def __init__(
        self, 
        embedding_dim: int,
        expert_dim: int = None,
        num_mixtures: int = 12,
        wavelet_type: str = 'shock',
        use_dual_kmote: bool = True,
        fusion_type: str = 'linear_mlp',
        use_dual_stream: bool = True,
        **kwargs
    ):
        super().__init__()
        
        # Set expert_dim default to embedding_dim if not provided
        if expert_dim is None:
            expert_dim = embedding_dim
        
        # Enforce architectural consistency like in KAN-MAMMOTE
        if use_dual_kmote and num_mixtures != expert_dim:
            print(f"⚠️  WARNING: num_mixtures ({num_mixtures}) != expert_dim ({expert_dim})")
            print(f"🔧 Setting num_mixtures = expert_dim = {expert_dim} for architectural consistency")
            num_mixtures = expert_dim
        
        # Validate fusion type
        if fusion_type not in ['linear_mlp', 'cross_attention', 'weighted_sum']:
            raise ValueError(f"fusion_type must be 'linear_mlp', 'cross_attention', or 'weighted_sum', got {fusion_type}")
        
        self.embedding_dim = embedding_dim
        self.expert_dim = expert_dim
        self.num_mixtures = num_mixtures
        self.wavelet_type = wavelet_type
        self.use_dual_kmote = use_dual_kmote
        self.fusion_type = fusion_type
        self.use_dual_stream = use_dual_stream
        
        # K-MOTE encoder for absolute time (if dual-stream enabled)
        if use_dual_stream:
            self.k_mote_abs = KMOTE(
                input_dim=1, 
                output_dim=expert_dim,
                wavelet_type=wavelet_type
            )
        else:
            self.k_mote_abs = None
        
        # Choose between dual K-MOTE or SM-Kernel for relative time
        if use_dual_kmote:
            print("🔧 Using K-MOTE for relative time encoding (dual K-MOTE mode)")
            self.k_mote_rel = KMOTE(
                input_dim=1, 
                output_dim=expert_dim,
                wavelet_type=wavelet_type
            )
            self.sm_kernel = None
            rel_feature_dim = expert_dim
        else:
            print("🔧 Using SM-Kernel for relative time encoding")
            self.k_mote_rel = None
            self.sm_kernel = SMKernelLayer(
                num_mixtures=num_mixtures, 
                input_dim=1
            )
            rel_feature_dim = num_mixtures
        
        # Configure fusion based on fusion_type and dual_stream setting
        if use_dual_stream:
            if fusion_type == 'cross_attention':
                # Cross-attention fusion requires same dimension for both streams
                if use_dual_kmote:
                    # Both streams have expert_dim, use cross-attention directly
                    self.fusion = CrossAttentionFusion(dim=expert_dim, num_heads=8)
                    fusion_output_dim = expert_dim
                else:
                    # Project rel_features to expert_dim for cross-attention
                    self.rel_proj = nn.Linear(rel_feature_dim, expert_dim)
                    self.fusion = CrossAttentionFusion(dim=expert_dim, num_heads=8)
                    fusion_output_dim = expert_dim
                    
            elif fusion_type == 'weighted_sum':
                # Weighted sum fusion between abs and rel features
                self.fusion = WeightedSumFusion(
                    abs_dim=expert_dim, 
                    rel_dim=rel_feature_dim, 
                    output_dim=expert_dim
                )
                fusion_output_dim = expert_dim
                    
            else:  # linear_mlp
                # Concatenate and project through MLP
                fusion_input_dim = expert_dim + rel_feature_dim
                self.fusion = nn.Sequential(
                    nn.Linear(fusion_input_dim, expert_dim),
                    nn.LayerNorm(expert_dim),
                    nn.GELU(),
                    nn.Linear(expert_dim, expert_dim)
                )
                fusion_output_dim = expert_dim
        else:
            # Single stream (only relative time)
            if fusion_type in ['cross_attention', 'weighted_sum']:
                print(f"⚠️  {fusion_type} requires dual stream, falling back to linear_mlp")
                fusion_type = 'linear_mlp'
            
            # Project relative features to expert_dim
            self.fusion = nn.Sequential(
                nn.Linear(rel_feature_dim, expert_dim),
                nn.LayerNorm(expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, expert_dim)
            )
            fusion_output_dim = expert_dim
        
        # Output projection to embedding_dim if needed
        if fusion_output_dim != embedding_dim:
            self.output_projection = nn.Sequential(
                nn.Linear(fusion_output_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        else:
            self.output_projection = nn.Identity()
        
        print(f"Initialized Enhanced KAN-MAMMOTE Lite:")
        print(f"  embedding_dim: {embedding_dim}")
        print(f"  expert_dim: {expert_dim}")
        print(f"  num_mixtures: {num_mixtures}")
        print(f"  wavelet_type: {wavelet_type}")
        print(f"  use_dual_kmote: {use_dual_kmote}")
        print(f"  fusion_type: {fusion_type}")
        print(f"  use_dual_stream: {use_dual_stream}")
        print(f"  parameters: ~{self.count_parameters():,}")
    
    def initialize_sm_kernel(self, delta_t_sample: torch.Tensor):
        """Initialize SM-Kernel from data statistics (if using SM-kernel mode)"""
        if self.sm_kernel is not None:
            self.sm_kernel.initialize_from_data(delta_t_sample)
        else:
            print("INFO: Skipping SM-kernel initialization (using dual K-MOTE mode)")
    
    def forward(self, timestamps: torch.Tensor = None, t_abs: torch.Tensor = None, t_rel: torch.Tensor = None, debug: bool = False) -> torch.Tensor:
        """
        Enhanced forward pass supporting dual K-MOTE and configurable fusion.
        
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
            print(f"\n{'='*60}")
            print(f"🔍 Enhanced KAN-MAMMOTE Lite DEBUG")
            print(f"{'='*60}")
        
        # Handle different calling conventions
        if timestamps is not None:
            # Legacy interface: Use timestamps as relative time
            t_rel = timestamps
            # Create dummy absolute time if needed
            if self.use_dual_stream and self.k_mote_abs is not None:
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
            print(f"🔧 CONFIGURATION:")
            print(f"   use_dual_kmote: {self.use_dual_kmote}")
            print(f"   fusion_type: {self.fusion_type}")
            print(f"   use_dual_stream: {self.use_dual_stream}")
        
        # ===== Get relative time features =====
        if self.use_dual_kmote:
            # Use K-MOTE for relative time
            v_k = self.k_mote_rel(t_rel)  # (B, [S], expert_dim)
            if debug or hasattr(self, '_debug_mode'):
                print(f"🎯 K-MOTE REL OUTPUT: {v_k.shape}")
        else:
            # Use SM-Kernel for relative time
            v_k = self.sm_kernel(t_rel)  # (B, [S], num_mixtures)
            if debug or hasattr(self, '_debug_mode'):
                print(f"🎯 SM-KERNEL OUTPUT: {v_k.shape}")
        
        # ===== Handle dual stream vs single stream =====
        if self.use_dual_stream and self.k_mote_abs is not None and t_abs is not None:
            # Encode absolute time with K-MOTE
            u_k = self.k_mote_abs(t_abs)  # (B, [S], expert_dim)
            
            if debug or hasattr(self, '_debug_mode'):
                print(f"🎯 K-MOTE ABS OUTPUT: {u_k.shape}")
            
            # ===== Apply fusion based on fusion_type =====
            if self.fusion_type == 'cross_attention':
                # Project rel_features to expert_dim if needed
                if not self.use_dual_kmote:
                    v_k = self.rel_proj(v_k)  # Project to expert_dim
                    if debug or hasattr(self, '_debug_mode'):
                        print(f"🔧 REL PROJECTED: {v_k.shape}")
                
                # Cross-attention fusion: abs_features query, rel_features key+value
                fused_output = self.fusion(u_k, v_k)  # (B, [S], expert_dim)
                
                if debug or hasattr(self, '_debug_mode'):
                    print(f"🎯 CROSS-ATTENTION OUTPUT: {fused_output.shape}")
                    
            elif self.fusion_type == 'weighted_sum':
                # Weighted sum fusion: learnable gating between abs and rel features
                fused_output = self.fusion(u_k, v_k)  # (B, [S], expert_dim)
                
                if debug or hasattr(self, '_debug_mode'):
                    print(f"🎯 WEIGHTED-SUM OUTPUT: {fused_output.shape}")
                    
            else:  # linear_mlp
                # Concatenate and project through MLP
                combined = torch.cat([u_k, v_k], dim=-1)  # (B, [S], expert_dim + rel_feature_dim)
                fused_output = self.fusion(combined)  # (B, [S], expert_dim)
                
                if debug or hasattr(self, '_debug_mode'):
                    print(f"🔗 CONCATENATED: {combined.shape}")
                    print(f"🎯 LINEAR-MLP OUTPUT: {fused_output.shape}")
        else:
            # Single stream (only relative time)
            fused_output = self.fusion(v_k)  # (B, [S], expert_dim)
            
            if debug or hasattr(self, '_debug_mode'):
                print(f"🎯 SINGLE STREAM OUTPUT: {fused_output.shape}")
        
        # Project to embedding_dim if needed
        final_output = self.output_projection(fused_output)  # (B, [S], embedding_dim)
        
        if debug or hasattr(self, '_debug_mode'):
            print(f"🎯 FINAL OUTPUT: {final_output.shape}")
            print(f"{'='*60}\n")
        
        return final_output
    
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
            'expert_dim': self.expert_dim,
            'num_mixtures': self.num_mixtures,
            'wavelet_type': self.wavelet_type,
            'use_dual_kmote': self.use_dual_kmote,
            'fusion_type': self.fusion_type,
            'use_dual_stream': self.use_dual_stream,
            'parameters': self.count_parameters()
        }


def create_kan_mammote_lite(
    embedding_dim: int,
    expert_dim: int = None,
    num_mixtures: int = 12,
    wavelet_type: str = 'shock',
    use_dual_kmote: bool = True,
    fusion_type: str = 'linear_mlp',
    use_dual_stream: bool = True,
    device: str = 'cpu'
):
    """
    Factory function to create Enhanced KAN-MAMMOTE Lite.
    
    Args:
        embedding_dim: Output dimension
        expert_dim: Dimension for K-MOTE experts (default: same as embedding_dim)
        num_mixtures: Number of Gaussian mixtures in SM-Kernel
        wavelet_type: Wavelet type for K-MOTE
        use_dual_kmote: Whether to use K-MOTE for both abs and rel time
        fusion_type: 'linear_mlp', 'cross_attention', or 'weighted_sum'
        use_dual_stream: Whether to use both t_abs and t_rel
        device: Device to place model on
    
    Returns:
        KAN_MAMMOTE_Lite model
    """
    model = KAN_MAMMOTE_Lite(
        embedding_dim=embedding_dim,
        expert_dim=expert_dim,
        num_mixtures=num_mixtures,
        wavelet_type=wavelet_type,
        use_dual_kmote=use_dual_kmote,
        fusion_type=fusion_type,
        use_dual_stream=use_dual_stream
    )
    return model.to(device)


# Alias for backward compatibility
KANMAMMOTELite = KAN_MAMMOTE_Lite
