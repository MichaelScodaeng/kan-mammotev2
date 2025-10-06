
import torch
import torch.nn as nn
from .k_mote import KMOTE
from .sm_kernel import SMKernelLayer


class SMKernelOnly(nn.Module):
    """
    SM-Kernel only for relative time encoding (ablation study)
    
    This encoder only uses the SM-Kernel component with relative time differences,
    ignoring absolute timestamps completely.
    """
    def __init__(self, embedding_dim: int, num_mixtures: int = 12, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_mixtures = num_mixtures
        
        # Only SM-Kernel for relative time
        self.sm_kernel = SMKernelLayer(num_mixtures=num_mixtures, input_dim=1)
        
        # Simple projection to match embedding_dim
        self.projection = nn.Linear(num_mixtures, embedding_dim)
        
    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """
        Use only relative time (t_rel) with SM-Kernel
        
        Args:
            t_abs: Absolute timestamps (ignored)
            t_rel: Relative time differences (used)
            
        Returns:
            Temporal embeddings based only on relative time
        """
        # Only use relative time with SM-Kernel
        sm_features = self.sm_kernel(t_rel)  # [batch, seq_len, num_mixtures]
        output = self.projection(sm_features)  # [batch, seq_len, embedding_dim]
        return output
    
    def initialize_sm_kernel(self, delta_t_sample: torch.Tensor):
        """Initialize SM-Kernel from data sample"""
        self.sm_kernel.initialize_from_data(delta_t_sample)


class KMOTEAbsOnly(nn.Module):
    """
    K-MOTE only for absolute time encoding (ablation study)
    
    This encoder only uses K-MOTE with absolute timestamps,
    ignoring relative time differences completely.
    """
    def __init__(self, embedding_dim: int, wavelet_type: str = 'shock', **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Only K-MOTE for absolute time
        self.k_mote = KMOTE(
            input_dim=1, 
            output_dim=embedding_dim, 
            wavelet_type=wavelet_type
        )
        
    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """
        Use only absolute time (t_abs) with K-MOTE
        
        Args:
            t_abs: Absolute timestamps (used)
            t_rel: Relative time differences (ignored)
            
        Returns:
            Temporal embeddings based only on absolute time
        """
        # Only use absolute time with K-MOTE
        kmote_features = self.k_mote(t_abs)  # [batch, seq_len, embedding_dim]
        return kmote_features


class KMOTERelOnly(nn.Module):
    """
    K-MOTE only for relative time encoding (ablation study)
    
    This encoder only uses K-MOTE with relative time differences,
    ignoring absolute timestamps completely.
    """
    def __init__(self, embedding_dim: int, wavelet_type: str = 'shock', **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Only K-MOTE for relative time
        self.k_mote = KMOTE(
            input_dim=1, 
            output_dim=embedding_dim, 
            wavelet_type=wavelet_type
        )
        
    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """
        Use only relative time (t_rel) with K-MOTE
        
        Args:
            t_abs: Absolute timestamps (ignored)
            t_rel: Relative time differences (used)
            
        Returns:
            Temporal embeddings based only on relative time
        """
        # Only use relative time with K-MOTE
        kmote_features = self.k_mote(t_rel)  # [batch, seq_len, embedding_dim]
        return kmote_features


class DualStreamBaseline(nn.Module):
    """
    Simple dual-stream baseline without Mamba (ablation study)
    
    This encoder combines K-MOTE (absolute) + SM-Kernel (relative) 
    with simple fusion, but without the Mamba2 component.
    """
    def __init__(self, embedding_dim: int, num_mixtures: int = 12, 
                 wavelet_type: str = 'shock', **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Dual stream components
        self.k_mote = KMOTE(
            input_dim=1, 
            output_dim=embedding_dim // 2, 
            wavelet_type=wavelet_type
        )
        self.sm_kernel = SMKernelLayer(num_mixtures=num_mixtures, input_dim=1)
        
        # Simple fusion without Mamba
        fusion_input_dim = (embedding_dim // 2) + num_mixtures
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """
        Dual-stream encoding without Mamba
        
        Args:
            t_abs: Absolute timestamps
            t_rel: Relative time differences
            
        Returns:
            Fused temporal embeddings
        """
        # Dual stream processing
        u_k = self.k_mote(t_abs)  # [batch, seq_len, embedding_dim//2]
        v_k = self.sm_kernel(t_rel)  # [batch, seq_len, num_mixtures]
        
        # Simple concatenation and fusion
        uv_concat = torch.cat([u_k, v_k], dim=-1)  # [batch, seq_len, fusion_input_dim]
        output = self.fusion_mlp(uv_concat)  # [batch, seq_len, embedding_dim]
        
        return output
    
    def initialize_sm_kernel(self, delta_t_sample: torch.Tensor):
        """Initialize SM-Kernel from data sample"""
        self.sm_kernel.initialize_from_data(delta_t_sample)