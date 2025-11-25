
import torch
import torch.nn as nn
from .k_mote import KMOTE


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
            use_scale=True, 
            use_layernorm=True,       # NEW: enable scale
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
            wavelet_type=wavelet_type,
            use_scale=True,     
            use_layernorm=True,   # NEW: enable scale
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

