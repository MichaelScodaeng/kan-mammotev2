# file: models/time_encoders/kan_mammote.py (Add the missing method)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .k_mote import KMOTE
from .sm_kernel import SMKernelLayer
from .controllable_mamba2 import ControllableMamba2

class KAN_MAMMOTE(nn.Module):
    # The __init__ method is already correct from our last fix.
    def __init__(self, embedding_dim: int, expert_dim: int, num_mixtures: int, 
                 mamba_d_state: int = 16, mamba_d_conv: int = 4, mamba_expand: int = 2, **kwargs):
        super().__init__()
        
        # Enforce that dimensions are multiples of 16 for hardware compatibility.
        if embedding_dim % 16 != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be a multiple of 16 for Mamba2 compatibility.")
        if mamba_d_state % 16 != 0:
            raise ValueError(f"mamba_d_state ({mamba_d_state}) must be a multiple of 16 for Mamba2 compatibility.")
        
        self.embedding_dim = embedding_dim
        self.k_mote = KMOTE(input_dim=1, output_dim=embedding_dim)
        self.sm_kernel = SMKernelLayer(num_mixtures=num_mixtures, input_dim=1)
        self.mamba2 = ControllableMamba2(
            d_model=self.embedding_dim,  # you already pad to /8
            d_state=32,                         # was 16; 32 aligns better
            d_conv=4,
            expand=2,
            headdim=16                          # NEW: ensures nheads is nice (d_model/16)
        )
        
         # --- START OF DEFINITIVE FIX ---
        # The Fusion MLP's input dimension MUST match the concatenated dimensions
        # of the K-MOTE output (embedding_dim) and the SM-Kernel output (num_mixtures).
        fusion_input_dim = num_mixtures
        
        # The MLP now needs to output TWO values per head: gamma (scale) and beta (shift).
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, embedding_dim), # Corrected input dimension
            nn.GELU(),
            nn.Linear(embedding_dim, self.mamba2.nheads * 2) # Output 2 * nheads for gamma and beta
        )

        print("Initialized KAN-MAMMOTE Framework (with alignment checks).")

    # --- START OF CORRECTION ---
    # Add this missing helper method
    def initialize_sm_kernel(self, delta_t_sample: torch.Tensor):
        """Passes the initialization call to the SM-Kernel module."""
        self.sm_kernel.initialize_from_data(delta_t_sample)
    # --- END OF CORRECTION ---
        
    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        # The forward pass is already correct
        u_k = self.k_mote(t_abs)
        v_k = self.sm_kernel(t_rel)
        uv_concat = v_k
        
        modulator_logits = self.fusion_mlp(uv_concat)
        
        gamma_logits, beta = modulator_logits.chunk(2, dim=-1)
        
        gamma = 2 * torch.sigmoid(gamma_logits)
        
        temporal_modulators = (gamma, beta)
        
        final_embedding = self.mamba2(u=u_k, temporal_modulators=temporal_modulators)
        return final_embedding