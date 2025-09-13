# file: kan_mammote.py

import torch
import torch.nn as nn

# Import our previously defined modules
from .k_mote import KMOTE
from .sm_kernel import SMKernelLayer
from .controllable_mamba2 import ControllableMamba2 # <-- Import your new class


class KAN_MAMMOTE(nn.Module):
    """
    The full KAN-MAMMOTE temporal encoding framework. (Simplified with ControllableMamba2)

    This module processes absolute and relative time in parallel streams and fuses them
    by dynamically modulating the content-aware gate of a Mamba2 SSM.

    Args:
        embedding_dim (int): The main dimension for all internal embeddings (D).
        mamba_d_state (int): The state dimension (N) for the Mamba2 block.
        mamba_d_conv (int): The convolution kernel size for the Mamba2 block.
        mamba_expand (int): The expansion factor for the Mamba2 block.
    """
    def __init__(self, embedding_dim: int, mamba_d_state: int = 16, mamba_d_conv: int = 4, mamba_expand: int = 2):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Stream 1: Absolute Time with K-MOTE
        self.k_mote = KMOTE(input_dim=1, output_dim=embedding_dim)

        # Stream 2: Relative Time with SM-Kernel
        self.sm_kernel = SMKernelLayer(num_mixtures=embedding_dim, input_dim=1)

        # Mamba2 Backbone is now an instance of our controllable wrapper
        self.mamba2 = ControllableMamba2(
            d_model=embedding_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand
        )
        
        # Fusion MLP for Temporal Modulation Gate
        # Its output size must match the number of heads in the Mamba2 block.
        self.fusion_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim), # Takes concatenated u_k and v_k
            nn.GELU(),
            nn.Linear(embedding_dim, self.mamba2.nheads) # Output matches Mamba's nheads
        )

        print("Initialized KAN-MAMMOTE Framework (using ControllableMamba2).")

    def initialize_sm_kernel(self, delta_t_sample: torch.Tensor):
        """Passes the initialization call to the SM-Kernel module."""
        self.sm_kernel.initialize_from_data(delta_t_sample)
        
    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """
        The simplified forward pass.

        Args:
            t_abs (torch.Tensor): Absolute timestamps. Shape: (B, S, 1).
            t_rel (torch.Tensor): Relative time differences (delta_t). Shape: (B, S, 1).
        Returns:
            torch.Tensor: The final, unified temporal embedding T_k. Shape: (B, S, embedding_dim).
        """
        # --- Step 1: Process streams in parallel ---
        u_k = self.k_mote(t_abs)      # Absolute embedding, shape: (B, S, D)
        v_k = self.sm_kernel(t_rel)  # Relative embedding, shape: (B, S, D)

        # --- Step 2: Compute the temporal modulation gate ---
        uv_concat = torch.cat([u_k, v_k], dim=-1)
        temporal_gate_logits = self.fusion_mlp(uv_concat)
        
        # A gate centered at 1 allows for both suppression and amplification.
        temporal_gate = 2 * torch.sigmoid(temporal_gate_logits) # Shape: (B, S, nheads)

        # --- Step 3: Call the ControllableMamba2 forward pass ---
        # Pass the absolute time embedding `u_k` as the main input and the
        # `temporal_gate` as the controller.
        final_embedding = self.mamba2(u=u_k, temporal_gate=temporal_gate)

        return final_embedding