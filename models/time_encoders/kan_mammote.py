# file: models/time_encoders/kan_mammote.py (Corrected Dimensions)

import torch
import torch.nn as nn

# Import our previously defined modules
from .k_mote import KMOTE
from .sm_kernel import SMKernelLayer
from .controllable_mamba2 import ControllableMamba2

class KAN_MAMMOTE(nn.Module):
    """
    The full KAN-MAMMOTE temporal encoding framework. (Corrected Dimensions)

    Args:
        embedding_dim (int): The main dimension for the Mamba backbone and the final output.
        expert_dim (int): The output dimension for the K-MOTE (absolute time) encoder.
        num_mixtures (int): The number of mixtures (output dimension) for the SM-Kernel.
        mamba_d_state (int): The state dimension (N) for the Mamba2 block.
        mamba_d_conv (int): The convolution kernel size for the Mamba2 block.
        mamba_expand (int): The expansion factor for the Mamba2 block.
    """
    def __init__(self, embedding_dim: int, expert_dim: int, num_mixtures: int, 
                 mamba_d_state: int = 16, mamba_d_conv: int = 4, mamba_expand: int = 2, **kwargs):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Stream 1: Absolute Time with K-MOTE
        # The Mamba backbone expects an input of size `embedding_dim`. 
        # So, K-MOTE's output must match this.
        self.k_mote = KMOTE(input_dim=1, output_dim=embedding_dim)

        # Stream 2: Relative Time with SM-Kernel
        self.sm_kernel = SMKernelLayer(num_mixtures=num_mixtures, input_dim=1)

        # Mamba2 Backbone is now an instance of our controllable wrapper
        self.mamba2 = ControllableMamba2(
            d_model=embedding_dim,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand
        )
        
        # --- START OF CORRECTION ---
        # The Fusion MLP's input dimension must match the concatenated dimensions
        # of the K-MOTE output and the SM-Kernel output.
        fusion_input_dim = embedding_dim + num_mixtures
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, embedding_dim), # Corrected input dimension
            nn.GELU(),
            nn.Linear(embedding_dim, self.mamba2.nheads) # Output matches Mamba's nheads
        )
        # --- END OF CORRECTION ---

        print("Initialized KAN-MAMMOTE Framework (using ControllableMamba2).")

    def initialize_sm_kernel(self, delta_t_sample: torch.Tensor):
        """Passes the initialization call to the SM-Kernel module."""
        self.sm_kernel.initialize_from_data(delta_t_sample)
        
    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """
        The simplified forward pass.
        """
        # --- Step 1: Process streams in parallel ---
        u_k = self.k_mote(t_abs)      # Absolute embedding, shape: (B, S, embedding_dim)
        v_k = self.sm_kernel(t_rel)  # Relative embedding, shape: (B, S, num_mixtures)

        # --- Step 2: Compute the temporal modulation gate ---
        uv_concat = torch.cat([u_k, v_k], dim=-1) # Shape: (B, S, embedding_dim + num_mixtures)
        temporal_gate_logits = self.fusion_mlp(uv_concat)
        temporal_gate = 2 * torch.sigmoid(temporal_gate_logits)

        # --- Step 3: Call the ControllableMamba2 forward pass ---
        # The primary input to Mamba MUST be u_k, which has `embedding_dim`.
        final_embedding = self.mamba2(u=u_k, temporal_gate=temporal_gate)

        return final_embedding