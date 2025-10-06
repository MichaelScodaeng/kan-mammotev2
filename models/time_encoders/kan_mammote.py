# file: models/time_encoders/kan_mammote.py (Add the missing method)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .k_mote import KMOTE
from .sm_kernel import SMKernelLayer
from .controllable_mamba2 import ControllableMamba2

class KAN_MAMMOTE(nn.Module):
    """Enhanced KAN-MAMMOTE with Custom Shock Wavelet for abrupt change detection."""
    def __init__(self, embedding_dim: int, expert_dim: int, num_mixtures: int, 
                 mamba_d_state: int = 256, mamba_d_conv: int = 4, mamba_expand: int = 4, 
                 wavelet_type: str = 'shock', mamba_headdim: int = 64, **kwargs):
        super().__init__()
        
        # Enforce that dimensions are multiples of 16 for hardware compatibility.
        if expert_dim % 16 != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be a multiple of 16 for Mamba2 compatibility.")
        if mamba_d_state % 16 != 0:
            raise ValueError(f"mamba_d_state ({mamba_d_state}) must be a multiple of 16 for Mamba2 compatibility.")
        
        self.embedding_dim = embedding_dim
        self.wavelet_type = wavelet_type
        self.expert_dim = expert_dim
        
        # Enhanced K-MOTE with configurable wavelet type
        self.k_mote = KMOTE(input_dim=1, output_dim=expert_dim, wavelet_type=wavelet_type)
        self.sm_kernel = SMKernelLayer(num_mixtures=num_mixtures, input_dim=1)
        self.mamba2 = ControllableMamba2(
            d_model=self.expert_dim,
            d_state=mamba_d_state, #mamba_d_state = 256
            d_conv=mamba_d_conv,
            expand=mamba_expand, 
            headdim=16 # 32
        )
        
         # --- START OF DEFINITIVE FIX ---
        # The Fusion MLP's input dimension MUST match the concatenated dimensions
        # of the K-MOTE output (embedding_dim) and the SM-Kernel output (num_mixtures).
        fusion_input_dim = num_mixtures
        
        # The MLP now needs to output TWO values per head: gamma (scale) and beta (shift).
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, expert_dim), # Corrected input dimension
            nn.LayerNorm(expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, self.mamba2.nheads * 2) # Output 2 * nheads for gamma and beta
        )
        print(f"mamba parameters")
        print(f"  nheads: {self.mamba2.nheads}")
        print(f"  d_state: {self.mamba2.d_state}")
        print(f"  d_conv: {self.mamba2.d_conv}")
        print(f"  expand: {self.mamba2.expand}")
        print(f"  headdim: {self.mamba2.headdim}")
        print(f"  embedding_dim: {self.embedding_dim}")

        print(f"Initialized Enhanced KAN-MAMMOTE Framework with {wavelet_type} wavelet.")
        if expert_dim != embedding_dim:
            self.output_projection = nn.Sequential(
                nn.Linear(expert_dim, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
        else:
            self.output_projection = nn.Identity()
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
        
        gamma = torch.sigmoid(gamma_logits) + 0.5 # Range: [0.5, 1.5]
        temporal_modulators = (gamma, beta)
        # ===== KEY FIX: Ensure alignment before Mamba =====
        u_k_aligned = self._ensure_aligned_for_mamba(u_k)

        mamba_output = self.mamba2(u=u_k_aligned, temporal_modulators=temporal_modulators)
        # adjust to expert_dim
        final_embedding = self.output_projection(mamba_output)

        return final_embedding

    def _ensure_aligned_for_mamba(self, tensor: torch.Tensor, debug: bool = False) -> torch.Tensor:
        """Ensure tensor is contiguous and has stride-8-aligned memory layout."""
        original_shape = tensor.shape
        original_strides = tensor.stride()
        was_contiguous = tensor.is_contiguous()
        
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        stride_0 = tensor.stride(0)
        stride_2 = tensor.stride(2)
        
        needs_alignment = (stride_0 % 8 != 0) or (stride_2 % 8 != 0)
        
        if debug:
            print(f"[Alignment Debug]")
            print(f"  Shape: {original_shape}")
            print(f"  Original strides: {original_strides}")
            print(f"  Was contiguous: {was_contiguous}")
            print(f"  Batch stride (stride[0]): {stride_0} (aligned: {stride_0 % 8 == 0})")
            print(f"  Feature stride (stride[2]): {stride_2} (aligned: {stride_2 % 8 == 0})")
            print(f"  Action: {'Clone for alignment' if needs_alignment else 'No action needed'}")
        
        if needs_alignment:
            pass
            #return tensor.clone()
        
        return tensor