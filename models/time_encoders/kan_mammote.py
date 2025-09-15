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
        # --- NEW WARNING BLOCK ---
        if embedding_dim % 8 != 0:
            print(f"Warning: embedding_dim ({embedding_dim}) is not a multiple of 8. This may cause errors or slow performance on modern GPUs.")
        if expert_dim % 8 != 0:
            print(f"Warning: expert_dim ({expert_dim}) is not a multiple of 8.")
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
        
        # --- START OF MODIFICATION ---
        # The Fusion MLP's input dimension is now ONLY num_mixtures.
        fusion_input_dim = num_mixtures
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, embedding_dim // 2), # A smaller hidden layer is fine
            nn.GELU(),
            nn.Linear(embedding_dim // 2, self.mamba2.nheads) # Output still matches Mamba's nheads
        )
        # --- END OF MODIFICATION ---

        print("Initialized KAN-MAMMOTE Framework (using ControllableMamba2).")

    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with aggressive CUDA alignment fixing.
        """
        # Handle input shapes and sequence length mismatches
        if t_abs.dim() == 2:
            t_abs = t_abs.unsqueeze(-1)
        if t_rel.dim() == 2:
            t_rel = t_rel.unsqueeze(-1)
        
        batch_size = t_abs.shape[0]
        seq_len_abs = t_abs.shape[1]
        seq_len_rel = t_rel.shape[1]
        
        # Handle sequence length mismatch
        if seq_len_abs != seq_len_rel:
            target_seq_len = max(seq_len_abs, seq_len_rel)
            
            if seq_len_abs < target_seq_len:
                if seq_len_abs == 1:
                    t_abs = t_abs.expand(batch_size, target_seq_len, -1)
                else:
                    padding = t_abs[:, -1:, :].expand(batch_size, target_seq_len - seq_len_abs, -1)
                    t_abs = torch.cat([t_abs, padding], dim=1)
            
            if seq_len_rel < target_seq_len:
                if seq_len_rel == 1:
                    t_rel = t_rel.expand(batch_size, target_seq_len, -1)
                else:
                    padding = t_rel[:, -1:, :].expand(batch_size, target_seq_len - seq_len_rel, -1)
                    t_rel = torch.cat([t_rel, padding], dim=1)

        # --- Step 1: Dual-Stream Processing ---
        u_k = self.k_mote(t_abs)      # Absolute embedding, shape: (B, S, embedding_dim)
        v_k = self.sm_kernel(t_rel)   # Relative embedding, shape: (B, S, num_mixtures)

        # --- Step 2: Temporal Gate from Relative Time ONLY ---
        temporal_gate = 2.0 * torch.sigmoid(self.fusion_mlp(v_k))

        # --- Step 3: AGGRESSIVE CUDA ALIGNMENT FIX ---
        if u_k.device.type == 'cuda':
            print(f"🔧 Applying aggressive CUDA alignment fix...")
            
            batch_size, seq_len, embed_dim = u_k.shape
            gate_dim = temporal_gate.shape[-1]
            
            # Force dimensions to be multiples of 8
            target_embed_dim = ((embed_dim + 7) // 8) * 8
            target_gate_dim = ((gate_dim + 7) // 8) * 8
            
            print(f"   Original: u_k={u_k.shape}, gate={temporal_gate.shape}")
            print(f"   Target dims: embed={target_embed_dim}, gate={target_gate_dim}")
            
            # AGGRESSIVE APPROACH: Create completely new tensors with proper alignment
            
            # Step 1: Create u_k with proper dimensions and strides
            u_k_new = torch.zeros(
                batch_size, seq_len, target_embed_dim,
                device=u_k.device,
                dtype=u_k.dtype,
                layout=torch.strided
            )
            u_k_new[:, :, :embed_dim] = u_k  # Copy original data
            
            # Step 2: Create temporal_gate with proper dimensions and strides  
            temporal_gate_new = torch.zeros(
                batch_size, seq_len, target_gate_dim,
                device=temporal_gate.device,
                dtype=temporal_gate.dtype,
                layout=torch.strided
            )
            temporal_gate_new[:, :, :gate_dim] = temporal_gate  # Copy original data
            
            # Step 3: Force optimal memory layout
            u_k_final = u_k_new.contiguous()
            temporal_gate_final = temporal_gate_new.contiguous()
            
            # Step 4: Additional stride enforcement
            if u_k_final.stride(0) % 8 != 0 or u_k_final.stride(2) % 8 != 0:
                print(f"   🔄 Recreating u_k for optimal strides...")
                u_k_optimal = torch.empty_strided(
                    (batch_size, seq_len, target_embed_dim),
                    (seq_len * target_embed_dim, target_embed_dim, 1),
                    device=u_k.device,
                    dtype=u_k.dtype
                )
                u_k_optimal.copy_(u_k_final)
                u_k_final = u_k_optimal
            
            if temporal_gate_final.stride(0) % 8 != 0 or temporal_gate_final.stride(2) % 8 != 0:
                print(f"   🔄 Recreating temporal_gate for optimal strides...")
                temporal_gate_optimal = torch.empty_strided(
                    (batch_size, seq_len, target_gate_dim),
                    (seq_len * target_gate_dim, target_gate_dim, 1),
                    device=temporal_gate.device,
                    dtype=temporal_gate.dtype
                )
                temporal_gate_optimal.copy_(temporal_gate_final)
                temporal_gate_final = temporal_gate_optimal
            
            print(f"   Final u_k: shape={u_k_final.shape}, strides={u_k_final.stride()}")
            print(f"   Final gate: shape={temporal_gate_final.shape}, strides={temporal_gate_final.stride()}")
            
            # Verify final alignment
            u_stride_ok = (u_k_final.stride(0) % 8 == 0) and (u_k_final.stride(2) % 8 == 0)
            g_stride_ok = (temporal_gate_final.stride(0) % 8 == 0) and (temporal_gate_final.stride(2) % 8 == 0)
            
            if u_stride_ok and g_stride_ok:
                print(f"   ✅ All strides now properly aligned!")
                u_k = u_k_final
                temporal_gate = temporal_gate_final
            else:
                print(f"   ❌ Stride alignment still failed. Falling back to CPU...")
                # CPU fallback
                original_device = u_k.device
                u_k = u_k.cpu()
                temporal_gate = temporal_gate.cpu()
                self.mamba2.cpu()
                
                final_embedding = self.mamba2(u=u_k, temporal_gate=temporal_gate)
                
                # Move back to GPU
                self.mamba2.to(original_device)
                final_embedding = final_embedding.to(original_device)
                return final_embedding[:, :, :self.embedding_dim]
        
        # --- Step 4: Call ControllableMamba2 ---
        try:
            final_embedding = self.mamba2(u=u_k, temporal_gate=temporal_gate)
            print(f"   ✅ Mamba2 call successful!")
        except Exception as e:
            print(f"   ❌ Mamba2 call failed: {e}")
            raise e

        # Remove padding if we added any
        if final_embedding.shape[-1] > self.embedding_dim:
            final_embedding = final_embedding[:, :, :self.embedding_dim]

        return final_embedding