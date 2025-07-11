# kan_mamote/src/models/continuous_mamba_block.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# Import your custom modules
from src.utils.config import KANMAMOTEConfig
from src.models.k_mote import K_MOTE # Assuming k_mote.py is correct as discussed
from src.layers.dynamic_mamba_ssm import DynamicMambaSSM # Your modified Mamba2
from src.layers.kan_base_layer import KANLayer # Your custom KANLayer, e.g., for FasterKAN below

# Assume FasterKAN is also a KANLayer as discussed
# If you have a separate, more complex FasterKAN, replace this.
# For simplicity, we'll assume a basic KANLayer here for the transformation.
class FasterKAN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, config: KANMAMOTEConfig):
        super().__init__()
        # Using a simple KANLayer with SplineBasis for the FasterKAN transformation
        # Note: This is an internal KAN, not one of the K-MOTE experts.
        # It's flexible, you could use a simple MLP or any other transformation here.
        self.kan_transform = KANLayer(in_features=input_dim, out_features=output_dim,
                                      basis_type='rkhs_gaussian', # Or 'fourier', 'wavelet', 'spline' if you re-add SplineBasis for this.
                                      config=config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.kan_transform(x)


class ContinuousMambaBlock(nn.Module):
    """
    A single block of the Continuous-Mamba, integrating K-MOTE time embeddings
    and dynamically influencing the Mamba SSM.
    """
    def __init__(self, d_model: int, config: KANMAMOTEConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.config = config
        self.layer_idx = layer_idx # Passed to Mamba2 for inference caching

        # K-MOTE for generating absolute time embeddings
        self.k_mote = K_MOTE(config)
        
        # FasterKAN to transform K-MOTE output into the delta_t_embedding that influences Mamba
        # The input to FasterKAN is K-MOTE's output (config.D_time).
        # The output of FasterKAN (delta_t_embedding) will also be config.D_time,
        # as it needs to be fed to dt_modulation_proj in DynamicMambaSSM.
        self.faster_kan_transform = FasterKAN(input_dim=config.D_time, output_dim=config.D_time, config=config)
        
        # Initial projection for input features (`uk`) to match `d_model`
        # This will be applied to the output of the previous block, or raw features for the first block.
        self.input_proj = nn.Linear(d_model, d_model) 

        # Dynamic Mamba SSM layer
        # Pass Mamba-specific kwargs from config
        mamba_kwargs = {
            "d_state": config.mamba_d_state,
            "d_conv": config.mamba_d_conv,
            "expand": config.mamba_expand,
            "headdim": config.mamba_headdim,
            "dt_min": config.mamba_dt_min,
            "dt_max": config.mamba_dt_max,
            "dt_init_floor": config.mamba_dt_init_floor,
            "bias": config.mamba_bias,
            "conv_bias": config.mamba_conv_bias,
            "chunk_size": config.mamba_chunk_size,
            "use_mem_eff_path": config.mamba_use_mem_eff_path,
            "layer_idx": layer_idx, # Pass unique layer_idx for cache
            "device": config.device,
            "dtype": config.dtype,
        }
        self.mamba_ssm = DynamicMambaSSM(
            d_model=d_model,
            k_mote_delta_t_embedding_dim=config.D_time, # K-MOTE's output dim
            **mamba_kwargs
        )

    def forward(self,
                uk_current_input: torch.Tensor,       # (batch_size, d_model) - output from previous layer/projected raw features
                tk_current: torch.Tensor,             # (batch_size, 1) - current timestamp
                tk_previous: torch.Tensor,            # (batch_size, 1) - previous timestamp
                hidden_state_prev_timestep: Optional[torch.Tensor] = None, # (batch_size, d_model) - for the current layer
                inference_params: Optional[dict] = None # For Mamba's inference cache
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # hk_current, expert_weights_loss, expert_selection_mask
        """
        Performs one step (for one time point in a sequence) of the Continuous-Mamba block.
        This block processes `uk_current_input` using Mamba, where Mamba's parameters
        are dynamically adjusted by the time context derived from `tk_current` and `tk_previous`.

        Args:
            uk_current_input: Features at the current timestamp, usually output from previous block
                              or projected raw features for the first block. Shape (batch_size, d_model).
            tk_current: Current timestamp. Shape (batch_size, 1).
            tk_previous: Previous timestamp. Shape (batch_size, 1).
            hidden_state_prev_timestep: Hidden state of *this specific Mamba layer* from the *previous timestep*.
                                        Shape (batch_size, d_model). Used for internal state updates.
                                        **Note:** For sequence-level forward, Mamba handles this internally.
                                        This is primarily relevant for step-by-step inference.
                                        For full sequence processing, it's None.
            inference_params: Dictionary for Mamba's inference cache.

        Returns:
            hk_current: The updated hidden state (output) of this Mamba block for the current timestep.
                        Shape (batch_size, d_model).
            expert_weights_for_loss: Expert weights from K-MOTE router for load balancing.
                                     Shape (batch_size, num_experts).
            expert_selection_mask: Binary mask indicating selected experts.
                                   Shape (batch_size, num_experts).
        """
        # Ensure input features are projected (if coming from a different source or initial layer)
        # This proj is applied to the input for THIS block, NOT the output of K-MOTE.
        processed_uk = self.input_proj(uk_current_input) # (batch_size, d_model)

        # 1. Generate K-MOTE absolute time embeddings for current and previous timestamps
        abs_time_embedding_tk, expert_weights_for_loss, expert_selection_mask = self.k_mote(tk_current)
        abs_time_embedding_tk_minus_1, _, _ = self.k_mote(tk_previous) # Don't need weights/mask from prev step

        # 2. Transform these embeddings with FasterKAN and compute delta_t_embedding
        # The FasterKAN layer expects (batch_size, input_dim) -> (batch_size, output_dim)
        transformed_tk = self.faster_kan_transform(abs_time_embedding_tk)
        transformed_tk_minus_1 = self.faster_kan_transform(abs_time_embedding_tk_minus_1)
        
        # This delta_t_embedding (shape: batch_size, config.D_time) will be used to modulate Mamba's dt_bias
        delta_t_embedding = transformed_tk - transformed_tk_minus_1

        # 3. Pass through Dynamic Mamba SSM
        # Mamba expects input `u` as (batch, seqlen, d_model) for sequence processing,
        # or (batch_seqlen, d_model) if seqlen is provided.
        # Here, `processed_uk` is (batch_size, d_model). We make it (batch_size, 1, d_model) for Mamba's sequence dim.
        # Same for `delta_t_embedding`.
        
        # Mamba's `forward` expects `u` and `delta_t_embedding` to have a sequence dimension.
        # Since this `ContinuousMambaBlock` handles ONE timestep at a time (when stacked),
        # we provide a sequence length of 1.
        
        # The `hidden_state_prev_timestep` is crucial if you are manually handling
        # Mamba's state for step-by-step decoding or custom state management.
        # However, the `DynamicMambaSSM` (Mamba2) `forward` method is designed for
        # full sequence processing and internally manages its state across the sequence.
        # For the common case of processing a sequence `(B, L, D)` in one go through Mamba's `forward`,
        # `hidden_state_prev_timestep` is not used in the main forward path of Mamba2, only in `step` (inference).
        # We will assume `KANMAMMOTE` passes `uk` as `(B, L, D)` to this block's mamba_ssm.
        
        # For this block:
        # `u` should be (batch_size, 1, d_model)
        # `delta_t_embedding` should be (batch_size, 1, k_mote_delta_t_embedding_dim)

        hk_current = self.mamba_ssm(
            u=processed_uk.unsqueeze(1), # (batch_size, 1, d_model)
            delta_t_embedding=delta_t_embedding.unsqueeze(1), # (batch_size, 1, D_time)
            inference_params=inference_params # Pass inference params if decoding
        )
        
        # Remove the sequence dimension of 1 for output
        hk_current = hk_current.squeeze(1) # (batch_size, d_model)

        return hk_current, expert_weights_for_loss, expert_selection_mask