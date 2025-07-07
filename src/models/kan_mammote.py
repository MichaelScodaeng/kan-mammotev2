# kan_mamote/src/models/kan_mamote.py

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

# Import configuration
from src.utils.config import KANMAMOTEConfig
# Import K-MOTE and ContinuousMambaBlock
from src.models.k_mote import K_MOTE
from src.models.c_mamba import ContinuousMambaBlock

class KAN_MAMOTE_Model(nn.Module):
    """
    The complete KAN-MAMOTE (Kernel-Adaptive-Neural-Mamba-Mixture-of-Time-Experts) model.
    This framework learns adaptive spatio-temporal representations for continuous-time dynamic systems.

    It comprises two primary synergistic modules:
    1. K-MOTE: For rich, adaptive point-in-time encoding of absolute and relative timestamps.
    2. Continuous-Mamba Integration: For sequential context and memory, dynamically adapting
       to irregular time differences between events.
    """
    def __init__(self, config: KANMAMOTEConfig):
        super().__init__()
        self.config = config

        # 1. K-MOTE Modules for Absolute and Relative Time Embeddings
        # Both K-MOTE modules operate on scalar time inputs (tk or delta_tk)
        # and optionally use auxiliary features for the router.
        self.k_mote_abs = K_MOTE(config)
        self.k_mote_rel = K_MOTE(config)

        # 2. Continuous-Mamba Block
        # Input to Continuous-Mamba is the concatenated time embeddings from K-MOTE:
        # - K-MOTE absolute embedding (D_time)
        # - K-MOTE relative embedding (D_time)
        # Total input dimension: 2 * D_time
        mamba_input_dim = 2 * config.D_time  # Concatenated absolute and relative time embeddings
        self.ct_mamba_block = ContinuousMambaBlock(input_dim=mamba_input_dim, config=config)

        # Optional: A final linear layer or task-specific head after Mamba's output
        # Mamba's output is hidden_dim_mamba.
        # This will depend on the downstream task (e.g., forecasting, classification).
        # For a general "embedding" output, this can be skipped or be an identity.
        # Let's add a placeholder for a task-specific head, which can be defined based on `output_dim`.
        # Assuming `output_dim` will be part of the config later for the final task.
        # For now, let's just make sure the output matches `hidden_dim_mamba`.
        # You can add: self.task_head = nn.Linear(config.hidden_dim_mamba, output_dim_for_task)

    def forward(self, 
                timestamps: torch.Tensor,         # (batch_size, seq_len, 1) - absolute timestamps
                event_features: torch.Tensor,     # (batch_size, seq_len, raw_event_feature_dim) - raw features at each timestamp
                initial_mamba_state: Optional[torch.Tensor] = None # (batch_size, state_dim_mamba)
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs the forward pass of the KAN-MAMOTE model for a sequence of events.

        Args:
            timestamps: Absolute timestamps of events in the sequence,
                        shape (batch_size, seq_len, 1).
            event_features: Raw features associated with each event,
                            shape (batch_size, seq_len, raw_event_feature_dim).
            initial_mamba_state: Optional initial hidden state for the Continuous-Mamba block,
                                 shape (batch_size, state_dim_mamba). Defaults to zeros.

        Returns:
            Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - final_embeddings: The comprehensive spatio-temporal embeddings for each event in sequence,
                                    shape (batch_size, seq_len, hidden_dim_mamba).
                - moe_losses_info: A tuple containing (abs_expert_weights_for_loss, rel_expert_weights_for_loss)
                                   used for MoE load balancing regularization.
        """
        batch_size, seq_len, _ = timestamps.shape
        device = timestamps.device

        # Lists to collect K-MOTE outputs for each time step
        u_k_sequence = [] # For Mamba input
        all_abs_expert_weights = [] # For load balancing loss
        all_rel_expert_weights = [] # For load balancing loss

        # Initial timestamp (for delta_t_0 calculation if needed, or assume first delta_t is 0)
        # We need a `prev_t` for `delta_t_k = t_k - prev_t`.
        # For k=0, `prev_t` can be `timestamps[:, 0:1, :]` or a special learnable `t_0_base`.
        # For simplicity, assume delta_t_0 is 0 or fixed for the first element.
        # Let's follow common practice for first delta_t.
        # Set delta_t for first element to 0 (or a small constant if timestamps are relative to start).
        # We need (batch_size, seq_len, 1) for delta_t_sequence.

        # Calculate delta_t_sequence
        delta_t_sequence = torch.zeros_like(timestamps) # (batch_size, seq_len, 1)
        if seq_len > 1:
            # Shift timestamps to get previous ones, pad with current first timestamp or zero
            prev_timestamps = torch.cat([timestamps[:, 0:1, :], timestamps[:, :-1, :]], dim=1)
            delta_t_sequence = timestamps - prev_timestamps
        # For seq_len = 1, delta_t_sequence remains zeros. This assumes delta_t_0 = 0.
        # Alternatively, delta_t_0 could be a small learned parameter if there's no "previous event."


        # Generate K-MOTE embeddings for the entire sequence
        # We run K-MOTE on absolute and relative timestamps for each step.
        # This will be looped over the sequence to get all u_k_sequence inputs for Mamba.
        
        # It's more efficient to process the whole sequence with K-MOTE if possible,
        # rather than looping over experts and batch size.
        # K-MOTE's forward takes (batch_size, 1) for timestamp.
        # To process a sequence, reshape timestamps and event_features
        
        timestamps_flat = timestamps.view(batch_size * seq_len, 1)
        delta_t_flat = delta_t_sequence.view(batch_size * seq_len, 1)
        event_features_flat = event_features.view(batch_size * seq_len, self.config.raw_event_feature_dim)

        # K-MOTE for absolute time
        # Returns (batch_size*seq_len, D_time), (batch_size*seq_len, num_experts), (batch_size*seq_len, num_experts)
        phi_abs_flat, abs_expert_weights_flat, _ = self.k_mote_abs(timestamps_flat, event_features_flat)
        
        # K-MOTE for relative time
        phi_rel_flat, rel_expert_weights_flat, _ = self.k_mote_rel(delta_t_flat, event_features_flat)

        # Concatenate for Mamba input: u_k
        # (batch_size*seq_len, D_time + D_time + raw_event_feature_dim)
        u_k_flat = torch.cat([phi_abs_flat, phi_rel_flat, event_features_flat], dim=-1)

        # Reshape u_k_flat and delta_t_flat back to sequence format for Mamba
        u_k_sequence_for_mamba = u_k_flat.view(batch_size, seq_len, -1)
        delta_t_sequence_for_mamba = delta_t_flat.view(batch_size, seq_len, 1)


        # 3. Continuous-Mamba Block Processing
        # Pass time embeddings from K-MOTE instead of raw features
        # The Mamba block expects: (time_embeddings, timestamps, initial_state)
        # time_embeddings are the concatenated K-MOTE outputs
        time_embeddings = torch.cat([phi_abs_flat, phi_rel_flat], dim=-1)  # Concatenate abs and rel embeddings
        time_embeddings = time_embeddings.view(batch_size, seq_len, -1)  # Reshape to sequence format
        
        # If we want to include raw event features, we can add them to the Mamba input dimension
        # or pass them separately. For now, let's keep it simple with just time embeddings.
        final_embeddings = self.ct_mamba_block(
            time_embeddings,  # (batch_size, seq_len, D_time + D_time) - concatenated K-MOTE embeddings
            timestamps,       # (batch_size, seq_len, 1) - raw timestamps for time differences
            initial_mamba_state
        )

        # Reshape expert weights for loss calculation
        abs_expert_weights_for_loss = abs_expert_weights_flat.view(batch_size, seq_len, self.config.num_experts)
        rel_expert_weights_for_loss = rel_expert_weights_flat.view(batch_size, seq_len, self.config.num_experts)

        return final_embeddings, (abs_expert_weights_for_loss, rel_expert_weights_for_loss)