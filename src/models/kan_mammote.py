# kan_mamote/src/models/kan_mammote.py

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

# Import your custom modules
from src.utils.config import KANMAMOTEConfig
from src.models.continuous_mamba_block import ContinuousMambaBlock
from src.models.regularization import KANMAMMOTE_RegularizationLosses

class KANMAMMOTE(nn.Module):
    """
    The full KAN-MAMMOTE model for continuous-time sequence modeling.
    Composed of multiple ContinuousMambaBlocks stacked sequentially.
    """
    def __init__(self, config: KANMAMOTEConfig):
        super().__init__()
        self.config = config
        
        # Initial linear projection for raw input features to match `d_model`
        self.initial_feature_proj = nn.Linear(config.input_feature_dim, config.d_model)

        # Stack multiple ContinuousMambaBlocks
        self.mamba_blocks = nn.ModuleList([
            ContinuousMambaBlock(
                d_model=config.d_model,
                config=config,
                layer_idx=i # Pass a unique layer_idx for Mamba's inference cache
            )
            for i in range(config.num_layers)
        ])

        # Final prediction head (e.g., linear layer for regression/classification)
        self.prediction_head = nn.Linear(config.d_model, config.output_dim_for_task)

        # Regularization losses module
        self.regularization_handler = KANMAMMOTE_RegularizationLosses(config)

    def forward(self,
                timestamps: torch.Tensor, # (batch_size, sequence_length)
                features: torch.Tensor,   # (batch_size, sequence_length, input_feature_dim)
                auxiliary_features: Optional[torch.Tensor] = None # (batch_size, sequence_length, aux_feature_dim)
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for the full KAN-MAMMOTE model.

        Args:
            timestamps: A tensor of shape (batch_size, sequence_length), containing tk.
            features: A tensor of shape (batch_size, sequence_length, input_feature_dim), containing uk.
            auxiliary_features: Optional. (batch_size, sequence_length, aux_feature_dim) auxiliary features.

        Returns:
            A tuple containing:
            - model_output: Tensor of shape (batch_size, sequence_length, output_dim_for_task).
            - regularization_losses: Dictionary of computed regularization losses.
        """
        batch_size, seq_len, _ = features.shape

        # Move tensors to device
        timestamps = timestamps.to(self.config.device, dtype=self.config.dtype)
        features = features.to(self.config.device, dtype=self.config.dtype)
        if auxiliary_features is not None:
            auxiliary_features = auxiliary_features.to(self.config.device, dtype=self.config.dtype)

        # Project initial raw features to model dimension
        current_features = self.initial_feature_proj(features) # (batch_size, seq_len, d_model)

        # To collect regularization terms (load balancing weights)
        all_expert_weights_for_loss = []
        all_expert_selection_masks = []

        # List to store outputs of the final Mamba block at each timestep
        sequence_outputs = []

        # Iterate through the sequence, processing timestep by timestep
        for i in range(seq_len):
            tk_current = timestamps[:, i].unsqueeze(1) # (batch_size, 1)
            uk_current = current_features[:, i, :]    # (batch_size, d_model)
            
            # Determine tk_previous (tk-1). For the first step (i=0), tk_previous can be tk_current.
            tk_previous = timestamps[:, i-1].unsqueeze(1) if i > 0 else tk_current
            
            aux_features_current = auxiliary_features[:, i, :] if auxiliary_features is not None else None

            # Process through stacked Mamba blocks for the current timestep
            # Each block's output becomes the input for the next block.
            # The hidden state for the MambaSSM is managed internally within each DynamicMambaSSM
            # when processing a single-timestep sequence (B, 1, D).
            
            current_timestep_input_to_layer = uk_current
            for l_idx, block in enumerate(self.mamba_blocks):
                # The `forward` of ContinuousMambaBlock is designed to take (B,D) for a single timestep
                # It internally expands to (B,1,D) for MambaSSM and then squeezes back.
                
                # Pass auxiliary features to K-MOTE if configured
                expert_aux_features = aux_features_current if self.config.use_aux_features_router else None

                # For each block's K-MOTE: (tk_current, tk_previous, uk_current)
                # The output from the block is the `uk` for the next block.
                block_output, expert_weights, expert_mask = block(
                    uk_current_input=current_timestep_input_to_layer, # Input to this block
                    tk_current=tk_current,
                    tk_previous=tk_previous,
                    # hidden_state_prev_timestep is managed internally by Mamba2's forward method for seq_len=1
                    # when it is used in sequence context, not step by step decoding.
                    # No need to pass layer_idx here as it's set in init.
                )
                current_timestep_input_to_layer = block_output # Output of this block is input to next
                
                # Collect expert weights for regularization (from the first layer's K-MOTE for simplicity,
                # or you could aggregate from all K-MOTE instances if desired, but typically one is enough).
                if l_idx == 0: # Collect from the first layer's K-MOTE
                    all_expert_weights_for_loss.append(expert_weights)
                    all_expert_selection_masks.append(expert_mask)

            sequence_outputs.append(current_timestep_input_to_layer) # Final output of top layer for this timestep

        # Stack outputs from all timesteps
        # (batch_size, seq_len, d_model)
        final_sequence_embedding = torch.stack(sequence_outputs, dim=1)

        # Apply final prediction head
        model_output = self.prediction_head(final_sequence_embedding) # (batch_size, seq_len, output_dim_for_task)

        # Calculate regularization losses
        # Load balancing loss: sum (or mean) collected weights
        if len(all_expert_weights_for_loss) > 0:
            # Concatenate collected weights across timesteps
            concatenated_expert_weights = torch.cat(all_expert_weights_for_loss, dim=0) # (batch*seq_len, num_experts)
            load_balance_loss = self.regularization_handler.compute_load_balance_loss(
                concatenated_expert_weights
            )
        else:
            load_balance_loss = torch.tensor(0.0, device=self.config.device, dtype=self.config.dtype)

        # Sobolev and Total Variation losses (currently stubs)
        sobolev_l2_loss = self.regularization_handler.compute_sobolev_l2_loss(self)
        total_variation_loss = self.regularization_handler.compute_total_variation_loss(self)

        regularization_losses = {
            "load_balance_loss": load_balance_loss,
            "sobolev_l2_loss": sobolev_l2_loss,
            "total_variation_loss": total_variation_loss
        }

        return model_output, regularization_losses