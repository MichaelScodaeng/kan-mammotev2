# kan_mamote/src/models/kan_mamote.py

import torch
import torch.nn as nn
from typing import Tuple, Optional, List

# Import configuration
from src.utils.config import KANMAMOTEConfig
# Import K-MOTE and ContinuousMambaBlock
from src.models.k_mote import K_MOTE

# Ensure ContinuousMambaBlock does NOT re-import or redefine KANMAMOTEConfig
from src.models.c_mamba import ContinuousMambaBlock

class KAN_MAMOTE_Model(nn.Module):
    """
    The complete KAN-MAMMOTE (Kernel-Adaptive-Neural-Mamba-Mixture-of-Time-Experts) model.
    
    TRUE KAN-MAMMOTE Architecture following the diagram:
    1. Independent K-MOTE embeddings for t_k and t_k-1
    2. Temporal differences in embedding space (t_k - t_k-1)  
    3. Faster-KAN processing of temporal differences → Δt embedding
    4. Continuous Mamba: current embedding as input, Δt embedding as delta parameter
    5. Output: Absolute-Relative t_k Embedding
    """
    def __init__(self, config: KANMAMOTEConfig):
        super().__init__()
        self.config = config

        # Import Faster-KAN layer for temporal difference processing
        from src.models.immediate_fasterkan_layer import FasterKANTemporalLayer

        # 1. K-MOTE Module for computing embeddings
        # Single K-MOTE module will be used for both t_k and t_k-1 independently
        self.k_mote = K_MOTE(config)

        # 2. Faster-KAN Layer for processing temporal differences
        # Takes temporal differences in embedding space (t_k - t_k-1) → Δt embedding
        self.faster_kan_layer = FasterKANTemporalLayer(
            input_dim=config.D_time,    # K-MOTE embedding dimension
            output_dim=config.D_time,   # Same dimension for Δt embedding
            grid_min=-2.0,
            grid_max=2.0,
            num_grids=8
        )

        # 3. TRUE KAN-MAMMOTE Continuous-Mamba Block
        # Now correctly follows the diagram pattern
        self.ct_mamba_block = ContinuousMambaBlock(
            input_dim=config.D_time,        # K-MOTE embedding dimension
            config=config,
            kmote_layer=self.k_mote,        # Pass K-MOTE reference for independent computations
            faster_kan_layer=self.faster_kan_layer  # Pass Faster-KAN reference for Δt processing
        )

        # Optional: Task-specific output head
        # Output dimension matches the final embedding from Continuous-Mamba
        self.output_dim = config.D_time  # Final embedding dimension

    def forward(self, 
                timestamps: torch.Tensor,         # (batch_size, seq_len, 1) - absolute timestamps
                event_features: torch.Tensor,     # (batch_size, seq_len, raw_event_feature_dim) - raw features
                initial_mamba_state: Optional[torch.Tensor] = None # Not used in new implementation
               ) -> Tuple[torch.Tensor, dict]:
        """
        TRUE KAN-MAMMOTE forward pass following the diagram exactly:
        
        Flow: Raw timestamps & features → ContinuousMambaBlock → Absolute-Relative t_k Embedding
        
        The ContinuousMambaBlock handles:
        1. Independent K-MOTE embeddings for t_k and t_k-1
        2. Temporal differences computation (t_k - t_k-1)
        3. Faster-KAN processing of differences → Δt embedding
        4. Continuous Mamba with current as input, Δt as delta parameter
        
        Args:
            timestamps: Raw absolute timestamps, shape (batch_size, seq_len, 1)
            event_features: Raw features at each timestamp, shape (batch_size, seq_len, raw_event_feature_dim)
            initial_mamba_state: Not used in the new implementation
            
        Returns:
            Tuple[torch.Tensor, dict]:
                - absolute_relative_embeddings: Final absolute-relative t_k embeddings,
                                               shape (batch_size, seq_len, D_time)
                - analysis_info: Dict with intermediate results for MoE loss and analysis
        """
        
        # TRUE KAN-MAMMOTE processing through the Continuous-Mamba Block
        # The block handles all the diagram logic internally
        absolute_relative_embeddings, analysis_info = self.ct_mamba_block(
            timestamps=timestamps,      # Raw timestamps (not pre-computed embeddings)
            features=event_features     # Raw features (not pre-computed embeddings)
        )
        
        # Extract expert weights for MoE load balancing loss
        # The analysis_info contains intermediate embeddings and expert usage information
        current_embeddings = analysis_info.get('current_embeddings')  # t_k embeddings
        previous_embeddings = analysis_info.get('previous_embeddings')  # t_k-1 embeddings
        
        # For compatibility with existing loss functions, we can extract expert weights
        # from the K-MOTE computations if needed (this would require modifying the block
        # to return expert weights, or we can compute them here for loss purposes)
        
        # Since K-MOTE expert weights are computed internally in the block,
        # we can add them to the analysis_info or compute them separately if needed for loss
        
        # For now, return the absolute-relative embeddings and analysis info
        return absolute_relative_embeddings, analysis_info