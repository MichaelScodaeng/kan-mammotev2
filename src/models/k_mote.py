# kan_mamote/src/models/k_mote.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional

# Import necessary components
from src.utils.config import KANMAMOTEConfig
from src.layers.kan_base_layer import KANLayer
from src.models.moe_router import MoERouter

class K_MOTE(nn.Module):
    """
    K-MOTE (Kernel-Mixture-of-Time-Experts) module.
    This is the core adaptive time encoding module, which uses a Mixture-of-Experts (MoE)
    architecture to dynamically select and combine specialized KAN-based experts.
    """
    def __init__(self, config: KANMAMOTEConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.D_time_per_expert = config.D_time_per_expert

        # 1. Initialize the MoE Router
        # Router input dimension: 1 for timestamp (tk) + raw_event_feature_dim if aux features are used.
        router_input_dim = 1 # for tk
        if config.use_aux_features_router:
            router_input_dim += config.raw_event_feature_dim # Add auxiliary features dimension
        
        self.router = MoERouter(input_dim=router_input_dim, num_experts=self.num_experts, config=config)

        # 2. Initialize the Specialized KAN-based Basis Function Experts
        # Each expert is a KANLayer that outputs a portion of the total D_time.
        # Input to each KANLayer is always the scalar timestamp (or its difference), so in_features is 1.
        self.experts = nn.ModuleDict({
            "fourier": KANLayer(in_features=1, out_features=self.D_time_per_expert, basis_type='fourier', config=config),
            "spline": KANLayer(in_features=1, out_features=self.D_time_per_expert, basis_type='spline', config=config),
            "rkhs_gaussian": KANLayer(in_features=1, out_features=self.D_time_per_expert, basis_type='rkhs_gaussian', config=config),
            "wavelet": KANLayer(in_features=1, out_features=self.D_time_per_expert, basis_type='wavelet', config=config),
        })
        self.expert_names = list(self.experts.keys()) # Keep order consistent

    def forward(self, 
                timestamp_input: torch.Tensor, 
                auxiliary_features: 'Optional[torch.Tensor]' = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Performs the forward pass of the K-MOTE module.

        Args:
            timestamp_input: The scalar timestamp (tk or delta_tk) for encoding, shape (batch_size, 1).
            auxiliary_features: Optional auxiliary temporal features for the router,
                                shape (batch_size, aux_feature_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - embedding: The final D_time-dimensional time embedding, shape (batch_size, D_time).
                - expert_weights_for_loss: Raw weights before Top-Ktop, for load balancing loss,
                                           shape (batch_size, num_experts).
                - expert_selection_mask: Binary mask indicating which experts were selected (for analysis),
                                         shape (batch_size, num_experts).
        """
        batch_size = timestamp_input.shape[0]

        # 1. Router computes logits and weights
        # logits: (batch_size, num_experts)
        # raw_weights: (batch_size, num_experts) - softmax probabilities
        router_logits, raw_weights = self.router(timestamp_input, auxiliary_features)
        
        # Store raw weights for load balancing loss later (before Top-Ktop)
        expert_weights_for_loss = raw_weights 

        # 2. Top-Ktop Expert Selection (Efficient Dispatch)
        # Get the top-Ktop expert indices for each item in the batch
        # topk_weights: (batch_size, K_top), topk_indices: (batch_size, K_top)
        topk_weights, topk_indices = torch.topk(raw_weights, self.config.K_top, dim=-1)
        
        # Create a mask for selected experts
        expert_selection_mask = torch.zeros_like(raw_weights).bool().scatter_(
            1, topk_indices, True
        ) # (batch_size, num_experts) - boolean mask where True means selected
        
        # Create a dispatch mask for weighted sum: non-selected experts get weight 0
        # This mask is used to zero out expert outputs.
        # It's important to use the raw_weights for actual output to maintain differentiability
        # and ensure the non-selected experts effectively contribute zero.
        dispatch_weights = torch.zeros_like(raw_weights).to(raw_weights.device)
        dispatch_weights.scatter_(1, topk_indices, topk_weights)
        
        # Re-normalize the selected weights if sum is important (optional, often done for consistency)
        # Or, directly use topk_weights and ensure non-selected get 0 contribution.
        # For simplicity, we just zero out non-selected experts' contributions.
        # The load balancing loss operates on raw_weights, so this is fine.
        
        # 3. Expert Processing and Weighted Combination
        expert_outputs_list = [] # List to collect outputs from all experts
        
        for i, expert_name in enumerate(self.expert_names):
            expert_module = self.experts[expert_name]
            
            # (batch_size, D_time_per_expert)
            expert_output = expert_module(timestamp_input)
            expert_outputs_list.append(expert_output)
        
        # Stack all expert outputs: (batch_size, num_experts, D_time_per_expert)
        all_expert_outputs = torch.stack(expert_outputs_list, dim=1)

        # Apply dispatch weights for sparse combination (Top-Ktop)
        # weights: (batch_size, num_experts, 1) - expanded for broadcasting
        # all_expert_outputs: (batch_size, num_experts, D_time_per_expert)
        
        # Create a zero tensor for non-selected experts' contributions
        weighted_expert_outputs = all_expert_outputs * dispatch_weights.unsqueeze(-1)
        
        # Sum across experts to get the final embedding for each batch item
        # (batch_size, D_time_per_expert)
        # The sum of all D_time_per_expert portions should combine to D_time
        
        # If all experts produce D_time_per_expert, then the final embedding is a concatenation
        # of the weighted sums of each D_time_per_expert slice.
        
        # Let's clarify the combination method:
        # "K-MOTE...dynamically selects experts and combines their specialized learnable basis function outputs."
        # This implies a weighted sum of their outputs, where the output dimension for each expert
        # aligns to form the total D_time.
        # E.g., if D_time = 128 and num_experts = 4, each expert produces 32 dimensions.
        # The final embedding is [E1_output_slice | E2_output_slice | E3_output_slice | E4_output_slice]
        # where each slice is weighted by its expert's dispatch weight.

        # Correct combination: Weighted sum across experts, then concatenate or combine.
        # If each expert produces a full D_time_per_expert, and D_time is the sum of these,
        # then the weights should apply to the *entire* expert output.
        
        # Final Embedding (batch_size, D_time)
        # Method 1: Weighted sum of all expert outputs if they all produce D_time
        # In our current setup, each expert produces `D_time_per_expert`.
        # So it's a weighted sum *within* that D_time_per_expert dimension.
        # This implies the KANLayer should produce `D_time` as output, and MoE takes care of mixing.
        
        # Re-evaluating K-MOTE output D_time:
        # "K-MOTE...producing a Dtime-dimensional embedding."
        # "Each expert maps to a portion of the Dtime-dimensional output space."
        # This means the final embedding is conceptually `[weighted_fourier_part | weighted_spline_part | ...]`
        # So each expert directly outputs its assigned portion of D_time, and these are then concatenated.
        # The weighting then implies selecting *which* portion gets updated or is primary.

        # Let's adjust for "Each expert maps to a portion of the D_time-dimensional output space."
        # This is a more common MoE setup where experts specialize in different parts of the output.
        # So, expert_output[i] is the D_time_per_expert for that expert.
        # The `raw_weights` (before Top-Ktop) indicate how much each expert influences its *assigned* output slice.
        
        final_embedding_parts = []
        for i, expert_name in enumerate(self.expert_names):
            expert_module = self.experts[expert_name]
            expert_output = expert_module(timestamp_input) # (batch_size, D_time_per_expert)
            
            # Apply individual expert's dispatch weight to its output portion
            # weights for this expert: (batch_size, 1)
            expert_weight_for_this_slice = dispatch_weights[:, i].unsqueeze(-1)
            weighted_slice = expert_output * expert_weight_for_this_slice
            final_embedding_parts.append(weighted_slice)

        # Concatenate all weighted parts to form the final D_time embedding
        # Result: (batch_size, D_time)
        final_embedding = torch.cat(final_embedding_parts, dim=-1)

        return final_embedding, expert_weights_for_loss, expert_selection_mask