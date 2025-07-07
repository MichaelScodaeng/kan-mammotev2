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
                auxiliary_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for K-MOTE (Kernel-Mixture-of-Time-Experts).

        Args:
            timestamp_input: (batch_size, 1) tensor of timestamps.
            auxiliary_features: (batch_size, aux_feature_dim) tensor or None.

        Returns:
            final_embedding: (batch_size, D_time) tensor.
            expert_weights_for_loss: (batch_size, num_experts) tensor (raw softmax weights, before Top-K).
            expert_selection_mask: (batch_size, num_experts) boolean tensor (True if expert selected).
        """
        batch_size = timestamp_input.shape[0]

        # 1. Router computes logits and weights
        router_logits, raw_weights = self.router(timestamp_input, auxiliary_features)
        expert_weights_for_loss = raw_weights

        # 2. Top-K selection
        topk_weights, topk_indices = torch.topk(raw_weights, self.config.K_top, dim=-1)
        expert_selection_mask = torch.zeros_like(raw_weights, dtype=torch.bool).scatter_(1, topk_indices, True)
        dispatch_weights = torch.zeros_like(raw_weights).scatter(1, topk_indices, topk_weights)

        # 3. Compute all expert outputs in one pass
        expert_outputs = [self.experts[name](timestamp_input) for name in self.expert_names]
        # (batch_size, num_experts, D_time_per_expert)
        all_expert_outputs = torch.stack(expert_outputs, dim=1)

        # 4. Apply dispatch weights (zero out non-selected experts)
        weighted_expert_outputs = all_expert_outputs * dispatch_weights.unsqueeze(-1)

        # 5. Concatenate weighted outputs to form the final embedding
        final_embedding = weighted_expert_outputs.view(batch_size, -1)

        return final_embedding, expert_weights_for_loss, expert_selection_mask