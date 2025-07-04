# kan_mamote/src/models/moe_router.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from src.utils.config import KANMAMOTEConfig

class MoERouter(nn.Module):
    """
    Implements the Mixture-of-Experts (MoE) gating mechanism (router).
    This router takes the current timestamp (tk) and optional auxiliary features,
    and outputs logits for each expert, which are then normalized into dispatch weights.
    """
    def __init__(self, input_dim: int, num_experts: int, config: KANMAMOTEConfig):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.config = config

        # The router is typically a small MLP
        layers = []
        current_dim = input_dim
        for hidden_dim in config.router_mlp_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU()) # Use ReLU for simplicity in router
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, num_experts)) # Output logits for each expert
        self.mlp = nn.Sequential(*layers)

    def forward(self, tk: torch.Tensor, auxiliary_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            tk: Current scalar timestamp, shape (batch_size, 1).
            auxiliary_features: Optional auxiliary temporal features, shape (batch_size, aux_feature_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - logits: Raw logits for each expert, shape (batch_size, num_experts).
                - weights: Softmax probabilities (dispatch weights) for each expert, shape (batch_size, num_experts).
        """
        router_input = tk
        if self.config.use_aux_features_router and auxiliary_features is not None:
            # Concatenate timestamp with auxiliary features
            router_input = torch.cat([tk, auxiliary_features], dim=-1)
        
        # Ensure router_input matches the expected input_dim
        if router_input.shape[-1] != self.input_dim:
            raise ValueError(f"Router input dimension mismatch. Expected {self.input_dim}, got {router_input.shape[-1]}.")

        logits = self.mlp(router_input)
        weights = F.softmax(logits, dim=-1) # Apply softmax to get normalized dispatch weights

        return logits, weights