# kan_mamote/src/models/moe_router.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from src.utils.config import KANMAMOTEConfig

class MoERouter(nn.Module):
    """
    Mixture-of-Experts Router that determines which experts to activate
    for a given timestamp input.
    
    This router aligns with the K-MOTE diagram, supporting:
    1. Four specific experts (Fourier, Spline, Gaussian, Wavelet)
    2. Top-K selection mechanism
    3. Optional auxiliary feature input
    """
    def __init__(self, input_dim: int, num_experts: int = 4, config: Optional[KANMAMOTEConfig] = None):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts  # Should be 4 for the four expert types
        self.config = config
        
        # Router network - converts timestamp (+ optional aux features) to expert weights
        self.router_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_experts),
        )
        
        # Add noise to encourage exploration (during training)
        self.noise_scale = 1e-2 if config is None else getattr(config, 'router_noise_scale', 1e-2)
        
        # Load balancing parameters
        self.use_load_balancing = True if config is None else getattr(config, 'use_load_balancing', True)
        self.balance_coefficient = 0.01 if config is None else getattr(config, 'balance_coefficient', 0.01)
        
    def forward(self, 
                timestamp_input: torch.Tensor, 
                auxiliary_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Route timestamps to appropriate experts.
        
        Args:
            timestamp_input: (batch_size, 1) or (batch_size,) tensor of timestamps
            auxiliary_features: Optional (batch_size, aux_dim) tensor of auxiliary features
            
        Returns:
            logits: (batch_size, num_experts) raw routing logits
            weights: (batch_size, num_experts) normalized expert weights (after softmax)
        """
        batch_size = timestamp_input.shape[0]
        
        # Ensure timestamp has correct shape
        if timestamp_input.dim() == 1:
            timestamp_input = timestamp_input.unsqueeze(1)  # (batch_size,) -> (batch_size, 1)
        elif timestamp_input.dim() > 2:
            timestamp_input = timestamp_input.squeeze()  # Handle extra dimensions
            if timestamp_input.dim() == 1:
                timestamp_input = timestamp_input.unsqueeze(1)
        
        # Prepare router input
        router_input = timestamp_input
        
        # Combine with auxiliary features if provided
        if auxiliary_features is not None and self.config and self.config.use_aux_features_router:
            # Ensure auxiliary features have correct shape
            if auxiliary_features.dim() == 1:
                auxiliary_features = auxiliary_features.unsqueeze(1)
            elif auxiliary_features.dim() > 2:
                auxiliary_features = auxiliary_features.view(batch_size, -1)  # Flatten extra dimensions
                
            # Concatenate timestamp with auxiliary features
            router_input = torch.cat([timestamp_input, auxiliary_features], dim=1)
        
        # Get routing logits
        logits = self.router_network(router_input)
        
        # Add noise during training to prevent mode collapse
        if self.training and self.noise_scale > 0:
            noise = torch.randn_like(logits) * self.noise_scale
            logits = logits + noise
        
        # Get normalized weights via softmax
        weights = F.softmax(logits, dim=-1)
        
        # Compute load balancing loss during training
        if self.training and self.use_load_balancing:
            # Compute mean expert activation across batch
            mean_expert_activation = weights.mean(dim=0)
            
            # Load balancing loss: encourage uniform distribution across experts
            target_distribution = torch.ones_like(mean_expert_activation) / self.num_experts
            load_balance_loss = F.kl_div(
                mean_expert_activation.log(), 
                target_distribution, 
                reduction='batchmean'
            ) * self.balance_coefficient
            
            # Register the loss so it can be accessed and included in total loss
            self.load_balance_loss = load_balance_loss
            
            # Add small gradient from load balancing loss without affecting forward pass
            weights = weights + 0 * load_balance_loss
        
        return logits, weights
    
    def get_load_balance_loss(self) -> torch.Tensor:
        """Return the load balancing loss if available."""
        if hasattr(self, 'load_balance_loss'):
            return self.load_balance_loss
        return torch.tensor(0.0, device=next(self.parameters()).device)