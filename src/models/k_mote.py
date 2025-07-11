# kan_mamote/src/models/k_mote.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional

# Import necessary components
from src.utils.config import KANMAMOTEConfig
from src.layers.kan_base_layer import KANLayer # Your custom KANLayer wrapper
from src.models.moe_router import MoERouter
# IMPORTANT: Import the actual MatrixKANLayer from its path
# Adjust this path based on where MatrixKANLayer.py is relative to k_mote.py
from src.models.kan.MatrixKANLayer import MatrixKANLayer 


class K_MOTE(nn.Module):
    """
    K-MOTE (Kernel-Mixture-of-Time-Experts) module.
    This is the core adaptive time encoding module, which uses a Mixture-of-Experts (MoE)
    architecture to dynamically select and combine specialized KAN-based experts.
    """
    def __init__(self, config: KANMAMOTEConfig):
        super().__init__()
        self.config = config
        
        self.num_experts = 4 # Fixed to 4 for the four expert types

        # 1. Initialize the MoE Router
        router_input_dim = 1
        if config.use_aux_features_router:
            router_input_dim += config.raw_event_feature_dim
        self.router = MoERouter(input_dim=router_input_dim, num_experts=self.num_experts, config=config)

        # 2. Initialize the Specialized KAN-based Basis Function Experts
        # Each expert will now output directly to config.D_time, as per paper's formula Σ ae * φe(tk)
        self.experts = nn.ModuleDict({
            # Fourier-KAN: Uses your custom KANLayer wrapper + FourierBasis
            "fourier": KANLayer(in_features=1, out_features=config.D_time, basis_type='fourier', config=config),
            
            # Spline-KAN: Directly uses the external MatrixKANLayer.
            # This avoids the redundant linear transformation from kan_base_layer.KANLayer.
            # Its input will be the timestamp itself (1 dimension).
            # Its output will be the full D_time dimension.
            "spline": MatrixKANLayer(
                in_dim=1, # The input is the scalar timestamp
                out_dim=config.D_time, # The output is the D_time-dimensional embedding
                num=config.spline_grid_size, # Number of grid intervals (G)
                k=config.spline_degree,      # Spline order (k)
                noise_scale=getattr(config, 'kan_noise_scale', 0.1), 
                scale_base_mu=getattr(config, 'kan_scale_base_mu', 0.0),
                scale_base_sigma=getattr(config, 'kan_scale_base_sigma', 1.0),
                grid_eps=getattr(config, 'kan_grid_eps', 0.02),
                grid_range=getattr(config, 'kan_grid_range', [-1, 1]),
                sp_trainable=getattr(config, 'kan_sp_trainable', True),
                sb_trainable=getattr(config, 'kan_sb_trainable', True),
                device=config.device
            ),
            
            # RKHS-KAN (Gaussian): Uses your custom KANLayer wrapper + GaussianKernelBasis
            "gaussian": KANLayer(in_features=1, out_features=config.D_time, basis_type='rkhs_gaussian', config=config),
            
            # Wavelet-KAN: Uses your custom KANLayer wrapper + WaveletBasis
            "wavelet": KANLayer(in_features=1, out_features=config.D_time, basis_type='wavelet', config=config),
        })
        self.expert_names = list(self.experts.keys())

        # Final projection: This is now just an Identity as each expert already outputs D_time.
        # LayerNorm remains for stability.
        self.final_projection = nn.Identity() 
        self.layer_norm = nn.LayerNorm(config.D_time)

    def forward(self, 
                timestamp_input: torch.Tensor, 
                auxiliary_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for K-MOTE (Kernel-Mixture-of-Time-Experts).
        """
        batch_size = timestamp_input.shape[0]

        router_logits, raw_weights = self.router(timestamp_input, auxiliary_features)
        expert_weights_for_loss = raw_weights

        topk_weights, topk_indices = torch.topk(raw_weights, self.config.K_top, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)
        expert_selection_mask = torch.zeros_like(raw_weights, dtype=torch.bool).scatter_(1, topk_indices, True)
        dispatch_weights = torch.zeros_like(raw_weights).scatter(1, topk_indices, topk_weights)

        # 3. Compute all expert outputs in one pass, handling MatrixKANLayer's tuple output
        processed_expert_outputs = []
        for name in self.expert_names:
            expert_module = self.experts[name]
            output = expert_module(timestamp_input) # Input is always (batch_size, 1)

            if isinstance(expert_module, MatrixKANLayer):
                # MatrixKANLayer returns (y, preacts, postacts, postspline)
                processed_expert_outputs.append(output[0]) # Take only the main output 'y'
            else:
                # Other KANLayer instances return just the tensor
                processed_expert_outputs.append(output)
        
        # Stack outputs: (batch_size, num_experts, D_time)
        all_expert_outputs = torch.stack(processed_expert_outputs, dim=1)

        # 4. Apply dispatch weights (zero out non-selected experts)
        weighted_expert_outputs = all_expert_outputs * dispatch_weights.unsqueeze(-1)

        # 5. Sum across experts to form the final embedding
        combined = weighted_expert_outputs.sum(dim=1)
        
        # 6. Final projection (now identity) and normalization
        final_embedding = self.layer_norm(self.final_projection(combined))

        return final_embedding, expert_weights_for_loss, expert_selection_mask