"""
Original Time Encoder

Traditional cosine-based time encoding from the original DyGMamba implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from .base_encoder import BaseTimeEncoder


class OriginalTimeEncoder(BaseTimeEncoder):
    """
    Original time encoder using cosine basis functions.
    
    This encoder transforms time differences into embeddings using trainable
    frequency parameters and cosine activations, similar to positional encoding.
    """
    
    def __init__(self, time_dim: int, parameter_requires_grad: bool = True, device: str = 'cpu'):
        """
        Initialize original time encoder.
        
        Args:
            time_dim: Dimension of output time encoding
            parameter_requires_grad: Whether parameters should be trainable
            device: Device to place encoder on
        """
        super().__init__(time_dim, device)
        
        # Trainable linear layer for time encoding
        self.w = nn.Linear(1, time_dim)
        
        # Initialize with geometric progression frequencies
        freq_init = (torch.from_numpy(1 / 10 ** np.linspace(0, 9, time_dim, dtype=np.float32))
                    .reshape(time_dim, -1))
        self.w.weight = nn.Parameter(freq_init)
        self.w.bias = nn.Parameter(torch.zeros(time_dim))

        # Control gradient computation
        if not parameter_requires_grad:
            self.w.weight.requires_grad = False
            self.w.bias.requires_grad = False

    # MODIFICATION: The forward signature is changed
    def forward(self, t_abs: torch.Tensor, t_rel: torch.Tensor) -> torch.Tensor:
        """
        Computes time encodings. Conforms to the unified interface.
        
        This encoder only uses relative time (t_rel) and ignores t_abs.
        """
        print("⚠️ WARNING: OriginalTimeEncoder only uses relative time (t_rel). t_abs is ignored.")
        print("t_rel", t_rel)
        # The 'timestamps' this encoder needs are the relative time deltas.
        timestamps = t_rel
        
        # --- The rest of the original logic is unchanged ---
        original_shape = timestamps.shape
        
        if timestamps.dim() == 1:
            timestamps = timestamps.unsqueeze(-1).unsqueeze(-1)
            single_batch = True
        elif timestamps.dim() == 2:
            timestamps = timestamps.unsqueeze(-1)
            single_batch = False
        else:
            single_batch = False
            
        output = torch.cos(self.w(timestamps))
        
        if single_batch and len(original_shape) == 1:
            output = output.squeeze(1)
            
        return output
        
    def get_config(self) -> dict:
        """Return configuration for reproducibility."""
        config = super().get_config()
        config.update({
            'parameter_requires_grad': self.w.weight.requires_grad
        })
        return config
