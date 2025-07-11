# kan_mamote/src/layers/kan_base_layer.py

import torch
import torch.nn as nn
from typing import Literal

# Import configuration and only the basis functions that this KANLayer will use
from src.utils.config import KANMAMOTEConfig
from .basis_functions import FourierBasis, GaussianKernelBasis, WaveletBasis, BaseBasisFunction

class KANLayer(nn.Module):
    """
    A generic KAN layer that applies a linear transformation followed by
    a learnable basis function from the specified type.
    This module represents one "edge" or connection in a traditional KAN graph,
    where the activation is a learnable function.
    
    This KANLayer is used for Fourier, RKHS-Gaussian, and Wavelet experts in K-MOTE.
    The Spline expert is handled directly by the external MatrixKANLayer.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 # 'spline' removed from Literal as it's handled separately in K_MOTE
                 basis_type: Literal['fourier', 'rkhs_gaussian', 'wavelet'], 
                 config: KANMAMOTEConfig):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis_type = basis_type
        self.config = config

        self.alpha_weights = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.alpha_bias = nn.Parameter(torch.zeros(out_features))

        if basis_type == 'fourier':
            self.basis_function: BaseBasisFunction = FourierBasis(out_features, config)
        # elif basis_type == 'spline': # This branch is intentionally removed
        #     self.basis_function: BaseBasisFunction = SplineBasis(out_features, config)
        elif basis_type == 'rkhs_gaussian':
            self.basis_function: BaseBasisFunction = GaussianKernelBasis(out_features, config)
        elif basis_type == 'wavelet':
            self.basis_function: BaseBasisFunction = WaveletBasis(out_features, config)
        else:
            # Update the error message to reflect the current supported types
            raise ValueError(f"Unsupported basis_type: {basis_type}. "
                             "Choose from 'fourier', 'rkhs_gaussian', 'wavelet'. "
                             "The 'spline' expert is handled directly by MatrixKANLayer in K_MOTE.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the KAN layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_features).
               For time encoding, `in_features` is typically 1 (the scalar timestamp).
               In a multi-layer KAN, `x` would be the output from the previous layer.

        Returns:
            Output tensor of shape (batch_size, out_features).
        """
        x_prime = torch.matmul(x, self.alpha_weights) + self.alpha_bias
        return self.basis_function(x_prime)