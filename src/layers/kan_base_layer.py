# kan_mamote/src/layers/kan_base_layer.py

import torch
import torch.nn as nn
from typing import Literal

# Import configuration and all defined basis functions
from src.utils.config import KANMAMOTEConfig
from .basis_functions import FourierBasis, SplineBasis, GaussianKernelBasis, WaveletBasis, BaseBasisFunction

class KANLayer(nn.Module):
    """
    A generic KAN layer that applies a linear transformation followed by
    a learnable basis function from the specified type.
    This module represents one "edge" or connection in a traditional KAN graph,
    where the activation is a learnable function.
    """
    def __init__(self, 
                 in_features: int, 
                 out_features: int, 
                 basis_type: Literal['fourier', 'spline', 'rkhs_gaussian', 'wavelet'], 
                 config: KANMAMOTEConfig):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.basis_type = basis_type
        self.config = config

        # The linear transformation: x_prime = x @ self.alpha_weights + self.alpha_bias
        # This maps the input (e.g., a timestamp or output from a previous KAN layer)
        # to a domain suitable for the basis function to operate on across `out_features` dimensions.
        # Initialize weights with small values for stability.
        self.alpha_weights = nn.Parameter(torch.randn(in_features, out_features) * 0.1)
        self.alpha_bias = nn.Parameter(torch.zeros(out_features))

        # Instantiate the specific basis function based on the `basis_type`
        # Each basis function is responsible for its own learnable parameters.
        if basis_type == 'fourier':
            self.basis_function: BaseBasisFunction = FourierBasis(out_features, config)
        elif basis_type == 'spline':
            self.basis_function: BaseBasisFunction = SplineBasis(out_features, config)
        elif basis_type == 'rkhs_gaussian':
            self.basis_function: BaseBasisFunction = GaussianKernelBasis(out_features, config)
        elif basis_type == 'wavelet':
            self.basis_function: BaseBasisFunction = WaveletBasis(out_features, config)
        else:
            raise ValueError(f"Unsupported basis_type: {basis_type}. "
                             "Choose from 'fourier', 'spline', 'rkhs_gaussian', 'wavelet'.")

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
        # Apply the linear transformation
        # x: (batch_size, in_features)
        # self.alpha_weights: (in_features, out_features)
        # Result: (batch_size, out_features)
        x_prime = torch.matmul(x, self.alpha_weights) + self.alpha_bias

        # Pass through the learnable basis function
        # The basis_function takes (batch_size, out_features) and returns (batch_size, out_features)
        return self.basis_function(x_prime)