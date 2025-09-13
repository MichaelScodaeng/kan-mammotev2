# file: models/k_mote.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from efficient_kan import KAN

# --- Expert 1: Spline/RBF KAN Layer ---

class SplineKANLayer(nn.Module):
    """
    An expert based on Kolmogorov-Arnold Networks, using either B-splines or RBFs.
    
    This expert is a general-purpose function approximator.
    - 'b_spline' mode uses the efficient-kan library directly.
    - 'rbf' mode implements a layer of Radial Basis Functions, inspired by Faster-KAN.
    """
    def __init__(self, input_dim: int, output_dim: int, grid_size: int = 5, basis_function: str = 'b_spline'):
        super().__init__()
        if basis_function not in ['b_spline', 'rbf']:
            raise ValueError("basis_function must be 'b_spline' or 'rbf'")
        self.basis_function = basis_function

        if self.basis_function == 'b_spline':
            self.kan = KAN([input_dim, output_dim], grid_size=grid_size)
        else: # RBF implementation
            self.centers = nn.Parameter(torch.randn(input_dim, output_dim))
            self.gammas = nn.Parameter(torch.randn(input_dim, output_dim))
            self.linear = nn.Linear(output_dim, output_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if self.basis_function == 'b_spline':
            return self.kan(t)
        else:
            # RBF forward pass
            # t shape: (B, S, D_in) -> (B, S, D_in, 1)
            # centers, gammas shape: (D_in, D_out) -> (1, 1, D_in, D_out)
            t = t.unsqueeze(-1)
            centers = self.centers.unsqueeze(0).unsqueeze(0)
            gammas = gammas.unsqueeze(0).unsqueeze(0)
            
            # Use squared Euclidean distance
            dist_sq = (t - centers).pow(2)
            # RBF activation: exp(-gamma * dist^2)
            rbf_out = torch.exp(-F.softplus(gammas) * dist_sq).sum(dim=2) # Sum over input_dim
            return self.linear(rbf_out)


# --- Expert 2: Fourier KAN Layer ---

class FourierKANLayer(nn.Module):
    """
    An expert specializing in periodic patterns using a Fourier series expansion.
    This is more direct than a generic KAN for learning periodicities.
    It models the function as: f(t) = Σ [a_k * cos(kωt + φ) + b_k * sin(kωt + φ)]
    """
    def __init__(self, input_dim: int, output_dim: int, n_harmonics: int = 10):
        super().__init__()
        self.output_dim = output_dim
        self.n_harmonics = n_harmonics

        # Learnable parameters for the Fourier series
        # We learn coefficients for each harmonic for each output dimension
        self.cos_coeffs = nn.Parameter(torch.randn(input_dim, n_harmonics, output_dim))
        self.sin_coeffs = nn.Parameter(torch.randn(input_dim, n_harmonics, output_dim))
        self.frequencies = nn.Parameter(torch.randn(input_dim, n_harmonics))
        self.phases = nn.Parameter(torch.randn(input_dim, n_harmonics))
        
        # A simple bias term
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B, S, D_in)
        # We need to broadcast across harmonics and output dimensions
        # t -> (B, S, D_in, 1, 1)
        # freqs/phases -> (1, 1, D_in, H, 1)
        # coeffs -> (1, 1, D_in, H, D_out)
        t = t.unsqueeze(-1).unsqueeze(-1)
        
        k_times_omega_t_plus_phi = (
            self.frequencies.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * t +
            self.phases.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )
        
        cos_term = torch.cos(k_times_omega_t_plus_phi) * self.cos_coeffs.unsqueeze(0).unsqueeze(0)
        sin_term = torch.sin(k_times_omega_t_plus_phi) * self.sin_coeffs.unsqueeze(0).unsqueeze(0)
        
        # Sum over harmonics and input dimensions
        output = (cos_term + sin_term).sum(dim=(2, 3)) + self.bias
        return output

# --- Expert 3: Wavelet KAN Layer ---

class WaveletKANLayer(nn.Module):
    """
    An expert specializing in transient, localized patterns using a learnable wavelet basis.
    We use the Morlet wavelet as the mother wavelet.
    """
    def __init__(self, input_dim: int, output_dim: int, n_wavelets: int = 16):
        super().__init__()
        self.n_wavelets = n_wavelets
        
        # Each wavelet has a learnable scale (dilation) and shift (translation)
        self.scales = nn.Parameter(torch.randn(input_dim, n_wavelets))
        self.shifts = nn.Parameter(torch.randn(input_dim, n_wavelets))
        
        # Linear layer to map wavelet activations to the output dimension
        self.linear = nn.Linear(input_dim * n_wavelets, output_dim)

    def morlet_wavelet(self, t):
        """ The Morlet wavelet function. """
        c = math.pi**(-0.25)
        return c * torch.exp(-0.5 * t**2) * torch.cos(5.0 * t)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t shape: (B, S, D_in)
        # scales/shifts shape: (D_in, W)
        # We want to compute: wavelet((t - shift) / scale)
        # t -> (B, S, D_in, 1)
        # scales/shifts -> (1, 1, D_in, W)
        t = t.unsqueeze(-1)
        scales = F.softplus(self.scales).unsqueeze(0).unsqueeze(0) # Ensure scales are positive
        shifts = self.shifts.unsqueeze(0).unsqueeze(0)

        # Apply wavelet transformation
        wavelet_input = (t - shifts) / scales
        wavelet_activations = self.morlet_wavelet(wavelet_input)
        
        # Reshape for the linear layer
        # (B, S, D_in, W) -> (B, S, D_in * W)
        B, S, _, _ = wavelet_activations.shape
        wavelet_activations = wavelet_activations.view(B, S, -1)
        
        return self.linear(wavelet_activations)


# --- The Main K-MOTE Module ---

class KMOTE(nn.Module):
    """
    Kolmogorov-Arnold Mixture-of-Time-Experts (K-MOTE) for absolute time encoding.

    This module combines multiple specialized KAN-based experts using a learned
    gating mechanism, allowing it to adaptively model diverse temporal patterns.
    """
    def __init__(self, input_dim: int, output_dim: int, use_layernorm: bool = True):
        super().__init__()
        self.output_dim = output_dim

        # Initialize the list of experts
        self.experts = nn.ModuleList([
            SplineKANLayer(input_dim, output_dim, basis_function='b_spline'),
            FourierKANLayer(input_dim, output_dim, n_harmonics=16),
            WaveletKANLayer(input_dim, output_dim, n_wavelets=16),
            SplineKANLayer(input_dim, output_dim, basis_function='rbf') # RBF version
        ])
        self.num_experts = len(self.experts)

        # Gating network: a small MLP that learns to weigh the experts
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.num_experts)
        )
        
        self.layer_norm = nn.LayerNorm(output_dim) if use_layernorm else nn.Identity()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the K-MOTE module.

        Args:
            t (torch.Tensor): Input tensor of absolute timestamps.
                              Shape: (batch_size, sequence_length, input_dim).

        Returns:
            torch.Tensor: The learned absolute time embedding.
                          Shape: (batch_size, sequence_length, output_dim).
        """
        # 1. Compute gating weights
        # gating_logits shape: (B, S, n_experts)
        gating_logits = self.gating_network(t)
        gating_weights = F.softmax(gating_logits, dim=-1) # (B, S, n_experts)

        # 2. Compute outputs from all experts
        expert_outputs = [expert(t) for expert in self.experts]
        
        # Stack expert outputs along a new dimension
        # expert_outputs is a list of tensors of shape (B, S, D_out)
        # stacked_outputs shape: (B, S, n_experts, D_out)
        stacked_outputs = torch.stack(expert_outputs, dim=2)

        # 3. Apply gating weights
        # gating_weights -> (B, S, n_experts, 1) for broadcasting
        # The weighted sum is over the 'n_experts' dimension (dim=2)
        weighted_sum = (gating_weights.unsqueeze(-1) * stacked_outputs).sum(dim=2)
        
        # 4. Apply final LayerNorm for stability
        output_embedding = self.layer_norm(weighted_sum)
        
        return output_embedding

if __name__ == '__main__':
    # --- Example Usage ---
    batch_size = 16
    seq_len = 100
    input_dim = 1 # for a single time value
    output_dim = 64

    # Instantiate the full K-MOTE encoder
    k_mote_encoder = KMOTE(input_dim=input_dim, output_dim=output_dim)
    
    # Create dummy absolute time tensor
    # Note: It's good practice to normalize time inputs before feeding them to the model
    dummy_time = torch.randn(batch_size, seq_len, input_dim)

    # Get the embedding
    absolute_time_embedding = k_mote_encoder(dummy_time)

    print("K-MOTE Encoder:")
    print("Input time shape:", dummy_time.shape)
    print("Output embedding shape:", absolute_time_embedding.shape)
    
    # Verify the output shape
    assert absolute_time_embedding.shape == (batch_size, seq_len, output_dim)
    print("\nShape verification successful!")