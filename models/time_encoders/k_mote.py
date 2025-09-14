# file: models/time_encoders/k_mote.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- CORRECTED IMPORT ---
# We explicitly import from the efficient_kan library to avoid ambiguity.
# This library's KAN class uses the `grid_size` argument.
try:
    from efficient_kan import KAN
except ImportError:
    print("FATAL ERROR: The 'efficient-kan' library is not installed. Please run 'pip install efficient-kan'")
    KAN = None # This will cause a failure, which is intended if the lib is missing.

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
            # This call is now correct because we are using the `efficient_kan` library.
            if KAN is None: raise ImportError("efficient_kan is required for b_spline mode.")
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
            t = t.unsqueeze(-1)
            centers = self.centers.unsqueeze(0).unsqueeze(0)
            gammas = self.gammas.unsqueeze(0).unsqueeze(0)
            dist_sq = (t - centers).pow(2)
            rbf_out = torch.exp(-F.softplus(gammas) * dist_sq).sum(dim=2)
            return self.linear(rbf_out)


# --- Expert 2: Fourier KAN Layer ---

class FourierKANLayer(nn.Module):
    """
    An expert specializing in periodic patterns using a Fourier series expansion.
    """
    def __init__(self, input_dim: int, output_dim: int, n_harmonics: int = 10):
        super().__init__()
        self.output_dim = output_dim
        self.n_harmonics = n_harmonics
        self.cos_coeffs = nn.Parameter(torch.randn(input_dim, n_harmonics, output_dim))
        self.sin_coeffs = nn.Parameter(torch.randn(input_dim, n_harmonics, output_dim))
        self.frequencies = nn.Parameter(torch.randn(input_dim, n_harmonics))
        self.phases = nn.Parameter(torch.randn(input_dim, n_harmonics))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1).unsqueeze(-1)
        k_times_omega_t_plus_phi = (
            self.frequencies.unsqueeze(0).unsqueeze(0).unsqueeze(-1) * t +
            self.phases.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        )
        cos_term = torch.cos(k_times_omega_t_plus_phi) * self.cos_coeffs.unsqueeze(0).unsqueeze(0)
        sin_term = torch.sin(k_times_omega_t_plus_phi) * self.sin_coeffs.unsqueeze(0).unsqueeze(0)
        output = (cos_term + sin_term).sum(dim=(2, 3)) + self.bias
        return output

# --- Expert 3: Wavelet KAN Layer ---

class WaveletKANLayer(nn.Module):
    """
    An expert specializing in transient, localized patterns using a learnable wavelet basis.
    """
    def __init__(self, input_dim: int, output_dim: int, n_wavelets: int = 16):
        super().__init__()
        self.n_wavelets = n_wavelets
        self.scales = nn.Parameter(torch.randn(input_dim, n_wavelets))
        self.shifts = nn.Parameter(torch.randn(input_dim, n_wavelets))
        self.linear = nn.Linear(input_dim * n_wavelets, output_dim)

    def morlet_wavelet(self, t):
        c = math.pi**(-0.25)
        return c * torch.exp(-0.5 * t**2) * torch.cos(5.0 * t)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)
        scales = F.softplus(self.scales).unsqueeze(0).unsqueeze(0)
        shifts = self.shifts.unsqueeze(0).unsqueeze(0)
        wavelet_input = (t - shifts) / scales
        wavelet_activations = self.morlet_wavelet(wavelet_input)
        B, S, _, _ = wavelet_activations.shape
        wavelet_activations = wavelet_activations.view(B, S, -1)
        return self.linear(wavelet_activations)


# --- The Main K-MOTE Module ---

class KMOTE(nn.Module):
    """
    Kolmogorov-Arnold Mixture-of-Time-Experts (K-MOTE) for absolute time encoding.
    """
    def __init__(self, input_dim: int, output_dim: int, use_layernorm: bool = True):
        super().__init__()
        self.output_dim = output_dim
        self.experts = nn.ModuleList([
            SplineKANLayer(input_dim, output_dim, basis_function='b_spline'),
            FourierKANLayer(input_dim, output_dim, n_harmonics=16),
            WaveletKANLayer(input_dim, output_dim, n_wavelets=16),
            SplineKANLayer(input_dim, output_dim, basis_function='rbf')
        ])
        self.num_experts = len(self.experts)
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.num_experts)
        )
        self.layer_norm = nn.LayerNorm(output_dim) if use_layernorm else nn.Identity()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        gating_logits = self.gating_network(t)
        gating_weights = F.softmax(gating_logits, dim=-1)
        expert_outputs = [expert(t) for expert in self.experts]
        stacked_outputs = torch.stack(expert_outputs, dim=2)
        weighted_sum = (gating_weights.unsqueeze(-1) * stacked_outputs).sum(dim=2)
        output_embedding = self.layer_norm(weighted_sum)
        return output_embedding

if __name__ == '__main__':
    batch_size = 16
    seq_len = 100
    input_dim = 1
    output_dim = 64
    k_mote_encoder = KMOTE(input_dim=input_dim, output_dim=output_dim)
    dummy_time = torch.randn(batch_size, seq_len, input_dim)
    absolute_time_embedding = k_mote_encoder(dummy_time)
    print("K-MOTE Encoder:")
    print("Input time shape:", dummy_time.shape)
    print("Output embedding shape:", absolute_time_embedding.shape)
    assert absolute_time_embedding.shape == (batch_size, seq_len, output_dim)
    print("\nShape verification successful!")