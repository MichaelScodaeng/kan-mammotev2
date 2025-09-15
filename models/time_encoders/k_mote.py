# file: models/time_encoders/k_mote.py (Corrected and Refactored)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from efficient_kan import KAN
except ImportError:
    print("FATAL ERROR: The 'efficient-kan' library is not installed. Please run 'pip install efficient-kan'")
    KAN = None

# --- Expert 1: Spline/RBF KAN Layer (Corrected) ---

class SplineKANLayer(nn.Module):
    """
    An expert based on Kolmogorov-Arnold Networks, using either B-splines or RBFs.
    """
    def __init__(self, input_dim: int, output_dim: int, grid_size: int = 5, basis_function: str = 'b_spline'):
        super().__init__()
        if basis_function not in ['b_spline', 'rbf']:
            raise ValueError("basis_function must be 'b_spline' or 'rbf'")
        self.basis_function = basis_function

        if self.basis_function == 'b_spline':
            if KAN is None: raise ImportError("efficient_kan is required for b_spline mode.")
            self.kan = KAN([input_dim, output_dim], grid_size=grid_size)
        else: # RBF implementation
            self.centers = nn.Parameter(torch.randn(input_dim, output_dim, grid_size))
            self.gammas = nn.Parameter(torch.randn(input_dim, output_dim, grid_size))
            self.linear = nn.Linear(input_dim * grid_size, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with conditional logic for 'b_spline' or 'rbf'.
        x: (batch_size, seq_len, input_dim)
        """
        # Ensure input is 3D
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        batch_size, seq_len, _ = x.shape
        
        # --- START OF CORRECTION ---
        if self.basis_function == 'b_spline':
            # Flatten to 2D for efficient-kan processing
            x_flat = x.view(-1, x.size(-1))  # (B * S, D_in)
            output_flat = self.kan(x_flat)   # (B * S, D_out)
            # Reshape back to 3D
            output = output_flat.view(batch_size, seq_len, -1)
        else:
            # RBF forward pass
            x_expanded = x.unsqueeze(-1).unsqueeze(-1) # (B, S, D_in, 1, 1)
            centers = self.centers.unsqueeze(0).unsqueeze(0)  # (1, 1, D_in, D_out, grid)
            gammas = self.gammas.unsqueeze(0).unsqueeze(0)    # (1, 1, D_in, D_out, grid)

            dist_sq = (x_expanded - centers).pow(2)
            rbf_out = torch.exp(-F.softplus(gammas) * dist_sq) # (B, S, D_in, D_out, grid)
            # Sum over the output_dim dimension to get final RBF activations
            rbf_activated = rbf_out.sum(dim=3) # (B, S, D_in, grid)
            
            # Flatten the last two dimensions for the linear layer
            rbf_flat = rbf_activated.view(batch_size, seq_len, -1) # (B, S, D_in * grid)
            output = self.linear(rbf_flat)
        # --- END OF CORRECTION ---
            
        return output


# --- Expert 2: Fourier KAN Layer (Refactored) ---

class FourierKANLayer(nn.Module):
    """
    An expert specializing in periodic patterns using a Fourier series expansion.
    """
    def __init__(self, input_dim: int, output_dim: int, n_harmonics: int = 16):
        super().__init__()
        self.output_dim = output_dim
        self.cos_coeffs = nn.Parameter(torch.randn(input_dim, n_harmonics, output_dim))
        self.sin_coeffs = nn.Parameter(torch.randn(input_dim, n_harmonics, output_dim))
        self.frequencies = nn.Parameter(torch.randn(input_dim, n_harmonics))
        self.bias = nn.Parameter(torch.randn(output_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2: t = t.unsqueeze(-1)
        
        t_expanded = t.unsqueeze(-1)  # (B, S, D_in, 1)
        freqs = self.frequencies.unsqueeze(0).unsqueeze(0) # (1, 1, D_in, n_harmonics)
        
        arg = freqs * t_expanded # (B, S, D_in, n_harmonics)
        
        cos_term = torch.cos(arg).unsqueeze(-1) * self.cos_coeffs.unsqueeze(0).unsqueeze(0)
        sin_term = torch.sin(arg).unsqueeze(-1) * self.sin_coeffs.unsqueeze(0).unsqueeze(0)
        
        output = (cos_term + sin_term).sum(dim=(2, 3)) + self.bias
        return output


# --- Expert 3: Wavelet KAN Layer (Refactored) ---

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
        if t.dim() == 2: t = t.unsqueeze(-1)
        batch_size, seq_len, _ = t.shape
        
        t_expanded = t.unsqueeze(-1)
        scales = F.softplus(self.scales).unsqueeze(0).unsqueeze(0)
        shifts = self.shifts.unsqueeze(0).unsqueeze(0)             
        
        wavelet_input = (t_expanded - shifts) / scales
        wavelet_activations = self.morlet_wavelet(wavelet_input)
        
        wavelet_flat = wavelet_activations.view(batch_size, seq_len, -1)
        output = self.linear(wavelet_flat)
        return output


# --- The Main K-MOTE Module (Simplified and Corrected) ---

class KMOTE(nn.Module):
    """
    Kolmogorov-Arnold Mixture-of-Time-Experts (K-MOTE) for absolute time encoding.
    """
    def __init__(self, input_dim: int, output_dim: int, use_layernorm: bool = True):
        super().__init__()
        self.output_dim = output_dim
        self.experts = nn.ModuleList([
            SplineKANLayer(input_dim, output_dim, basis_function='b_spline', grid_size=8),
            FourierKANLayer(input_dim, output_dim, n_harmonics=16),
            WaveletKANLayer(input_dim, output_dim, n_wavelets=16),
            SplineKANLayer(input_dim, output_dim, basis_function='rbf', grid_size=8)
        ])
        self.num_experts = len(self.experts)
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.num_experts)
        )
        self.layer_norm = nn.LayerNorm(output_dim) if use_layernorm else nn.Identity()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Ensure input is 3D: (B, S, D_in)
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        gating_logits = self.gating_network(t)
        gating_weights = F.softmax(gating_logits, dim=-1) # (B, S, num_experts)
        
        # Each expert is guaranteed to return (B, S, D_out)
        expert_outputs = [expert(t) for expert in self.experts]
        
        stacked_outputs = torch.stack(expert_outputs, dim=-1) # (B, S, D_out, num_experts)
        gating_weights = gating_weights.unsqueeze(-2)         # (B, S, 1, num_experts)
        
        weighted_sum = (gating_weights * stacked_outputs).sum(dim=-1) # (B, S, D_out)
        
        output_embedding = self.layer_norm(weighted_sum)
        return output_embedding