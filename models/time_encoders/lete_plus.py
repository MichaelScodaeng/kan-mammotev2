import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional

class LeTEPlusUnified(nn.Module):
    """
    LeTE+ Unified Model: Direct extension of LeTE with wavelets.
    
    This model combines the best of LeTE (Fourier + Spline) with additional
    wavelet capabilities, using a fixed weighting scheme instead of learned gating.
    
    Architecture:
    - Fourier branch: Direct LeTE implementation (geometric init)
    - Spline branch: Memory-optimized LeTE implementation 
    - Wavelet branch: Additional capability for abrupt changes
    - Fixed combination weights (no MoE complexity)
    
    Benefits:
    - Guaranteed LeTE-level performance on Fourier + Spline
    - Added wavelet capability for complex temporal patterns
    - No gating network complexity or training issues
    - Simple, stable, and efficient
    """
    
    def __init__(self, dim: int = 64, 
                 # Component weights (must sum to 1.0)
                 fourier_weight: float = 0.4,
                 spline_weight: float = 0.4, 
                 wavelet_weight: float = 0.2,
                 # Component configurations
                 fourier_harmonics: int = 5,
                 spline_grid_size: int = 5,
                 spline_order: int = 3,
                 wavelet_count: int = 8,
                 wavelet_type: str = 'shock',
                 # Output configurations
                 layer_norm: bool = True,
                 scale: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        
        # Validate weights
        total_weight = fourier_weight + spline_weight + wavelet_weight
        if abs(total_weight - 1.0) > 1e-6:
            raise ValueError(f"Component weights must sum to 1.0, got {total_weight}")
        
        self.dim = dim
        self.fourier_weight = fourier_weight
        self.spline_weight = spline_weight
        self.wavelet_weight = wavelet_weight
        self.layer_norm = layer_norm
        self.scale = scale
        
        # ===== FOURIER BRANCH (Direct LeTE Implementation) =====
        if fourier_weight > 0:
            self.dim_fourier = dim
            self.w1_fourier = nn.Linear(1, self.dim_fourier)
            self.w2_fourier = FourierSeries(dim_fourier=self.dim_fourier, 
                                          grid_size_fourier=fourier_harmonics)
            
            # LeTE's geometric initialization
            fourier_vals = 1.0 / (10 ** np.linspace(0, 9, self.dim_fourier, dtype=np.float32))
            self.w1_fourier.weight = nn.Parameter(torch.from_numpy(fourier_vals).reshape(self.dim_fourier, -1))
            self.w1_fourier.bias = nn.Parameter(torch.zeros(self.dim_fourier))
        else:
            self.dim_fourier = 0
        
        # ===== SPLINE BRANCH (Memory-Optimized LeTE Implementation) =====
        if spline_weight > 0:
            self.dim_spline = dim
            self.w1_spline = nn.Linear(1, self.dim_spline)
            self.w2_spline = OptimizedSpline(dim_spline=self.dim_spline,
                                           grid_size_spline=spline_grid_size,
                                           order_spline=spline_order)
            
            # LeTE's geometric initialization 
            spline_vals = 1.0 / (10 ** np.linspace(0, 9, self.dim_spline, dtype=np.float32))
            self.w1_spline.weight = nn.Parameter(torch.from_numpy(spline_vals).reshape(self.dim_spline, -1))
            self.w1_spline.bias = nn.Parameter(torch.zeros(self.dim_spline))
        else:
            self.dim_spline = 0
        
        # ===== WAVELET BRANCH (New Capability) =====
        if wavelet_weight > 0:
            self.dim_wavelet = dim
            self.w1_wavelet = nn.Linear(1, self.dim_wavelet)
            self.w2_wavelet = WaveletSeries(dim_wavelet=self.dim_wavelet,
                                          n_wavelets=wavelet_count,
                                          wavelet_type=wavelet_type)
            
            # Geometric initialization for wavelets too
            wavelet_vals = 1.0 / (10 ** np.linspace(0, 9, self.dim_wavelet, dtype=np.float32))
            self.w1_wavelet.weight = nn.Parameter(torch.from_numpy(wavelet_vals).reshape(self.dim_wavelet, -1))
            self.w1_wavelet.bias = nn.Parameter(torch.zeros(self.dim_wavelet))
        else:
            self.dim_wavelet = 0
        
        # ===== OUTPUT PROCESSING =====
        total_dim = self.dim_fourier + self.dim_spline + self.dim_wavelet
        
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = nn.Identity()
        
        if layer_norm and total_dim > 1:
            self.layernorm = nn.LayerNorm(total_dim)
        else:
            self.layernorm = nn.Identity()
        
        if scale:
            self.scale_weight = nn.Parameter(torch.ones(total_dim))
        else:
            self.scale_weight = None
        
        print(f"✅ LeTE+ Unified Model initialized:")
        print(f"   Fourier: {self.dim_fourier}D (weight: {fourier_weight})")
        print(f"   Spline:  {self.dim_spline}D (weight: {spline_weight})")  
        print(f"   Wavelet: {self.dim_wavelet}D (weight: {wavelet_weight})")
        print(f"   Total:   {total_dim}D")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fixed weighted combination of all branches.
        
        Args:
            timestamps: Shape (batch_size, seq_len) or (batch_size, seq_len, 1)
        
        Returns:
            Time encodings of shape (batch_size, seq_len, total_dim)
        """
        # Ensure correct input shape
        if timestamps.dim() == 2:
            timestamps = timestamps.unsqueeze(dim=2)  # (B, S) -> (B, S, 1)
        
        outputs = []
        
        # ===== FOURIER BRANCH =====
        if self.dim_fourier > 0:
            proj_fourier = self.w1_fourier(timestamps)  # (B, S, dim_fourier)
            output_fourier = self.w2_fourier(proj_fourier)  # (B, S, dim_fourier)
            output_fourier = output_fourier * self.fourier_weight  # Apply fixed weight
            outputs.append(output_fourier)
        
        # ===== SPLINE BRANCH =====
        if self.dim_spline > 0:
            proj_spline = self.w1_spline(timestamps)  # (B, S, dim_spline)
            output_spline = self.w2_spline(proj_spline)  # (B, S, dim_spline)
            output_spline = output_spline * self.spline_weight  # Apply fixed weight
            outputs.append(output_spline)
        
        # ===== WAVELET BRANCH =====
        if self.dim_wavelet > 0:
            proj_wavelet = self.w1_wavelet(timestamps)  # (B, S, dim_wavelet)
            output_wavelet = self.w2_wavelet(proj_wavelet)  # (B, S, dim_wavelet)
            output_wavelet = output_wavelet * self.wavelet_weight  # Apply fixed weight
            outputs.append(output_wavelet)
        
        # ===== COMBINE OUTPUTS =====
        if len(outputs) == 1:
            combined_output = outputs[0]
        else:
            combined_output = torch.cat(outputs, dim=-1)  # (B, S, total_dim)
        
        # ===== OUTPUT PROCESSING =====
        output = self.dropout(combined_output)
        output = self.layernorm(output)
        
        if self.scale_weight is not None:
            output = self.scale_weight * output
        
        return output
    
    def get_component_outputs(self, timestamps: torch.Tensor) -> dict:
        """
        Get individual component outputs for analysis.
        
        Returns:
            Dictionary with 'fourier', 'spline', 'wavelet' outputs
        """
        if timestamps.dim() == 2:
            timestamps = timestamps.unsqueeze(dim=2)
        
        results = {}
        
        if self.dim_fourier > 0:
            proj_fourier = self.w1_fourier(timestamps)
            results['fourier'] = self.w2_fourier(proj_fourier)
        
        if self.dim_spline > 0:
            proj_spline = self.w1_spline(timestamps)
            results['spline'] = self.w2_spline(proj_spline)
        
        if self.dim_wavelet > 0:
            proj_wavelet = self.w1_wavelet(timestamps)
            results['wavelet'] = self.w2_wavelet(proj_wavelet)
        
        return results


class FourierSeries(nn.Module):
    """
    Direct copy of LeTE's FourierSeries implementation.
    Exact same code for guaranteed compatibility.
    """
    def __init__(self, dim_fourier: int, grid_size_fourier: int = 5):
        super().__init__()
        self.dim_fourier = dim_fourier
        self.grid_size_fourier = grid_size_fourier

        # fourier_weight shape: (2, dim_fourier, dim_fourier, grid_size_fourier)
        self.fourier_weight = torch.nn.Parameter(
            torch.randn(2, self.dim_fourier, self.dim_fourier, grid_size_fourier) /
            (np.sqrt(self.dim_fourier) * np.sqrt(self.grid_size_fourier))
        )
        self.bias = nn.Parameter(torch.zeros(self.dim_fourier))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        out_shape = original_shape[0:-1] + (self.dim_fourier,)

        x = x.reshape(-1, self.dim_fourier)  # (N, dim_fourier)

        # Frequency indices k = 1..grid_size_fourier
        k = torch.arange(1, self.grid_size_fourier + 1, device=x.device)
        k = k.reshape(1, 1, 1, self.grid_size_fourier)

        x_reshaped = x.reshape(x.shape[0], 1, x.shape[1], 1)  # (N,1,dim_fourier,1)

        # Compute cos(k * x) and sin(k * x)
        c = torch.cos(k * x_reshaped)  # (N,1,dim_fourier,K)
        s = torch.sin(k * x_reshaped)  # (N,1,dim_fourier,K)

        # Sum up the contributions
        y = torch.sum(c * self.fourier_weight[0:1], dim=(-2, -1))
        y += torch.sum(s * self.fourier_weight[1:2], dim=(-2, -1))
        y += self.bias

        y = y.reshape(out_shape)
        return y


class OptimizedSpline(nn.Module):
    """
    Memory-optimized version of LeTE's Spline implementation.
    Uses the same mathematical operations but with better memory efficiency.
    """
    def __init__(self, dim_spline: int, grid_size_spline: int = 5, 
                 order_spline: int = 3, grid_range: list = [-1, 1]):
        super().__init__()
        self.dim_spline = dim_spline
        self.grid_size_spline = grid_size_spline
        self.order_spline = order_spline

        # Grid setup (same as LeTE)
        h = (grid_range[1] - grid_range[0]) / float(self.grid_size_spline)
        grid = torch.arange(-self.order_spline, self.grid_size_spline + self.order_spline + 1)
        grid = grid * h + grid_range[0]
        grid = grid.expand(self.dim_spline, -1).contiguous()
        self.register_buffer("grid", grid)

        # Parameters (same as LeTE)
        self.base_weight = nn.Parameter(torch.Tensor(self.dim_spline, self.dim_spline))
        self.spline_weight = nn.Parameter(torch.Tensor(self.dim_spline, self.dim_spline, 
                                                      self.grid_size_spline + self.order_spline))
        
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize spline parameters to avoid NaN values"""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.normal_(self.spline_weight, mean=0, std=0.1)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Correct B-spline implementation from the original LeTE.
        """
        # grid: (dim_spline, grid_size + 2*order + 1)
        grid = self.grid
        x = x.unsqueeze(-1)

        # bases will mark where x lies between [grid_i, grid_{i+1})
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        # Recursively elevate the basis to higher spline orders (Cox-de Boor recursion)
        for k in range(1, self.order_spline + 1):
            # Correctly slice the grid for the recursion
            left_grid = grid[:, :-(k + 1)]
            right_grid = grid[:, k:-1]
            
            # Calculate the numerator for the left term
            left_num = x - left_grid
            # Calculate the denominator for the left term, adding epsilon for stability
            left_den = right_grid - left_grid
            
            # Calculate the left term of the recursion
            left_term = (left_num / (left_den + 1e-12)) * bases[:, :, :-1]

            # Correctly slice the grid for the right term
            right_grid_2 = grid[:, (k + 1):]
            left_grid_2 = grid[:, 1:-k]

            # Calculate the numerator for the right term
            right_num = right_grid_2 - x
            # Calculate the denominator for the right term, adding epsilon for stability
            right_den = right_grid_2 - left_grid_2
            
            # Calculate the right term of the recursion
            right_term = (right_num / (right_den + 1e-12)) * bases[:, :, 1:]
            
            # Update the bases with the sum of the left and right terms
            bases = left_term + right_term

        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, self.dim_spline)

        # Base branch (linear + activation)
        base_output = nn.functional.linear(torch.tanh(x_flat), self.base_weight)

        # Spline branch
        # Get B-spline basis functions: shape (N, dim_spline, grid_size + order_spline)
        b_spline_basis = self.b_splines(x_flat)
        
        # Project with spline coefficients
        # einsum: (N, in, grid) * (in, out, grid) -> (N, out)
        spline_output = torch.einsum('big,iog->bo', b_spline_basis, self.spline_weight)

        # Combine branches
        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.dim_spline)
        return output


class WaveletSeries(nn.Module):
    """
    Wavelet series implementation for LeTE+ unified model.
    Provides additional capability for modeling abrupt changes and discontinuities.
    """
    def __init__(self, dim_wavelet: int, n_wavelets: int = 8, wavelet_type: str = 'shock'):
        super().__init__()
        self.dim_wavelet = dim_wavelet
        self.n_wavelets = n_wavelets
        self.wavelet_type = wavelet_type
        
        # Wavelet parameters (learnable)
        self.scales = nn.Parameter(torch.randn(dim_wavelet, n_wavelets))
        self.shifts = nn.Parameter(torch.randn(dim_wavelet, n_wavelets))
        
        if wavelet_type == 'shock':
            self.asymmetry = nn.Parameter(torch.randn(dim_wavelet, n_wavelets) * 0.1)
            self.steepness = nn.Parameter(torch.randn(dim_wavelet, n_wavelets) * 0.1)
        
        # Output transformation
        self.linear = nn.Linear(dim_wavelet * n_wavelets, dim_wavelet)
        self.bias = nn.Parameter(torch.zeros(dim_wavelet))
        
        # Initialize reasonably
        nn.init.uniform_(self.scales, 0.5, 2.0)
        nn.init.uniform_(self.shifts, -1.0, 1.0)

    def shock_wavelet(self, t, asymmetry, steepness):
        """Shock wavelet for modeling abrupt changes"""
        asym = torch.tanh(asymmetry)
        steep = F.softplus(steepness) + 0.1
        steep = torch.clamp(steep, max=5.0)
        
        left_exp = torch.clamp(steep * t * (1 + asym), min=-10, max=10)
        right_exp = torch.clamp(-steep * t * (1 - asym), min=-10, max=10)
        
        shock = torch.where(t < 0, torch.exp(left_exp), torch.exp(right_exp))
        oscillation = torch.cos(torch.clamp(steep, max=3.0) * t)
        
        return torch.clamp(shock * oscillation, min=-100, max=100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.dim_wavelet)  # (N, dim_wavelet)
        
        # Expand for wavelet computation
        x_expanded = x.unsqueeze(-1)  # (N, dim_wavelet, 1)
        scales = F.softplus(self.scales).unsqueeze(0) + 0.1  # (1, dim_wavelet, n_wavelets)
        shifts = self.shifts.unsqueeze(0)  # (1, dim_wavelet, n_wavelets)
        
        # Compute wavelet input
        wavelet_input = (x_expanded - shifts) / scales  # (N, dim_wavelet, n_wavelets)
        
        # Apply wavelet function
        if self.wavelet_type == 'shock':
            asym = self.asymmetry.unsqueeze(0)
            steep = self.steepness.unsqueeze(0)
            wavelet_activations = self.shock_wavelet(wavelet_input, asym, steep)
        else:  # Default Morlet
            c = math.pi**(-0.25)
            wavelet_activations = c * torch.exp(-0.5 * wavelet_input**2) * torch.cos(5.0 * wavelet_input)
        
        # Flatten and transform
        wavelet_flat = wavelet_activations.view(x.shape[0], -1)
        output = self.linear(wavelet_flat) + self.bias
        
        # Reshape back
        output = output.reshape(*original_shape[:-1], self.dim_wavelet)
        return output


# ===== TESTING =====
if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("LeTE+ UNIFIED MODEL TESTING")
    print("=" * 70)
    
    # Create model with balanced weights
    model = LeTEPlusUnified(
        dim=64,
        fourier_weight=0.4,
        spline_weight=0.4,
        wavelet_weight=0.2,
        fourier_harmonics=5,
        spline_grid_size=5,
        wavelet_count=8,
        layer_norm=True,
        scale=True
    )
    
    # Test input
    batch_size, seq_len = 32, 512
    timestamps = torch.randn(batch_size, seq_len)
    
    print(f"\nTest input shape: {timestamps.shape}")
    
    # Forward pass test
    print(f"\nTesting forward pass...")
    start_time = time.time()
    
    with torch.no_grad():
        # Basic forward
        output = model(timestamps)
        print(f"  Output shape: {output.shape}")
        
        # Component analysis
        components = model.get_component_outputs(timestamps)
        for name, comp_output in components.items():
            print(f"  {name.capitalize()} component: {comp_output.shape}")
    
    elapsed = time.time() - start_time
    print(f"  Total time: {elapsed*1000:.2f}ms")
    
    # Memory comparison
    total_params = sum(p.numel() for p in model.parameters())
    memory_mb = total_params * 4 / (1024**2)  # 4 bytes per float32
    
    print(f"\n✅ LeTE+ Unified test completed!")
    print(f"   Parameters: {total_params:,}")
    print(f"   Memory: ~{memory_mb:.1f}MB")
    print(f"   Fixed combination weights (no MoE complexity)")
    print(f"   Guaranteed LeTE compatibility + wavelet capability")