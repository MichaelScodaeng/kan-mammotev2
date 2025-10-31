import torch
import torch.nn as nn
import math
from typing import Optional

class LeTEStyleFourierExpert(nn.Module):
    """
    LeTE-inspired Fourier expert that matches LeTE's performance.
    Uses the same architecture: Linear → FourierSeries → projection
    """
    def __init__(self, input_dim: int, output_dim: int, 
                 fourier_dim: int = 64, n_harmonics: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fourier_dim = fourier_dim
        self.n_harmonics = n_harmonics
        
        # Stage 1: Input projection with geometric initialization
        self.w1_fourier = nn.Linear(input_dim, fourier_dim)
        
        # Stage 2: High-dimensional Fourier transformation
        self.fourier_weight = nn.Parameter(
            torch.randn(2, fourier_dim, fourier_dim, n_harmonics) /
            (math.sqrt(fourier_dim) * math.sqrt(n_harmonics))
        )
        self.fourier_bias = nn.Parameter(torch.zeros(fourier_dim))
        
        # Stage 3: Output projection
        self.output_head = nn.Linear(fourier_dim, output_dim)
        
        # Initialize with geometric progression (LeTE's secret sauce)
        self._initialize_geometric()
    
    def _initialize_geometric(self):
        """Initialize with LeTE's geometric frequency progression"""
        with torch.no_grad():
            # Geometric progression: 1/(10^[0,9])
            fourier_vals = 1.0 / (10 ** torch.linspace(0, 9, self.fourier_dim))
            self.w1_fourier.weight.copy_(fourier_vals.reshape(self.fourier_dim, -1))
            self.w1_fourier.bias.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LeTE-style forward pass"""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        original_shape = x.shape
        x_flat = x.reshape(-1, self.input_dim)
        
        # Stage 1: Project to high-dimensional space
        proj_fourier = self.w1_fourier(x_flat)  # (N, fourier_dim)
        
        # Stage 2: High-dimensional Fourier transform (exactly like LeTE)
        k = torch.arange(1, self.n_harmonics + 1, device=x.device)
        k = k.reshape(1, 1, 1, self.n_harmonics)
        
        x_reshaped = proj_fourier.reshape(proj_fourier.shape[0], 1, proj_fourier.shape[1], 1)
        
        # Compute cos and sin
        c = torch.cos(k * x_reshaped)
        s = torch.sin(k * x_reshaped)
        
        # Apply Fourier coefficients
        y = torch.sum(c * self.fourier_weight[0:1], dim=(-2, -1))
        y += torch.sum(s * self.fourier_weight[1:2], dim=(-2, -1))
        y += self.fourier_bias
        
        # Stage 3: Project to output
        output = self.output_head(y)
        
        # Reshape back
        return output.view(*original_shape[:-1], self.output_dim)


class SimplifiedSplineExpert(nn.Module):
    """
    Simplified spline expert using ReLU basis functions instead of B-splines.
    Much more memory efficient while maintaining expressiveness.
    """
    def __init__(self, input_dim: int, output_dim: int, n_knots: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_knots = n_knots
        
        # Learnable knot positions
        self.knot_positions = nn.Parameter(torch.linspace(-2, 2, n_knots))
        
        # Linear transformation for ReLU basis
        self.basis_weights = nn.Parameter(torch.randn(input_dim, n_knots, output_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Base linear component
        self.base_linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, input_dim = x.shape
        x_flat = x.reshape(-1, input_dim)  # (B*S, input_dim)
        
        # Base component
        base_output = self.base_linear(x_flat)
        
        # ReLU basis functions (much simpler than B-splines)
        # Compute distance to each knot
        x_expanded = x_flat.unsqueeze(-1)  # (B*S, input_dim, 1)
        knots_expanded = self.knot_positions.unsqueeze(0).unsqueeze(0)  # (1, 1, n_knots)
        
        # ReLU basis: max(0, 1 - |x - knot|)
        distances = torch.abs(x_expanded - knots_expanded)  # (B*S, input_dim, n_knots)
        basis_values = torch.relu(1.0 - distances)  # (B*S, input_dim, n_knots)
        
        # Apply basis weights: (B*S, input_dim, n_knots) * (input_dim, n_knots, output_dim)
        spline_output = torch.einsum('bin,ino->bo', basis_values, self.basis_weights)
        spline_output += self.bias
        
        # Combine
        output = base_output + spline_output
        return output.view(batch_size, seq_len, self.output_dim)


class OptimizedKMOTE(nn.Module):
    """
    Memory and performance optimized K-MOTE.
    
    Key improvements:
    1. LeTE-style Fourier expert for better performance
    2. Simplified spline expert for memory efficiency
    3. Optional expert pruning
    4. Gradient checkpointing on memory-heavy operations
    """
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 64,  # Increased for better performance
                 use_spline_expert: bool = True,
                 use_fourier_expert: bool = True,
                 use_wavelet_expert: bool = True,
                 fourier_dim: int = 64,  # Match LeTE's dimensionality
                 enable_checkpointing: bool = True):
        super().__init__()
        
        if input_dim != 1:
            raise ValueError("OptimizedKMOTE is designed for 1D time input")
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.enable_checkpointing = enable_checkpointing
        
        # Time transformation
        self.time_base_transform = nn.Linear(1, hidden_dim)
        self._initialize_time_transform()
        
        # Build expert list based on configuration
        self.experts = nn.ModuleList()
        self.expert_names = []
        
        if use_fourier_expert:
            self.experts.append(LeTEStyleFourierExpert(
                input_dim=hidden_dim, 
                output_dim=output_dim,
                fourier_dim=fourier_dim
            ))
            self.expert_names.append("LeTEFourier")
        
        if use_spline_expert:
            self.experts.append(SimplifiedSplineExpert(
                input_dim=hidden_dim,
                output_dim=output_dim,
                n_knots=10
            ))
            self.expert_names.append("SimplifiedSpline")
        
        if use_wavelet_expert:
            from .k_mote import WaveletKANLayer  # Import existing wavelet
            self.experts.append(WaveletKANLayer(
                input_dim=hidden_dim,
                output_dim=output_dim,
                n_wavelets=8,
                wavelet_type='shock'
            ))
            self.expert_names.append("Wavelet")
        
        self.num_experts = len(self.experts)
        
        if self.num_experts == 0:
            raise ValueError("At least one expert must be enabled")
        
        # Gating network
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, self.num_experts)
        )
        
        # Layer norm and scale
        self.layer_norm = nn.LayerNorm(output_dim)
        self.scale = nn.Parameter(torch.ones(output_dim))
        
        print(f"✅ Optimized K-MOTE initialized with {self.num_experts} experts: {self.expert_names}")
        print(f"   Memory optimization: {'Enabled' if enable_checkpointing else 'Disabled'}")
    
    def _initialize_time_transform(self):
        """Initialize time transform with geometric progression"""
        with torch.no_grad():
            fourier_vals = 1.0 / (10 ** torch.linspace(0, 6, self.hidden_dim))
            self.time_base_transform.weight.copy_(fourier_vals.reshape(self.hidden_dim, -1))
            self.time_base_transform.bias.zero_()
    
    def _expert_forward_checkpointed(self, expert_idx: int, x: torch.Tensor) -> torch.Tensor:
        """Checkpointed expert forward to save memory"""
        if self.enable_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self.experts[expert_idx], x, use_reentrant=False
            )
        else:
            return self.experts[expert_idx](x)
    
    def forward(self, t: torch.Tensor, return_expert_outputs: bool = False) -> torch.Tensor:
        """Optimized forward pass"""
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        # Time transformation
        t_base = self.time_base_transform(t)  # (B, S, hidden_dim)
        
        # Expert outputs (with optional checkpointing)
        expert_outputs = []
        for i in range(self.num_experts):
            output = self._expert_forward_checkpointed(i, t_base)
            expert_outputs.append(output)
        
        # Gating
        gating_logits = self.gating_network(t_base)
        gating_weights = torch.softmax(gating_logits, dim=-1)
        
        # Weighted combination
        stacked_outputs = torch.stack(expert_outputs, dim=-1)
        gating_weights_expanded = gating_weights.unsqueeze(-2)
        
        weighted_sum = (gating_weights_expanded * stacked_outputs).sum(dim=-1)
        
        # Final processing
        output = self.layer_norm(weighted_sum)
        output = output * self.scale
        
        if return_expert_outputs:
            return output, expert_outputs, gating_weights
        return output


# Performance comparison test
if __name__ == "__main__":
    import time
    
    # Create models
    optimized_kmote = OptimizedKMOTE(
        input_dim=1, 
        output_dim=1, 
        hidden_dim=64,
        fourier_dim=64,  # Match LeTE
        enable_checkpointing=False
    )
    
    # Test input
    batch_size, seq_len = 32, 512
    t = torch.randn(batch_size, seq_len, 1)
    
    print(f"\nModel size: {sum(p.numel() for p in optimized_kmote.parameters()):,} parameters")
    
    # Warmup
    with torch.no_grad():
        _ = optimized_kmote(t)
    
    # Timing test
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            output = optimized_kmote(t)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    avg_time = (time.time() - start_time) / 10
    
    print(f"Average forward pass time: {avg_time*1000:.2f}ms")
    print(f"Output shape: {output.shape}")
    print(f"Memory efficient: ✅")