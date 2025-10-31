import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple

# Import optimized experts
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Dict, Tuple, Optional, List

# Import the optimized implementations
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from optimized_spline import MemoryOptimizedSplineKAN, EfficientFourierKAN
    print("✅ Imported optimized K-MOTE components.")
except ImportError:
    # Fallback minimal implementations
    print("⚠️ Using fallback implementations for optimized components")
    
    class MemoryOptimizedSplineKAN(nn.Module):
        def __init__(self, input_dim: int = 1, hidden_dim: int = 64, grid_size: int = 5, **kwargs):
            super().__init__()
            self.linear = nn.Linear(input_dim, hidden_dim)
            
        def forward(self, x):
            return torch.tanh(self.linear(x))
    
    class EfficientFourierKAN(nn.Module):
        def __init__(self, input_dim: int = 1, hidden_dim: int = 64, fourier_modes: int = 5, **kwargs):
            super().__init__()
            self.linear = nn.Linear(input_dim, hidden_dim)
            
        def forward(self, x):
            return torch.sin(self.linear(x))

# Import the original, stable Spline implementation from LeTE
try:
    from .LeTE_original import Spline as LeTESpline
    print("✅ Imported stable LeTE Spline for expert replacement.")
except ImportError:
    print("❌ Could not import LeTE Spline. Using a fallback.")
    # Define a minimal fallback if LeTE_original.py is not found
    class LeTESpline(nn.Module):
        def __init__(self, dim_spline, **kwargs):
            super().__init__()
            self.linear = nn.Linear(dim_spline, dim_spline)
        def forward(self, x):
            return torch.tanh(self.linear(x))

class EnhancedWaveletKAN(nn.Module):
    """
    Enhanced wavelet expert with LeTE-style geometric initialization.
    Optimized for memory efficiency and better performance.
    """
    def __init__(self, input_dim: int, output_dim: int, n_wavelets: int = 8, 
                 wavelet_type: str = 'shock', use_geometric_init: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_wavelets = n_wavelets
        self.wavelet_type = wavelet_type
        
        # Learnable wavelet parameters
        self.scales = nn.Parameter(torch.randn(input_dim, n_wavelets))
        self.shifts = nn.Parameter(torch.randn(input_dim, n_wavelets))
        
        if wavelet_type == 'adaptive_morlet':
            self.frequencies = nn.Parameter(torch.randn(input_dim, n_wavelets))
            self.sharpness = nn.Parameter(torch.randn(input_dim, n_wavelets))
        elif wavelet_type == 'shock':
            self.asymmetry = nn.Parameter(torch.randn(input_dim, n_wavelets) * 0.1)
            self.steepness = nn.Parameter(torch.randn(input_dim, n_wavelets) * 0.1)
        
        # Linear transformation (with better initialization)
        self.linear = nn.Linear(input_dim * n_wavelets, output_dim)
        
        if use_geometric_init:
            self._initialize_geometric()
    
    def _initialize_geometric(self):
        """Initialize with LeTE-style geometric progression for better performance"""
        with torch.no_grad():
            # Initialize linear layer with geometric progression
            if self.linear.weight.shape[1] > 1:
                # Create frequency-like initialization for wavelets
                n_features = self.linear.weight.shape[1]
                freq_vals = 1.0 / (10 ** torch.linspace(0, 6, n_features))
                self.linear.weight.copy_(torch.randn_like(self.linear.weight) * freq_vals.unsqueeze(0))
            
            # Initialize scales and shifts with reasonable values
            nn.init.uniform_(self.scales, 0.5, 2.0)  # Reasonable scale range
            nn.init.uniform_(self.shifts, -1.0, 1.0)  # Reasonable shift range

    def shock_wavelet(self, t, asymmetry, steepness):
        """Optimized shock wavelet computation"""
        asym = torch.tanh(asymmetry)
        steep = F.softplus(steepness) + 0.1
        steep = torch.clamp(steep, max=5.0)
        
        left_exponent = torch.clamp(steep * t * (1 + asym), min=-10, max=10)
        right_exponent = torch.clamp(-steep * t * (1 - asym), min=-10, max=10)
        
        shock_profile = torch.where(t < 0, torch.exp(left_exponent), torch.exp(right_exponent))
        
        freq = torch.clamp(steep, max=3.0)
        oscillation = torch.cos(freq * t)
        
        result = torch.clamp(shock_profile * oscillation, min=-100, max=100)
        return result

    def adaptive_morlet_wavelet(self, t, freq, sharpness):
        """Adaptive Morlet wavelet"""
        c = math.pi**(-0.25)
        sharp = F.softplus(sharpness) + 0.1
        freq_param = F.softplus(freq) + 1.0
        return c * torch.exp(-sharp * t**2) * torch.cos(freq_param * t)

    def morlet_wavelet(self, t):
        """Standard Morlet wavelet"""
        c = math.pi**(-0.25)
        return c * torch.exp(-0.5 * t**2) * torch.cos(5.0 * t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        # Expand dimensions for wavelet computation
        x_expanded = x.unsqueeze(-1)  # (B, S, input_dim, 1)
        scales = F.softplus(self.scales).unsqueeze(0).unsqueeze(0) + 0.1  # (1, 1, input_dim, n_wavelets)
        shifts = self.shifts.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim, n_wavelets)
        
        # Compute wavelet input
        wavelet_input = (x_expanded - shifts) / scales
        
        # Apply wavelet function based on type
        if self.wavelet_type == 'adaptive_morlet':
            freq = self.frequencies.unsqueeze(0).unsqueeze(0)
            sharp = self.sharpness.unsqueeze(0).unsqueeze(0)
            wavelet_activations = self.adaptive_morlet_wavelet(wavelet_input, freq, sharp)
        elif self.wavelet_type == 'shock':
            asym = self.asymmetry.unsqueeze(0).unsqueeze(0)
            steep = self.steepness.unsqueeze(0).unsqueeze(0)
            wavelet_activations = self.shock_wavelet(wavelet_input, asym, steep)
        else:  # Default to Morlet
            wavelet_activations = self.morlet_wavelet(wavelet_input)
        
        # Flatten and apply linear transformation
        wavelet_flat = wavelet_activations.view(batch_size, seq_len, -1)
        output = self.linear(wavelet_flat)
        
        return output


class SequentialKMOTE(nn.Module):
    """
    Sequential K-MOTE: Memory-efficient Mixture of Experts with sequential evaluation.
    
    Key features:
    - Sequential expert evaluation (one at a time) for memory efficiency
    - LeTE-style initialization for all components
    - FIXED: Replaced buggy spline expert with a stable, trainable LeTE-style spline.
    - Balanced expert parameter counts
    - Proper gradient flow and training dynamics
    """
    def __init__(self, input_dim: int = 1, output_dim: int = 1,
                 hidden_dim: int = 64,  # Match LeTE's dimension
                 use_spline_expert: bool = True,
                 use_fourier_expert: bool = True,
                 use_wavelet_expert: bool = True,
                 # Expert configurations
                 spline_grid_size: int = 5,
                 fourier_harmonics: int = 8,
                 wavelet_count: int = 8,
                 wavelet_type: str = 'shock',
                 # Training configurations
                 gating_temperature: float = 1.0,
                 balanced_init: bool = True,
                 use_layer_norm: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        
        if input_dim != 1:
            raise ValueError("SequentialKMOTE is designed for 1D time input")
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.gating_temperature = gating_temperature
        self.balanced_init = balanced_init
        self.dropout = dropout
        
        # ===== TIME TRANSFORMATION (LeTE-style) =====
        self.time_transform = nn.Linear(input_dim, hidden_dim)
        self._initialize_time_transform()
        
        # ===== BUILD EXPERT LIST =====
        self.experts = nn.ModuleList()
        self.expert_names = []
        
        if use_spline_expert:
            # FIX: Use the stable, original LeTE Spline implementation and add a projection head
            # This is critical for ensuring the model is trainable and output shapes match.
            self.experts.append(nn.Sequential(
                LeTESpline(
                    dim_spline=hidden_dim,
                    grid_size_spline=spline_grid_size,
                    order_spline=3
                ),
                nn.Linear(hidden_dim, output_dim) # Project from hidden_dim to output_dim
            ))
            self.expert_names.append("LeTE_Spline")
        
        if use_fourier_expert:
            self.experts.append(EfficientFourierKAN(
                input_dim=hidden_dim,
                output_dim=output_dim,
                intermediate_dim=64,  # High-dimensional like LeTE
                n_harmonics=fourier_harmonics
            ))
            self.expert_names.append("EfficientFourier")
        
        if use_wavelet_expert:
            self.experts.append(EnhancedWaveletKAN(
                input_dim=hidden_dim,
                output_dim=output_dim,
                n_wavelets=wavelet_count,
                wavelet_type=wavelet_type,
                use_geometric_init=True
            ))
            self.expert_names.append("EnhancedWavelet")
        
        self.num_experts = len(self.experts)
        
        if self.num_experts == 0:
            raise ValueError("At least one expert must be enabled")
        
        # ===== GATING NETWORK =====
        self.gating_network = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(64, self.num_experts)
        )
        
        # ===== OUTPUT PROCESSING =====
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = nn.Identity()
        
        self.output_scale = nn.Parameter(torch.ones(output_dim))
        
        # ===== INITIALIZATION =====
        if balanced_init:
            self._initialize_balanced_gating()
        
        print(f"✅ Sequential K-MOTE initialized:")
        print(f"   Experts: {self.expert_names}")
        print(f"   Hidden dim: {hidden_dim}")
        print(f"   Total parameters: {sum(p.numel() for p in self.parameters()):,}")
        self._print_expert_sizes()
    
    def _initialize_time_transform(self):
        """Initialize time transform with LeTE's geometric progression"""
        with torch.no_grad():
            # Geometric frequency initialization: 1/(10^[0,9])
            fourier_vals = 1.0 / (10 ** torch.linspace(0, 9, self.hidden_dim))
            self.time_transform.weight.copy_(fourier_vals.reshape(self.hidden_dim, -1))
            self.time_transform.bias.zero_()
    
    def _initialize_balanced_gating(self):
        """Initialize gating network to balanced weights"""
        with torch.no_grad():
            # Initialize final gating layer to output balanced logits
            self.gating_network[-1].weight.zero_()
            self.gating_network[-1].bias.zero_()  # This gives equal probabilities after softmax
    
    def _print_expert_sizes(self):
        """Print parameter counts for each expert"""
        print("   Expert parameter counts:")
        for i, expert in enumerate(self.experts):
            param_count = sum(p.numel() for p in expert.parameters())
            print(f"     {self.expert_names[i]}: {param_count:,} parameters")
    
    def forward(self, t: torch.Tensor, return_expert_info: bool = False) -> torch.Tensor:
        """
        Sequential forward pass: experts evaluated one at a time for memory efficiency.
        
        Args:
            t: Input timestamps of shape (batch_size, seq_len) or (batch_size, seq_len, 1)
            return_expert_info: If True, return (output, expert_outputs, gating_weights)
        
        Returns:
            output: Final time encoding of shape (batch_size, seq_len, output_dim)
            expert_outputs: List of individual expert outputs (if return_expert_info=True)
            gating_weights: Gating weights of shape (batch_size, seq_len, num_experts) (if return_expert_info=True)
        """
        if t.dim() == 2:
            t = t.unsqueeze(-1)  # Add feature dimension: (B, S) -> (B, S, 1)
        
        batch_size, seq_len, _ = t.shape
        
        # ===== STEP 1: TIME TRANSFORMATION =====
        t_transformed = self.time_transform(t)  # (B, S, hidden_dim)
        
        # ===== STEP 2: GATING COMPUTATION (FIRST) =====
        gating_logits = self.gating_network(t_transformed)  # (B, S, num_experts)
        gating_weights = F.softmax(gating_logits / self.gating_temperature, dim=-1)
        
        # ===== STEP 3: SEQUENTIAL EXPERT EVALUATION =====
        # Initialize result accumulator
        result = torch.zeros(batch_size, seq_len, self.output_dim, device=t.device, dtype=t.dtype)
        expert_outputs = [] if return_expert_info else None
        
        for i, expert in enumerate(self.experts):
            # Run expert
            expert_output = expert(t_transformed)  # (B, S, output_dim)
            
            # Store for return if requested
            if return_expert_info:
                expert_outputs.append(expert_output.clone())
            
            # Apply gating weight and accumulate
            weight = gating_weights[:, :, i:i+1]  # (B, S, 1)
            result = result + (weight * expert_output)  # Use out-of-place addition
            
            # CRITICAL: Free memory immediately to prevent accumulation
            del expert_output
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # ===== STEP 4: OUTPUT PROCESSING =====
        output = self.layer_norm(result)
        output = output * self.output_scale
        
        if return_expert_info:
            return output, expert_outputs, gating_weights
        return output
    
    def get_expert_utilization(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute expert utilization statistics for analysis.
        
        Returns:
            utilization: Mean gating weights of shape (num_experts,)
        """
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        t_transformed = self.time_transform(t)
        gating_logits = self.gating_network(t_transformed)
        gating_weights = F.softmax(gating_logits / self.gating_temperature, dim=-1)
        
        # Compute mean utilization across batch and sequence
        utilization = gating_weights.mean(dim=(0, 1))  # (num_experts,)
        return utilization
    
    def get_memory_usage(self) -> dict:
        """Get estimated memory usage breakdown"""
        param_memory = sum(p.numel() * 4 for p in self.parameters()) / (1024**2)  # MB
        
        expert_memories = []
        for expert in self.experts:
            expert_params = sum(p.numel() * 4 for p in expert.parameters()) / (1024**2)
            expert_memories.append(expert_params)
        
        return {
            "total_parameters_mb": param_memory,
            "expert_parameters_mb": expert_memories,
            "expert_names": self.expert_names,
            "estimated_peak_memory_mb": max(expert_memories) + 50,  # Peak + overhead
            "sequential_benefit": f"vs {sum(expert_memories) + 50:.1f}MB parallel"
        }


# ===== TESTING AND COMPARISON =====
if __name__ == "__main__":
    import time
    
    print("=" * 70)
    print("SEQUENTIAL K-MOTE TESTING")
    print("=" * 70)
    
    # Create model
    model = SequentialKMOTE(
        input_dim=1,
        output_dim=1,
        hidden_dim=64,
        use_spline_expert=True,
        use_fourier_expert=True,
        use_wavelet_expert=True,
        fourier_harmonics=8,
        balanced_init=True
    )
    
    # Test input
    batch_size, seq_len = 32, 512
    t = torch.randn(batch_size, seq_len, 1)
    
    print(f"\nTest input shape: {t.shape}")
    
    # Memory usage analysis
    memory_info = model.get_memory_usage()
    print(f"\nMemory Analysis:")
    for key, value in memory_info.items():
        print(f"  {key}: {value}")
    
    # Forward pass test
    print(f"\nTesting forward pass...")
    start_time = time.time()
    
    with torch.no_grad():
        # Test basic forward
        output = model(t)
        print(f"  Basic forward: {output.shape}")
        
        # Test with expert info
        output, expert_outputs, gating_weights = model(t, return_expert_info=True)
        print(f"  With expert info: output {output.shape}, {len(expert_outputs)} experts")
        
        # Expert utilization
        utilization = model.get_expert_utilization(t)
        print(f"  Expert utilization: {utilization.tolist()}")
    
    elapsed = time.time() - start_time
    print(f"  Total time: {elapsed*1000:.2f}ms")
    
    print(f"\n✅ Sequential K-MOTE test completed successfully!")
    print(f"   Memory efficient: Peak ~{memory_info['estimated_peak_memory_mb']:.1f}MB")
    print(f"   vs Parallel: ~{memory_info['sequential_benefit'].split()[-1]}")