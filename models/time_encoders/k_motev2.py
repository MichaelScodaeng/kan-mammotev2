# file: models/time_encoders/k_motev2.py
# K-MOTE v2: Weighted Concatenation Architecture (LeTE-inspired)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch.cuda.amp as amp
from typing import List, Optional


class LeTESpline(nn.Module):
    """
    LeTE-style B-spline implementation for K-MOTE v2
    """
    def __init__(self, dim_spline: int, grid_size_spline: int = 5, order_spline: int = 3, grid_range: list = [-1, 1]):
        super().__init__()
        self.dim_spline = dim_spline
        self.grid_size_spline = grid_size_spline
        self.order_spline = order_spline

        # Compute grid spacing
        h = (grid_range[1] - grid_range[0]) / float(self.grid_size_spline)

        # Build the grid for each dimension
        grid = torch.arange(-self.order_spline, self.grid_size_spline + self.order_spline + 1)
        grid = grid * h + grid_range[0]
        grid = grid.expand(self.dim_spline, -1).contiguous()
        self.register_buffer("grid", grid)

        # Base weight for the linear+activation branch
        self.base_weight = nn.Parameter(
            torch.randn(self.dim_spline, self.dim_spline) / math.sqrt(self.dim_spline)
        )
        
        # Spline coefficients
        self.spline_weight = nn.Parameter(
            torch.randn(self.dim_spline, self.dim_spline, self.grid_size_spline + self.order_spline) /
            (math.sqrt(self.dim_spline) * math.sqrt(self.grid_size_spline + self.order_spline))
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.dim_spline)

        # Base branch: tanh + linear
        base_output = nn.functional.linear(torch.tanh(x), self.base_weight)

        # If the input batch is empty, return a zero-like tensor
        if x.size(0) == 0:
            spline_output = torch.zeros_like(base_output)
        else:
            # Evaluate B-spline basis
            b_splines_val = self.b_splines(x).view(x.size(0), -1)
            w = self.spline_weight.view(self.dim_spline, -1)
            spline_output = nn.functional.linear(b_splines_val, w)

        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.dim_spline)
        return output

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.grid
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)

        for k in range(1, self.order_spline + 1):
            left_num = (x - grid[:, :-(k + 1)])
            left_den = (grid[:, k:-1] - grid[:, :-(k + 1)])
            right_num = (grid[:, k + 1:] - x)
            right_den = (grid[:, k + 1:] - grid[:, 1:-k])

            eps = 1e-12
            left_den = left_den + eps
            right_den = right_den + eps
            
            left_term = torch.where(torch.abs(left_den) > eps, 
                                  (left_num / left_den) * bases[:, :, :-1], 
                                  torch.zeros_like(bases[:, :, :-1]))
            right_term = torch.where(torch.abs(right_den) > eps,
                                   (right_num / right_den) * bases[:, :, 1:],
                                   torch.zeros_like(bases[:, :, 1:]))

            bases = left_term + right_term

        return bases.contiguous()


class LeTEFourierSeries(nn.Module):
    """
    LeTE-style Fourier Series implementation for K-MOTE v2
    """
    def __init__(self, dim_fourier: int, grid_size_fourier: int = 5):
        super().__init__()
        self.dim_fourier = dim_fourier
        self.grid_size_fourier = grid_size_fourier

        self.fourier_weight = torch.nn.Parameter(
            torch.randn(2, self.dim_fourier, self.dim_fourier, grid_size_fourier) /
            (math.sqrt(self.dim_fourier) * math.sqrt(self.grid_size_fourier))
        )

        self.bias = nn.Parameter(torch.zeros(self.dim_fourier))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        out_shape = original_shape[0:-1] + (self.dim_fourier,)

        x = x.reshape(-1, self.dim_fourier)

        k = torch.arange(1, self.grid_size_fourier + 1, device=x.device)
        k = k.reshape(1, 1, 1, self.grid_size_fourier)

        x_reshaped = x.reshape(x.shape[0], 1, x.shape[1], 1)

        c = torch.cos(k * x_reshaped)
        s = torch.sin(k * x_reshaped)

        y = torch.sum(c * self.fourier_weight[0:1], dim=(-2, -1))
        y += torch.sum(s * self.fourier_weight[1:2], dim=(-2, -1))

        y += self.bias

        y = y.reshape(out_shape)
        return y


class EnhancedWaveletKAN(nn.Module):
    """
    Enhanced Wavelet KAN expert for K-MOTE v2
    """
    def __init__(self, input_dim: int, output_dim: int, n_wavelets: int = 8, 
                 wavelet_type: str = 'shock', use_geometric_init: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_wavelets = n_wavelets
        self.wavelet_type = wavelet_type
        
        self.scales = nn.Parameter(torch.randn(input_dim, n_wavelets))
        self.shifts = nn.Parameter(torch.randn(input_dim, n_wavelets))
        
        if wavelet_type == 'adaptive_morlet':
            self.frequencies = nn.Parameter(torch.randn(input_dim, n_wavelets))
            self.sharpness = nn.Parameter(torch.randn(input_dim, n_wavelets))
        elif wavelet_type == 'shock':
            self.asymmetry = nn.Parameter(torch.randn(input_dim, n_wavelets) * 0.1)
            self.steepness = nn.Parameter(torch.randn(input_dim, n_wavelets) * 0.1)
        
        self.linear = nn.Linear(input_dim * n_wavelets, output_dim)
        
        if use_geometric_init:
            self._initialize_geometric()
    
    def _initialize_geometric(self):
        with torch.no_grad():
            if self.linear.weight.shape[1] > 1:
                n_features = self.linear.weight.shape[1]
                freq_vals = 1.0 / (10 ** torch.linspace(0, 9, n_features))
                self.linear.weight.copy_(torch.randn_like(self.linear.weight) * freq_vals.unsqueeze(0))
            
            nn.init.uniform_(self.scales, 0.5, 2.0)
            nn.init.uniform_(self.shifts, -1.0, 1.0)

    def shock_wavelet(self, t, asymmetry, steepness):
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
        c = math.pi**(-0.25)
        sharp = F.softplus(sharpness) + 0.1
        freq_param = F.softplus(freq) + 1.0
        return c * torch.exp(-sharp * t**2) * torch.cos(freq_param * t)

    def morlet_wavelet(self, t):
        c = math.pi**(-0.25)
        return c * torch.exp(-0.5 * t**2) * torch.cos(5.0 * t)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        batch_size, seq_len, _ = x.shape
        
        x_expanded = x.unsqueeze(-1)
        scales = F.softplus(self.scales).unsqueeze(0).unsqueeze(0) + 0.1
        shifts = self.shifts.unsqueeze(0).unsqueeze(0)
        
        wavelet_input = (x_expanded - shifts) / scales
        
        if self.wavelet_type == 'adaptive_morlet':
            freq = self.frequencies.unsqueeze(0).unsqueeze(0)
            sharp = self.sharpness.unsqueeze(0).unsqueeze(0)
            wavelet_activations = self.adaptive_morlet_wavelet(wavelet_input, freq, sharp)
        elif self.wavelet_type == 'shock':
            asym = self.asymmetry.unsqueeze(0).unsqueeze(0)
            steep = self.steepness.unsqueeze(0).unsqueeze(0)
            wavelet_activations = self.shock_wavelet(wavelet_input, asym, steep)
        else:
            wavelet_activations = self.morlet_wavelet(wavelet_input)
        
        wavelet_flat = wavelet_activations.view(batch_size, seq_len, -1)
        output = self.linear(wavelet_flat)
        
        return output


class WeightedConcatKMOTE(nn.Module):
    """
    K-MOTE v2: Weighted Concatenation Architecture
    
    Key Innovation: LeTE-inspired concatenation with learnable gating weights.
    
    Instead of weighted sum of expert outputs, we:
    1. Concatenate expert outputs of different dimensions
    2. Apply learnable gating weights to the concatenated features
    3. Use LeTE-style post-processing (LayerNorm + Scale)
    
    Args:
        input_dim (int): Input dimension (must be 1 for time encoding)
        output_dim (int): Output dimension of the final encoding
        expert_ratios (List[float]): Dimension allocation ratios for experts [spline, fourier, wavelet]
        hidden_dim (int): Hidden dimension for time transformation
        gating_mode (str): Gating strategy - 'per_expert', 'dense', or 'hybrid'
        layer_norm (bool): Apply LayerNorm like LeTE
        scale (bool): Apply learnable scaling like LeTE
        use_gradient_checkpointing (bool): Use gradient checkpointing for memory efficiency
        use_amp (bool): Use automatic mixed precision
        wavelet_type (str): Type of wavelet for wavelet expert
        gating_temp (float): Temperature for gating softmax
    """
    
    def __init__(self, 
                 input_dim: int = 1,
                 output_dim: int = 192,
                 expert_ratios: Optional[List[float]] = None,
                 hidden_dim: int = 64,
                 gating_mode: str = 'per_expert',  # 'per_expert', 'dense', 'hybrid'
                 layer_norm: bool = True,
                 scale: bool = True,
                 use_gradient_checkpointing: bool = True,
                 use_amp: bool = False,
                 wavelet_type: str = 'shock',
                 gating_temp: float = 2.0):
        
        super().__init__()
        
        # Validate inputs
        if input_dim != 1:
            raise ValueError("K-MOTE v2 requires input_dim=1 for time encoding")
        
        if gating_mode not in ['per_expert', 'dense', 'hybrid']:
            raise ValueError("gating_mode must be 'per_expert', 'dense', or 'hybrid'")
        
        # Store configuration
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.gating_mode = gating_mode
        self.layer_norm_enabled = layer_norm
        self.scale_enabled = scale
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_amp = use_amp
        self.temperature = gating_temp
        
        # Calculate expert dimensions
        if expert_ratios is None:
            # LeTE-inspired default allocation: Spline > Fourier > Wavelet
            expert_ratios = [0.4, 0.35, 0.25]
        
        if len(expert_ratios) != 3:
            raise ValueError("expert_ratios must have exactly 3 elements [spline, fourier, wavelet]")
        
        if abs(sum(expert_ratios) - 1.0) > 1e-6:
            raise ValueError("expert_ratios must sum to 1.0")
        
        # Calculate expert output dimensions
        self.expert_dims = [max(1, int(output_dim * ratio)) for ratio in expert_ratios]
        
        # Ensure total dimensions match output_dim
        actual_total = sum(self.expert_dims)
        if actual_total != output_dim:
            # Adjust the largest expert to make exact match
            largest_idx = expert_ratios.index(max(expert_ratios))
            self.expert_dims[largest_idx] += (output_dim - actual_total)
        
        print(f"K-MOTE v2 Expert Dimensions: Spline={self.expert_dims[0]}, "
              f"Fourier={self.expert_dims[1]}, Wavelet={self.expert_dims[2]} "
              f"(Total: {sum(self.expert_dims)})")
        
        # ===== SEPARATE TIME TRANSFORMATIONS FOR EACH EXPERT =====
        # Each expert gets its own time transformation that outputs hidden_dim * ratio
        self.expert_time_transforms = nn.ModuleList([
            nn.Linear(input_dim, int(hidden_dim * expert_ratios[0])),  # Spline
            nn.Linear(input_dim, int(hidden_dim * expert_ratios[1])),  # Fourier 
            nn.Linear(input_dim, int(hidden_dim * expert_ratios[2]))   # Wavelet
        ])
        self._initialize_expert_time_transforms()
        
        # ===== INDEPENDENT GATING TRANSFORMATION =====
        # Separate scale-invariant transform just for gating decisions
        self.gating_transform = nn.Linear(input_dim, 32)  # Small but sufficient for gating
        self._initialize_gating_transform()
        
        # ===== EXPERT MODULES =====
        self.experts = nn.ModuleList([
            # Expert 1: Spline (smooth trends, largest allocation)
            nn.Sequential(
                LeTESpline(dim_spline=int(hidden_dim * expert_ratios[0])),
                nn.Linear(int(hidden_dim * expert_ratios[0]), self.expert_dims[0])
            ),
            
            # Expert 2: Fourier (periodic patterns, medium allocation)
            nn.Sequential(
                LeTEFourierSeries(dim_fourier=int(hidden_dim * expert_ratios[1]), grid_size_fourier=5),
                nn.Linear(int(hidden_dim * expert_ratios[1]), self.expert_dims[1])
            ),
            
            # Expert 3: Wavelet (abrupt changes, smallest allocation)
            nn.Sequential(
                EnhancedWaveletKAN(int(hidden_dim * expert_ratios[2]), int(hidden_dim * expert_ratios[2]), n_wavelets=5, wavelet_type=wavelet_type),
                nn.Linear(int(hidden_dim * expert_ratios[2]), self.expert_dims[2])
            ),
        ])
        
        self.num_experts = len(self.experts)
        
        # Store expert hidden dimensions for gating
        self.expert_hidden_dims = [
            int(hidden_dim * expert_ratios[0]),  # Spline
            int(hidden_dim * expert_ratios[1]),  # Fourier 
            int(hidden_dim * expert_ratios[2])   # Wavelet
        ]
        
        # ===== GATING NETWORK =====
        # Use independent gating transform (32 dims) for gating input
        gating_input_dim = 32
        
        if gating_mode == 'per_expert':
            # Simple per-expert gating (broadcasted to expert dimensions)
            self.gating_network = nn.Sequential(
                nn.Linear(gating_input_dim, 64),
                nn.GELU(),
                nn.Linear(64, self.num_experts)
            )
        elif gating_mode == 'dense':
            # Dense gating (one weight per concatenated dimension)
            self.gating_network = nn.Sequential(
                nn.Linear(gating_input_dim, 128),
                nn.GELU(),
                nn.Linear(128, output_dim)
            )
        else:  # hybrid
            # Hybrid: both per-expert and dense gating
            self.expert_gating = nn.Sequential(
                nn.Linear(gating_input_dim, 64),
                nn.GELU(),
                nn.Linear(64, self.num_experts)
            )
            self.dense_gating = nn.Sequential(
                nn.Linear(gating_input_dim, 128),
                nn.GELU(),
                nn.Linear(128, output_dim)
            )
        
        # ===== POST-PROCESSING (LeTE-style) =====
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_dim)
        else:
            self.layer_norm = nn.Identity()
        
        if scale:
            self.scale_weight = nn.Parameter(torch.ones(output_dim))
        
        print(f"Initialized K-MOTE v2 with {gating_mode} gating, "
              f"hidden_dim={hidden_dim}, output_dim={output_dim}")
        print(f"Features: LayerNorm={layer_norm}, Scale={scale}, "
              f"Checkpointing={use_gradient_checkpointing}, AMP={use_amp}")
    
    def _initialize_expert_time_transforms(self):
        """
        Initialize separate time transformations for each expert with LeTE-style geometric progression.
        Creates frequencies from 1.0 down to 1e-9 for multi-scale time encoding.
        """
        with torch.no_grad():
            for i, transform in enumerate(self.expert_time_transforms):
                output_dim = transform.out_features
                # Create geometric progression of frequencies for this expert
                frequencies = 1.0 / (10 ** torch.linspace(0, 9, output_dim, dtype=torch.float32))
                transform.weight.copy_(frequencies.unsqueeze(1))
                transform.bias.zero_()
    
    def _initialize_gating_transform(self):
        """
        Initialize gating transformation with scale-invariant geometric progression.
        """
        with torch.no_grad():
            output_dim = self.gating_transform.out_features
            # Create geometric progression for gating features
            frequencies = 1.0 / (10 ** torch.linspace(0, 5, output_dim, dtype=torch.float32))  # Shorter range for gating
            self.gating_transform.weight.copy_(frequencies.unsqueeze(1))
            self.gating_transform.bias.zero_()
    
    def _compute_gating_weights(self, t_transformed: torch.Tensor) -> torch.Tensor:
        """
        Compute gating weights based on the selected gating mode.
        
        Args:
            t_transformed: Time-transformed input (B, S, hidden_dim)
            
        Returns:
            Gating weights tensor (B, S, output_dim)
        """
        if self.gating_mode == 'per_expert':
            # Per-expert gating: compute expert weights and broadcast
            expert_weights = self.gating_network(t_transformed)  # (B, S, num_experts)
            expert_weights = F.softmax(expert_weights / self.temperature, dim=-1)
            
            # Broadcast to match concatenated dimensions
            gating_weights = []
            for i, expert_dim in enumerate(self.expert_dims):
                weight = expert_weights[:, :, i:i+1]  # (B, S, 1)
                broadcasted = weight.repeat(1, 1, expert_dim)  # (B, S, expert_dim)
                gating_weights.append(broadcasted)
            
            return torch.cat(gating_weights, dim=-1)  # (B, S, output_dim)
            
        elif self.gating_mode == 'dense':
            # Dense gating: one weight per output dimension
            dense_weights = self.gating_network(t_transformed)  # (B, S, output_dim)
            return torch.sigmoid(dense_weights)  # Use sigmoid for element-wise gating
            
        else:  # hybrid
            # Hybrid: combine per-expert and dense gating
            expert_weights = self.expert_gating(t_transformed)  # (B, S, num_experts)
            expert_weights = F.softmax(expert_weights / self.temperature, dim=-1)
            
            dense_weights = self.dense_gating(t_transformed)  # (B, S, output_dim)
            dense_weights = torch.sigmoid(dense_weights)
            
            # Broadcast expert weights
            broadcasted_expert = []
            for i, expert_dim in enumerate(self.expert_dims):
                weight = expert_weights[:, :, i:i+1]
                broadcasted = weight.repeat(1, 1, expert_dim)
                broadcasted_expert.append(broadcasted)
            
            expert_broadcast = torch.cat(broadcasted_expert, dim=-1)  # (B, S, output_dim)
            
            # Combine both gating mechanisms
            return expert_broadcast * dense_weights
    
    def _checkpoint_expert(self, expert, t_input, expert_idx):
        """
        Apply gradient checkpointing to expert computation if enabled.
        """
        def _run_expert(inp):
            with amp.autocast(enabled=(self.use_amp and torch.cuda.is_available())):
                return expert(inp)
        
        if self.use_gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(_run_expert, t_input, use_reentrant=False)
        else:
            return _run_expert(t_input)
    
    def forward(self, t: torch.Tensor, return_weights: bool = False) -> torch.Tensor:
        """
        Forward pass of K-MOTE v2.
        
        Args:
            t: Input timestamps (B, S) or (B, S, 1)
            return_weights: If True, return gating weights along with output
            
        Returns:
            Time encoding (B, S, output_dim) or tuple with gating weights
        """
        # Ensure correct input shape
        if t.dim() == 2:
            t = t.unsqueeze(-1)  # (B, S, 1)
        
        batch_size, seq_len = t.shape[:2]
        
        # Step 1: Separate time transformations for each expert
        expert_transformed = []
        for i, transform in enumerate(self.expert_time_transforms):
            transformed = transform(t)  # (B, S, expert_hidden_dim[i])
            expert_transformed.append(transformed)
        
        # Step 2: Independent gating transformation
        gating_features = self.gating_transform(t)  # (B, S, 32) - independent of experts
        
        # Step 3: Expert processing (parallel with different output dimensions)
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            output = self._checkpoint_expert(expert, expert_transformed[i], i)
            expert_outputs.append(output)
        
        # Step 4: Concatenation
        concatenated = torch.cat(expert_outputs, dim=-1)  # (B, S, output_dim)
        
        # Step 5: Gating (using independent gating features)
        gating_weights = self._compute_gating_weights(gating_features)  # (B, S, output_dim)
        
        # Step 6: Weighted concatenation
        gated_output = concatenated * gating_weights  # (B, S, output_dim)
        
        # Step 6: LeTE-style post-processing
        output = self.layer_norm(gated_output)
        
        if self.scale_enabled:
            output = output * self.scale_weight
        
        if return_weights:
            return output, gating_weights
        return output
    
    def get_expert_contributions(self, t: torch.Tensor) -> dict:
        """
        Get individual expert contributions for analysis.
        
        Returns:
            Dictionary with expert outputs and gating weights
        """
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        # Get separate time transformations for each expert
        expert_transformed = []
        for i, transform in enumerate(self.expert_time_transforms):
            transformed = transform(t)
            expert_transformed.append(transformed)
        
        # Get independent gating features
        gating_features = self.gating_transform(t)
        
        # Get individual expert outputs
        expert_outputs = {}
        for i, expert in enumerate(self.experts):
            name = ['spline', 'fourier', 'wavelet'][i]
            expert_outputs[name] = expert(expert_transformed[i])
        
        # Get gating weights (using independent gating features)
        gating_weights = self._compute_gating_weights(gating_features)
        
        return {
            'expert_outputs': expert_outputs,
            'gating_weights': gating_weights,
            'expert_dims': self.expert_dims,
            'expert_hidden_dims': self.expert_hidden_dims,
            'gating_features': gating_features
        }
    
    # Utility methods for training
    def make_grad_scaler(self):
        """Return GradScaler for AMP training if enabled."""
        if self.use_amp and torch.cuda.is_available():
            return torch.cuda.amp.GradScaler()
        return None
    
    def get_parameter_count(self) -> dict:
        """Get detailed parameter count breakdown."""
        total = sum(p.numel() for p in self.parameters())
        
        breakdown = {
            'expert_time_transforms': sum(p.numel() for p in self.expert_time_transforms.parameters()),
            'gating_transform': sum(p.numel() for p in self.gating_transform.parameters()),
            'experts': sum(p.numel() for p in self.experts.parameters()),
            'gating': 0,
            'post_processing': 0,
            'total': total
        }
        
        # Count gating parameters
        if hasattr(self, 'gating_network'):
            breakdown['gating'] = sum(p.numel() for p in self.gating_network.parameters())
        else:  # hybrid mode
            breakdown['gating'] = (sum(p.numel() for p in self.expert_gating.parameters()) +
                                 sum(p.numel() for p in self.dense_gating.parameters()))
        
        # Count post-processing parameters
        if hasattr(self, 'layer_norm') and hasattr(self.layer_norm, 'parameters'):
            breakdown['post_processing'] += sum(p.numel() for p in self.layer_norm.parameters())
        if hasattr(self, 'scale_weight'):
            breakdown['post_processing'] += self.scale_weight.numel()
        
        return breakdown


# Alias for easier import
KMOTEv2 = WeightedConcatKMOTE


def create_kmote_v2(output_dim: int = 192, 
                   expert_allocation: str = 'lete_style',
                   gating_mode: str = 'per_expert',
                   **kwargs) -> WeightedConcatKMOTE:
    """
    Factory function to create K-MOTE v2 with common configurations.
    
    Args:
        output_dim: Output dimension
        expert_allocation: 'equal', 'lete_style', or 'custom'
        gating_mode: 'per_expert', 'dense', or 'hybrid'
        **kwargs: Additional arguments for WeightedConcatKMOTE
        
    Returns:
        Configured WeightedConcatKMOTE instance
    """
    if expert_allocation == 'equal':
        expert_ratios = [1/3, 1/3, 1/3]
    elif expert_allocation == 'lete_style':
        expert_ratios = [0.4, 0.35, 0.25]  # Spline > Fourier > Wavelet
    elif expert_allocation == 'fourier_focused':
        expert_ratios = [0.25, 0.5, 0.25]  # Focus on Fourier
    else:  # custom - expect expert_ratios in kwargs
        expert_ratios = kwargs.pop('expert_ratios', [0.4, 0.35, 0.25])
    
    return WeightedConcatKMOTE(
        output_dim=output_dim,
        expert_ratios=expert_ratios,
        gating_mode=gating_mode,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Testing K-MOTE v2...")
    
    # Create model
    model = create_kmote_v2(
        output_dim=192,
        expert_allocation='lete_style',
        gating_mode='per_expert',
        hidden_dim=64,
        layer_norm=True,
        scale=True
    )
    
    # Test forward pass
    batch_size, seq_len = 32, 100
    timestamps = torch.randn(batch_size, seq_len)
    
    output = model(timestamps)
    print(f"Input shape: {timestamps.shape}")
    print(f"Output shape: {output.shape}")
    
    # Parameter count
    param_count = model.get_parameter_count()
    print(f"\nParameter breakdown:")
    for key, count in param_count.items():
        print(f"  {key}: {count:,}")
    
    # Expert analysis
    contributions = model.get_expert_contributions(timestamps[:4, :10])  # Small subset
    print(f"\nExpert dimensions: {contributions['expert_dims']}")
    print(f"Gating weights shape: {contributions['gating_weights'].shape}")
    
    print("\nK-MOTE v2 test completed successfully!")