# file: models/time_encoders/k_mote.py (Corrected and Refactored)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Note: We implement custom B-splines instead of using efficient_kan for better control
# and numerical stability in temporal modeling applications

# --- Expert 1: Spline/RBF KAN Layer (Corrected with Learnable Projection) ---

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SplineKANLayer(nn.Module):
    """
    An expert based on Kolmogorov-Arnold Networks, using either B-splines or RBFs.
    This version includes a powerful MLP-based global branch for global trend fitting,
    and a local branch (B-spline or RBF) for fine-grained, local adjustments.
    """
    def __init__(self, input_dim: int, output_dim: int, grid_size: int = 8, basis_function: str = 'b_spline'):
        super().__init__()
        if basis_function not in ['b_spline', 'rbf']:
            raise ValueError("basis_function must be 'b_spline' or 'rbf'")
        self.basis_function = basis_function
        self.grid_size = grid_size
        self.output_dim = output_dim

        # This layer learns to scale/shift the input for the LOCAL branch
        self.input_projection = nn.Linear(input_dim, input_dim)

        # --- FIX: A powerful MLP to learn the GLOBAL trend ---
        hidden_dim = 32 # A small hidden layer
        self.global_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Initialize the final layer of the global branch to be small
        nn.init.xavier_uniform_(self.global_branch[-1].weight, gain=0.1)
        nn.init.zeros_(self.global_branch[-1].bias)

        # --- FIX: Removed the redundant self.base_weight parameter ---
        # self.base_weight = nn.Parameter(torch.Tensor(self.output_dim, input_dim)) # This is no longer needed

        if self.basis_function == 'b_spline':
            # Parameters for the LOCAL spline branch
            self.grid_range = [-2, 2]
            self.order_spline = 3
            self.spline_weight = nn.Parameter(torch.Tensor(
                input_dim, self.output_dim, self.grid_size + self.order_spline))
            nn.init.xavier_uniform_(self.spline_weight)
            
        else: # RBF implementation
            # Parameters for the LOCAL RBF branch
            internal_range = [-2, 2]
            centers_init = torch.linspace(internal_range[0], internal_range[1], grid_size, dtype=torch.float32)
            centers_init = centers_init.unsqueeze(0).unsqueeze(0).repeat(input_dim, output_dim, 1)
            self.centers = nn.Parameter(centers_init)
            self.gammas = nn.Parameter(torch.ones(input_dim, output_dim, grid_size))
            self.local_linear = nn.Linear(input_dim * grid_size, output_dim) # Renamed to avoid confusion

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions. Expects 'x' to be already projected.
        """
        input_dim = x.shape[2]
        grid_points = torch.linspace(self.grid_range[0], self.grid_range[1], 
                                   self.grid_size + self.order_spline, 
                                   device=x.device).unsqueeze(0).expand(input_dim, -1)
        bandwidth = (self.grid_range[1] - self.grid_range[0]) / self.grid_size
        distances_sq = (x.unsqueeze(-1) - grid_points.unsqueeze(0).unsqueeze(0)) ** 2
        bases = torch.exp(-distances_sq / (2 * bandwidth ** 2))
        bases = bases / (bases.sum(dim=-1, keepdim=True) + 1e-8)
        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that combines the global MLP branch with the local spline/RBF branch.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch_size, seq_len, input_dim = x.shape
        
        # --- FIX: Rewritten forward logic ---

        # 1. Calculate the global trend using the powerful MLP branch on the raw input
        global_output = self.global_branch(x)

        # 2. Project the input for the local branch to handle data scaling
        x_proj = self.input_projection(x)
        
        # 3. Calculate the local correction using the specialized branch
        if self.basis_function == 'b_spline':
            b_splines_val = self.b_splines(x_proj)
            spline_output = torch.einsum('bsik,iok->bso', b_splines_val, self.spline_weight)
            local_output = spline_output.view(batch_size, seq_len, self.output_dim)
            
        else: # RBF forward pass
            x_expanded = x_proj.unsqueeze(-1).unsqueeze(-1)
            centers = self.centers.unsqueeze(0).unsqueeze(0)
            gammas = self.gammas.unsqueeze(0).unsqueeze(0)

            dist_sq = (x_expanded - centers).pow(2)
            rbf_out = torch.exp(-F.softplus(gammas) * dist_sq)
            rbf_activated = rbf_out.sum(dim=3)
            
            rbf_flat = rbf_activated.view(batch_size, seq_len, -1)
            local_output = self.local_linear(rbf_flat)
        
        # 4. Combine the global trend and the local correction
        output = global_output + local_output
            
        return output


# --- Expert 2: Fourier KAN Layer (Unchanged) ---

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
        
        t_expanded = t.unsqueeze(-1)
        freqs = self.frequencies.unsqueeze(0).unsqueeze(0)
        
        arg = freqs * t_expanded
        
        cos_term = torch.cos(arg).unsqueeze(-1) * self.cos_coeffs.unsqueeze(0).unsqueeze(0)
        sin_term = torch.sin(arg).unsqueeze(-1) * self.sin_coeffs.unsqueeze(0).unsqueeze(0)
        
        output = (cos_term + sin_term).sum(dim=(2, 3)) + self.bias
        return output


# --- Expert 3: Enhanced Wavelet KAN Layer (Unchanged) ---

class WaveletKANLayer(nn.Module):
    """
    Enhanced wavelet expert with multiple wavelet types optimized for abrupt changes.
    """
    def __init__(self, input_dim: int, output_dim: int, n_wavelets: int = 16, wavelet_type: str = 'shock'):
        super().__init__()
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

    def adaptive_morlet_wavelet(self, t, freq, sharpness):
        c = math.pi**(-0.25)
        sharp = F.softplus(sharpness) + 0.1
        freq_param = F.softplus(freq) + 1.0
        return c * torch.exp(-sharp * t**2) * torch.cos(freq_param * t)

    def mexican_hat_wavelet(self, t):
        c = (2 / (math.sqrt(3) * math.pi**(1/4)))
        return c * (1 - t**2) * torch.exp(-t**2 / 2)

    def haar_wavelet(self, t):
        result = torch.zeros_like(t)
        mask1 = (t >= 0) & (t < 0.5)
        mask2 = (t >= 0.5) & (t < 1.0)
        result[mask1] = 1.0
        result[mask2] = -1.0
        return result

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

    def morlet_wavelet(self, t):
        c = math.pi**(-0.25)
        return c * torch.exp(-0.5 * t**2) * torch.cos(5.0 * t)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2: 
            t = t.unsqueeze(-1)
        
        batch_size, seq_len, _ = t.shape
        
        t_expanded = t.unsqueeze(-1)
        scales = F.softplus(self.scales).unsqueeze(0).unsqueeze(0) + 0.1
        shifts = self.shifts.unsqueeze(0).unsqueeze(0)
        
        wavelet_input = (t_expanded - shifts) / scales
        
        if self.wavelet_type == 'adaptive_morlet':
            freq = self.frequencies.unsqueeze(0).unsqueeze(0)
            sharp = self.sharpness.unsqueeze(0).unsqueeze(0)
            wavelet_activations = self.adaptive_morlet_wavelet(wavelet_input, freq, sharp)
        elif self.wavelet_type == 'mexican_hat':
            wavelet_activations = self.mexican_hat_wavelet(wavelet_input)
        elif self.wavelet_type == 'haar':
            wavelet_activations = self.haar_wavelet(wavelet_input)
        elif self.wavelet_type == 'shock':
            asym = self.asymmetry.unsqueeze(0).unsqueeze(0)
            steep = self.steepness.unsqueeze(0).unsqueeze(0)
            wavelet_activations = self.shock_wavelet(wavelet_input, asym, steep)
        else:
            wavelet_activations = self.morlet_wavelet(wavelet_input)
        
        wavelet_flat = wavelet_activations.view(batch_size, seq_len, -1)
        output = self.linear(wavelet_flat)
        return output


# --- The Main K-MOTE Module (Unchanged, but now benefits from fixed experts) ---

class KMOTE(nn.Module):
    """
    Enhanced K-MOTE with conditional LayerNorm for stability and gating temperature
    to encourage decisive expert selection.
    """
    def __init__(self, input_dim: int, output_dim: int, wavelet_type: str = 'shock', use_layernorm: bool = True,
                 gating_temp: float = 1.0):
        super().__init__()
        self.output_dim = output_dim
        self.temperature = gating_temp

        self.experts = nn.ModuleList([
            SplineKANLayer(input_dim, output_dim, basis_function='b_spline', grid_size=8),
            FourierKANLayer(input_dim, output_dim, n_harmonics=16),
            WaveletKANLayer(input_dim, output_dim, n_wavelets=16, wavelet_type=wavelet_type),
            SplineKANLayer(input_dim, output_dim, basis_function='rbf', grid_size=8)
        ])
        
        self.num_experts = len(self.experts)
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.num_experts)
        )
        
        # --- FIX: Conditionally apply LayerNorm ONLY when output_dim > 1 ---
        if use_layernorm and output_dim > 1:
            self.layer_norm = nn.LayerNorm(output_dim)
            print(f"Initialized K-MOTE with LayerNorm, Gating Temperature={self.temperature}.")
        else:
            self.layer_norm = nn.Identity()
            print(f"Initialized K-MOTE (No LayerNorm for 1D output), Gating Temperature={self.temperature}.")

    def forward(self, t: torch.Tensor, return_weights: bool = False) -> torch.Tensor:
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        gating_logits = self.gating_network(t)
        gating_weights = F.softmax(gating_logits / self.temperature, dim=-1)
        
        expert_outputs = [expert(t) for expert in self.experts]
        
        stacked_outputs = torch.stack(expert_outputs, dim=-1)
        gating_weights = gating_weights.unsqueeze(-2)
        
        weighted_sum = (gating_weights * stacked_outputs).sum(dim=-1)
        
        output_embedding = self.layer_norm(weighted_sum)
        
        if return_weights:
            return output_embedding, gating_weights.squeeze(-2)
        return output_embedding