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

        # ===== REMOVE: input_projection (now handled at K-MOTE level) =====
        # self.input_projection = nn.Linear(input_dim, input_dim)
        # ===== END REMOVE =====

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
            # Parameters for the LOCAL spline branch - following LeTE's approach
            self.num_knots = self.grid_size + 3  # Similar to LeTE's knot count
            # Learnable knot positions in [0, 1] range (like LeTE)
            self.knots = nn.Parameter(torch.linspace(0, 1, self.num_knots))
            # Learnable coefficients (like LeTE's coeffs)
            self.spline_coeffs = nn.Parameter(torch.randn(self.output_dim, self.num_knots) * 0.1)
            
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
        Compute B-spline basis functions following LeTE's approach.
        Uses sigmoid normalization and L1 distance like LeTE.
        """
        # Normalize input to [0, 1] range using sigmoid (like LeTE)
        x_norm = torch.sigmoid(x)  # Maps any real input to [0, 1]
        
        # Handle different input dimensions properly
        original_shape = x_norm.shape
        
        # Flatten to 2D for processing: (batch_size * seq_len, input_dim)
        if x_norm.dim() > 2:
            x_flat = x_norm.view(-1, x_norm.shape[-1])  # Flatten all but last dim
        else:
            x_flat = x_norm
        
        # Extract the last dimension (should be 1 for time encoding)
        if x_flat.shape[-1] == 1:
            x_values = x_flat.squeeze(-1)  # (batch_size * seq_len,)
        else:
            # Take mean across features if multiple features
            x_values = x_flat.mean(dim=-1)  # (batch_size * seq_len,)
        
        # Reshape for broadcasting with knots
        x_values = x_values.unsqueeze(-1)  # (batch_size * seq_len, 1)
        knots = self.knots.unsqueeze(0)  # (1, num_knots)
        
        # Compute distances using L1 norm (like LeTE)
        distances = torch.abs(x_values - knots)  # (batch_size * seq_len, num_knots)
        
        # RBF kernel as B-spline approximation (like LeTE)
        basis = torch.exp(-distances * 5.0)  # (batch_size * seq_len, num_knots)
        
        # Apply learnable coefficients using matrix multiplication (like LeTE)
        # basis: (batch_size * seq_len, num_knots)
        # spline_coeffs: (output_dim, num_knots)
        # Want output: (batch_size * seq_len, output_dim)
        spline_output = torch.matmul(basis, self.spline_coeffs.T)  # (batch_size * seq_len, output_dim)
        
        # Reshape back to original batch/sequence dimensions
        if len(original_shape) == 3:  # (batch_size, seq_len, input_dim)
            spline_output = spline_output.view(original_shape[0], original_shape[1], self.output_dim)
        elif len(original_shape) == 2:  # (batch_size, input_dim)
            spline_output = spline_output.view(original_shape[0], self.output_dim)
        else:  # (batch_size,)
            spline_output = spline_output.view(original_shape[0], self.output_dim)
        
        return spline_output

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

        # ===== REMOVE: input_projection (now handled at K-MOTE level) =====
        # 2. Use input directly for local branch (preprocessing done at K-MOTE level)
        x_proj = x  # No additional projection needed
        # ===== END REMOVE =====
        
        # 3. Calculate the local correction using the specialized branch
        if self.basis_function == 'b_spline':
            # Use the corrected B-spline implementation (following LeTE)
            spline_output = self.b_splines(x_proj)  # Direct output, no einsum needed
            local_output = spline_output  # Already in correct shape
            
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
    Backward-compatible memory-efficient Fourier expert for periodic patterns.
    """
    def __init__(self, input_dim: int, output_dim: int, n_harmonics: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_harmonics = n_harmonics
        
        # KEEP SAME parameter structure for compatibility with existing models
        self.cos_coeffs = nn.Parameter(torch.randn(input_dim, n_harmonics, output_dim) * 0.1)
        self.sin_coeffs = nn.Parameter(torch.randn(input_dim, n_harmonics, output_dim) * 0.1)
        self.frequencies = nn.Parameter(torch.randn(input_dim, n_harmonics) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2: 
            t = t.unsqueeze(-1)
        
        batch_size, seq_len, input_dim = t.shape
        
        # MEMORY FIX: Process in flattened form to avoid 5D tensors
        t_flat = t.view(-1, input_dim)  # (B*S, input_dim)
        
        # Compute frequency arguments efficiently
        t_expanded = t_flat.unsqueeze(-1)  # (B*S, input_dim, 1)
        freqs = self.frequencies.unsqueeze(0)  # (1, input_dim, n_harmonics)
        
        args = t_expanded * freqs  # (B*S, input_dim, n_harmonics)
        
        # Compute trigonometric functions (still manageable size)
        cos_vals = torch.cos(args)  # (B*S, input_dim, n_harmonics)
        sin_vals = torch.sin(args)  # (B*S, input_dim, n_harmonics)
        
        # MEMORY FIX: Use einsum to avoid creating huge intermediate tensors
        cos_output = torch.einsum('bih,iho->bo', cos_vals, self.cos_coeffs)  # (B*S, output_dim)
        sin_output = torch.einsum('bih,iho->bo', sin_vals, self.sin_coeffs)  # (B*S, output_dim)
        
        # Combine and add bias
        output = cos_output + sin_output + self.bias
        
        # Reshape back to original dimensions
        output = output.view(batch_size, seq_len, self.output_dim)
        
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
    Final, architecturally correct K-MOTE.
    This version fully aligns with the LeTE/Time2Vec design by expanding the
    time dimension in the initial linear layer BEFORE it is passed to the experts.
    """
    def __init__(self, input_dim: int, output_dim: int,
                 wavelet_type: str = 'shock',
                 use_layernorm: bool = True,
                 use_scale: bool = True,
                 gating_temp: float = 1.0):
        super().__init__()
        
        # This architecture is designed for a 1D time input.
        if input_dim != 1:
            raise ValueError("K-MOTE requires input_dim=1 for the time input.")

        self.output_dim = output_dim
        self.temperature = gating_temp
        self.use_scale = use_scale

        # ===== CRITICAL CHANGE 1: The Initial Linear Transformation =====
        # This layer now expands the 1D time input to the full output dimension.
        # This is the 'w*t + b' step that creates the multi-channel time representation.
        self.time_linear_transform = nn.Linear(input_dim, output_dim)
        self._initialize_time_transform() # Apply the special initialization
        # =============================================================

        if use_scale:
            self.scale = nn.Parameter(torch.ones(output_dim))

        # ===== CRITICAL CHANGE 2: Experts now operate on high-dimensional input =====
        # The input_dim for all experts is now `output_dim`, as they receive the
        # already-transformed time vector.
        self.experts = nn.ModuleList([
            #SplineKANLayer(output_dim, output_dim, basis_function='b_spline', grid_size=8),
            FourierKANLayer(output_dim, output_dim, n_harmonics=16),
            WaveletKANLayer(output_dim, output_dim, n_wavelets=16, wavelet_type=wavelet_type),
            SplineKANLayer(output_dim, output_dim, basis_function='rbf', grid_size=8)
        ])
        # ===========================================================================
        
        self.num_experts = len(self.experts)
        # ===== CRITICAL CHANGE 3: Gating network also uses the transformed time =====
        # It needs to decide which expert to use based on the rich, multi-channel representation.
        self.gating_network = nn.Sequential(
            nn.Linear(output_dim, 64), # Input is now output_dim
            nn.GELU(),
            nn.Linear(64, self.num_experts)
        )
        # =========================================================================
        
        if use_layernorm and output_dim > 1:
            self.layer_norm = nn.LayerNorm(output_dim)
            print(f"Initialized K-MOTE with LayerNorm and LeTE-style architecture.")
        else:
            self.layer_norm = nn.Identity()

    def _initialize_time_transform(self):
        """
        Initializes the time transformation layer with a geometric progression of frequencies,
        following the LeTE / Transformer paper's methodology.
        
        CRITICAL: These weights REMAIN LEARNABLE to ensure scale-invariance.
        The initialization provides a strong inductive bias, while the trainability
        allows the model to fine-tune frequencies and adapt to input scale.
        """
        # The torch.no_grad() context is good practice for initialization.
        # It prevents this operation from being tracked in the computation graph.
        # It does NOT freeze the parameters from future gradient updates.
        with torch.no_grad():
            # Create frequencies from 1.0 down to 1.0 / 10^9
            frequencies = 1.0 / (10 ** torch.linspace(0, 9, self.output_dim, dtype=torch.float32))
            
            # Copy these values into the weight tensor. The .weight attribute itself
            # is still a learnable nn.Parameter.
            self.time_linear_transform.weight.copy_(frequencies.unsqueeze(1))
            
            # Initialize bias to zero
            self.time_linear_transform.bias.zero_()

    def forward(self, t: torch.Tensor, return_weights: bool = False) -> torch.Tensor:
        if t.dim() == 2:
            t = t.unsqueeze(-1) # Ensure shape (B, S, 1)

        # 1. Apply the LeTE-style linear transformation FIRST.
        # This is the crucial step that creates the scale-invariant, multi-scale representation.
        # Shape: (B, S, 1) -> (B, S, output_dim)
        t_transformed = self.time_linear_transform(t)
        
        # 2. Pass this high-dimensional representation to the gating network and all experts.
        gating_logits = self.gating_network(t_transformed)
        gating_weights = F.softmax(gating_logits / self.temperature, dim=-1)
        
        expert_outputs = [expert(t_transformed) for expert in self.experts]
        
        # 3. Combine, Normalize, and Scale as before.
        stacked_outputs = torch.stack(expert_outputs, dim=-1)
        gating_weights = gating_weights.unsqueeze(-2)
        
        weighted_sum = (gating_weights * stacked_outputs).sum(dim=-1)
        
        output_embedding = self.layer_norm(weighted_sum)
        
        if self.use_scale:
            output_embedding = output_embedding * self.scale
        
        if return_weights:
            return output_embedding, gating_weights.squeeze(-2)
        return output_embedding