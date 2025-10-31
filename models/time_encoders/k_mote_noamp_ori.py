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
    def __init__(self, input_dim: int, output_dim: int, grid_size: int = 5, basis_function: str = 'b_spline', order: int = 3, grid_range: list = [-1, 1]):
        super().__init__()
        if basis_function not in ['b_spline', 'rbf']:
            raise ValueError("basis_function must be 'b_spline' or 'rbf'")
        self.basis_function = basis_function
        self.grid_size = grid_size
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.order = order

        if self.basis_function == 'b_spline':

            h = (grid_range[1] - grid_range[0]) / float(self.grid_size)
            
            grid = torch.arange(-self.order, self.grid_size + self.order + 1)
            grid = grid * h + grid_range[0]
            grid = grid.expand(self.input_dim, -1).contiguous()
            self.register_buffer("grid", grid)
            
            # Base weight for the linear+activation branch (like LeTE)
            self.base_weight = nn.Parameter(
                torch.randn(self.input_dim, self.output_dim) / math.sqrt(self.input_dim)
            )
            
            self.spline_weight = nn.Parameter(
                torch.randn(self.input_dim, self.output_dim, self.grid_size + self.order) /
                (math.sqrt(self.input_dim) * math.sqrt(self.grid_size + self.order))
            )
            
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
        TRUE Cox-de Boor B-spline implementation (copied exactly from LeTE).
        This replaces the incorrect RBF kernel implementation.
        """
        # grid: (input_dim, grid_size + 2*order + 1)
        grid = self.grid
        x = x.unsqueeze(-1)  # shape: (N, input_dim, 1)

        # bases will mark where x lies between [grid_i, grid_{i+1})
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # shape: (N, input_dim, ?)

        # Recursively elevate the basis to higher spline orders (Cox-de Boor recursion)
        for k in range(1, self.order + 1):
            # Segment lengths in the left and right directions
            left_num = (x - grid[:, :-(k + 1)])
            left_den = (grid[:, k:-1] - grid[:, :-(k + 1)])

            right_num = (grid[:, k + 1:] - x)
            right_den = (grid[:, k + 1:] - grid[:, 1:-k])

            bases = (left_num / left_den) * bases[:, :, :-1] + (right_num / right_den) * bases[:, :, 1:]

        return bases.contiguous()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using TRUE B-spline implementation (copied from LeTE).
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        original_shape = x.shape
        batch_size, seq_len, input_dim = x.shape
        
        if self.basis_function == 'b_spline':
            x_flat = x.reshape(-1, input_dim)  # flatten all but the last dimension

            # Base branch: tanh + linear (like LeTE)
            base_output = F.linear(torch.tanh(x_flat), self.base_weight.T)  # shape: (N, output_dim)

            # If the input batch is empty, return a zero-like tensor
            if x_flat.size(0) == 0:
                spline_output = torch.zeros_like(base_output)
            else:
                B = self.b_splines(x_flat).reshape(x_flat.size(0), -1)  # (N, I*(G+O))
                W = self.spline_weight.permute(0, 2, 1).reshape(-1, self.output_dim)  # (I*(G+O), O)
                spline_output = B @ W  # (N, O)

            output = base_output + spline_output
            output = output.reshape(batch_size, seq_len, self.output_dim)
            
        else: # RBF forward pass
            x_expanded = x.unsqueeze(-1).unsqueeze(-1)
            centers = self.centers.unsqueeze(0).unsqueeze(0)
            gammas = self.gammas.unsqueeze(0).unsqueeze(0)

            dist_sq = (x_expanded - centers).pow(2)
            rbf_out = torch.exp(-F.softplus(gammas) * dist_sq)
            rbf_activated = rbf_out.sum(dim=3)
            
            rbf_flat = rbf_activated.view(batch_size, seq_len, -1)
            output = self.local_linear(rbf_flat)
            
        return output


# --- Expert 2: Fourier KAN Layer (Unchanged) ---

class FourierKANLayer(nn.Module):
    """
    Memory-efficient Fourier expert following LeTE's vectorized design.
    Processes all input dimensions together instead of per-dimension loops.
    """
    def __init__(self, input_dim: int, output_dim: int, n_harmonics: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_harmonics = n_harmonics
        
        self.fourier_weight = nn.Parameter(
            torch.randn(2, input_dim, output_dim, n_harmonics) / 
            (math.sqrt(input_dim) * math.sqrt(n_harmonics))
        )
        
        # Learnable base output (like LeTE)
        self.base_weight = nn.Parameter(torch.randn(input_dim, output_dim) / math.sqrt(input_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using vectorized operations (like LeTE).
        
        Args:
            t: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, output_dim)
        """
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        batch_size, seq_len, input_dim = t.shape

        t_flat = t.reshape(-1, input_dim)  # (B*S, input_dim)
        
        base_output = torch.matmul(t_flat, self.base_weight)
        
        k = torch.arange(1, self.n_harmonics + 1, device=t.device, dtype=t.dtype)
        args = t_flat.unsqueeze(-1) * k  # (B*S, input_dim, n_harmonics)
        
        c = torch.cos(args)  # (B*S, 1, input_dim, n_harmonics)
        s = torch.sin(args)  # (B*S, 1, input_dim, n_harmonics)

        cos_output = torch.einsum('bih,ioh->bo', c, self.fourier_weight[0])
        sin_output = torch.einsum('bih,ioh->bo', s, self.fourier_weight[1])

        output = base_output + cos_output + sin_output + self.bias  # (B*S, output_dim)
        
        # Reshape back to original dimensions
        output = output.view(batch_size, seq_len, self.output_dim)
        
        return output

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
    Hybrid K-MOTE with configurable time transformation strategy.
    
    Supports three architectures:
    - transform_mode='shared': Single time transform shared by all experts (MoE approach)
    - transform_mode='per_expert': Per-expert time transforms (LeTE-style specialization)
    - transform_mode='adapter' (DEFAULT): Shared base + lightweight expert adapters
    """
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = None,
                 wavelet_type: str = 'shock',
                 use_layernorm: bool = True,
                 use_scale: bool = True,
                 gating_temp: float = 1.0,
                 transform_mode: str = 'adapter',
                 adapter_type: str = 'affine'):
        super().__init__()
        self.adapter_type = adapter_type
        # This architecture is designed for a 1D time input.
        if input_dim != 1:
            raise ValueError("K-MOTE requires input_dim=1 for the time input.")
        
        if transform_mode not in ['shared', 'per_expert', 'adapter']:
            raise ValueError("transform_mode must be 'shared', 'per_expert', or 'adapter'")
        
         # ===== FIX: Only validate adapter_type when in adapter mode =====
        if transform_mode == 'adapter':
            if adapter_type not in ['affine', 'linear']:
                raise ValueError(f"adapter_type must be 'affine' or 'linear', got {adapter_type}")
        # ===== END FIX =====

        self.output_dim = output_dim
        self.hidden_dim = output_dim//3 if hidden_dim is not None else output_dim
        self.temperature = gating_temp
        self.use_scale = use_scale
        self.transform_mode = transform_mode
        

        if self.transform_mode == 'adapter':
            if self.adapter_type not in ['affine', 'linear']:
                raise ValueError("adapter_type must be 'affine' or 'linear'")
        # ===== CONFIGURABLE TIME TRANSFORMATION =====
        if transform_mode == 'shared':
            # Option A: Single shared transform (MoE approach)
            self.time_linear_transform = nn.Linear(input_dim, self.hidden_dim)
            self._initialize_shared_transform()
            
        elif transform_mode == 'per_expert':
            # Option B: Per-expert transforms (LeTE-style specialization)
            self.num_experts = 3  # Will be used for initialization
            self.time_transforms = nn.ModuleList([
                nn.Linear(input_dim, self.hidden_dim) for _ in range(self.num_experts)
            ])
            self._initialize_expert_transforms()
            
        else:  # transform_mode == 'adapter'
            # Option C: Shared base + expert adapters (DEFAULT, best balance)
            self.num_experts = 3
            self.time_base_transform = nn.Linear(input_dim, self.hidden_dim)
            self._initialize_shared_transform()  # Initialize base with geometric progression
            
            if adapter_type == 'affine':
                # Lightweight affine adapters (scale + shift, like LayerNorm)
                self.expert_scales = nn.ParameterList([
                    nn.Parameter(torch.ones(self.hidden_dim)) for _ in range(self.num_experts)
                ])
                self.expert_shifts = nn.ParameterList([
                    nn.Parameter(torch.zeros(self.hidden_dim)) for _ in range(self.num_experts)
                ])
            else:  # adapter_type == 'linear'
                # Small linear adapters
                self.expert_adapters = nn.ModuleList([
                    nn.Linear(self.hidden_dim, self.hidden_dim) for _ in range(self.num_experts)
                ])
                # Initialize adapters to near-identity
                for adapter in self.expert_adapters:
                    nn.init.eye_(adapter.weight)
                    nn.init.zeros_(adapter.bias)
        # =============================================================

        if use_scale:
            self.scale = nn.Parameter(torch.ones(output_dim))

        # ===== EXPERTS: Now use hidden_dim for input =====
        self.experts = nn.ModuleList([
            SplineKANLayer(self.hidden_dim, output_dim, basis_function='b_spline', grid_size=5, order=3, grid_range=[-1, 1]),
            FourierKANLayer(self.hidden_dim, output_dim, n_harmonics=8),
            WaveletKANLayer(self.hidden_dim, output_dim, n_wavelets=8, wavelet_type=wavelet_type),
        ])
        # ===========================================================================
        
        self.num_experts = len(self.experts)
        
        # ===== GATING NETWORK: Uses hidden_dim input =====
        self.gating_network = nn.Sequential(
            nn.Linear(self.hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.num_experts)
        )
        # ============================================
        
        if use_layernorm and output_dim > 1:
            self.layer_norm = nn.LayerNorm(output_dim)
            if transform_mode == 'adapter':
                print(f"Initialized K-MOTE with {adapter_type} adapters (DEFAULT), hidden_dim={self.hidden_dim}")
            else:
                transform_type = "shared" if transform_mode == 'shared' else "per-expert"
                print(f"Initialized K-MOTE with {transform_type} transform, hidden_dim={self.hidden_dim}")
        else:
            self.layer_norm = nn.Identity()

    def _initialize_shared_transform(self):
        """
        Initializes the shared time transformation layer with a geometric progression of frequencies,
        following the LeTE / Transformer paper's methodology.
        
        CRITICAL: These weights REMAIN LEARNABLE to ensure scale-invariance.
        The initialization provides a strong inductive bias, while the trainability
        allows the model to fine-tune frequencies and adapt to input scale.
        """
        with torch.no_grad():
            # Create frequencies from 1.0 down to 1.0 / 10^9
            frequencies = 1.0 / (10 ** torch.linspace(0, 9, self.hidden_dim, dtype=torch.float32))
            
            # Determine which transform to initialize based on mode
            if hasattr(self, 'time_linear_transform'):
                # For 'shared' mode
                self.time_linear_transform.weight.copy_(frequencies.unsqueeze(1))
                self.time_linear_transform.bias.zero_()
            elif hasattr(self, 'time_base_transform'):
                # For 'adapter' mode
                self.time_base_transform.weight.copy_(frequencies.unsqueeze(1))
                self.time_base_transform.bias.zero_()
    
    def _initialize_expert_transforms(self):
        """
        Initializes per-expert time transforms with different frequency ranges.
        This allows each expert to specialize in different temporal scales.
        
        Expert specialization:
        - Expert 0 (Spline): Low-to-medium frequencies (smooth trends)
        - Expert 1 (Fourier): Medium frequencies (periodic patterns)
        - Expert 2 (Wavelet): Medium-to-high frequencies (abrupt changes)
        """
        with torch.no_grad():
            # Expert 0 (Spline): Focus on low-to-medium frequencies (0 to 6)
            freqs_spline = 1.0 / (10 ** torch.linspace(0, 6, self.hidden_dim, dtype=torch.float32))
            self.time_transforms[0].weight.copy_(freqs_spline.unsqueeze(1))
            self.time_transforms[0].bias.zero_()
            
            # Expert 1 (Fourier): Focus on medium frequencies (2 to 8)
            freqs_fourier = 1.0 / (10 ** torch.linspace(2, 8, self.hidden_dim, dtype=torch.float32))
            self.time_transforms[1].weight.copy_(freqs_fourier.unsqueeze(1))
            self.time_transforms[1].bias.zero_()
            
            # Expert 2 (Wavelet): Focus on medium-to-high frequencies (4 to 9)
            freqs_wavelet = 1.0 / (10 ** torch.linspace(4, 9, self.hidden_dim, dtype=torch.float32))
            self.time_transforms[2].weight.copy_(freqs_wavelet.unsqueeze(1))
            self.time_transforms[2].bias.zero_()

    def forward(self, t: torch.Tensor, return_weights: bool = False) -> torch.Tensor:
        if t.dim() == 2:
            t = t.unsqueeze(-1) # Ensure shape (B, S, 1)

        # 1. Apply time transformation(s) based on mode
        if self.transform_mode == 'shared':
            # Option A: Shared transform - all experts get the same transformed input
            t_transformed = self.time_linear_transform(t)  # (B, S, hidden_dim)
            
            # Pass to gating and experts
            gating_logits = self.gating_network(t_transformed)
            gating_weights = F.softmax(gating_logits / self.temperature, dim=-1)
            
            expert_outputs = [expert(t_transformed) for expert in self.experts]
            
        elif self.transform_mode == 'per_expert':
            # Option B: Per-expert transforms - each expert gets its own specialized input
            t_transformed = [transform(t) for transform in self.time_transforms]  # List of (B, S, hidden_dim)
            
            # Each expert processes its specialized input
            expert_outputs = [expert(t_trans) for expert, t_trans in zip(self.experts, t_transformed)]
            
            # Gating network uses the average of all transformed inputs
            t_for_gating = torch.stack(t_transformed, dim=0).mean(dim=0)  # (B, S, hidden_dim)
            gating_logits = self.gating_network(t_for_gating)
            gating_weights = F.softmax(gating_logits / self.temperature, dim=-1)
            
        else:  # transform_mode == 'adapter'
            # Option C: Shared base + expert adapters (DEFAULT)
            t_base = self.time_base_transform(t)  # (B, S, hidden_dim)
            
            # Apply expert-specific adaptations
            expert_outputs = []
            for i, expert in enumerate(self.experts):
                if self.adapter_type == 'affine':
                    # Simple affine adaptation (scale + shift)
                    t_adapted = t_base * self.expert_scales[i] + self.expert_shifts[i]
                else:  # adapter_type == 'linear'
                    # Linear adapter
                    t_adapted = self.expert_adapters[i](t_base)
                
                expert_outputs.append(expert(t_adapted))
            
            # Gating network uses the shared base representation
            gating_logits = self.gating_network(t_base)
            gating_weights = F.softmax(gating_logits / self.temperature, dim=-1)
        
        # 2. Combine expert outputs using gating weights
        stacked_outputs = torch.stack(expert_outputs, dim=-1)  # (B, S, output_dim, num_experts)
        gating_weights = gating_weights.unsqueeze(-2)  # (B, S, 1, num_experts)
        
        weighted_sum = (gating_weights * stacked_outputs).sum(dim=-1)  # (B, S, output_dim)
        
        # 3. Normalize and scale
        output_embedding = self.layer_norm(weighted_sum)
        
        if self.use_scale:
            output_embedding = output_embedding * self.scale
        
        if return_weights:
            return output_embedding, gating_weights.squeeze(-2)
        return output_embedding