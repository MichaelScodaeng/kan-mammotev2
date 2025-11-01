# file: models/time_encoders/k_mote.py (Corrected and Refactored)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.cuda.amp as amp



class LeTESpline(nn.Module):
    """
    Args:
        dim_spline (int): Dimension of the spline input (and typically output).
        grid_size_spline (int): Number of spline knots (excluding boundary). Defaults to 5.
        order_spline (int): Spline order (degree). Defaults to 3.
        grid_range (list): Value range of the grid. Defaults to [-1, 1].

    Attributes:
        dim_spline (int): Dimension of the input/output.
        grid_size_spline (int): Number of spline knots in each dimension.
        order_spline (int): Spline order (degree).
        grid (torch.Tensor): The registered buffer containing grid points of shape (dim_spline, grid_size_spline + 2 * order_spline + 1).
        base_weight (torch.nn.Parameter): Linear weight for the base branch (dim_spline x dim_spline).
        spline_weight (torch.nn.Parameter): Learnable B-spline coefficients (dim_spline x dim_spline x (grid_size_spline + order_spline)).
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
        x = x.reshape(-1, self.dim_spline)  # flatten all but the last dimension

        # Base branch: tanh + linear
        base_output = nn.functional.linear(torch.tanh(x), self.base_weight)  # shape: (N, dim_spline)

        # If the input batch is empty, return a zero-like tensor
        if x.size(0) == 0:
            spline_output = torch.zeros_like(base_output)
        else:
            # Evaluate B-spline basis
            # b_splines(x).shape -> (N, dim_spline, grid_size_spline + order_spline)
            # Flatten for linear
            b_splines_val = self.b_splines(x).view(x.size(0), -1)

            # Reshape spline_weight for linear operation
            w = self.spline_weight.view(self.dim_spline, -1)
            spline_output = nn.functional.linear(b_splines_val, w)

        output = base_output + spline_output
        output = output.reshape(*original_shape[:-1], self.dim_spline)
        return output

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # A: (dim_spline, batch_size, grid_size_spline + order_spline)
        A = self.b_splines(x).transpose(0, 1)
        # B: (dim_spline, batch_size, dim_spline)
        B = y.transpose(0, 1)

        # Compute pseudo-inverse of A
        A_pinv = torch.pinverse(A)  # shape: (dim_spline, grid_size_spline+order_spline, batch_size)

        # Solve the least squares problem using the pseudo-inverse
        # shape: (dim_spline, grid_size_spline+order_spline, batch_size)
        solution = torch.bmm(A_pinv, B)

        # Permute to (batch_size, dim_spline, grid_size_spline+order_spline) -> (dim_spline, dim_spline, grid_size_spline+order_spline)
        result = solution.permute(2, 0, 1).contiguous()
        return result

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        # grid: (dim_spline, grid_size_spline + 2*order_spline + 1)
        grid = self.grid
        x = x.unsqueeze(-1)  # shape: (N, dim_spline, 1)

        # bases will mark where x lies between [grid_i, grid_{i+1})
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)  # shape: (N, dim_spline, ?)

        # Recursively elevate the basis to higher spline orders
        for k in range(1, self.order_spline + 1):
            # Segment lengths in the left and right directions
            left_num = (x - grid[:, :-(k + 1)])
            left_den = (grid[:, k:-1] - grid[:, :-(k + 1)])

            right_num = (grid[:, k + 1:] - x)
            right_den = (grid[:, k + 1:] - grid[:, 1:-k])

            # Avoid division by zero by adding small epsilon
            eps = 1e-12
            left_den = left_den + eps
            right_den = right_den + eps
            
            # Handle potential NaN/Inf in division
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
    Exact LeTE FourierSeries implementation following lete_encoder.py
    """
    def __init__(self, dim_fourier: int, grid_size_fourier: int = 5):
        super().__init__()
        self.dim_fourier = dim_fourier
        self.grid_size_fourier = grid_size_fourier

        # Exact LeTE fourier_weight shape: (2, dim_fourier, dim_fourier, grid_size_fourier)
        #   - 2 corresponds to cosine and sine parts, respectively
        #   - The layer outputs dim_fourier features given dim_fourier inputs
        self.fourier_weight = torch.nn.Parameter(
            torch.randn(2, self.dim_fourier, self.dim_fourier, grid_size_fourier) /
            (math.sqrt(self.dim_fourier) * math.sqrt(self.grid_size_fourier))
        )

        # Bias term, one per output dimension
        self.bias = nn.Parameter(torch.zeros(self.dim_fourier))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        out_shape = original_shape[0:-1] + (self.dim_fourier,)

        # Flatten everything except the last dim
        x = x.reshape(-1, self.dim_fourier)  # (N, dim_fourier)

        # Frequency indices k = 1..grid_size_fourier
        k = torch.arange(1, self.grid_size_fourier + 1, device=x.device)
        k = k.reshape(1, 1, 1, self.grid_size_fourier)  # shape: (1,1,1,K)

        # Reshape x to broadcast with k
        x_reshaped = x.reshape(x.shape[0], 1, x.shape[1], 1)  # (N,1,dim_fourier,1)

        # Compute cos(k * x) and sin(k * x)
        c = torch.cos(k * x_reshaped)  # (N,1,dim_fourier,K)
        s = torch.sin(k * x_reshaped)  # (N,1,dim_fourier,K)

        # Sum up the contributions with the learned Fourier coefficients
        # fourier_weight[0:1] -> cosine part, fourier_weight[1:2] -> sine part
        y = torch.sum(c * self.fourier_weight[0:1], dim=(-2, -1))
        y += torch.sum(s * self.fourier_weight[1:2], dim=(-2, -1))

        # Add bias
        y += self.bias

        # Reshape back to original shape
        y = y.reshape(out_shape)
        return y
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
            # TRUE B-SPLINE IMPLEMENTATION (copied from LeTE)
            # Compute grid spacing
            h = (grid_range[1] - grid_range[0]) / float(self.grid_size)
            
            # Build the grid for each input dimension
            grid = torch.arange(-self.order, self.grid_size + self.order + 1)
            grid = grid * h + grid_range[0]
            grid = grid.expand(self.input_dim, -1).contiguous()
            self.register_buffer("grid", grid)
            
            # Base weight for the linear+activation branch (like LeTE)
            self.base_weight = nn.Parameter(
                torch.randn(self.input_dim, self.output_dim) / math.sqrt(self.input_dim)
            )
            
            # Spline coefficients (like LeTE)
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
            # TRUE B-SPLINE FORWARD (exactly like LeTE)
            x_flat = x.reshape(-1, input_dim)  # flatten all but the last dimension

            # Base branch: tanh + linear (like LeTE)
            base_output = F.linear(torch.tanh(x_flat), self.base_weight.T)  # shape: (N, output_dim)

            # If the input batch is empty, return a zero-like tensor
            if x_flat.size(0) == 0:
                spline_output = torch.zeros_like(base_output)
            else:
                # CLEARER: Explicit reshape instead of view chains
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
        
        # CHANGE: Shared Fourier weights across all input dimensions (like LeTE)
        # Shape: (2, input_dim, output_dim, n_harmonics)
        # First dimension: [0] = cos weights, [1] = sin weights
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
        
        # Reshape for efficient computation
        t_flat = t.reshape(-1, input_dim)  # (B*S, input_dim)
        
        # Step 1: Base transformation (linear component)
        # (B*S, input_dim) @ (input_dim, output_dim) = (B*S, output_dim)
        base_output = torch.matmul(t_flat, self.base_weight)
        
         # FIXED: Corrected Fourier computation
        k = torch.arange(1, self.n_harmonics + 1, device=t.device, dtype=t.dtype)
        args = t_flat.unsqueeze(-1) * k  # (B*S, input_dim, n_harmonics)
        
        # Compute cos and sin (vectorized across all features)
        c = torch.cos(args)  # (B*S, 1, input_dim, n_harmonics)
        s = torch.sin(args)  # (B*S, 1, input_dim, n_harmonics)
        
        # ✅ FIXED: Correct einsum contraction (3D input, not 4D)
        # c: (B*S, input_dim, n_harmonics)
        # fourier_weight[0]: (input_dim, output_dim, n_harmonics)
        # Result: (B*S, output_dim)
        cos_output = torch.einsum('bih,ioh->bo', c, self.fourier_weight[0])
        sin_output = torch.einsum('bih,ioh->bo', s, self.fourier_weight[1])
        # ✅ END FIX
        
        # Combine all components
        output = base_output + cos_output + sin_output + self.bias  # (B*S, output_dim)
        
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
    Hybrid K-MOTE with configurable time transformation strategy.
    
    Supports three architectures:
    - transform_mode='shared': Single time transform shared by all experts (MoE approach)
    - transform_mode='per_expert': Per-expert time transforms (LeTE-style specialization)
    - transform_mode='adapter' (DEFAULT): Shared base + lightweight expert adapters
    
    Memory optimization:
    - use_gradient_checkpointing: Trade compute for memory by checkpointing expert computations
    """
    def __init__(self, input_dim: int, output_dim: int,
                 hidden_dim: int = 64, # Default to a reasonable hidden_dim
                 wavelet_type: str = 'shock',
                 use_layernorm: bool = True,
                 use_scale: bool = True,
                 gating_temp: float = 1.0,
                 transform_mode: str = 'per_expert',
                 adapter_type: str = 'affine',
                 use_gradient_checkpointing: bool = True,
                 use_amp: bool = False):
        super().__init__()
        self.adapter_type = adapter_type
        self.use_gradient_checkpointing = use_gradient_checkpointing
        # Optionally enable mixed-precision (AMP) for internal expert computations
        self.use_amp = use_amp
        
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
        self.hidden_dim = hidden_dim if hidden_dim is not None else output_dim//3
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
                print("Using AFFINE adapters for K-MOTE experts.")
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

        # ===== EXPERTS: Now use hidden_dim for input and ensure consistent output =====
        self.experts = nn.ModuleList([
            # Expert 1: LeTE Spline
            nn.Sequential(
                LeTESpline(dim_spline=self.hidden_dim),
                nn.Linear(self.hidden_dim, self.output_dim)
            ),
            
            # Expert 2: LeTE Fourier (exact LeTE implementation)
            nn.Sequential(
                LeTEFourierSeries(dim_fourier=self.hidden_dim, grid_size_fourier=5),
                nn.Linear(self.hidden_dim, self.output_dim)
            ),

            # Expert 3: Enhanced Wavelet KAN
            nn.Sequential(
                EnhancedWaveletKAN(self.hidden_dim, self.hidden_dim, n_wavelets=5, wavelet_type=wavelet_type),
                nn.Linear(self.hidden_dim, self.output_dim)
            ),
        ])
        # ===========================================================================
        
        self.num_experts = len(self.experts)
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
            if use_gradient_checkpointing:
                print(f"   ├─ Gradient checkpointing: ENABLED (per-expert level)")
            else:
                print(f"   ├─ Gradient checkpointing: disabled")
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

    def _checkpoint_expert(self, expert, t_input, skip_checkpointing=False):
        """
        Helper function to conditionally apply gradient checkpointing to expert forward pass.
        
        Args:
            expert: The expert module to run
            t_input: Input tensor for the expert
            skip_checkpointing: If True, always use normal forward pass
            
        Returns:
            Output from the expert
        """
        # Helper to run expert under autocast when requested
        def _run_expert(inp):
            # Use CUDA autocast only when requested and CUDA is available
            with amp.autocast(enabled=(self.use_amp and torch.cuda.is_available())):
                return expert(inp)

        if self.use_gradient_checkpointing and self.training and not skip_checkpointing:
            # Use gradient checkpointing to save memory during training.
            # Wrap the expert call so autocast is active during recomputation as well.
            return torch.utils.checkpoint.checkpoint(_run_expert, t_input, use_reentrant=False)
        else:
            # Normal forward pass with optional autocast
            return _run_expert(t_input)

    # --- AMP convenience helpers for training scripts ---
    def make_grad_scaler(self):
        """Return a GradScaler if AMP is enabled and CUDA is available, else None."""
        if self.use_amp and torch.cuda.is_available():
            return torch.cuda.amp.GradScaler()
        return None

    def scaled_backward(self, loss: torch.Tensor, scaler):
        """Perform backward using the provided scaler when available."""
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

    def scaler_step(self, scaler, optimizer):
        """Perform optimizer step using scaler when available."""
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
    
    def _forward_impl(self, t, return_weights=False):
        """Internal forward implementation (for checkpointing the entire K-MOTE)"""
        if t.dim() == 2:
            t = t.unsqueeze(-1) # Ensure shape (B, S, 1)

        # 1. Apply time transformation(s) based on mode
        if self.transform_mode == 'shared':
            # Option A: Shared transform - all experts get the same transformed input
            t_transformed = self.time_linear_transform(t)  # (B, S, hidden_dim)
            
            # Pass to gating and experts
            gating_logits = self.gating_network(t_transformed)
            gating_weights = F.softmax(gating_logits / self.temperature, dim=-1)
            
            # Apply checkpointing to expert computations (skip for Fourier expert - index 1)
            expert_outputs = [
                self._checkpoint_expert(expert, t_transformed, skip_checkpointing=(i == 1)) 
                for i, expert in enumerate(self.experts)
            ]
            
        elif self.transform_mode == 'per_expert':
            # Option B: Per-expert transforms - each expert gets its own specialized input
            t_transformed = [transform(t) for transform in self.time_transforms]  # List of (B, S, hidden_dim)
            
            # Each expert processes its specialized input with checkpointing (skip for Fourier expert - index 1)
            expert_outputs = [
                self._checkpoint_expert(expert, t_trans, skip_checkpointing=(i == 1)) 
                for i, (expert, t_trans) in enumerate(zip(self.experts, t_transformed))
            ]
            
            # Gating network uses the average of all transformed inputs
            t_for_gating = torch.stack(t_transformed, dim=0).mean(dim=0)  # (B, S, hidden_dim)
            gating_logits = self.gating_network(t_for_gating)
            gating_weights = F.softmax(gating_logits / self.temperature, dim=-1)
            
        else:  # transform_mode == 'adapter'
            # Option C: Shared base + expert adapters (DEFAULT)
            t_base = self.time_base_transform(t)  # (B, S, hidden_dim)
            
            # Apply expert-specific adaptations and checkpointing
            expert_outputs = []
            for i, expert in enumerate(self.experts):
                if self.adapter_type == 'affine':
                    # Simple affine adaptation (scale + shift)
                    t_adapted = t_base * self.expert_scales[i] + self.expert_shifts[i]
                else:  # adapter_type == 'linear'
                    # Linear adapter
                    t_adapted = self.expert_adapters[i](t_base)
                
                # Apply checkpointing to expert computations (skip for Fourier expert - index 1)
                expert_output = self._checkpoint_expert(expert, t_adapted, skip_checkpointing=(i == 1))
                expert_outputs.append(expert_output)
            
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

    def forward(self, t: torch.Tensor, return_weights: bool = False):
        """
        Forward pass with per-expert gradient checkpointing.
        
        Checkpointing strategy: Checkpoint individual expert forward passes only.
        """
        if return_weights:
            return self._forward_impl(t, return_weights=True)
        return self._forward_impl(t, return_weights=False)