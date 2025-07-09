# kan_mamote/src/layers/basis_functions.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple, Literal

# Import configuration and spline matrix utility
from src.utils.config import KANMAMOTEConfig
from src.utils.spline_matrix_utils import compute_bspline_matrix

class BaseBasisFunction(nn.Module, ABC):
    """
    Abstract base class for all learnable basis functions in K-MOTE.
    Each basis function takes a (linearly transformed) time input and
    produces an embedding of `output_dim`.
    """
    def __init__(self, output_dim: int, config: KANMAMOTEConfig):
        super().__init__()
        self.output_dim = output_dim
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the basis function value for input x.
        Args:
            x: A tensor of shape (batch_size, self.output_dim).
               Each element x[b, d] is a linearly transformed timestamp
               that the basis function for output dimension 'd' operates on.
        Returns:
            A tensor of shape (batch_size, self.output_dim), representing
            the basis function's output.
        """
        pass

class FourierBasis(BaseBasisFunction):
    """
    Implements a learnable Fourier basis function.
    phi(x) = sum_{j=1}^{K'} (A_j * cos(omega_j * x + phi_j))
    where A_j, omega_j, phi_j are learnable parameters.
    """
    def __init__(self, output_dim: int, config: KANMAMOTEConfig):
        super().__init__(output_dim, config)
        self.num_harmonics = config.fourier_k_prime

        # Parameters are initialized as nn.Parameters if learnable, or registered as buffers if fixed.
        if config.fourier_learnable_params:
            # Frequencies (omega_j): Initialized to encourage learning from common patterns.
            # Shape: (output_dim, num_harmonics)
            self.frequencies = nn.Parameter(
                torch.randn(output_dim, self.num_harmonics) * 0.1 + # Small random noise
                (torch.arange(1, self.num_harmonics + 1).float() * torch.pi).unsqueeze(0).repeat(output_dim, 1) # Bias towards integer multiples of pi
            )
            # Amplitudes (A_j)
            self.amplitudes = nn.Parameter(torch.ones(output_dim, self.num_harmonics) * 0.5) # Initialized smaller
            # Phases (phi_j)
            self.phases = nn.Parameter(torch.zeros(output_dim, self.num_harmonics))
        else:
            # Fixed parameters for comparison/ablation
            self.register_buffer('frequencies', 
                                 (torch.arange(1, self.num_harmonics + 1).float() * torch.pi).unsqueeze(0).repeat(output_dim, 1))
            self.register_buffer('amplitudes', torch.ones(output_dim, self.num_harmonics) * 0.5)
            self.register_buffer('phases', torch.zeros(output_dim, self.num_harmonics))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, output_dim). Linearly transformed time input.
        Returns:
            Tensor of shape (batch_size, output_dim).
        """
        # x_expanded: (batch_size, output_dim, 1)
        x_expanded = x.unsqueeze(-1)
        
        # Parameters expanded to (1, output_dim, num_harmonics) for broadcasting
        freq_expanded = self.frequencies.unsqueeze(0)
        amp_expanded = self.amplitudes.unsqueeze(0)
        phase_expanded = self.phases.unsqueeze(0)

        # Compute A_j * cos(omega_j * x + phi_j)
        # Result shape: (batch_size, output_dim, num_harmonics)
        term = amp_expanded * torch.cos(freq_expanded * x_expanded + phase_expanded)
        
        # Sum over harmonics to get final output: (batch_size, output_dim)
        return term.sum(dim=-1)


class SplineBasis(BaseBasisFunction):
    """
    Implements a learnable B-spline basis function using MatrixKAN principles.
    This includes precomputing the Psi^k matrix and using matrix multiplications
    for the forward pass, handling knot intervals to some extent.
    """
    def __init__(self, output_dim: int, config: KANMAMOTEConfig):
        super().__init__(output_dim, config)
        self.grid_size = config.spline_grid_size
        self.spline_degree = config.spline_degree # Corresponds to 'degree' in KAN literature, 'k-1' for order 'k'
        self.use_matrix_kan_optimized_spline = config.use_matrix_kan_optimized_spline

        # Ensure degree is valid for MatrixKAN (order k >= 1, so degree >= 0)
        if self.spline_degree < 0:
            raise ValueError("Spline degree must be non-negative.")

        # Number of control points. For uniform splines, typically `num_intervals + degree`.
        # num_intervals = grid_size - 1. So, `(grid_size - 1) + spline_degree`.
        self.num_control_points = (self.grid_size - 1) + self.spline_degree + 1 # +1 is crucial for count
        self.control_points = nn.Parameter(torch.randn(output_dim, self.num_control_points) * 0.1) # Small init

        # Knot sequence. MatrixKAN typically uses uniform knots.
        # For B-spline of degree `d` defined on `N` intervals, `N + 2d` knots are needed.
        # If `grid_size` is number of distinct grid points (defining `grid_size-1` intervals)
        # then `(grid_size - 1) + 2 * spline_degree` knots.
        # Let's align with general B-spline theory for number of knots
        # Example: for cubic (degree 3), 4 control points are active per interval.
        self.register_buffer('knots', 
                             torch.linspace(0, 1, self.grid_size + self.spline_degree * 2).to(config.device))

        # Precompute the Psi^k (bspline_matrix)
        # `compute_bspline_matrix` takes `k_order`. If `spline_degree` is `d`, then `k_order = d + 1`.
        # So, for spline_degree `self.spline_degree`, we call `compute_bspline_matrix(self.spline_degree + 1)`.
        if self.use_matrix_kan_optimized_spline:
            try:
                psi_k = compute_bspline_matrix(self.spline_degree + 1)
                self.register_buffer('psi_k_matrix', psi_k.to(config.device))
                # print(f"SplineBasis: Precomputed MatrixKAN Psi^k matrix of shape {psi_k.shape}")
            except Exception as e:
                print(f"WARNING: Error computing Psi^k matrix: {e}. Falling back to non-MatrixKAN spline concept.")
                self.use_matrix_kan_optimized_spline = False
        
        if not self.use_matrix_kan_optimized_spline:
            # Fallback for plotting or simple differentiable spline (not MatrixKAN)
            # This is a highly simplified differentiable placeholder, not a true B-spline.
            # It allows the model to run and train for initial debugging.
            self.fallback_linear = nn.Linear(output_dim, output_dim)
            nn.init.zeros_(self.fallback_linear.weight)
            nn.init.zeros_(self.fallback_linear.bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, output_dim). Linearly transformed time input.
               Each element x[b, d] is a scalar time value.
        Returns:
            Tensor of shape (batch_size, output_dim).
        """
        batch_size, output_dim = x.shape
        device = x.device

        if not self.use_matrix_kan_optimized_spline:
            # Simplified fallback for B-spline for initial testing
            # This is NOT a B-spline, just a differentiable function of x scaled by control points.
            # Replace with a proper PyTorch B-spline if MatrixKAN is not used.
            x_normalized = (x - x.min()) / (x.max() - x.min() + 1e-6) # Normalize x to [0,1] for stability
            return self.fallback_linear(x_normalized) * self.control_points[:, :1] # Use first control point conceptually


        # --- MatrixKAN Forward Pass ---
        # 1. Normalize input `x` to `u` in [0, 1] based on its position within the B-spline domain.
        # The B-spline is effectively defined over `knots[degree]` to `knots[-degree-1]`.
        # Clamp `x` to this active domain to avoid out-of-bounds issues.
        knots = getattr(self, 'knots').to(x.device)  # Ensure knots are on same device as input
        x_clamped = torch.clamp(x, knots[self.spline_degree], knots[-self.spline_degree - 1])
        
        # Calculate `u` (normalized position within the active spline domain [0,1])
        domain_min = knots[self.spline_degree]
        domain_max = knots[-self.spline_degree - 1]
        u_input = (x_clamped - domain_min) / (domain_max - domain_min + 1e-6) # shape: (batch_size, output_dim)
        u_input = torch.clamp(u_input, 0, 1) # Ensure u is strictly within [0,1]

        # 2. Construct the power basis tensor: [1, u, u^2, ..., u^(spline_degree)]
        # This is (batch_size, output_dim, spline_degree + 1)
        power_basis = torch.stack([u_input**i for i in range(self.spline_degree + 1)], dim=-1)
        
        # 3. Compute B-spline basis function values for `u`
        # (batch_size, output_dim, spline_degree + 1) @ (spline_degree + 1, spline_degree + 1)
        # -> (batch_size, output_dim, spline_degree + 1)
        # These are `B_j(u)` values, where j runs over the relevant local basis functions.
        psi_k_matrix = self.psi_k_matrix
        if isinstance(psi_k_matrix, nn.Module):
            psi_k_matrix = next(psi_k_matrix.parameters())
        psi_k_matrix = psi_k_matrix.to(x.device)  # Ensure matrix is on same device as input
        bspline_basis_vals = torch.matmul(power_basis, psi_k_matrix)

        # 4. Combine with control points. This is `sum_{j} C_j * B_j(x)`
        # This is the most intricate part in PyTorch for efficiency, requiring precise gathering
        # of the active `spline_degree + 1` control points for each `x` based on its knot interval.

        # Find the segment index for each `x` in `x_clamped`.
        # `searchsorted` finds the insertion point, effectively the knot index (j) such that knots[j-1] <= x < knots[j].
        # We need to reshape `x_clamped` to (batch_size * output_dim) and `knots` to (num_knots) for `searchsorted`.
        # The result `segment_indices` will be for the flattened array.
        # Ensure self.knots is a Tensor (not a Module)
        knots_tensor = self.knots
        if isinstance(knots_tensor, nn.Module):
            knots_tensor = next(knots_tensor.parameters())
        knots_tensor = knots_tensor.to(x.device)  # Ensure knots are on same device as input
        segment_indices_flat = torch.searchsorted(knots_tensor, x_clamped.flatten()) # shape: (batch_size * output_dim)
        
        # Adjust indices to be relative to the *active* knot range for control point lookup.
        # For a uniform B-spline, the j-th interval corresponds to control points C_j to C_{j+degree}.
        # The `searchsorted` index `j` usually corresponds to `knots[j]`.
        # We need to subtract the initial `spline_degree` offset from the knot sequence.
        # This `j` should be `0` for the first active interval.
        segment_indices_adjusted = segment_indices_flat - self.spline_degree # `j` in C_j

        # Clamp these adjusted indices to ensure they are within valid range of control points.
        # The control points are indexed from `0` to `self.num_control_points - 1`.
        segment_indices_adjusted = torch.clamp(segment_indices_adjusted, 0, self.num_control_points - self.spline_degree - 1)
        
        # Create offsets for the (spline_degree + 1) active control points per segment
        offsets = torch.arange(self.spline_degree + 1, device=x.device).unsqueeze(0).unsqueeze(0) # (1, 1, spline_degree + 1)
        
        # Expand `segment_indices_adjusted` to match output_dim and offsets for gathering
        # (batch_size * output_dim, 1, 1) + (1, 1, spline_degree + 1) -> (batch_size * output_dim, 1, spline_degree + 1)
        indices_to_gather_flat = segment_indices_adjusted.unsqueeze(-1) + offsets
        
        # Ensure indices are within control points bounds
        indices_to_gather_flat = torch.clamp(indices_to_gather_flat, 0, self.num_control_points - 1)

        # Expand control_points for gathering
        # `control_points` shape: (output_dim, num_control_points)
        # We need to gather independently for each `output_dim`.
        # This needs to be done iteratively per output_dim or with a more complex `gather`.
        
        # More efficient gathering:
        # Create `output_dim_indices` (0 to output_dim-1) for matching
        output_dim_indices = torch.arange(output_dim, device=device).view(1, output_dim, 1).repeat(batch_size, 1, self.spline_degree + 1)

        # Reshape segment_indices_adjusted to (batch_size, output_dim, 1)
        segment_indices_adjusted_reshaped = segment_indices_adjusted.view(batch_size, output_dim, 1)

        # Full gather indices (batch_size, output_dim, spline_degree + 1, 2)
        # (batch_idx, output_dim_idx, relative_offset_idx) -> (control_points[output_dim_idx, control_point_idx])
        full_indices_to_gather = segment_indices_adjusted_reshaped + offsets
        full_indices_to_gather = torch.clamp(full_indices_to_gather, 0, self.num_control_points - 1)

        # The gather operation must be for each output_dim separately.
        # `active_control_points` will be (batch_size, output_dim, spline_degree + 1)
        
        # A more straightforward gather for multiple dimensions:
        active_control_points = self.control_points[
            output_dim_indices, full_indices_to_gather
        ]

        # Final weighted sum: `sum (B_j(u) * C_j)`
        # (batch_size, output_dim, spline_degree + 1) * (batch_size, output_dim, spline_degree + 1)
        # -> sum over last dim to get (batch_size, output_dim)
        output = (bspline_basis_vals * active_control_points).sum(dim=-1)
        
        return output


class GaussianKernelBasis(BaseBasisFunction):
    """
    Implements a Gaussian Mixture Kernel basis function.
    The parameters for weights, means, and standard deviations are directly learnable.
    The "tiny KAN" aspect described in the proposal ("generated by a tiny KAN")
    would be an additional layer of modulation *on top of these parameters*
    based on global context, which can be added later.
    For now, these are the core learnable parameters as per the base RKHS definition.
    """
    def __init__(self, output_dim: int, config: KANMAMOTEConfig):
        super().__init__(output_dim, config)
        self.num_mixture_components = config.rkhs_num_mixture_components
        
        # Learnable parameters for each Gaussian component (per output dimension)
        # Shape: (output_dim, num_mixture_components)
        self.raw_weights = nn.Parameter(torch.randn(output_dim, self.num_mixture_components) * 0.1)
        self.means = nn.Parameter(torch.randn(output_dim, self.num_mixture_components) * 2.0) # Means could be over broader range
        self.raw_stds = nn.Parameter(torch.randn(output_dim, self.num_mixture_components) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, output_dim). Linearly transformed time input.
               Each x[b, d] is a scalar time value for output dimension d.
        Returns:
            Tensor of shape (batch_size, output_dim).
        """
        # Ensure weights and stds are positive using softplus for numerical stability
        weights = F.softplus(self.raw_weights.unsqueeze(0)) # (1, output_dim, num_mixture_components)
        means = self.means.unsqueeze(0)                     # (1, output_dim, num_mixture_components)
        stds = F.softplus(self.raw_stds.unsqueeze(0)) + 1e-6 # (1, output_dim, num_mixture_components); Add epsilon to avoid zero std

        x_expanded = x.unsqueeze(-1) # (batch_size, output_dim, 1)

        # Compute the Gaussian terms: exp(-0.5 * ((x - mean)^2 / std^2))
        # Result shape: (batch_size, output_dim, num_mixture_components)
        gaussian_terms = torch.exp(-0.5 * ((x_expanded - means) / stds)**2)
        
        # Weighted sum over mixture components to get final output: (batch_size, output_dim)
        return (weights * gaussian_terms).sum(dim=-1)


class WaveletBasis(BaseBasisFunction):
    """
    Implements a learnable Wavelet basis function.
    phi(x) = sum (W_j * mother_wavelet_func((x - T_j) / S_j))
    where W_j, T_j (translation), S_j (scale) are learnable parameters.
    """
    def __init__(self, output_dim: int, config: KANMAMOTEConfig):
        super().__init__(output_dim, config)
        self.num_wavelets = config.wavelet_num_wavelets
        self.mother_wavelet_type = config.wavelet_mother_type

        # Learnable parameters for each wavelet component
        # Shape: (output_dim, num_wavelets)
        if config.wavelet_learnable_params:
            self.weights = nn.Parameter(torch.randn(output_dim, self.num_wavelets) * 0.1)
            self.raw_scales = nn.Parameter(torch.randn(output_dim, self.num_wavelets) * 0.1) # Will apply softplus/exp
            self.translations = nn.Parameter(torch.randn(output_dim, self.num_wavelets) * 0.1)
        else:
            self.register_buffer('weights', torch.ones(output_dim, self.num_wavelets) * 0.1)
            self.register_buffer('raw_scales', torch.ones(output_dim, self.num_wavelets) * 0.1)
            self.register_buffer('translations', torch.zeros(output_dim, self.num_wavelets))

        # Define the mother wavelet function
        if self.mother_wavelet_type == 'mexican_hat':
            # Mexican Hat wavelet (2nd derivative of Gaussian) is simple and differentiable
            # psi(t) = (1 - t^2) * exp(-0.5 * t^2)
            self._mother_wavelet_func = lambda t: (1 - t**2) * torch.exp(-0.5 * t**2)
        elif self.mother_wavelet_type == 'morlet':
            # Morlet wavelet: exp(-t^2/2) * cos(omega_0 * t)
            # A common choice for omega_0 is ~5-6 for good time-frequency localization
            self._mother_wavelet_func = lambda t: torch.exp(-0.5 * t**2) * torch.cos(5.0 * t)
        else:
            raise NotImplementedError(f"Wavelet type {self.mother_wavelet_type} not implemented for direct computation.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, output_dim). Linearly transformed time input.
        Returns:
            Tensor of shape (batch_size, output_dim).
        """
        # Ensure scales are positive for numerical stability
        scales = F.softplus(self.raw_scales.unsqueeze(0)) + 1e-6 # (1, output_dim, num_wavelets)
        
        x_expanded = x.unsqueeze(-1) # (batch_size, output_dim, 1)

        # Apply transformation: (x - translation) / scale
        # (batch_size, output_dim, 1) - (1, output_dim, num_wavelets) -> (batch_size, output_dim, num_wavelets)
        transformed_x = (x_expanded - self.translations.unsqueeze(0)) / scales

        # Apply mother wavelet function
        wavelet_terms = self._mother_wavelet_func(transformed_x) # (batch_size, output_dim, num_wavelets)

        # Weighted sum over wavelets
        # (1, output_dim, num_wavelets) * (batch_size, output_dim, num_wavelets) -> sum over last dim
        return (self.weights.unsqueeze(0) * wavelet_terms).sum(dim=-1)