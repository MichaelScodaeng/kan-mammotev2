# kan_mamote/src/utils/config.py

from dataclasses import dataclass, field
import torch
from typing import List, Literal # For type hints

@dataclass
class KANMAMOTEConfig:
    """
    Configuration dataclass for the KAN-MAMOTE model and its components.
    Centralizes all hyperparameters for easy management and experimentation.
    """
    # --- Global Model Settings ---
    D_time: int = 128  # Total dimension of time embedding from K-MOTE (must be divisible by num_experts)
    hidden_dim_mamba: int = 256 # Mamba's internal hidden dimension
    state_dim_mamba: int = 64  # Mamba's state dimension (for Continuous-Time Mamba's SSM state)
    K_top: int = 2     # Number of experts to activate in K-MOTE (Top-Ktop MoE dispatch)
    num_experts: int = 4 # Fixed number of experts (Fourier, Spline, RKHS, Wavelet)
    use_aux_features_router: bool = True # If router input includes auxiliary features (from raw_event_features)
    raw_event_feature_dim: int = 16 # Dimensionality of raw event features (if used in concatenation)

    # --- K-MOTE Router Settings ---
    # Input to router is (timestamp + aux_features_dim). Output is num_experts logits.
    router_mlp_dims: List[int] = field(default_factory=lambda: [64, 32]) # Hidden dims for router MLP

    # --- KANLayer Settings (Shared by all experts, D_time_per_expert computed below) ---
    
    # --- Expert 1: Advanced Fourier-KAN ---
    fourier_k_prime: int = 16 # Number of harmonics for Fourier series
    fourier_learnable_params: bool = True # If frequencies, amplitudes, phases are learnable

    # --- Expert 2: Spline-KAN (Leveraging MatrixKAN principles) ---
    spline_grid_size: int = 10 # Number of grid points for B-splines (determines the knots distribution)
    spline_degree: int = 3 # Degree of B-splines (e.g., 3 for cubic splines). Corresponds to 'k-1' in some contexts.
    # When using MatrixKAN's `bspline_matrix(k_order)`, `k_order` corresponds to `spline_degree + 1`.
    use_matrix_kan_optimized_spline: bool = True # Flag to enable MatrixKAN optimizations for SplineBasis

    # --- Expert 3: Parameterized RKHS/GaussianKernel KAN ---
    rkhs_num_mixture_components: int = 5 # Number of Gaussian components per output dim
    # Note: The "tiny KAN" described in the proposal for RKHS expert is for *modulating*
    # parameters based on context, not generating them from scratch. For initial implementation,
    # RKHS parameters (weights, means, stds) are directly learnable.
    # The 'anchor_points' concept implicitly relates to the 'means' of the Gaussians.

    # --- Expert 4: Wavelet-KAN ---
    wavelet_num_wavelets: int = 8 # Number of wavelets per output dim
    wavelet_mother_type: Literal['mexican_hat', 'morlet'] = 'mexican_hat' # Type of mother wavelet function
    wavelet_learnable_params: bool = True # If scales, translations, weights are learnable

    # --- Regularization Parameters ---
    lambda_sobolev_l2: float = 0.01      # For smoothness of expert functions
    lambda_total_variation_l1: float = 0.001 # For sparsity in derivatives (sharp transitions)
    lambda_moe_load_balancing: float = 0.01 # To prevent expert collapse (encourages balanced usage)

    # --- Training Configuration ---
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # --- Data Configuration (placeholder for now, will be detailed in data/*) ---
    # Example: sequence_length for time series tasks
    max_sequence_length: int = 256 

    def __post_init__(self):
        """
        Post-initialization method to perform checks and calculate derived properties.
        """
        if self.D_time % self.num_experts != 0:
            raise ValueError(f"D_time ({self.D_time}) must be divisible by num_experts ({self.num_experts}) for even split.")
        self.D_time_per_expert = self.D_time // self.num_experts

        # Ensure that if using auxiliary features for the router, raw_event_feature_dim is defined.
        if self.use_aux_features_router and self.raw_event_feature_dim is None:
            raise ValueError("raw_event_feature_dim must be specified if use_aux_features_router is True.")

# Example usage (will be instantiated in train.py or main script)
# config = KANMAMOTEConfig()