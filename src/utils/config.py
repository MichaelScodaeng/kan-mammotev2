from dataclasses import dataclass

@dataclass
class KANMAMOTEConfig:
    # Global Model
    D_time: int = 128  # Dimension of time embedding
    hidden_dim_mamba: int = 256 # Mamba's internal hidden dimension
    state_dim_mamba: int = 64  # Mamba's state dimension
    K_top: int = 2     # Number of experts to activate (Top-Ktop MoE)
    use_aux_features_router: bool = True # Router input: tk + aux_features

    # K-MOTE Router
    router_mlp_dims: list[int] = (64, 32) # Router MLP hidden dims

    # Expert 1: Advanced Fourier-KAN
    fourier_k_prime: int = 16 # Number of harmonics
    fourier_learnable_params: bool = True # If frequencies are learnable

    # Expert 2: Spline-KAN
    spline_grid_size: int = 10 # Number of grid points for B-splines
    spline_k: int = 3 # Degree of B-splines
    use_matrix_kan: bool = True # Use MatrixKAN-like parallelization

    # Expert 3: Parameterized RKHS/GaussianKernel KAN
    rkhs_num_anchor_points: int = 8
    rkhs_num_mixture_components: int = 5
    rkhs_tiny_kan_dims: list[int] = (32, 16) # Tiny KAN for kernel params

    # Expert 4: Wavelet-KAN
    wavelet_num_wavelets: int = 8 # Number of wavelets
    wavelet_mother_type: str = 'daubechies4' # e.g., Daubechies-4
    wavelet_learnable_params: bool = True # If scale/translation are learnable

    # Regularization
    lambda_sobolev_l2: float = 0.01
    lambda_total_variation_l1: float = 0.001
    lambda_moe_load_balancing: float = 0.01

    # Training
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    device: str = 'cuda' # or 'cpu'
    # ... (and other training, data, output configs)