# kan_mamote/src/utils/config.py

import torch

class KANMAMOTEConfig:
    """
    Configuration class for the KAN-MAMMOTE model and its components.
    """
    def __init__(self):
        # Global Model Parameters
        self.d_model = 128           # Main model dimension (hidden_dim for Mamba)
        self.D_time = 64             # Output dimension of K-MOTE (time embedding dimension)
        self.num_layers = 2          # Number of ContinuousMambaBlocks to stack
        self.input_feature_dim = 10  # Dimension of your raw input features (uk)
        self.output_dim_for_task = 1 # Dimension of the final prediction (e.g., 1 for regression)

        # K-MOTE Parameters
        self.K_top = 2               # Number of top experts to select in K-MOTE router
        self.use_aux_features_router = False # Whether router uses auxiliary features
        self.raw_event_feature_dim = 0 # Dummy if use_aux_features_router is False

        # Router Parameters (MoERouter)
        self.router_noise_scale = 1e-2 # Noise scale for router during training
        self.use_load_balancing = True # Enable load balancing loss for router
        self.balance_coefficient = 0.01 # Coefficient for load balancing loss

        # KANLayer (kan_base_layer.py) and Basis Function Parameters (basis_functions.py)
        # These apply to Fourier, Gaussian, Wavelet basis types which use KANLayer wrapper.
        self.kan_noise_scale = 0.1
        self.kan_scale_base_mu = 0.0
        self.kan_scale_base_sigma = 1.0
        self.kan_grid_eps = 0.02
        self.kan_grid_range = [-1, 1]
        self.kan_sp_trainable = True # Spline part trainable
        self.kan_sb_trainable = True # Base part trainable

        # --- MISSING ATTRIBUTES FOR BASIS FUNCTIONS (Added/Updated) ---
        # FourierBasis parameters
        self.fourier_k_prime = 10 # Number of harmonics for FourierBasis
        self.fourier_learnable_params = True # <--- ADDED THIS ONE

        # RKHS / GaussianKernelBasis parameters
        self.rkhs_num_mixture_components = 10 # Number of Gaussian components
        self.rkhs_learnable_params = True # <--- ADDED THIS ONE (for consistency, even if not explicitly demanded yet)

        # WaveletBasis parameters (already there, good)
        self.wavelet_num_wavelets = 10 # Number of wavelet components
        self.wavelet_mother_type = 'mexican_hat' # 'mexican_hat' or 'morlet'
        self.wavelet_learnable_params = True # Whether wavelet weights, scales, translations are learnable

        # SplineBasis (MatrixKANLayer) Specific Parameters (already there, good)
        self.spline_grid_size = 5    # Number of grid intervals (G)
        self.spline_degree = 3       # Spline order (k) - for MatrixKANLayer (k >= 0)

        # Mamba2 Parameters (dynamic_mamba_ssm.py) - passed as kwargs (already there, good)
        self.mamba_d_state = 128
        self.mamba_d_conv = 4
        self.mamba_expand = 2
        self.mamba_headdim = 32
        self.mamba_dt_min = 0.001
        self.mamba_dt_max = 0.1
        self.mamba_dt_init_floor = 1e-4
        self.mamba_bias = False
        self.mamba_conv_bias = True
        self.mamba_chunk_size = 256
        self.mamba_use_mem_eff_path = True
        self.mamba_layer_idx = None # Will be set per layer in KANMAMMOTE

        # Regularization Loss Coefficients (for KANMAMMOTE's total loss)
        self.lambda_sobolev_l2 = 0.01
        self.lambda_total_variation = 0.01

        # General Training Parameters
        self.learning_rate = 1e-3
        self.num_epochs = 50
        self.batch_size = 32
        self.sequence_length = 100 # For dummy data generation
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float32 # Use torch.bfloat16 for BFloat16 if supported