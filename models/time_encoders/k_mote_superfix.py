# file: models/time_encoders/k_mote.py (Further Memory-Optimized Version)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Expert 1: Spline/RBF KAN Layer (Memory-Optimized) ---

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
        self.input_dim = input_dim

        # --- Global MLP branch for global trend ---
        hidden_dim = 32
        self.global_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        nn.init.xavier_uniform_(self.global_branch[-1].weight, gain=0.1)
        nn.init.zeros_(self.global_branch[-1].bias)

        if self.basis_function == 'b_spline':
            self.num_knots = self.grid_size + 3
            self.knots = nn.Parameter(torch.linspace(0, 1, self.num_knots))
            self.spline_coeffs = nn.Parameter(torch.randn(self.output_dim, self.num_knots) * 0.1)
            
        else:
            # RBF parameters - MEMORY EFFICIENT structure
            internal_range = [-2, 2]
            centers_init = torch.linspace(internal_range[0], internal_range[1], grid_size)
            self.centers = nn.Parameter(centers_init.unsqueeze(0).repeat(input_dim, 1))
            self.gammas = nn.Parameter(torch.ones(input_dim, grid_size))
            self.local_linear = nn.Linear(input_dim * grid_size, output_dim)

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions following LeTE's approach.
        Uses sigmoid normalization and L1 distance like LeTE.
        OPTIMIZED: Process in chunks to reduce memory.
        """
        x_norm = torch.sigmoid(x)
        original_shape = x_norm.shape
        
        if x_norm.dim() > 2:
            x_flat = x_norm.reshape(-1, x_norm.shape[-1])
        else:
            x_flat = x_norm
        
        if x_flat.shape[-1] == 1:
            x_values = x_flat.squeeze(-1)
        else:
            x_values = x_flat.mean(dim=-1)
        
        x_values = x_values.unsqueeze(-1)
        knots = self.knots.unsqueeze(0)
        
        # Process in chunks if input is large
        chunk_size = 10000  # Process 10k elements at a time
        total_size = x_values.shape[0]
        
        if total_size > chunk_size:
            outputs = []
            for i in range(0, total_size, chunk_size):
                end_idx = min(i + chunk_size, total_size)
                x_chunk = x_values[i:end_idx]
                
                distances = torch.abs(x_chunk - knots)
                basis = torch.exp(-distances * 5.0)
                output_chunk = torch.matmul(basis, self.spline_coeffs.T)
                outputs.append(output_chunk)
            
            spline_output = torch.cat(outputs, dim=0)
        else:
            distances = torch.abs(x_values - knots)
            basis = torch.exp(-distances * 5.0)
            spline_output = torch.matmul(basis, self.spline_coeffs.T)
        
        if len(original_shape) == 3:
            spline_output = spline_output.view(original_shape[0], original_shape[1], self.output_dim)
        elif len(original_shape) == 2:
            spline_output = spline_output.view(original_shape[0], self.output_dim)
        else:
            spline_output = spline_output.view(original_shape[0], self.output_dim)
        
        return spline_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass that combines the global MLP branch with the local spline/RBF branch.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch_size, seq_len, input_dim = x.shape
        
        global_output = self.global_branch(x)
        
        if self.basis_function == 'b_spline':
            local_output = self.b_splines(x)
            
        else:
            # Memory-efficient RBF implementation with chunking
            x_flat = x.reshape(-1, input_dim)
            chunk_size = 10000
            total_size = x_flat.shape[0]
            
            if total_size > chunk_size:
                outputs = []
                for i in range(0, total_size, chunk_size):
                    end_idx = min(i + chunk_size, total_size)
                    x_chunk = x_flat[i:end_idx]
                    
                    x_expanded = x_chunk.unsqueeze(-1)
                    centers = self.centers.unsqueeze(0)
                    gammas = self.gammas.unsqueeze(0)
                    
                    dist_sq = (x_expanded - centers).pow(2)
                    rbf_out = torch.exp(-F.softplus(gammas) * dist_sq)
                    rbf_flat = rbf_out.view(-1, input_dim * self.grid_size)
                    output_chunk = self.local_linear(rbf_flat)
                    outputs.append(output_chunk)
                
                local_output_flat = torch.cat(outputs, dim=0)
            else:
                x_expanded = x_flat.unsqueeze(-1)
                centers = self.centers.unsqueeze(0)
                gammas = self.gammas.unsqueeze(0)
                
                dist_sq = (x_expanded - centers).pow(2)
                rbf_out = torch.exp(-F.softplus(gammas) * dist_sq)
                rbf_flat = rbf_out.view(-1, input_dim * self.grid_size)
                local_output_flat = self.local_linear(rbf_flat)
            
            local_output = local_output_flat.view(batch_size, seq_len, self.output_dim)
        
        output = global_output + local_output
            
        return output


# --- Expert 2: Fourier KAN Layer (Further Optimized) ---

class FourierKANLayer(nn.Module):
    """Ultra memory-efficient Fourier expert using per-feature processing."""
    def __init__(self, input_dim: int, output_dim: int, n_harmonics: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_harmonics = n_harmonics
        
        self.cos_coeffs = nn.Parameter(torch.randn(input_dim, n_harmonics, output_dim) * 0.1)
        self.sin_coeffs = nn.Parameter(torch.randn(input_dim, n_harmonics, output_dim) * 0.1)
        self.frequencies = nn.Parameter(torch.randn(input_dim, n_harmonics) * 0.1)
        self.bias = nn.Parameter(torch.zeros(output_dim))
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        batch_size, seq_len, input_dim = t.shape
        total_size = batch_size * seq_len
        t_flat = t.reshape(-1, input_dim)
        
        # Process in chunks for very large inputs
        chunk_size = 10000
        
        if total_size > chunk_size:
            outputs = []
            for i in range(0, total_size, chunk_size):
                end_idx = min(i + chunk_size, total_size)
                t_chunk = t_flat[i:end_idx]
                
                cos_output = torch.zeros(t_chunk.shape[0], self.output_dim, 
                                        device=t.device, dtype=t.dtype)
                sin_output = torch.zeros(t_chunk.shape[0], self.output_dim, 
                                        device=t.device, dtype=t.dtype)
                
                for j in range(input_dim):
                    t_j = t_chunk[:, j]
                    freq_j = self.frequencies[j]
                    
                    args_j = t_j.unsqueeze(-1) * freq_j.unsqueeze(0)
                    cos_vals_j = torch.cos(args_j)
                    sin_vals_j = torch.sin(args_j)
                    
                    cos_output += torch.matmul(cos_vals_j, self.cos_coeffs[j])
                    sin_output += torch.matmul(sin_vals_j, self.sin_coeffs[j])
                
                chunk_output = cos_output + sin_output + self.bias
                outputs.append(chunk_output)
            
            output = torch.cat(outputs, dim=0)
        else:
            cos_output = torch.zeros(total_size, self.output_dim, 
                                    device=t.device, dtype=t.dtype)
            sin_output = torch.zeros(total_size, self.output_dim, 
                                    device=t.device, dtype=t.dtype)
            
            for i in range(input_dim):
                t_i = t_flat[:, i]
                freq_i = self.frequencies[i]
                
                args_i = t_i.unsqueeze(-1) * freq_i.unsqueeze(0)
                cos_vals_i = torch.cos(args_i)
                sin_vals_i = torch.sin(args_i)
                
                cos_output += torch.matmul(cos_vals_i, self.cos_coeffs[i])
                sin_output += torch.matmul(sin_vals_i, self.sin_coeffs[i])
            
            output = cos_output + sin_output + self.bias
        
        output = output.view(batch_size, seq_len, self.output_dim)
        return output


# --- Expert 3: Enhanced Wavelet KAN Layer (HEAVILY OPTIMIZED) ---

class WaveletKANLayer(nn.Module):
    """
    Enhanced wavelet expert with multiple wavelet types optimized for abrupt changes.
    HEAVILY OPTIMIZED: Processes features and wavelets sequentially with chunking.
    """
    def __init__(self, input_dim: int, output_dim: int, n_wavelets: int = 16, wavelet_type: str = 'shock'):
        super().__init__()
        self.input_dim = input_dim
        self.n_wavelets = n_wavelets
        self.wavelet_type = wavelet_type
        self.output_dim = output_dim
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
        """
        HEAVILY OPTIMIZED: Process with chunking and gradient checkpointing support.
        """
        if t.dim() == 2: 
            t = t.unsqueeze(-1)
        
        batch_size, seq_len, input_dim = t.shape
        total_size = batch_size * seq_len
        
        # Flatten for processing
        t_flat = t.reshape(total_size, input_dim)
        
        # Process in chunks to handle large sequences
        chunk_size = 5000  # Smaller chunks for wavelets
        
        if total_size > chunk_size:
            output_chunks = []
            
            for chunk_start in range(0, total_size, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_size)
                t_chunk = t_flat[chunk_start:chunk_end]
                chunk_len = t_chunk.shape[0]
                
                # Pre-allocate for this chunk
                wavelet_features = torch.zeros(
                    chunk_len, input_dim * self.n_wavelets,
                    device=t.device, dtype=t.dtype
                )
                
                # Process each input dimension separately
                for i in range(input_dim):
                    t_i = t_chunk[:, i]
                    scales_i = F.softplus(self.scales[i]) + 0.1
                    shifts_i = self.shifts[i]
                    
                    # Process wavelets for this dimension
                    for j in range(self.n_wavelets):
                        wavelet_input_ij = (t_i - shifts_i[j]) / scales_i[j]
                        
                        # Compute wavelet activation
                        if self.wavelet_type == 'adaptive_morlet':
                            activation = self.adaptive_morlet_wavelet(
                                wavelet_input_ij, self.frequencies[i, j], self.sharpness[i, j]
                            )
                        elif self.wavelet_type == 'mexican_hat':
                            activation = self.mexican_hat_wavelet(wavelet_input_ij)
                        elif self.wavelet_type == 'haar':
                            activation = self.haar_wavelet(wavelet_input_ij)
                        elif self.wavelet_type == 'shock':
                            activation = self.shock_wavelet(
                                wavelet_input_ij, self.asymmetry[i, j], self.steepness[i, j]
                            )
                        else:
                            activation = self.morlet_wavelet(wavelet_input_ij)
                        
                        feature_idx = i * self.n_wavelets + j
                        wavelet_features[:, feature_idx] = activation
                
                # Project this chunk
                chunk_output = self.linear(wavelet_features)
                output_chunks.append(chunk_output)
            
            # Concatenate all chunks
            output = torch.cat(output_chunks, dim=0)
        else:
            # Small input - process normally
            wavelet_features = torch.zeros(
                total_size, input_dim * self.n_wavelets,
                device=t.device, dtype=t.dtype
            )
            
            for i in range(input_dim):
                t_i = t_flat[:, i]
                scales_i = F.softplus(self.scales[i]) + 0.1
                shifts_i = self.shifts[i]
                
                for j in range(self.n_wavelets):
                    wavelet_input_ij = (t_i - shifts_i[j]) / scales_i[j]
                    
                    if self.wavelet_type == 'adaptive_morlet':
                        activation = self.adaptive_morlet_wavelet(
                            wavelet_input_ij, self.frequencies[i, j], self.sharpness[i, j]
                        )
                    elif self.wavelet_type == 'mexican_hat':
                        activation = self.mexican_hat_wavelet(wavelet_input_ij)
                    elif self.wavelet_type == 'haar':
                        activation = self.haar_wavelet(wavelet_input_ij)
                    elif self.wavelet_type == 'shock':
                        activation = self.shock_wavelet(
                            wavelet_input_ij, self.asymmetry[i, j], self.steepness[i, j]
                        )
                    else:
                        activation = self.morlet_wavelet(wavelet_input_ij)
                    
                    feature_idx = i * self.n_wavelets + j
                    wavelet_features[:, feature_idx] = activation
            
            output = self.linear(wavelet_features)
        
        # Reshape back
        output = output.view(batch_size, seq_len, self.output_dim)
        return output


# --- The Main K-MOTE Module ---

class KMOTE(nn.Module):
    """
    Final, architecturally correct K-MOTE with memory optimizations.
    This version fully aligns with the LeTE/Time2Vec design by expanding the
    time dimension in the initial linear layer BEFORE it is passed to the experts.
    """
    def __init__(self, input_dim: int, output_dim: int,
                 wavelet_type: str = 'shock',
                 use_layernorm: bool = True,
                 use_scale: bool = True,
                 gating_temp: float = 1.0):
        super().__init__()
        
        if input_dim != 1:
            raise ValueError("K-MOTE requires input_dim=1 for the time input.")

        self.output_dim = output_dim
        self.temperature = gating_temp
        self.use_scale = use_scale

        self.time_linear_transform = nn.Linear(input_dim, output_dim)
        self._initialize_time_transform()

        if use_scale:
            self.scale = nn.Parameter(torch.ones(output_dim))

        self.experts = nn.ModuleList([
            #SplineKANLayer(output_dim, output_dim, basis_function='b_spline', grid_size=5),
            FourierKANLayer(output_dim, output_dim, n_harmonics=5),
            WaveletKANLayer(output_dim, output_dim, n_wavelets=5, wavelet_type=wavelet_type),
            SplineKANLayer(output_dim, output_dim, basis_function='rbf', grid_size=5)
        ])
        
        self.num_experts = len(self.experts)
        
        self.gating_network = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.GELU(),
            nn.Linear(64, self.num_experts)
        )
        
        if use_layernorm and output_dim > 1:
            self.layer_norm = nn.LayerNorm(output_dim)
            print(f"Initialized K-MOTE with LayerNorm and LeTE-style architecture (memory-optimized).")
        else:
            self.layer_norm = nn.Identity()

    def _initialize_time_transform(self):
        """
        Initializes the time transformation layer with a geometric progression of frequencies,
        following the LeTE / Transformer paper's methodology.
        """
        with torch.no_grad():
            frequencies = 1.0 / (10 ** torch.linspace(0, 9, self.output_dim, dtype=torch.float32))
            self.time_linear_transform.weight.copy_(frequencies.unsqueeze(1))
            self.time_linear_transform.bias.zero_()

    def forward(self, t: torch.Tensor, return_weights: bool = False) -> torch.Tensor:
        if t.dim() == 2:
            t = t.unsqueeze(-1)

        # Apply LeTE-style transformation
        t_transformed = self.time_linear_transform(t)
        
        # Compute gating weights
        gating_logits = self.gating_network(t_transformed)
        gating_weights = F.softmax(gating_logits / self.temperature, dim=-1)
        
        # Process experts sequentially to save memory during list comprehension
        # This avoids keeping all expert outputs in memory simultaneously
        batch_size, seq_len, _ = t_transformed.shape
        weighted_sum = torch.zeros(batch_size, seq_len, self.output_dim, 
                                   device=t.device, dtype=t.dtype)
        
        for expert_idx, expert in enumerate(self.experts):
            expert_output = expert(t_transformed)
            # Get weight for this expert and add weighted contribution
            expert_weight = gating_weights[:, :, expert_idx:expert_idx+1]
            weighted_sum += expert_weight * expert_output
            
            # Clear intermediate tensors
            del expert_output
        
        output_embedding = self.layer_norm(weighted_sum)
        
        if self.use_scale:
            output_embedding = output_embedding * self.scale
        
        if return_weights:
            return output_embedding, gating_weights
        return output_embedding