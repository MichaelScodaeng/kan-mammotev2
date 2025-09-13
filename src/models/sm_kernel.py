import torch
import torch.nn as nn
import gpytorch
import math

class SMKernelLayer(nn.Module):
    """
    A learnable layer that uses the Spectral Mixture (SM) Kernel to encode
    relative time differences (delta_t).

    This module acts as a powerful feature extractor for temporal signals, capable
    of discovering and modeling complex periodic and non-periodic patterns.

    Args:
        num_mixtures (int): The number of Gaussian components in the mixture. This
                            is equivalent to the output dimension of the embedding.
        input_dim (int): The input dimension of the time signal. For delta_t, this is 1.
        use_layernorm (bool): Whether to apply LayerNorm to the output for stability.
    """
    def __init__(self, num_mixtures: int, input_dim: int = 1, use_layernorm: bool = True):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.input_dim = input_dim

        # We instantiate the SpectralMixtureKernel from GPyTorch.
        # This object acts as a convenient container for the kernel's learnable
        # parameters: mixture_weights, mixture_means, and mixture_scales.
        # GPyTorch handles the constraints (e.g., positivity) for these parameters.
        self.kernel = gpytorch.kernels.SpectralMixtureKernel(
            num_mixtures=num_mixtures, 
            ard_num_dims=input_dim
        )

        # Optional LayerNorm for stabilizing the output activations
        self.layer_norm = nn.LayerNorm(num_mixtures) if use_layernorm else nn.Identity()

        print(f"Initialized SMKernelLayer with {num_mixtures} mixtures (output dimension).")

    def initialize_from_data(self, delta_t_sample: torch.Tensor):
        """
        Initializes the kernel's parameters based on the frequency spectrum of sample data.
        This provides a much better starting point for training than random initialization.

        Args:
            delta_t_sample (torch.Tensor): A sample tensor of delta_t values from the
                                           training set. Shape: (batch_size, seq_len, 1).
        """
        print("Initializing SM-Kernel from data spectrum...")
        if not isinstance(delta_t_sample, torch.Tensor):
            raise TypeError("delta_t_sample must be a PyTorch tensor.")
        if delta_t_sample.dim() != 3 or delta_t_sample.shape[-1] != 1:
            raise ValueError("delta_t_sample must have shape (batch_size, seq_len, 1).")

        # Flatten the sample to 1D for FFT
        delta_t_flat = delta_t_sample.reshape(-1).cpu().numpy()

        # Compute the periodogram (power spectral density) using FFT
        # This tells us which frequencies are most prominent in the data.
        freqs = torch.fft.fftfreq(len(delta_t_flat))
        fft_vals = torch.fft.fft(torch.tensor(delta_t_flat, dtype=torch.float32))
        power_spectrum = torch.abs(fft_vals)**2

        # Find the top `num_mixtures` peaks in the power spectrum
        # These peaks correspond to the dominant frequencies (means)
        # We only look at the positive frequencies
        positive_freq_indices = freqs > 0
        positive_freqs = freqs[positive_freq_indices]
        positive_power = power_spectrum[positive_freq_indices]
        
        # Find the indices of the top-k highest power values
        num_peaks = min(self.num_mixtures, len(positive_power))
        peak_indices = torch.topk(positive_power, k=num_peaks).indices
        
        # Get the frequencies corresponding to these peaks
        top_freqs = positive_freqs[peak_indices]

        # GPyTorch stores means and scales in a constrained space (raw parameters).
        # We need to set these raw parameters. We will initialize means based on top_freqs
        # and scales with a reasonable default.
        with torch.no_grad():
            # Initialize means based on FFT peaks
            self.kernel.raw_mixture_means.zero_()
            self.kernel.raw_mixture_means[:num_peaks, 0] = top_freqs
            
            # Initialize scales to a value that corresponds to a wide bandwidth,
            # allowing the model to adjust later. The value is heuristic.
            # A smaller raw value corresponds to a wider bandwidth.
            self.kernel.raw_mixture_scales.fill_(-1.0) 
            
            # Initialize weights uniformly
            self.kernel.raw_mixture_weights.fill_(1.0 / self.num_mixtures)

        print(f"SM-Kernel initialized with top frequencies: {top_freqs.tolist()}")


    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Computes the relative time embedding using the SM-Kernel formula.

        Instead of building a full kernel matrix, we efficiently evaluate the
        kernel function k(delta_t) for each input time gap, treating it as
        a learnable feature map.

        Args:
            delta_t (torch.Tensor): Input tensor of relative time differences.
                                    Shape: (batch_size, sequence_length, 1).

        Returns:
            torch.Tensor: The learned temporal embedding.
                          Shape: (batch_size, sequence_length, num_mixtures).
        """
        # Ensure input has a feature dimension
        if delta_t.dim() == 2:
            delta_t = delta_t.unsqueeze(-1)
        
        # GPyTorch stores parameters in a transformed space for constraints.
        # We access the "real" parameter values here. `softplus` ensures positivity.
        weights = self.kernel.mixture_weights
        means = self.kernel.mixture_means
        scales = self.kernel.mixture_scales

        # Reshape for broadcasting:
        # delta_t: (B, S, 1, 1)
        # weights: (1, 1, Q, 1)
        # means:   (1, 1, Q, D) -> (1, 1, Q, 1) for D=1
        # scales:  (1, 1, Q, D) -> (1, 1, Q, 1) for D=1
        delta_t = delta_t.unsqueeze(-2) # Add mixture dimension

        # --- Manual application of the SM-Kernel formula ---
        # k(τ) = Σ w_q * exp(-2π²τ²v_q) * cos(2πτμ_q)
        # We compute this for each mixture component `q` without summing.

        # 1. Exponential (RBF-like) part for decay/trend
        # The term inside exp is -(1/2) * ((τ - μ)/l)^2, where l is lengthscale.
        # In SM-Kernel, μ=0 and scale = 1/l^2. Here, scales are variance, so v_q = scale^2
        # GPyTorch formula uses `2 * pi^2 * delta_t^2 * scales`, which is variance v_q.
        dist_sq = delta_t.pow(2)
        exp_term = torch.exp(-2 * (math.pi**2) * dist_sq * scales)

        # 2. Cosine (Periodic) part
        cos_term = torch.cos(2 * math.pi * delta_t * means)

        # 3. Combine and weigh
        # The output for each dimension q is its respective component's value
        embedding = weights * exp_term * cos_term
        
        # Squeeze the last dimension which was the input_dim (1)
        embedding = embedding.squeeze(-1)
        
        # 4. Apply LayerNorm for stability
        embedding = self.layer_norm(embedding)

        return embedding