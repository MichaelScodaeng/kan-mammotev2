# Practical SM-Kernel Initialization Strategies
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq

# --- Ensure project root is importable ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def analyze_time_series_for_init(timestamps, values):
    """
    Analyze real time series data to initialize SM kernel parameters.
    This is what you'd do in practice with your actual data.
    """
    # 1. Frequency analysis using FFT (more robust)
    dt = np.mean(np.diff(timestamps))
    freqs = fftfreq(len(values), dt)
    fft_vals = np.abs(fft(values))
    
    # Find peaks in frequency spectrum
    peak_indices = signal.find_peaks(fft_vals[1:len(fft_vals)//2], 
                                   height=np.max(fft_vals)*0.1)[0] + 1
    dominant_freqs = freqs[peak_indices][:5]  # Top 5 frequencies
    
    if len(dominant_freqs) == 0:
        # Fallback: use autocorrelation
        autocorr = np.correlate(values, values, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        
        min_idx = np.argmin(autocorr[1:min(20, len(autocorr))]) + 1
        if min_idx < len(timestamps):
            period_est = timestamps[min_idx] * 2
            dominant_freqs = [1.0 / period_est if period_est > 0 else 0.5]
        else:
            dominant_freqs = [0.5]
    
    # 2. Estimate lengthscales from autocorrelation decay
    autocorr = np.correlate(values, values, mode='full')
    autocorr = autocorr[autocorr.size // 2:]
    autocorr = autocorr / autocorr[0]
    
    decay_idx = np.where(autocorr < 0.37)[0]  # 1/e decay
    if len(decay_idx) > 0 and decay_idx[0] < len(timestamps):
        lengthscale_est = timestamps[decay_idx[0]]
    else:
        lengthscale_est = (timestamps[-1] - timestamps[0]) / 3
    
    # 3. Data statistics
    data_variance = np.var(values)
    
    return {
        'dominant_frequencies': list(dominant_freqs),
        'lengthscale': lengthscale_est,
        'variance': data_variance,
        'mean_level': np.mean(values),
        'fft_spectrum': (freqs[:len(freqs)//2], fft_vals[:len(fft_vals)//2])
    }

def practical_sm_initialization(model, data_stats=None, num_mixtures=None):
    """
    Initialize SM kernel based on common heuristics used in practice.
    
    Strategy:
    1. One low-freq component for long-term trends/decay
    2. Several components covering different frequency bands
    3. Conservative lengthscales (not too short/long)
    4. Reasonable weight distribution
    """
    Q = num_mixtures or model.num_mixtures
    
    with torch.no_grad():
        if data_stats is not None:
            # Data-driven initialization
            freqs = data_stats['dominant_frequencies']
            base_lengthscale = data_stats['lengthscale']
            data_var = data_stats['variance']
            
            # Component 0: Long-term/aperiodic
            model.kernel.raw_mixture_means.data[0] = torch.tensor([0.01])  # Very low freq
            model.kernel.raw_mixture_scales.data[0] = torch.tensor([np.log(base_lengthscale**2)])
            model.kernel.raw_mixture_weights.data[0] = torch.tensor([np.log(data_var * 0.5)])
            
            # Remaining components: Cover detected frequencies
            for i in range(1, Q):
                if i-1 < len(freqs):
                    freq = abs(freqs[i-1])
                else:
                    # Fill with geometric spacing
                    freq = 0.1 * (2.0 ** i)
                
                scale = base_lengthscale / (2 ** i)  # Decreasing lengthscales
                weight = data_var / (2 ** (i+1))     # Decreasing weights
                
                model.kernel.raw_mixture_means.data[i] = torch.tensor([freq])
                model.kernel.raw_mixture_scales.data[i] = torch.tensor([np.log(scale**2)])
                model.kernel.raw_mixture_weights.data[i] = torch.tensor([np.log(weight)])
        
        else:
            # Generic heuristic initialization (no data available)
            print("Using generic heuristic initialization...")
            
            # Component 0: Broad, slow decay
            model.kernel.raw_mixture_means.data[0] = torch.tensor([0.01])
            model.kernel.raw_mixture_scales.data[0] = torch.tensor([-1.0])  # softplus(-1) ≈ 0.31
            model.kernel.raw_mixture_weights.data[0] = torch.tensor([0.0])   # softplus(0) = 0.69
            
            # Remaining: Geometric frequency spacing
            for i in range(1, Q):
                freq = 0.1 * (1.5 ** i)  # Geometric progression
                scale_raw = -2.0 - 0.5 * i  # Decreasing scales
                weight_raw = -0.5 * i       # Decreasing weights
                
                model.kernel.raw_mixture_means.data[i] = torch.tensor([freq])
                model.kernel.raw_mixture_scales.data[i] = torch.tensor([scale_raw])
                model.kernel.raw_mixture_weights.data[i] = torch.tensor([weight_raw])

def demo_practical_initialization():
    """Demo with synthetic 'observed' time series."""
    # Simulate some "observed" data
    t = np.linspace(0, 10, 200)
    
    # Synthetic data: trend + periodic + noise
    observed = (0.5 * np.exp(-t/3) +           # Decay
               0.3 * np.sin(2*np.pi*0.5*t) +   # 0.5 Hz oscillation  
               0.2 * np.sin(2*np.pi*1.2*t) +   # 1.2 Hz oscillation
               0.1 * np.random.randn(len(t)))   # Noise
    
    # Analyze the data
    stats = analyze_time_series_for_init(t, observed)
    print("Data Analysis Results:")
    print(f"  Dominant frequencies: {stats['dominant_frequencies']}")
    print(f"  Estimated lengthscale: {stats['lengthscale']:.3f}")
    print(f"  Data variance: {stats['variance']:.3f}")
    
    try:
        # Initialize SM kernel with these stats
        from models.time_encoders.sm_kernel import SMKernelLayer
        model = SMKernelLayer(num_mixtures=4)
        practical_sm_initialization(model, stats)
        
        print("\nInitialized SM Parameters:")
        print(f"  Means: {model.kernel.mixture_means.data.squeeze().tolist()}")
        print(f"  Scales: {model.kernel.mixture_scales.data.squeeze().tolist()}")
        print(f"  Weights: {model.kernel.mixture_weights.data.squeeze().tolist()}")
    except ImportError:
        print("\nCould not import SMKernelLayer for demo, but analysis completed successfully.")

if __name__ == "__main__":
    demo_practical_initialization()