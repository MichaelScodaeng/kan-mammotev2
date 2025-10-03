# Realistic SM-Kernel Test: Learn from Time Series Data
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import gpytorch

# --- Ensure project root is importable BEFORE importing project modules ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from models.time_encoders.sm_kernel import SMKernelLayer
except ImportError:
    print("Warning: Could not import SMKernelLayer. Please ensure sm_kernel is in models/time_encoders.")
    sys.exit(1)

def generate_realistic_time_series(n_points=100):
    """Generate realistic time series with known patterns."""
    t = torch.linspace(0, 10, n_points)
    
    # True underlying patterns
    trend = 0.5 * torch.exp(-t/4)                               # Long-term decay
    seasonal = 0.3 * torch.sin(2*np.pi*0.8*t)                   # 0.8 Hz cycle
    fast_osc = 0.15 * torch.sin(2*np.pi*2.1*t) * torch.exp(-t/6) # Damped high-freq
    noise = 0.1 * torch.randn(n_points)                         # Observation noise
    
    y = trend + seasonal + fast_osc + noise
    
    return t, y, {
        'trend': trend,
        'seasonal': seasonal, 
        'fast_osc': fast_osc,
        'noise': noise
    }

def test_sm_kernel_on_real_data():
    """Test SM kernel by fitting to realistic time series."""
    print("Testing SM-Kernel on Realistic Time Series Data")
    print("=" * 50)
    
    # Generate data
    t_train, y_train, components = generate_realistic_time_series(80)
    t_test = torch.linspace(0, 12, 100)  # Extrapolate slightly
    
    # Plot the data
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(t_train, y_train, 'ko', markersize=3, label='Observed Data')
    plt.plot(t_train, components['trend'], '--', label='True Trend')
    plt.plot(t_train, components['seasonal'], '--', label='True Seasonal')
    plt.plot(t_train, components['fast_osc'], '--', label='True Fast Osc')
    plt.title('Synthetic Time Series Data')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Initialize SM kernel using practical heuristics
    sm_model = SMKernelLayer(num_mixtures=4)
    
    # Use the practical initialization from the previous file
    from practical_sm_initialization import analyze_time_series_for_init, practical_sm_initialization
    
    stats = analyze_time_series_for_init(t_train.numpy(), y_train.numpy())
    practical_sm_initialization(sm_model, stats)
    
    print("Initial SM Parameters (from data analysis):")
    print(f"  Frequencies: {sm_model.kernel.mixture_means.data.squeeze().tolist()}")
    print(f"  Lengthscales: {torch.sqrt(sm_model.kernel.mixture_scales.data.squeeze()).tolist()}")
    print(f"  Weights: {sm_model.kernel.mixture_weights.data.squeeze().tolist()}")
    
    # Set up GP model for proper training
    class SMGaussianProcess(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = sm_model.kernel  # Use our SM kernel
            
        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
    # Set up likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SMGaussianProcess(t_train, y_train, likelihood)
    
    # Training mode
    model.train()
    likelihood.train()
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    # Train the model
    print("\nTraining SM-Kernel GP...")
    for i in range(100):
        optimizer.zero_grad()
        output = model(t_train)
        loss = -mll(output, y_train)
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print(f"Iter {i:3d}: Loss = {loss.item():.3f}")
    
    # Prediction
    model.eval()
    likelihood.eval()
    
    with torch.no_grad():
        pred_dist = likelihood(model(t_test))
        pred_mean = pred_dist.mean
        pred_std = pred_dist.stddev
    
    print("\nFinal SM Parameters (after training):")
    print(f"  Frequencies: {model.covar_module.mixture_means.data.squeeze().tolist()}")
    print(f"  Lengthscales: {torch.sqrt(model.covar_module.mixture_scales.data.squeeze()).tolist()}")  
    print(f"  Weights: {model.covar_module.mixture_weights.data.squeeze().tolist()}")
    print(f"  Noise: {likelihood.noise.item():.3f}")
    
    # Plot results
    plt.subplot(2, 2, 2)
    plt.plot(t_train, y_train, 'ko', markersize=3, label='Training Data')
    plt.plot(t_test, pred_mean, 'b-', label='SM-Kernel Prediction')
    plt.fill_between(t_test, pred_mean - 2*pred_std, pred_mean + 2*pred_std, 
                     alpha=0.3, label='95% Confidence')
    plt.title('SM-Kernel GP Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Show learned kernel
    plt.subplot(2, 2, 3)
    tau_range = torch.linspace(0, 5, 100).unsqueeze(-1)
    with torch.no_grad():
        K_learned = []
        for tau in tau_range:
            # Evaluate the lazy tensor first, then extract scalar
            kernel_val = model.covar_module(torch.zeros(1,1), tau.unsqueeze(0)).evaluate()
            K_learned.append(kernel_val.item())
        K_learned = torch.tensor(K_learned)
    
    plt.plot(tau_range.squeeze(), K_learned, 'b-', linewidth=2, label='Learned K(τ)')
    plt.title('Learned Covariance Function')
    plt.xlabel('τ (Time Lag)')
    plt.ylabel('Covariance')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Component analysis
    plt.subplot(2, 2, 4)
    for i in range(4):
        with torch.no_grad():
            # Calculate individual component contributions
            weight = model.covar_module.mixture_weights[i]
            mean = model.covar_module.mixture_means[i, 0]
            scale = model.covar_module.mixture_scales[i, 0]
            
            component = weight * torch.exp(-2*(np.pi**2)*tau_range.squeeze()**2*scale) * \
                       torch.cos(2*np.pi*tau_range.squeeze()*mean)
            
            plt.plot(tau_range.squeeze(), component, '--', 
                    label=f'Component {i+1} (f={mean.item():.2f})')
    
    plt.title('SM Components')
    plt.xlabel('τ (Time Lag)')
    plt.ylabel('Component Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('realistic_sm_kernel_test.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return model, stats

if __name__ == "__main__":
    test_sm_kernel_on_real_data()