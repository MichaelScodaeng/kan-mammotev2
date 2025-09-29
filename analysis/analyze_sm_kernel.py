import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Import your SMKernelLayer ---
try:
    from models.time_encoders.sm_kernel import SMKernelLayer
    print("✅ Successfully imported SMKernelLayer")
except ImportError as e:
    print(f"❌ Error: Could not import SMKernelLayer: {e}")
    print("Make sure 'sm_kernel.py' is in the correct path.")
    sys.exit(1)

# Create output directory for saving figures
os.makedirs('analysis_figures_sm_kernel', exist_ok=True)


# --- 1. Model Wrapper for the Reconstruction Task ---

class SMKernelReconstructor(nn.Module):
    """
    An Encoder-Decoder model to test the information capacity of the SMKernelLayer.
    - The Encoder is the SMKernelLayer.
    - The Decoder is a simple linear layer.
    """
    def __init__(self, num_mixtures: int = 32, input_dim: int = 1):
        super().__init__()
        self.encoder = SMKernelLayer(num_mixtures=num_mixtures, input_dim=input_dim)
        self.decoder = nn.Linear(num_mixtures, 1) # Projects embedding back to a scalar

    def forward(self, delta_t: torch.Tensor) -> torch.Tensor:
        """
        Takes a delta_t tensor (time differences), encodes it, and reconstructs it.
        
        Args:
            delta_t: Time differences, can be 1D, 2D, or 3D tensor
        Returns:
            reconstruction: Reconstructed signal
        """
        # Ensure proper shape for SMKernelLayer: (batch, seq_len, input_dim)
        original_shape = delta_t.shape
        
        if delta_t.dim() == 1:
            # Shape: [seq_len] -> [1, seq_len, 1]
            delta_t = delta_t.unsqueeze(0).unsqueeze(-1)
        elif delta_t.dim() == 2:
            # Shape: [batch, seq_len] -> [batch, seq_len, 1]
            delta_t = delta_t.unsqueeze(-1)
        # If already 3D, assume it's correct
        
        # Encode the time signal into a rich embedding
        embedding = self.encoder(delta_t)  # Output: (batch, seq_len, num_mixtures)
        
        # Decode the embedding back to the original signal's shape
        reconstruction = self.decoder(embedding)  # Output: (batch, seq_len, 1)
        
        # Restore original shape
        if len(original_shape) == 1:
            return reconstruction.squeeze()
        elif len(original_shape) == 2:
            return reconstruction.squeeze(-1)
        else:
            return reconstruction


# --- 2. Helper Functions for Training and Plotting ---

def convert_timestamps_to_delta_t(t_data):
    """
    Convert absolute timestamps to relative time differences (staleness).
    This is what SM-Kernel expects as input.
    
    Args:
        t_data: Absolute timestamps [seq_len]
    Returns:
        delta_t: Time differences from current time [seq_len]
    """
    # For analysis purposes, we'll use the last timestamp as "current time"
    current_time = t_data[-1]
    delta_t = current_time - t_data  # Staleness (how old each timestamp is)
    
    # Make sure delta_t is non-negative (staleness should be >= 0)
    delta_t = torch.abs(delta_t)
    
    return delta_t

def convert_timestamps_to_delta_t_uniform(t_data):
    """Create uniform staleness to test pure SM-Kernel capability without position bias"""
    uniform_staleness = torch.ones_like(t_data) * 1.0  # Constant staleness
    return uniform_staleness

def convert_timestamps_to_delta_t_centered(t_data):
    """Use center point as reference instead of last point"""
    current_time = t_data[len(t_data)//2]  # Middle point as "current"
    delta_t = torch.abs(current_time - t_data)
    return delta_t

def convert_timestamps_to_delta_t_normalized(t_data):
    """Normalize staleness to [0, max_staleness] to prevent exponential explosion"""
    current_time = t_data[-1]
    delta_t = torch.abs(current_time - t_data)
    # Normalize to [0, 3] to prevent exponential explosion in SM-Kernel
    max_allowed_staleness = 3.0
    delta_t_normalized = (delta_t / delta_t.max()) * max_allowed_staleness
    return delta_t_normalized

def convert_timestamps_to_delta_t_local_differences(t_data):
    """Use local time differences instead of global staleness"""
    # Create local time differences between consecutive points
    # Fix: prepend should be a tensor with same shape as first element
    local_diffs = torch.abs(torch.diff(t_data, prepend=t_data[0:1]))
    # Add small constant to avoid zero staleness
    local_diffs = local_diffs + 0.1
    return local_diffs


def prepare_data_for_sm_kernel(t_data):
    """
    Prepare data in the format expected by SM-Kernel initialization.
    
    Args:
        t_data: Absolute timestamps [seq_len]
    Returns:
        delta_t_formatted: Properly shaped for SM-Kernel [1, seq_len, 1]
    """
    delta_t = convert_timestamps_to_delta_t(t_data)
    
    # SM-Kernel expects shape: (batch_size, seq_len, input_dim)
    delta_t_formatted = delta_t.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
    
    return delta_t_formatted

def train_reconstructor(model, t_train, y_train, epochs=3000, lr=2e-4):
    """
    Trains the SMKernelReconstructor model on the training data.
    """
    print(f"--- Training on {len(t_train)} data points ---")
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    # Convert timestamps to delta_t format for SM-Kernel
    delta_t_train = prepare_data_for_sm_kernel(t_train)
    print(f"Prepared delta_t shape: {delta_t_train.shape}")
    
    # Initialize the kernel from the training data's spectrum
    try:
        model.encoder.initialize_from_data(delta_t_train)
        print("✅ SM-Kernel initialized from data spectrum")
    except Exception as e:
        print(f"⚠️  Warning: Could not initialize from data: {e}")
        print("Using default initialization")
    
    # Convert delta_t back to 1D for training
    delta_t_train_1d = delta_t_train.squeeze()
    
    best_loss = float('inf')
    patience = 0
    max_patience = epochs // 10
    
    for epoch in range(epochs):
        model.train()
        
        # Forward pass using delta_t instead of absolute time
        y_pred = model(delta_t_train_1d)
        
        # Ensure shapes match
        if y_pred.shape != y_train.shape:
            if y_pred.dim() > y_train.dim():
                y_pred = y_pred.squeeze()
            elif y_pred.dim() < y_train.dim():
                y_pred = y_pred.unsqueeze(-1)
        
        loss = loss_fn(y_pred, y_train)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"⚠️  Warning: NaN/Inf loss at epoch {epoch+1}, stopping training")
            break
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Early stopping
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
            
        if patience > max_patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 500 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.6f}")
            
    print(f"--- Training Complete, Final Loss: {best_loss:.6f} ---")
    return model

def plot_reconstruction(model, t_full, y_full, t_train, title, filename):
    """
    Plots the ground truth vs. reconstruction and shows the train/test split.
    """
    model.eval()
    with torch.no_grad():
        # Convert full timeline to delta_t for reconstruction
        delta_t_full = convert_timestamps_to_delta_t(t_full)
        reconstructed_full = model(delta_t_full)
        
        # Calculate test performance
        split_idx = len(t_train)
        t_test = t_full[split_idx:]
        y_test = y_full[split_idx:]
        
        if len(t_test) > 0:
            delta_t_test = convert_timestamps_to_delta_t(t_test)
            reconstructed_test = model(delta_t_test)
            test_loss = nn.MSELoss()(reconstructed_test, y_test)
        else:
            test_loss = torch.tensor(0.0)

    plt.figure(figsize=(14, 8))
    
    # Main plot
    plt.subplot(2, 1, 1)
    plt.plot(t_full.numpy(), y_full.numpy(), label='Ground Truth Signal', 
             color='black', linewidth=3, alpha=0.8)
    plt.plot(t_full.numpy(), reconstructed_full.numpy(), 
             label='SM-Kernel Reconstruction', color='red', linestyle='--', linewidth=2)
    
    if len(t_train) < len(t_full):
        plt.axvline(x=t_train[-1], color='gray', linestyle=':', linewidth=2, 
                   label=f'Train/Test Split')
    
    plt.title(f"{title}\nTest MSE (on unseen data): {test_loss.item():.5f}", fontsize=14)
    plt.xlabel("Absolute Time (t)")
    plt.ylabel("Signal Value y(t)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    # Delta_t visualization
    plt.subplot(2, 1, 2)
    delta_t_full = convert_timestamps_to_delta_t(t_full)
    plt.plot(t_full.numpy(), delta_t_full.numpy(), 
             label='Delta_t (Staleness)', color='blue', alpha=1.0)
    plt.title("Time Differences (Staleness) Used by SM-Kernel")
    plt.xlabel("Absolute Time (t)")
    plt.ylabel("Delta_t (Staleness)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()


# --- 3. Synthetic Data Generation ---

def generate_quasi_periodic_signal(t):
    """A damped sine wave - perfect for SM-Kernel's spectral mixture approach."""
    return torch.exp(-0.1 * torch.abs(t)) * torch.sin(2 * t + 0.5)

def generate_mixed_signal(t):
    """A mix of trend and quasi-periodic - tests SM-Kernel's multi-scale capability."""
    trend = 0.05 * t
    periodic = torch.exp(-0.1 * torch.abs(t)) * torch.sin(2 * t)
    noise = 0.1 * torch.randn_like(t)
    return trend + 0.8 * periodic + noise

def generate_abrupt_change_signal(t):
    """Step function - stress test for SM-Kernel."""
    signal = torch.zeros_like(t)
    signal[t > -1] = 0.5
    signal[t > 1] = -0.3
    signal[t > 3] = 0.8
    return signal

def generate_multi_scale_signal(t):
    """Multiple frequency components - ideal for spectral mixture kernel."""
    component1 = torch.sin(0.5 * t)  # Low frequency
    component2 = 0.5 * torch.sin(2 * t)  # Medium frequency  
    component3 = 0.3 * torch.sin(5 * t) * torch.exp(-0.05 * torch.abs(t))  # High freq, damped
    return component1 + component2 + component3


# --- 4. Main Analysis Functions ---

def run_reconstruction_analysis():
    """
    Main function for SM-Kernel Reconstruction Analysis.
    Tests how well SM-Kernel can encode and reconstruct different signal types.
    """
    print("\n" + "="*60)
    print("SM-KERNEL ANALYSIS: RECONSTRUCTION TASK")
    print("Testing expressiveness & generalization capability")
    print("="*60 + "\n")

    # Define signals that should showcase SM-Kernel's strengths
    signals = {
        "Quasi-Periodic (Damped Sine)": generate_quasi_periodic_signal,
        "Multi-Scale (Multiple Frequencies)": generate_multi_scale_signal, 
        "Mixed Signal (Trend + Periodic + Noise)": generate_mixed_signal,
        "Abrupt Changes (Stress Test)": generate_abrupt_change_signal,
    }

    # Create timeline
    t_full = torch.linspace(-5, 5, 500)
    split_index = int(len(t_full) * 1.0)  # 70% for training
    t_train = t_full[:split_index]

    trained_models = {}
    results_summary = {}

    for name, func in signals.items():
        print(f"\n{'='*50}")
        print(f"Testing on: {name}")
        print(f"{'='*50}")
        
        # Generate signal
        y_full = func(t_full)
        y_train = y_full[:split_index]
        
        print(f"Signal statistics:")
        print(f"  Full signal range: [{y_full.min():.3f}, {y_full.max():.3f}]")
        print(f"  Training points: {len(y_train)}")
        print(f"  Test points: {len(y_full) - len(y_train)}")
        
        # Create and train the model
        model = SMKernelReconstructor(num_mixtures=16)  # Reduced mixtures for stability
        trained_model = train_reconstructor(model, t_train, y_train, epochs=10000, lr=2e-4)
        trained_models[name] = trained_model
        
        # Evaluate performance
        model.eval()
        with torch.no_grad():
            delta_t_full = convert_timestamps_to_delta_t(t_full)
            y_reconstructed = model(delta_t_full)
            
            # Calculate losses
            train_loss = nn.MSELoss()(y_reconstructed[:len(y_train)], y_train)
            test_loss = nn.MSELoss()(y_reconstructed[len(y_train):], y_full[len(y_train):])
            
            results_summary[name] = {
                'train_loss': train_loss.item(),
                'test_loss': test_loss.item(),
                'generalization_gap': test_loss.item() - train_loss.item()
            }
        
        # Plot results
        safe_filename = name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '')
        plot_reconstruction(
            trained_model, 
            t_full, 
            y_full,
            t_train,
            title=f"SM-Kernel Reconstruction: {name}",
            filename=f"analysis_figures_sm_kernel/reconstruction_{safe_filename}.png"
        )
    
    # Print summary
    print(f"\n{'='*80}")
    print("RECONSTRUCTION RESULTS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Signal Type':<35} {'Train Loss':<12} {'Test Loss':<12} {'Gen. Gap':<12}")
    print("-" * 80)
    
    for name, metrics in results_summary.items():
        print(f"{name:<35} {metrics['train_loss']:<12.6f} {metrics['test_loss']:<12.6f} {metrics['generalization_gap']:<12.6f}")
    
    return trained_models

def run_bias_analysis():
    """
    Comprehensive analysis to test different staleness conversion methods
    and identify which one eliminates position-dependent bias.
    """
    print("\n" + "="*60)
    print("SM-KERNEL BIAS ANALYSIS")
    print("Testing different staleness conversion methods")
    print("="*60 + "\n")

    # Test signal: Simple sine wave (should be reconstructable everywhere)
    t_test = torch.linspace(-5, 5, 400)
    y_test = torch.sin(t_test)
    
    # Different staleness methods to test
    staleness_methods = {
        "Original (Last as Current)": convert_timestamps_to_delta_t,
        "Uniform Staleness": convert_timestamps_to_delta_t_uniform,
        "Centered Reference": convert_timestamps_to_delta_t_centered,
        "Normalized Staleness": convert_timestamps_to_delta_t_normalized,
        "Local Differences": convert_timestamps_to_delta_t_local_differences,
    }
    
    results = {}
    
    # Test each method
    for method_name, convert_func in staleness_methods.items():
        print(f"\n[INFO] Testing: {method_name}")
        
        # Convert timestamps using this method
        delta_t = convert_func(t_test)
        
        print(f"  Delta_t range: [{delta_t.min():.3f}, {delta_t.max():.3f}]")
        print(f"  Delta_t mean: {delta_t.mean():.3f}, std: {delta_t.std():.3f}")
        
        # Train SM-Kernel model
        model = SMKernelReconstructor(num_mixtures=8)  # Smaller for faster testing
        
        try:
            # Format data for SM-Kernel initialization
            delta_t_formatted = delta_t.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
            model.encoder.initialize_from_data(delta_t_formatted)
            print("  ✅ SM-Kernel initialized successfully")
        except Exception as e:
            print(f"  ⚠️  Warning: Initialization failed: {e}")
        
        # Train the model
        final_loss = train_reconstructor_simple(model, delta_t, y_test, epochs=1500, lr=2e-4)
        
        # Evaluate reconstruction quality across different regions
        model.eval()
        with torch.no_grad():
            y_pred = model(delta_t)
            
            # Split into regions to check for bias
            n_points = len(t_test)
            left_region = slice(0, n_points//3)
            middle_region = slice(n_points//3, 2*n_points//3)
            right_region = slice(2*n_points//3, n_points)
            
            # Calculate MSE for each region
            mse_left = F.mse_loss(y_pred[left_region], y_test[left_region]).item()
            mse_middle = F.mse_loss(y_pred[middle_region], y_test[middle_region]).item()
            mse_right = F.mse_loss(y_pred[right_region], y_test[right_region]).item()
            mse_overall = F.mse_loss(y_pred, y_test).item()
            
            # Calculate bias metric (variance in regional performance)
            regional_losses = [mse_left, mse_middle, mse_right]
            bias_metric = np.std(regional_losses) / np.mean(regional_losses)
            
            results[method_name] = {
                'overall_loss': mse_overall,
                'left_loss': mse_left,
                'middle_loss': mse_middle,
                'right_loss': mse_right,
                'bias_metric': bias_metric,
                'delta_t_range': (delta_t.min().item(), delta_t.max().item())
            }
            
        print(f"  📊 Results:")
        print(f"    Overall Loss: {mse_overall:.6f}")
        print(f"    Regional Loss - Left: {mse_left:.6f}, Middle: {mse_middle:.6f}, Right: {mse_right:.6f}")
        print(f"    Bias Metric: {bias_metric:.4f} (lower = less biased)")
        
        # Create visualization
        plt.figure(figsize=(15, 10))
        
        # Plot 1: Original signal and reconstruction
        plt.subplot(3, 1, 1)
        plt.plot(t_test.numpy(), y_test.numpy(), 'k-', linewidth=3, label='Ground Truth', alpha=0.8)
        plt.plot(t_test.numpy(), y_pred.numpy(), 'r--', linewidth=2, label='SM-Kernel Reconstruction')
        plt.axvline(x=t_test[n_points//3], color='gray', linestyle=':', alpha=0.7, label='Region Boundaries')
        plt.axvline(x=t_test[2*n_points//3], color='gray', linestyle=':', alpha=0.7)
        plt.title(f'{method_name}: Signal Reconstruction\nOverall MSE: {mse_overall:.6f}, Bias: {bias_metric:.4f}')
        plt.xlabel('Time (t)')
        plt.ylabel('Signal Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Delta_t visualization
        plt.subplot(3, 1, 2)
        plt.plot(t_test.numpy(), delta_t.numpy(), 'b-', linewidth=2, label='Delta_t (Staleness)')
        plt.title(f'Staleness Pattern: {method_name}')
        plt.xlabel('Time (t)')
        plt.ylabel('Delta_t (Staleness)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Regional error comparison
        plt.subplot(3, 1, 3)
        regions = ['Left', 'Middle', 'Right']
        losses = [mse_left, mse_middle, mse_right]
        colors = ['red', 'orange', 'green']
        bars = plt.bar(regions, losses, color=colors, alpha=0.7)
        plt.title('Regional Reconstruction Error')
        plt.ylabel('MSE Loss')
        plt.yscale('log')  # Log scale to better see differences
        
        # Add value labels on bars
        for bar, loss in zip(bars, losses):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.1, 
                    f'{loss:.4f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        safe_method_name = method_name.replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(f'analysis_figures_sm_kernel/bias_analysis_{safe_method_name}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("BIAS ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<25} {'Overall Loss':<12} {'Bias Metric':<12} {'Best Region':<12}")
    print("-" * 80)
    
    for method, metrics in results.items():
        regional_losses = [metrics['left_loss'], metrics['middle_loss'], metrics['right_loss']]
        best_region = ['Left', 'Middle', 'Right'][np.argmin(regional_losses)]
        
        print(f"{method:<25} {metrics['overall_loss']:<12.6f} {metrics['bias_metric']:<12.4f} {best_region:<12}")
    
    # Find best method (lowest bias)
    best_method = min(results.keys(), key=lambda k: results[k]['bias_metric'])
    print(f"\n🏆 LEAST BIASED METHOD: {best_method}")
    print(f"   Bias Metric: {results[best_method]['bias_metric']:.4f}")
    print(f"   Overall Loss: {results[best_method]['overall_loss']:.6f}")
    
    return results

def train_reconstructor_simple(model, delta_t, y_true, epochs=1500, lr=2e-4):
    """Simplified training function for bias analysis"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        y_pred = model(delta_t)
        
        # Ensure shapes match
        if y_pred.shape != y_true.shape:
            if y_pred.dim() > y_true.dim():
                y_pred = y_pred.squeeze()
        
        loss = loss_fn(y_pred, y_true)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    Warning: NaN/Inf loss at epoch {epoch+1}")
            break
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if (epoch + 1) % 500 == 0:
            print(f"    Epoch {epoch+1}: Loss {loss.item():.6f}")
    
    return loss.item()

def run_interpretability_analysis(trained_models):
    """
    Analyze the learned spectral mixture components.
    """
    print("\n" + "="*60)
    print("SM-KERNEL INTERPRETABILITY ANALYSIS")
    print("Visualizing learned spectral mixture components")
    print("="*60 + "\n")

    # Use the model trained on multi-scale signal (should have rich components)
    model_name = "Multi-Scale (Multiple Frequencies)"
    model = trained_models.get(model_name)
    
    if model is None:
        print(f"⚠️  Model '{model_name}' not found, using first available model")
        model = list(trained_models.values())[0]
        model_name = list(trained_models.keys())[0]

    print(f"Analyzing model trained on: {model_name}")
    
    # Extract learned parameters from the SM-Kernel
    kernel = model.encoder.kernel
    
    try:
        weights = kernel.mixture_weights.squeeze().detach()
        means = kernel.mixture_means.squeeze().detach() 
        scales = kernel.mixture_scales.squeeze().detach()
        
        print(f"Learned parameters:")
        print(f"  Number of mixtures: {len(weights)}")
        print(f"  Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
        print(f"  Frequency range: [{means.min():.4f}, {means.max():.4f}]")
        print(f"  Scale range: [{scales.min():.4f}, {scales.max():.4f}]")
        
    except Exception as e:
        print(f"❌ Error extracting parameters: {e}")
        return
    
    # Visualize components over a range of delta_t values
    delta_t_range = torch.linspace(0, 8, 800).unsqueeze(-1)  # [800, 1]
    
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Individual mixture components
    plt.subplot(3, 1, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(weights)))
    
    all_components = []
    for i in range(len(weights)):
        w, m, s = weights[i], means[i], scales[i]
        
        # SM-Kernel formula: w * exp(-2π²Δt²s) * cos(2πΔtm)
        dist_sq = delta_t_range.pow(2).squeeze()
        exp_term = torch.exp(-2 * (np.pi**2) * dist_sq * s)
        cos_term = torch.cos(2 * np.pi * delta_t_range.squeeze() * m)
        component = w * exp_term * cos_term
        
        all_components.append(component)
        
        # Only plot significant components for clarity
        if torch.abs(w) > 0.1:
            plt.plot(delta_t_range.squeeze().numpy(), component.numpy(), 
                    color=colors[i], alpha=1.0, 
                    label=f'Mix {i}: w={w:.3f}, f={m:.2f}, s={s:.3f}')
    
    plt.title(f"Individual Spectral Mixture Components\n(Model: {model_name})", fontsize=14)
    plt.xlabel("Time Difference (Δt)")
    plt.ylabel("Component Value")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.5)
    
    # Plot 2: Sum of all components (the learned kernel)
    plt.subplot(3, 1, 2)
    all_components_tensor = torch.stack(all_components, dim=0)
    kernel_sum = all_components_tensor.sum(dim=0)
    
    plt.plot(delta_t_range.squeeze().numpy(), kernel_sum.numpy(), 
             color='black', linewidth=3)
    plt.title("Learned Spectral Mixture Kernel k(Δt)", fontsize=14)
    plt.xlabel("Time Difference (Δt)")
    plt.ylabel("Kernel Value")
    plt.grid(True, alpha=0.5)
    
    # Plot 3: Parameter distribution
    plt.subplot(3, 1, 3)
    x_pos = np.arange(len(weights))
    width = 0.25
    
    plt.bar(x_pos - width, weights.numpy(), width, label='Weights', alpha=1.0)
    plt.bar(x_pos, means.numpy() / means.max().item(), width, label='Frequencies (normalized)', alpha=1.0)
    plt.bar(x_pos + width, scales.numpy() / scales.max().item(), width, label='Scales (normalized)', alpha=1.0)
    
    plt.title("Learned Parameter Distribution", fontsize=14)
    plt.xlabel("Mixture Component Index")
    plt.ylabel("Parameter Value (normalized)")
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig("analysis_figures_sm_kernel/interpretability_components.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

    # Print top components
    print(f"\nTop 5 mixture components by weight:")
    top_indices = torch.argsort(torch.abs(weights), descending=True)[:5]
    for rank, idx in enumerate(top_indices):
        print(f"  {rank+1}. Component {idx}: weight={weights[idx]:.4f}, "
              f"freq={means[idx]:.3f}, scale={scales[idx]:.4f}")


# --- Main Execution ---
if __name__ == '__main__':
    print("🚀 SM-Kernel Comprehensive Analysis")
    print("="*60)
    
    try:
        # Test SM-Kernel instantiation first
        test_kernel = SMKernelLayer(num_mixtures=4, input_dim=1)
        print("✅ SM-Kernel successfully instantiated")
        
        # Run analyses
        trained_models = run_reconstruction_analysis()
        run_interpretability_analysis(trained_models)
        
        print("\n✨ SM-Kernel analysis complete!")
        print("📁 Figures saved in 'analysis_figures_sm_kernel' directory.")
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Enhanced main function with bias analysis"""
    print("🔍 SM-KERNEL COMPREHENSIVE ANALYSIS")
    print("=" * 50)
    
    # Critical: Bias analysis first
    print("\n🚨 BIAS ANALYSIS (Testing staleness conversion methods)")
    print("-" * 50)
    run_bias_analysis()
    
    # Original analysis
    print("\n📊 STANDARD RECONSTRUCTION ANALYSIS")  
    print("-" * 50)
    try:
        # Test basic instantiation
        test_kernel = SMKernelLayer(num_mixtures=4, input_dim=1)
        print("✅ SM-Kernel successfully instantiated")
        
        # Run analyses
        trained_models = run_reconstruction_analysis()
        run_interpretability_analysis(trained_models)
        
        print("\n✨ SM-Kernel analysis complete!")
        print("📁 Figures saved in 'analysis_figures_sm_kernel' directory.")
        
    except Exception as e:
        print(f"❌ Analysis failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()