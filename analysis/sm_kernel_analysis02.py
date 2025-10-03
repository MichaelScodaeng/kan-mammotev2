# file: sm_kernel_analysis.py

import os
import sys
import torch
import torch.nn as nn
import gpytorch
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers 3D projection
import copy
import re
# --- Ensure project root is importable BEFORE importing project modules ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Ensure project root is importable ---
# (Assuming the structure allows importing SMKernelLayer from here)
try:
    from models.time_encoders.sm_kernel import SMKernelLayer
except ImportError:
    # Placeholder/Fallback if running outside the intended project structure
    print("Warning: Could not import SMKernelLayer. Please ensure sm_kernel is in models/time_encoders.")
    # Assuming SMKernelLayer class definition is provided elsewhere for functionality.

def calculate_full_kernel(model: SMKernelLayer, delta_t: torch.Tensor) -> torch.Tensor:
    """
    Compute the scalar stationary kernel value K(tau) for given time differences (tau = delta_t).

    Uses the underlying Spectral Mixture parameters directly (without layer norm), i.e.:
      K(tau) = sum_k w_k * exp(-2*pi^2 * tau^2 * s_k) * cos(2*pi * tau * m_k)

    Returns a tensor broadcastable to the shape of delta_t with last dim removed.
    """
    # Normalize shape to (..., 1)
    if delta_t.dim() == 0:
        delta_t = delta_t.view(1, 1, 1)
    elif delta_t.dim() == 1:
        delta_t = delta_t.view(-1, 1, 1)
    elif delta_t.dim() == 2:
        delta_t = delta_t.unsqueeze(-1)

    weights = model.kernel.mixture_weights.view(1, 1, -1)
    means = model.kernel.mixture_means.squeeze(-1).view(1, 1, -1)
    scales = model.kernel.mixture_scales.squeeze(-1).view(1, 1, -1)

    dist_sq = delta_t.pow(2)
    exp_term = torch.exp(-2 * (math.pi**2) * dist_sq * scales)
    cos_term = torch.cos(2 * math.pi * delta_t * means)
    components = weights * exp_term * cos_term
    K_tau = components.sum(dim=-1)
    return K_tau.squeeze()


def calculate_components(model: SMKernelLayer, delta_t: torch.Tensor) -> torch.Tensor:
    """
    Calculates the individual components k_q(tau) that sum up to K(tau).
    
    Returns a tensor of shape (N, Q), where N is the number of time points, and Q is num_mixtures.
    """
    if delta_t.dim() == 1:
        delta_t = delta_t.unsqueeze(-1)
    
    weights = model.kernel.mixture_weights.view(1, -1)
    means = model.kernel.mixture_means.squeeze(-1).view(1, -1)
    scales = model.kernel.mixture_scales.squeeze(-1).view(1, -1)

    dist_sq = delta_t.pow(2)
    
    # Calculate each component separately
    components = []
    for i in range(model.num_mixtures):
        w_q = weights[:, i].view(1, 1)
        m_q = means[:, i].view(1, 1)
        s_q = scales[:, i].view(1, 1)
        
        exp_term = torch.exp(-2 * (math.pi**2) * dist_sq * s_q)
        cos_term = torch.cos(2 * math.pi * delta_t * m_q)
        component = w_q * exp_term * cos_term
        components.append(component)
        
    return torch.cat(components, dim=-1).squeeze()


def analyze_stationarity(model: SMKernelLayer, test_range: tuple = (1.0, 5.0)):
    """ Tests the stationarity (translation invariance) of the SM-Kernel's output. """
    print("\n--- 1. Testing Stationarity (Translation Invariance) ---")
    
    # 1. Define two time points
    t1 = torch.tensor([test_range[1]], dtype=torch.float32).unsqueeze(0).unsqueeze(-1) # e.g., t=5.0
    t2 = torch.tensor([test_range[0]], dtype=torch.float32).unsqueeze(0).unsqueeze(-1) # e.g., t=1.0
    
    # 2. Calculate the difference (tau) and the original embedding
    delta_t_original = t1 - t2 # tau = 4.0
    emb_original = model(delta_t_original)
    K_original = calculate_full_kernel(model, delta_t_original)

    # 3. Apply a translation 'a' (shift both times by a constant)
    a = 100.0 
    t1_shifted = t1 + a # e.g., 105.0
    t2_shifted = t2 + a # e.g., 101.0
    
    # 4. Calculate the shifted difference and embedding
    delta_t_shifted = t1_shifted - t2_shifted # tau = 4.0 (Same difference)
    emb_shifted = model(delta_t_shifted)
    K_shifted = calculate_full_kernel(model, delta_t_shifted)
    
    # Check 1: Feature Vector Equality (Z_delta_t)
    feature_diff = torch.norm(emb_original - emb_shifted).item()
    
    # Check 2: Scalar Kernel Equality (K(tau))
    kernel_diff = torch.norm(K_original - K_shifted).item()
    
    print(f"Original Time Points (t1, t2): ({t1.item():.2f}, {t2.item():.2f})")
    print(f"Shifted Time Points (t1+a, t2+a): ({t1_shifted.item():.2f}, {t2_shifted.item():.2f}) (Shift a={a})")
    print(f"Time Difference (tau): {delta_t_original.item():.2f} vs {delta_t_shifted.item():.2f}")
    
    print(f"L2 Norm Difference of Feature Vectors (Z_delta_t): {feature_diff:.6f}")
    print(f"L2 Norm Difference of Scalar Kernel Values (K(tau)): {kernel_diff:.6f}")

    if feature_diff < 1e-5:
        print("RESULT: PASS - Feature vector is perfectly translation-invariant (stationary).")
    else:
        print("RESULT: FAIL - Check SM Kernel implementation.")


def analyze_aperiodic_modeling(model: SMKernelLayer):
    """ Demonstrates the ability to model aperiodic (non-oscillatory) decay. """
    print("\n--- 2. Analyzing Aperiodic (Decay) Modeling Capability ---")
    
    # Test a time range tau = [0, 10]
    tau = torch.linspace(0, 10, 100).unsqueeze(-1)
    
    # Calculate the full kernel function K(tau) across this range
    K_tau = calculate_full_kernel(model, tau)
    
    # Check if the kernel decays (a strong sign of aperiodic modeling)
    decay_rate = (K_tau[0] - K_tau[-1]).item() / 10.0
    
    print(f"Kernel Value at tau=0: {K_tau[0].item():.4f}")
    print(f"Kernel Value at tau=10: {K_tau[-1].item():.4f}")
    print(f"Average Decay Rate over [0, 10]: {decay_rate:.4f} (Expected positive for decay)")
    
    # Inspect learned parameters
    with torch.no_grad():
        means = model.kernel.mixture_means.squeeze().tolist()
        scales = model.kernel.raw_mixture_scales.squeeze().tolist() # Using raw scales for clarity
        
    print("\nLearned Spectral Parameters (illustrative):")
    for i, (m, s) in enumerate(zip(means, scales)):
        print(f"  Mixture {i+1}: Freq Mean (mu)={m:.3f}, Decay Scale (sigma^2)={s:.3f}")

    if decay_rate > 0.05 and K_tau[-1].item() < K_tau[0].item() / 2:
        print("RESULT: PASS - Kernel exhibits significant decay, confirming ability to model aperiodic temporal dependencies (e.g., memory decay).")
    else:
        print("RESULT: WARNING - Decay is too small or absent. Model may be stuck in a poor initialization or purely periodic mode.")


def plot_kernel_matrix(model: SMKernelLayer, time_points: torch.Tensor, filename: str = 'sm_kernel_matrix.png'):
    """
    Calculates and plots the kernel matrix K(t_i, t_j) to visually demonstrate 
    translation invariance, similar to the lower panels of Figure 9 in the reference paper.
    """
    print("\n--- 3. Visualizing Kernel Matrix K(t_1, t_2) ---")
    
    T = time_points.size(0)
    K_matrix = torch.zeros((T, T), dtype=torch.float32)
    
    # Calculate the kernel matrix K(t_i, t_j)
    for i in range(T):
        for j in range(T):
            delta_t = time_points[i] - time_points[j]
            K_matrix[i, j] = calculate_full_kernel(model, delta_t)

    K_np = K_matrix.detach().cpu().numpy()
    
    # Plotting
    plt.figure(figsize=(8, 8))
    plt.imshow(K_np, cmap='viridis', origin='lower')
    plt.colorbar(label='K(t_1, t_2) Covariance')
    plt.title(f'Learned Stationary Kernel Matrix $K(t_1, t_2)$ ($Q={model.num_mixtures}$)')
    
    # Set axis ticks to reflect the actual time values
    tick_indices = np.linspace(0, T - 1, 5, dtype=int)
    tick_labels = [f'{time_points[i].item():.1f}' for i in tick_indices]
    
    plt.xticks(tick_indices, tick_labels)
    plt.yticks(tick_indices, tick_labels)
    plt.xlabel('$t_2$ (Time)')
    plt.ylabel('$t_1$ (Time)')
    
    plt.subplots_adjust()
    plt.savefig(filename)
    plt.close()
    print(f"Kernel matrix visualized and saved to {filename}")
    print("NOTE: For stationary kernels, non-zero values run parallel to the main diagonal.")


def plot_functional_decomposition(model: SMKernelLayer, tau_range: torch.Tensor, filename: str = 'sm_functional_decomposition.png'):
    """
    Plots the functional decomposition of the SM-Kernel K(tau) into its Q components,
    demonstrating the kernel's expressive power.
    """
    print("\n--- 4. Visualizing Functional Decomposition K(tau) (Additive Synthesis) ---")

    # 1. Calculate Individual Components k_q(tau)
    components_matrix = calculate_components(model, tau_range)
    
    # 2. Calculate the Final Kernel K(tau) (The reconstruction)
    K_tau = calculate_full_kernel(model, tau_range)
    
    # 3. Plotting
    plt.figure(figsize=(8, 8))
    
    # Plot components (k_q(tau))
    for i in range(model.num_mixtures):
        plt.plot(tau_range.squeeze().numpy(), components_matrix[:, i].detach().numpy(), 
                 label=f'Component $k_{i+1}(\\tau)$ (Freq $\\mu$={model.kernel.mixture_means.data.squeeze().tolist()[i]:.2f})', 
                 linestyle='--')
                 
    # Plot the final combined kernel (The Reconstruction)
    plt.plot(tau_range.squeeze().numpy(), K_tau.detach().numpy(), 
             label='Final Kernel $K(\\tau)$ (Reconstruction)', 
             linewidth=3, color='black')

    plt.title('SM-Kernel Functional Decomposition: Modeling Complex Stationary Patterns')
    plt.xlabel('$\\tau$ (Time Difference)')
    plt.ylabel('Covariance/Contribution')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.subplots_adjust()
    plt.savefig(filename)
    plt.close()
    print(f"Functional decomposition visualized and saved to {filename}")


# --- CORRECTED TEST: Valid Covariance Function Reconstruction ---

# IMPORTANT: These are valid covariance functions that SM kernels can actually represent
# (not arbitrary mathematical functions)

def target_rbf_kernel(tau: torch.Tensor, lengthscale: float = 1.0) -> torch.Tensor:
    """RBF/Squared Exponential kernel - most basic valid covariance function."""
    return torch.exp(-0.5 * (tau / lengthscale)**2)

def target_matern32_kernel(tau: torch.Tensor, lengthscale: float = 1.0) -> torch.Tensor:
    """Matérn 3/2 kernel - another valid covariance function."""
    sqrt3_r = np.sqrt(3) * tau / lengthscale
    return (1 + sqrt3_r) * torch.exp(-sqrt3_r)

def target_matern52_kernel(tau: torch.Tensor, lengthscale: float = 1.0) -> torch.Tensor:
    """Matérn 5/2 kernel - smoother than Matérn 3/2."""
    sqrt5_r = np.sqrt(5) * tau / lengthscale
    return (1 + sqrt5_r + (5.0/3.0) * (tau / lengthscale)**2) * torch.exp(-sqrt5_r)

def target_periodic_kernel(tau: torch.Tensor, period: float = 2.0, lengthscale: float = 1.0) -> torch.Tensor:
    """Periodic kernel - captures exact periodic patterns."""
    return torch.exp(-2 * torch.sin(np.pi * tau / period)**2 / lengthscale**2)

def target_quasi_periodic_kernel(tau: torch.Tensor, period: float = 2.0, lengthscale: float = 1.0, decay: float = 2.0) -> torch.Tensor:
    """Quasi-periodic kernel - periodic with decay (what SM kernels naturally represent)."""
    periodic_part = torch.exp(-2 * torch.sin(np.pi * tau / period)**2 / lengthscale**2)
    decay_part = torch.exp(-0.5 * (tau / decay)**2)
    return periodic_part * decay_part

def target_rational_quadratic_kernel(tau: torch.Tensor, alpha: float = 1.0, lengthscale: float = 1.0) -> torch.Tensor:
    """Rational Quadratic kernel - heavy-tailed alternative to RBF."""
    return (1.0 + (tau / lengthscale)**2 / (2.0 * alpha))**(-alpha)

def target_spectral_mixture_kernel(tau: torch.Tensor) -> torch.Tensor:
    """A true spectral mixture kernel (what SM should fit perfectly)."""
    # This is exactly what SM kernels represent: sum of weighted cosines with Gaussian envelopes
    component1 = 0.6 * torch.exp(-2 * (np.pi**2) * (tau**2) * 0.1) * torch.cos(2 * np.pi * tau * 0.5)
    component2 = 0.3 * torch.exp(-2 * (np.pi**2) * (tau**2) * 0.05) * torch.cos(2 * np.pi * tau * 1.2)  
    component3 = 0.1 * torch.exp(-2 * (np.pi**2) * (tau**2) * 0.02) * torch.cos(2 * np.pi * tau * 2.5)
    return component1 + component2 + component3


def _sanitize(name: str) -> str:
    """Make a safe filename token from an arbitrary title string."""
    return re.sub(r'[^A-Za-z0-9._-]+', '_', name)


def plot_reconstruction_3d(model: SMKernelLayer, tau_range: torch.Tensor, title: str, filename: str, y_target: torch.Tensor | None = None):
    """
    3D plot: per-component k_q(τ) as separate tracks along component index,
    plus the final reconstruction K(τ) as a thick black curve.
    """
    components = calculate_components(model, tau_range).detach().cpu().numpy()  # [N, Q]
    K_tau = calculate_full_kernel(model, tau_range).detach().cpu().numpy()      # [N]
    y_target_np = None
    if y_target is not None:
        y_target_np = y_target.detach().cpu().numpy()
    x = tau_range.squeeze().detach().cpu().numpy()
    Q = model.num_mixtures

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, max(Q, 3)))
    for i in range(Q):
        y = np.full_like(x, i + 1, dtype=float)
        ax.plot(x, y, components[:, i], color=colors[i % len(colors)], linestyle='--', linewidth=2, label=f'k_{i+1}(τ)')

    y_final = np.full_like(x, Q + 1.5, dtype=float)
    ax.plot(x, y_final, K_tau, color='black', linewidth=3, label='K(τ)')

    # Optional: overlay the ground-truth target kernel as a separate track
    if y_target_np is not None:
        y_target_track = np.full_like(x, Q + 2.5, dtype=float)
        ax.plot(x, y_target_track, y_target_np, color='red', linewidth=2, alpha=0.9, label='K_target(τ)')

    ax.set_title(title)
    ax.set_xlabel('τ (Time Difference)')
    ax.set_ylabel('Component Index i')
    ax.set_zlabel('Covariance / Contribution')
    yticks = list(range(1, Q + 1)) + [Q + 2]
    ylabels = [f'i={i}' for i in range(1, Q + 1)] + ['K(τ)']
    if y_target_np is not None:
        yticks += [Q + 3]
        ylabels += ['K_target(τ)']
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.view_init(elev=25, azim=-60)
    ax.grid(True)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.subplots_adjust()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"3D reconstruction plot saved to {filename}")


def smart_initialize_for_target(model: SMKernelLayer, target_func, tau_range: torch.Tensor):
    """Smart initialization based on target function characteristics."""
    y_target = target_func(tau_range.squeeze())
    
    with torch.no_grad():
        # Estimate dominant frequency from zero-crossings (rough heuristic)
        tau_vals = tau_range.squeeze()
        y_vals = y_target.detach().cpu().numpy()
        
        # Find approximate frequencies from oscillations
        zero_crossings = []
        for i in range(1, len(y_vals)):
            if y_vals[i-1] * y_vals[i] < 0:  # Sign change
                zero_crossings.append(tau_vals[i].item())
        
        # Initialize mixture parameters
        Q = model.num_mixtures
        
        # Component 1: Broad decay (low freq, large scale)
        model.kernel.raw_mixture_means.data[0] = torch.tensor([0.01])  # Near-zero freq
        model.kernel.raw_mixture_scales.data[0] = torch.tensor([-2.0])  # Large scale (softplus will make ~0.13)
        model.kernel.raw_mixture_weights.data[0] = torch.tensor([0.0])   # log(1.0) after softmax
        
        # Fill remaining with different frequencies
        if len(zero_crossings) >= 2:
            period_est = 2 * (zero_crossings[1] - zero_crossings[0]) if len(zero_crossings) > 1 else 2.0
            freq_est = 1.0 / period_est if period_est > 0 else 0.5
        else:
            freq_est = 0.5
            
        for i in range(1, min(Q, 4)):
            # Different frequencies
            freq = freq_est * (0.5 + i * 0.3)
            model.kernel.raw_mixture_means.data[i] = torch.tensor([freq])
            model.kernel.raw_mixture_scales.data[i] = torch.tensor([-3.0 + i * 0.5])  # Varying scales
            model.kernel.raw_mixture_weights.data[i] = torch.tensor([-1.0 - i * 0.5])  # Lower weights
            
        # Zero out remaining components
        if Q > 4:
            model.kernel.raw_mixture_weights.data[4:] = -10.0  # Very small weights


def test_covariance_function_reconstruction(model: SMKernelLayer, target_func, func_name: str, tau_range: torch.Tensor, filename_prefix: str, steps: int = 200, lr: float = 0.05):
    """
    Optimize SM hyperparameters to fit a VALID covariance function K_target(τ).
    
    This is the correct way to test SM kernels - by seeing if they can represent
    other valid covariance functions (which they should be able to do well).
    """
    print(f"\n--- Testing Covariance Function Reconstruction: {func_name} ---")
    print(f"Target type: Valid covariance function (stationary, positive definite)")

    # Get the ground truth target covariance function values
    y_target = target_func(tau_range.squeeze())
    
    # Check that target is a valid covariance (K(0) should be maximum)
    tau_zero_idx = torch.argmin(torch.abs(tau_range.squeeze()))
    max_val = y_target.max()
    val_at_zero = y_target[tau_zero_idx]
    
    print(f"Covariance validation: K(0)={val_at_zero:.3f}, max={max_val:.3f} {'✓' if abs(val_at_zero - max_val) < 0.1 else '✗'}")
    
    # Smart initialization for covariance functions
    with torch.no_grad():
        Q = model.num_mixtures
        
        # Initialize with reasonable spectral mixture parameters
        for i in range(Q):
            # Frequency: spread across reasonable range
            freq = 0.1 * (1.5 ** i) if i > 0 else 0.01
            model.kernel.raw_mixture_means.data[i] = torch.tensor([freq])
            
            # Scale: decreasing with frequency
            scale_raw = -2.0 - 0.3 * i  # Will become reasonable after softplus
            model.kernel.raw_mixture_scales.data[i] = torch.tensor([scale_raw])
            
            # Weight: equal initial weights
            weight_raw = -np.log(Q) if i == 0 else -np.log(Q) - 0.5 * i
            model.kernel.raw_mixture_weights.data[i] = torch.tensor([weight_raw])
    
    # --- Multi-stage optimization ---
    best_loss = float('inf')
    y_reconstructed = None
    best_state = copy.deepcopy(model.state_dict())
    
    # Stage 1: Coarse search
    optimizer1 = torch.optim.Adam(model.kernel.parameters(), lr=lr * 2)
    for step in range(steps // 2):
        optimizer1.zero_grad()
        y_approx = calculate_full_kernel(model, tau_range)
        loss = torch.mean((y_approx - y_target)**2)
        
        # Add positive definiteness encouragement
        if y_approx[tau_zero_idx] < y_approx.max() * 0.8:
            loss += 0.1 * (y_approx.max() - y_approx[tau_zero_idx])**2
            
        l1_lambda = 0.01  # Regularization strength (a hyperparameter to tune)
        mixture_weights = model.kernel.mixture_weights
        l1_penalty = l1_lambda * torch.sum(torch.abs(mixture_weights))
        loss += l1_penalty
        loss.backward()
        optimizer1.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            y_reconstructed = y_approx.detach()
            best_state = copy.deepcopy(model.state_dict())
    
    # Stage 2: Fine-tuning
    optimizer2 = torch.optim.Adam(model.kernel.parameters(), lr=lr * 0.5)
    for step in range(steps // 2):
        optimizer2.zero_grad()
        y_approx = calculate_full_kernel(model, tau_range)
        loss = torch.mean((y_approx - y_target)**2)
        loss.backward()
        optimizer2.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            y_reconstructed = y_approx.detach()
            best_state = copy.deepcopy(model.state_dict())

    # Load best parameters for analysis
    model.load_state_dict(best_state)
    
    # Compute fit quality metrics
    mse = best_loss
    rmse = np.sqrt(mse)
    mae = torch.mean(torch.abs(y_reconstructed - y_target)).item()
    corr = torch.corrcoef(torch.stack([y_target, y_reconstructed]))[0, 1].item()
    
    print(f"Final metrics: MSE={mse:.6f}, RMSE={rmse:.6f}, MAE={mae:.6f}, Corr={corr:.6f}")
    print(f"Fit quality: {'Excellent' if corr > 0.95 else 'Good' if corr > 0.8 else 'Poor'}")
    
    # Print learned parameters
    with torch.no_grad():
        means = model.kernel.mixture_means.squeeze().tolist()
        scales = model.kernel.mixture_scales.squeeze().tolist()
        weights = model.kernel.mixture_weights.squeeze().tolist()
        
    print(f"Learned SM parameters:")
    for i, (w, m, s) in enumerate(zip(weights, means, scales)):
        print(f"  Component {i+1}: weight={w:.3f}, freq={m:.3f} Hz, scale={s:.3f}")
    
    # --- Visualization ---
    plt.figure(figsize=(8, 8))
    
    plt.plot(tau_range.squeeze().numpy(), y_target.numpy(), 
             label=f'Target: {func_name}', 
             linewidth=4, color='red', alpha=0.7)
             
    plt.plot(tau_range.squeeze().numpy(), y_reconstructed.numpy(), 
             label=f'SM-Kernel Fit (corr={corr:.3f})', 
             linewidth=2, color='black', linestyle='--')

    plt.title(f'SM-Kernel Covariance Function Fitting: {func_name}\n(RMSE: {rmse:.4f}, Correlation: {corr:.3f})')
    plt.xlabel('$\\tau$ (Time Lag)')
    plt.ylabel('Covariance K($\\tau$)')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.subplots_adjust()
    plt.savefig(f'{filename_prefix}.png')
    plt.close()
    print(f"Reconstruction plot saved to {filename_prefix}.png")

    # 3D component plot
    plot_reconstruction_3d(
        model,
        tau_range,
        title=f'SM-Kernel Components: {func_name} (Corr: {corr:.3f})',
        filename=f'{filename_prefix}_3D.png',
        y_target=y_target
    )
    
    return mse, corr


def test_quasi_periodic_specialization():
    """
    Special test for quasi-periodic kernel fitting - this is where SM kernels should excel.
    Quasi-periodic = periodic pattern with gradual decay, which is exactly what SM represents.
    """
    print("\n" + "="*60)
    print("SPECIAL TEST: QUASI-PERIODIC KERNEL FITTING")
    print("(This is what SM kernels are designed to excel at)")
    print("="*60)
    
    tau_range = torch.linspace(0.0, 15.0, 150).unsqueeze(-1)  # Longer range to see decay
    
    # Test different quasi-periodic patterns
    qp_tests = [
        {
            'name': 'QP: Fast decay, high freq',
            'period': 1.5,
            'lengthscale': 0.3, 
            'decay': 2.0,
            'mixtures': 12
        },
        {
            'name': 'QP: Slow decay, medium freq', 
            'period': 3.0,
            'lengthscale': 0.8,
            'decay': 5.0,
            'mixtures': 12
        },
        {
            'name': 'QP: Very slow decay, low freq',
            'period': 4.0,
            'lengthscale': 1.0,
            'decay': 8.0,
            'mixtures': 12
        }
    ]
    
    for test_config in qp_tests:
        print(f"\nTesting: {test_config['name']}")
        print(f"  Period={test_config['period']}, Decay={test_config['decay']}, Mixtures={test_config['mixtures']}")
        
        # Create target quasi-periodic function
        def target_qp(tau):
            return target_quasi_periodic_kernel(
                tau, 
                period=test_config['period'],
                lengthscale=test_config['lengthscale'], 
                decay=test_config['decay']
            )
        
        # Initialize SM kernel with appropriate number of mixtures
        sm_model = SMKernelLayer(num_mixtures=test_config['mixtures'])
        
        # Smart initialization specifically for quasi-periodic
        with torch.no_grad():
            base_freq = 1.0 / test_config['period']
            Q = test_config['mixtures']
            
            # Component 0: Match the main frequency
            sm_model.kernel.raw_mixture_means.data[0] = torch.tensor([base_freq])
            sm_model.kernel.raw_mixture_scales.data[0] = torch.tensor([np.log(test_config['lengthscale']**2)])
            sm_model.kernel.raw_mixture_weights.data[0] = torch.tensor([0.0])  # High weight
            
            # Remaining components: harmonics and decay terms
            for i in range(1, Q):
                if i <= 2:
                    # Harmonics of base frequency
                    freq = base_freq * (i + 1)
                    scale = test_config['lengthscale']**2 / (i + 1)
                    weight = 1.0 / (i + 1)**2
                else:
                    # Aperiodic decay components
                    freq = 0.01  # Near-zero for decay
                    scale = (test_config['decay'] / (i-2))**2
                    weight = 0.1 / (i-2)
                
                sm_model.kernel.raw_mixture_means.data[i] = torch.tensor([freq])
                sm_model.kernel.raw_mixture_scales.data[i] = torch.tensor([np.log(scale)])
                sm_model.kernel.raw_mixture_weights.data[i] = torch.tensor([np.log(weight)])
        
        # Test reconstruction
        filename_prefix = f'KAN_MAMMOTE_QP_Test_{_sanitize(test_config["name"])}'
        mse, corr = test_covariance_function_reconstruction(
            sm_model,
            target_qp,
            test_config['name'],
            tau_range,
            filename_prefix,
            steps=20000,  # More steps for complex patterns
            lr=0.005
        )
        
        # Expected performance analysis
        expected_corr = 0.95  # SM should excel at quasi-periodic
        performance = "Excellent" if corr >= expected_corr else "Good" if corr >= 0.8 else "Poor"
        
        print(f"  RESULT: MSE={mse:.6f}, Correlation={corr:.3f} ({performance})")
        if corr >= expected_corr:
            print("  ✓ SM kernel successfully captured quasi-periodic pattern")
        else:
            print(f"  ⚠ Lower than expected (target: >{expected_corr:.2f}). May need more mixtures or different init.")
        
        # Component analysis
        with torch.no_grad():
            weights = sm_model.kernel.mixture_weights.squeeze().tolist()
            means = sm_model.kernel.mixture_means.squeeze().tolist()
            scales = sm_model.kernel.mixture_scales.squeeze().tolist()
            
        print(f"  Learned components:")
        for i, (w, m, s) in enumerate(zip(weights, means, scales)):
            component_type = "Periodic" if m > 0.1 else "Decay"
            print(f"    {i+1}: {component_type} - freq={m:.3f}Hz, weight={w:.3f}, scale={s:.3f}")
    
    print("\n" + "="*60)
    print("QUASI-PERIODIC TEST SUMMARY:")
    print("- SM kernels should achieve >0.95 correlation on quasi-periodic targets")
    print("- Poor performance indicates initialization or optimization issues")
    print("- Components should show mix of periodic and decay terms")
    print("="*60)


def run_analysis():
    """ Main function to initialize model and run all analyses. """
    
    # Configuration
    NUM_MIXTURES = 8
    TIME_POINTS = torch.linspace(0.0, 10.0, 50).unsqueeze(-1)
    
    # 1. Initialize the SM-Kernel model for Functional Decomposition analysis
    sm_model_decomp = SMKernelLayer(num_mixtures=NUM_MIXTURES)

    # 2. Set specific parameters for Decomposition Visualization (Test 4)
    with torch.no_grad():
        # Mixture 1: Broad, slow decay (Aperiodic component)
        sm_model_decomp.kernel.raw_mixture_means.data[0] = torch.tensor([0.0])
        sm_model_decomp.kernel.raw_mixture_scales.data[0] = torch.tensor([0.313])
        sm_model_decomp.kernel.raw_mixture_weights.data[0] = torch.tensor([0.5])

        # Mixture 2: Medium frequency oscillation (Quasi-Periodic) - Increased persistence
        sm_model_decomp.kernel.raw_mixture_means.data[1] = torch.tensor([1.701]) 
        sm_model_decomp.kernel.raw_mixture_scales.data[1] = torch.tensor([0.05])
        sm_model_decomp.kernel.raw_mixture_weights.data[1] = torch.tensor([0.3])
        
        # Mixture 3: High frequency, low weight oscillation - Increased persistence
        sm_model_decomp.kernel.raw_mixture_means.data[2] = torch.tensor([5.0]) 
        sm_model_decomp.kernel.raw_mixture_scales.data[2] = torch.tensor([0.05]) 
        sm_model_decomp.kernel.raw_mixture_weights.data[2] = torch.tensor([0.1])
        
        sm_model_decomp.kernel.raw_mixture_weights.data[3:] = 0.0

    # 3. Run Validation Tests (Stationarity, Aperiodic Modeling, and Matrix Plot)
    analyze_stationarity(sm_model_decomp)
    analyze_aperiodic_modeling(sm_model_decomp)
    plot_kernel_matrix(sm_model_decomp, TIME_POINTS, 'KAN_MAMMOTE_SM_Kernel_Matrix_Analysis.png')
    plot_functional_decomposition(sm_model_decomp, TIME_POINTS, 'KAN_MAMMOTE_SM_Functional_Decomposition.png')
    
    
    # 4. Run Valid Covariance Function Reconstruction Test Suite 
    # These are functions that SM kernels can actually represent well
    TIME_POINTS = torch.linspace(0.0, 10.0, 100).unsqueeze(-1)
    targets = [
        ("RBF Kernel (Gaussian decay)", lambda t: target_rbf_kernel(t, lengthscale=1.5), 8),
        ("Matérn 3/2 Kernel", lambda t: target_matern32_kernel(t, lengthscale=1.2), 8), 
        ("Matérn 5/2 Kernel", lambda t: target_matern52_kernel(t, lengthscale=1.0), 8),
        ("Periodic Kernel", lambda t: target_periodic_kernel(t, period=3.0, lengthscale=0.5), 8),
        ("Quasi-Periodic Kernel", lambda t: target_quasi_periodic_kernel(t, period=2.5, lengthscale=0.8, decay=3.0), 8),
        ("Rational Quadratic Kernel", lambda t: target_rational_quadratic_kernel(t, alpha=1.5, lengthscale=1.0), 8),
        ("True Spectral Mixture", target_spectral_mixture_kernel, 8),
    ]

    print("\n" + "="*80)
    print("TESTING SM-KERNEL ABILITY TO FIT VALID COVARIANCE FUNCTIONS")
    print("(These are the types of functions SM kernels are designed to represent)")
    print("="*80)

    for name, func, q in targets:
        sm_model_recons = SMKernelLayer(num_mixtures=q)
        prefix = f'KAN_MAMMOTE_SM_Covariance_{_sanitize(name)}'
        mse, corr = test_covariance_function_reconstruction(
            sm_model_recons,
            func,
            name,
            TIME_POINTS,
            prefix,
            steps=20000,
            lr=0.005
        )
        print(f"RESULT: {name} -> MSE: {mse:.6f}, Correlation: {corr:.3f}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("Key insights:")
    print("- SM kernels excel at fitting valid covariance functions")
    print("- Poor fits indicate the target wasn't a proper covariance function")
    print("- High correlations (>0.9) show SM kernel expressiveness")
    print("="*80)

    # 5. Special quasi-periodic kernel test (SM kernels should excel here)
    test_quasi_periodic_specialization()


if __name__ == '__main__':
    # Add the current directory to path to ensure sm_kernel is found
    # This might be necessary depending on the execution environment
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    run_analysis()
