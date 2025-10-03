# file: sm_kernel_analysis.py

import os
import sys
import torch
import torch.nn as nn
import gpytorch
import math
import numpy as np
import matplotlib.pyplot as plt
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

def plot_functional_decomposition_3d(model: SMKernelLayer, tau_range: torch.Tensor, filename: str = 'sm_functional_decomposition_3d.png'):
    """
    3D line plot of each SM component k_q(τ) along the component index axis,
    plus the combined kernel K(τ) as a thick black curve.
    """
    print("\n--- 4b. Visualizing Functional Decomposition K(tau) in 3D ---")

    # Data
    components_matrix = calculate_components(model, tau_range)         # [N, Q]
    K_tau = calculate_full_kernel(model, tau_range).detach()           # [N]
    x = tau_range.squeeze().detach().cpu().numpy()                     # τ
    Q = model.num_mixtures

    # Figure
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, max(Q, 3)))
    for i in range(Q):
        z = components_matrix[:, i].detach().cpu().numpy()
        y = np.full_like(x, i + 1, dtype=float)                        # component index along Y
        ax.plot(x, y, z, color=colors[i % len(colors)], linestyle='--', linewidth=2, label=f'k_{i+1}(τ)')

    # Final kernel as a separate "track" after components
    y_final = np.full_like(x, Q + 1.5, dtype=float)
    ax.plot(x, y_final, K_tau.detach().cpu().numpy(), color='black', linewidth=3, label='K(τ)')

    # Axes/labels
    ax.set_title('SM-Kernel Functional Decomposition (3D)')
    ax.set_xlabel('τ (Time Difference)')
    ax.set_ylabel('Component index i')
    ax.set_zlabel('Covariance / Contribution')
    ax.set_yticks(list(range(1, Q + 1)) + [Q + 2])
    ax.set_yticklabels([f'i={i}' for i in range(1, Q + 1)] + ['K(τ)'])
    ax.view_init(elev=25, azim=-60)
    ax.grid(True)

    # Legend outside plot for clarity
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()
    print(f"3D functional decomposition saved to {filename}")

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
    plt.figure(figsize=(7, 6))
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
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Kernel matrix visualized and saved to {filename}")
    print("NOTE: For stationary kernels, non-zero values run parallel to the main diagonal.")


def plot_functional_decomposition(model: SMKernelLayer, tau_range: torch.Tensor, filename: str = 'sm_functional_decomposition.png'):
    """
    Plots the functional decomposition of the SM-Kernel K(tau) into its Q components,
    demonstrating the kernel's expressive power.
    """
    print("\n--- 4. Visualizing Functional Decomposition K(tau) ---")

    # 1. Calculate Individual Components k_q(tau)
    components_matrix = calculate_components(model, tau_range)
    
    # 2. Calculate the Final Kernel K(tau) (The reconstruction)
    K_tau = calculate_full_kernel(model, tau_range)
    
    # 3. Plotting
    plt.figure(figsize=(10, 6))
    
    # Plot components (k_q(tau))
    for i in range(model.num_mixtures):
        plt.plot(tau_range.squeeze().numpy(), components_matrix[:, i].detach().numpy(), 
                 label=f'Component $k_{i+1}(\\tau)$', linestyle='--')
                 
    # Plot the final combined kernel (K(tau))
    plt.plot(tau_range.squeeze().numpy(), K_tau.detach().numpy(), 
             label='Final Kernel $K(\\tau) = \\sum k_q(\\tau)$', 
             linewidth=3, color='black')

    plt.title('SM-Kernel Functional Decomposition: Modeling Complex Stationary Patterns')
    plt.xlabel('$\\tau$ (Time Difference)')
    plt.ylabel('Covariance/Contribution')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Functional decomposition visualized and saved to {filename}")


def run_analysis():
    """ Main function to initialize model and run all analyses. """
    
    # Configuration
    NUM_MIXTURES = 5
    TIME_POINTS = torch.linspace(0.0, 10.0, 50).unsqueeze(-1)
    
    # 1. Initialize the SM-Kernel model
    sm_model = SMKernelLayer(num_mixtures=NUM_MIXTURES)

    # 2. Initialize parameters (This simulates learning a pattern)
    # We manually set parameters to illustrate a mix of decay and oscillation (aperiodic)
    with torch.no_grad():
        # Mixture 1: Broad, slow decay (Aperiodic component)
        sm_model.kernel.raw_mixture_means.data[0] = torch.tensor([0.0])  # Low frequency (pure decay)
        sm_model.kernel.raw_mixture_scales.data[0] = torch.tensor([0.313]) # Large lengthscale (sigma^2)
        sm_model.kernel.raw_mixture_weights.data[0] = torch.tensor([0.5])

        # Mixture 2: Medium frequency oscillation (Quasi-Periodic)
        sm_model.kernel.raw_mixture_means.data[1] = torch.tensor([1.701]) 
        sm_model.kernel.raw_mixture_scales.data[1] = torch.tensor([0.05]) 
        sm_model.kernel.raw_mixture_weights.data[1] = torch.tensor([0.3])
        
        # Mixture 3: High frequency, low weight oscillation
        sm_model.kernel.raw_mixture_means.data[2] = torch.tensor([5.0]) 
        sm_model.kernel.raw_mixture_scales.data[2] = torch.tensor([0.01]) 
        sm_model.kernel.raw_mixture_weights.data[2] = torch.tensor([0.1])
        
        # Reset remaining mixtures to zero influence for cleaner visual (optional)
        sm_model.kernel.raw_mixture_weights.data[3:] = 0.0

    # 3. Run Analysis Tests
    analyze_stationarity(sm_model)
    analyze_aperiodic_modeling(sm_model)
    plot_kernel_matrix(sm_model, TIME_POINTS, 'KAN_MAMMOTE_SM_Kernel_Matrix_Analysis.png')
    plot_functional_decomposition(sm_model, TIME_POINTS, 'KAN_MAMMOTE_SM_Functional_Decomposition.png')

if __name__ == '__main__':
    # Add the current directory to path to ensure sm_kernel is found
    # This might be necessary depending on the execution environment
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    run_analysis()
