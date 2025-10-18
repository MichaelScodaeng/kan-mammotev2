# file: analysis/analyze_kan_mammote.py

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d

# Add project root to path to import our models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.replicate_figure11 import ReconstructionAnalysisModel
# Helper functions to generate synthetic data
from analysis_tools.synthetic_data import (
    generate_smooth_trend_data,
    generate_periodic_data,
    generate_abrupt_change_data,
    generate_localized_event_data,
    generate_mixed_pattern_data
)

# --- 1. Training and Data Generation ---

def train_reconstruction_model(model, t_abs, t_rel, y_true, epochs=10000, lr=5e-4, patience=500):
    """Trains the ReconstructionAnalysisModel."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    # Reshape for model: (Batch=1, Seq_len, Dim=1)
    t_abs = t_abs.unsqueeze(0).unsqueeze(-1)
    t_rel = t_rel.unsqueeze(0).unsqueeze(-1)
    y_true = y_true.unsqueeze(0).unsqueeze(-1)

    best_loss = float('inf')
    patience_counter = 0
    
    with tqdm(range(epochs), desc="Training Reconstruction Model") as pbar:
        for epoch in pbar:
            model.train()
            y_pred = model(t_abs, t_rel)
            loss = loss_fn(y_pred, y_true)

            if torch.isnan(loss):
                print("NaN loss detected. Stopping.")
                break

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            
            pbar.set_postfix({'Loss': f'{best_loss:.6f}', 'Patience': f'{patience_counter}/{patience}'})
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    print(f"Final reconstruction loss: {best_loss:.6f}")

def get_time_inputs_from_series(t_series):
    """Generates t_abs and t_rel from a time series."""
    t_abs = t_series
    # Calculate t_rel = t_k - t_{k-1}
    t_rel = torch.diff(t_abs, prepend=t_abs.new_tensor([t_abs[0]]))
    return t_abs, t_rel

# --- 2. Visualization ---

def plot_analysis_figure(model, t_abs, t_rel, y_true, components, title):
    """Creates the multi-panel analysis figure like in the LeTE paper."""
    model.eval()
    with torch.no_grad():
        # Reshape for model: (Batch=1, Seq_len, Dim=1)
        t_abs_in = t_abs.unsqueeze(0).unsqueeze(-1)
        t_rel_in = t_rel.unsqueeze(0).unsqueeze(-1)
        y_pred, gating_weights = model(t_abs_in, t_rel_in, return_gating_weights=True)
        y_pred = y_pred.squeeze().cpu().numpy()
        gating_weights = gating_weights.squeeze().cpu().numpy()

    t_numpy = t_abs.cpu().numpy()
    y_true_numpy = y_true.cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(18, 15), sharex=True)
    fig.suptitle(title, fontsize=20)

    # Panel 1: Signal Decomposition
    axes[0].plot(t_numpy, y_true_numpy, 'k-', linewidth=3, label='Full Signal (Ground Truth)', alpha=0.9)
    colors = ['green', 'blue', 'red', 'purple']
    patterns = ['Smooth (B-Spline)', 'Periodic (Fourier)', 'Abrupt (Wavelet)', 'Localized (RBF)']
    for i, (name, data) in enumerate(components.items()):
        axes[0].plot(t_numpy, data.cpu().numpy(), linestyle='--', label=f'{patterns[i]} Component', color=colors[i], alpha=0.8)
    axes[0].set_title("Signal Decomposition and Ground Truth", fontsize=14)
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.4)

    # Panel 2: Model Reconstruction
    axes[1].plot(t_numpy, y_true_numpy, 'k-', linewidth=3, label='Ground Truth', alpha=0.9)
    axes[1].plot(t_numpy, y_pred, 'c--', linewidth=2.5, label='KAN-MAMMOTE Reconstruction')
    axes[1].set_title("Model Reconstruction Performance", fontsize=14)
    axes[1].legend(loc='upper left')
    axes[1].grid(True, alpha=0.4)
    
    # Panel 3: MoE Gating Weights Analysis
    expert_names = ['B-Spline', 'Fourier', 'Wavelet', 'RBF']
    for i in range(gating_weights.shape[1]):
        axes[2].plot(t_numpy, gating_weights[:, i], label=f'{expert_names[i]} Expert Weight', color=colors[i], linewidth=2)
    axes[2].set_title("K-MOTE Absolute Time Expert Gating Weights", fontsize=14)
    axes[2].set_xlabel("Time (t)", fontsize=12)
    axes[2].set_ylabel("Gating Weight (Softmax)", fontsize=12)
    axes[2].legend(loc='upper left')
    axes[2].grid(True, alpha=0.4)
    axes[2].set_ylim(0, 1)

    # Add annotations to connect patterns to gating
    # Example: Find where the shock component is largest
    shock_region = components['abrupt'].abs() > 0.1
    axes[2].fill_between(t_numpy, 0, 1, where=shock_region.cpu().numpy(), color='red', alpha=0.15, transform=axes[2].get_xaxis_transform(), label='Abrupt Change Region')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the figure
    safe_title = title.replace(' ', '_').lower()
    os.makedirs('analysis_output', exist_ok=True)
    plt.savefig(f'analysis_output/{safe_title}.png', dpi=300)
    plt.show()

# --- 3. Main Execution ---

def main():
    # --- Part 1: Synthetic Data Analysis ---
    print("\n--- Running Analysis on Synthetic Mixed-Pattern Data ---")
    t_series = torch.linspace(-8, 8, 500)
    
    # Generate components and the final mixed signal
    smooth_comp = generate_smooth_trend_data(t_series)
    periodic_comp = generate_periodic_data(t_series)
    abrupt_comp = generate_abrupt_change_data(t_series)
    localized_comp = generate_localized_event_data(t_series)
    
    y_true_mixed = smooth_comp + periodic_comp + abrupt_comp + localized_comp
    
    components = {
        'smooth': smooth_comp,
        'periodic': periodic_comp,
        'abrupt': abrupt_comp,
        'localized': localized_comp
    }

    t_abs, t_rel = get_time_inputs_from_series(t_series)

    # Initialize the analysis model
    # Smaller dims for faster training in this toy example
    model = ReconstructionAnalysisModel(
        embedding_dim=64,
        expert_dim=32,
        mamba_d_state=64,
        mamba_expand=2,
        use_controllable_mamba=True
    )
    
    train_reconstruction_model(model, t_abs, t_rel, y_true_mixed)
    plot_analysis_figure(model, t_abs, t_rel, y_true_mixed, components, 
                         "KAN-MAMMOTE Reconstruction on Synthetic Mixed Data")

    # --- Part 2: Real Data Analysis (Conceptual) ---
    print("\n--- Running Analysis on Simulated Real Data ---")
    # In a real scenario, you would load timestamps from a dataset.
    # Here, we simulate a node's activity for demonstration.
    
    # Simulate bursts of activity (non-periodic) and a weekly periodic signal
    base_time = torch.arange(0, 100, 0.1)
    burst1_times = base_time[(base_time > 10) & (base_time < 15)] + torch.randn(50) * 0.1
    burst2_times = base_time[(base_time > 60) & (base_time < 62)] + torch.randn(20) * 0.05
    periodic_times = torch.arange(0, 100, 7) # Every 7 days
    
    # Combine and sort to get the final timestamp sequence
    real_t_series = torch.sort(torch.cat([burst1_times, burst2_times, periodic_times]))[0]

    # Create the ground truth signal by smoothing a spike train
    spike_train = torch.zeros(int(real_t_series.max())*10 + 1)
    indices = (real_t_series * 10).long()
    spike_train[indices] = 1
    y_true_real_smoothed = torch.tensor(gaussian_filter1d(spike_train.numpy(), sigma=10), dtype=torch.float32)
    t_series_real_dense = torch.linspace(0, real_t_series.max(), len(y_true_real_smoothed))

    t_abs_real, t_rel_real = get_time_inputs_from_series(t_series_real_dense)

    # Train a new model on this data
    model_real = ReconstructionAnalysisModel(embedding_dim=64, expert_dim=32)
    train_reconstruction_model(model_real, t_abs_real, t_rel_real, y_true_real_smoothed)
    
    # For visualization, we can't decompose into known components, so we pass empty dict
    plot_analysis_figure(model_real, t_abs_real, t_rel_real, y_true_real_smoothed, {},
                         "KAN-MAMMOTE Reconstruction on Simulated Node Activity")


if __name__ == '__main__':
    main()