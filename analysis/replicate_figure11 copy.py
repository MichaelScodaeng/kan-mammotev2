# file: analysis/replicate_figure11.py
# Description: A script to faithfully replicate the reconstruction experiment 
# from Section G.4, Figure 11 of the LeTE paper.

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# --- 1. SETUP: Add project root to path to import our models ---
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.time_encoders.kan_mammote import KAN_MAMMOTE
except ImportError as e:
    print(f"ERROR: Could not import KAN_MAMMOTE. Please check your file paths.")
    print(f"Details: {e}")
    sys.exit(1)

os.makedirs('analysis_figure11_replication', exist_ok=True)


# --- 2. DATA GENERATION (Following Your Logic) ---

def generate_periodic_signal(t):
    """Generates a complex periodic signal by summing sinusoids."""
    y_periodic = (1.0 * torch.sin(2 * torch.pi * t / 25) +
                  0.6 * torch.cos(2 * torch.pi * t / 10) +
                  0.3 * torch.sin(2 * torch.pi * t / 3.5) +
                  0.1 * torch.randn_like(t)) # Add a little noise
    return y_periodic

def generate_non_periodic_signal(t, probability=0.03, spike_height=2.0):
    """Generates a signal of random spikes based on a probability."""
    spikes = torch.rand_like(t) < probability
    y_non_periodic = torch.zeros_like(t)
    # Give spikes random heights for more variety
    y_non_periodic[spikes] = spike_height * (0.5 + 0.5 * torch.rand_like(y_non_periodic[spikes]))
    return y_non_periodic

def generate_mixed_signal(t):
    """Generates a mixed signal by combining periodic and non-periodic data."""
    y_periodic = generate_periodic_signal(t)
    y_non_periodic = generate_non_periodic_signal(t, probability=0.02, spike_height=2.5)
    return y_periodic + y_non_periodic


# --- 3. MODEL ARCHITECTURES FOR THE EXPERIMENT ---

class Time2Vec(nn.Module):
    """A simple, correct implementation of Time2Vec (FTE) for our baseline."""
    def __init__(self, in_features, out_features):
        super(Time2Vec, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(torch.randn(out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        v1 = self.f(torch.matmul(tau, self.w) + self.b)
        v2 = torch.matmul(tau, self.w0) + self.b0
        return torch.cat([v1, v2], -1)

class ReconstructionModel(nn.Module):
    """
    Generic Encoder-Decoder wrapper for the analysis.
    It can wrap either the Time2Vec baseline or the full KAN_MAMMOTE model.
    """
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        # The simple linear decoder, as described in the LeTE paper
        self.decoder = nn.Linear(encoder.embedding_dim, 1)

    def forward(self, t_abs, t_rel=None, return_gating_weights=False):
        # Handle different encoder types
        if isinstance(self.encoder, KAN_MAMMOTE):
            # KAN_MAMMOTE needs both t_abs and t_rel
            embedding = self.encoder(t_abs, t_rel)
        elif isinstance(self.encoder, Time2Vec):
            # Time2Vec baseline only uses t_abs
            embedding = self.encoder(t_abs)
        else:
            raise TypeError("Unsupported encoder type")

        reconstruction = self.decoder(embedding)
        
        # This part is only for KAN-MAMMOTE to analyze gating
        if return_gating_weights and isinstance(self.encoder, KAN_MAMMOTE):
            with torch.no_grad():
                _, gating_weights = self.encoder.k_mote_abs(t_abs, return_weights=True)
            return reconstruction, gating_weights

        return reconstruction


# --- 4. TRAINING AND PLOTTING ---

def train_model(model, t_abs, y_true, epochs=8000, lr=5e-4, patience=400):
    """Universal training loop for any reconstruction model."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()

    # Generate t_rel from t_abs
    t_rel = torch.diff(t_abs, prepend=t_abs.new_tensor([t_abs[0]]))
    
    # Reshape for model: (Batch=1, Seq_len, Dim=1)
    t_abs = t_abs.unsqueeze(0).unsqueeze(-1)
    t_rel = t_rel.unsqueeze(0).unsqueeze(-1)
    y_true = y_true.unsqueeze(0).unsqueeze(-1)

    best_loss = float('inf')
    patience_counter = 0
    
    with tqdm(range(epochs), desc=f"Training {model.encoder.__class__.__name__}") as pbar:
        for epoch in pbar:
            model.train()
            # Pass both t_abs and t_rel; the model will use what it needs
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
    print(f"Final loss for {model.encoder.__class__.__name__}: {best_loss:.6f}\n")

def plot_figure11_replication(t, y_true, fte_model, kan_mammote_model, signal_type):
    """Generates a plot that mimics the layout of LeTE's Figure 11."""
    fte_model.eval()
    kan_mammote_model.eval()

    t_abs, t_rel = t, torch.diff(t, prepend=t.new_tensor([t[0]]))
    t_abs_in = t_abs.unsqueeze(0).unsqueeze(-1)
    t_rel_in = t_rel.unsqueeze(0).unsqueeze(-1)
    
    with torch.no_grad():
        y_pred_fte = fte_model(t_abs_in).squeeze().cpu().numpy()
        y_pred_kan, gating_weights = kan_mammote_model(t_abs_in, t_rel_in, return_gating_weights=True)
        y_pred_kan = y_pred_kan.squeeze().cpu().numpy()

    y_true_np = y_true.cpu().numpy()
    t_np = t.cpu().numpy()
    
    residuals_fte = y_true_np - y_pred_fte
    residuals_kan = y_true_np - y_pred_kan

    fig, axes = plt.subplots(2, 2, figsize=(20, 10), sharex=True)
    fig.suptitle(f"Reconstruction Analysis on Synthetic '{signal_type}' Data", fontsize=20)

    # --- Row 1: Reconstructions ---
    axes[0, 0].plot(t_np, y_true_np, 'k-', label='Ground Truth')
    axes[0, 0].plot(t_np, y_pred_fte, 'c--', label='FTE (Baseline) Rec.')
    axes[0, 0].set_title('FTE (Baseline) Reconstruction', fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.5, linestyle=':')

    axes[0, 1].plot(t_np, y_true_np, 'k-', label='Ground Truth')
    axes[0, 1].plot(t_np, y_pred_kan, 'm--', label='KAN-MAMMOTE Rec.')
    axes[0, 1].set_title('KAN-MAMMOTE Reconstruction', fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.5, linestyle=':')

    # --- Row 2: Residuals (Error) ---
    axes[1, 0].plot(t_np, residuals_fte, 'r-', label='FTE Residuals')
    axes[1, 0].set_title('FTE Residuals (Error)', fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.5, linestyle=':')
    axes[1, 0].set_xlabel("Time (t)")
    axes[1, 0].set_ylabel("Error (True - Pred)")
    
    axes[1, 1].plot(t_np, residuals_kan, 'r-', label='KAN-MAMMOTE Residuals')
    axes[1, 1].set_title('KAN-MAMMOTE Residuals (Error)', fontsize=14)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.5, linestyle=':')
    axes[1, 1].set_xlabel("Time (t)")

    # Set common y-limits for residuals to make comparison fair
    max_residual = max(np.abs(residuals_fte).max(), np.abs(residuals_kan).max()) * 1.1
    axes[1, 0].set_ylim(-max_residual, max_residual)
    axes[1, 1].set_ylim(-max_residual, max_residual)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"analysis_figure11_replication/reconstruction_{signal_type.lower()}.png", dpi=300)
    plt.show()

# --- 5. MAIN EXECUTION SCRIPT ---

if __name__ == '__main__':
    EMBEDDING_DIM = 64 # Dimension for both encoders
    
    # Define the time axis for all experiments
    t = torch.linspace(0, 100, 1000)

    # Define the datasets to test
    datasets = {
        "Periodic": generate_periodic_signal(t),
        "Non-Periodic": generate_non_periodic_signal(t),
        "Mixed": generate_mixed_signal(t)
    }

    for signal_type, y_data in datasets.items():
        print("\n" + "="*70)
        print(f"--- Starting Analysis for '{signal_type}' Data ---")
        print("="*70)

        # 1. Initialize and Train the FTE (Time2Vec) Baseline Model
        print("--- Training Baseline: FTE (Time2Vec) ---")
        fte_encoder = Time2Vec(in_features=1, out_features=EMBEDDING_DIM)
        # Add embedding_dim attribute for compatibility with the wrapper
        fte_encoder.embedding_dim = EMBEDDING_DIM 
        fte_model = ReconstructionModel(encoder=fte_encoder)
        train_model(fte_model, t, y_data)

        # 2. Initialize and Train the KAN-MAMMOTE Model
        print("\n--- Training Main Model: KAN-MAMMOTE ---")
        kan_mammote_encoder = KAN_MAMMOTE(
            embedding_dim=EMBEDDING_DIM,
            expert_dim=32, # Must be a multiple of 16
            num_mixtures=32,
            mamba_d_state=64, # Must be a multiple of 16
            mamba_expand=2,
            use_kmote_for_relative=True, # Dual-K-MOTE mode
            use_controllable_mamba=True
        )
        kan_mammote_model = ReconstructionModel(encoder=kan_mammote_encoder)
        train_model(kan_mammote_model, t, y_data)
        
        # 3. Generate the comparison plot for this dataset
        plot_figure11_replication(t, y_data, fte_model, kan_mammote_model, signal_type)