import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add the project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from models.time_encoders.k_mote import (
    SplineKANLayer, FourierKANLayer, WaveletKANLayer, KMOTE
)

# Global training configuration
MAX_EPOCHS = 1000
HIDDEN_DIM = 64
OUTPUT_DIM = 1

class SimpleWrapper(nn.Module):
    """Wrapper for KAN layers to match LeTE/FTE regressor structure"""
    def __init__(self, kan_layer, input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM):
        super().__init__()
        # This layer is not strictly necessary if input is already 1D, but good practice
        self.time_transform = nn.Linear(input_dim, hidden_dim)
        self.kan = kan_layer
        self.output_head = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        x = self.time_transform(x)
        x = self.kan(x)
        x = self.output_head(x)
        return x

def create_models():
    """Create simple model instances with consistent architecture"""
    models = {
        'B-SplineKAN': SimpleWrapper(SplineKANLayer(input_dim=HIDDEN_DIM, output_dim=HIDDEN_DIM)),
        'FourierKAN': SimpleWrapper(FourierKANLayer(input_dim=HIDDEN_DIM, output_dim=HIDDEN_DIM)),
        'WaveletKAN': SimpleWrapper(WaveletKANLayer(input_dim=HIDDEN_DIM, output_dim=HIDDEN_DIM)),
        'K-MOTE': KMOTE(input_dim=1, output_dim=OUTPUT_DIM, hidden_dim=HIDDEN_DIM)
    }
    return models

def train_simple_model(model, X, y, epochs=50):
    """Train a model with simple training loop"""
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    criterion = nn.MSELoss()
    
    # Ensure consistent shapes: [batch, sequence, features]
    if X.dim() == 2:
        X = X.unsqueeze(0) # (500, 1) -> (1, 500, 1)
    if y.dim() == 1:
        y = y.unsqueeze(0).unsqueeze(-1) # (500) -> (1, 500, 1)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        final_pred = model(X)
        final_loss = criterion(final_pred, y)
    
    return final_loss.item()

# Define mathematical functions
def func_sin(t):
    return torch.sin(t)

def func_modulated_sin(t):
    return (1 + torch.sin(t)) * torch.sin(2*t)

def func_softplus(t):
    return torch.log(1 + torch.exp(t))

def func_swish(t):
    return t / (1 + torch.exp(-t))

def func_step(t):
    return torch.where(t > 0, torch.ones_like(t), torch.zeros_like(t))

def func_cubic(t):
    return t**3

def func_mixed(t):
    return torch.sin(t) + torch.where(t > 2, torch.ones_like(t), torch.zeros_like(t))

# --- NEW: Expert-specific synthetic functions ---
def generate_smooth_trend_data(t):
    """Data that should favor B-Spline expert - smooth polynomial trends"""
    return 0.1 * t**3 - 0.5 * t**2 + 0.3 * t + 0.2

def generate_periodic_data(t):
    """Data that should favor Fourier expert - complex periodic patterns"""
    return (torch.sin(2 * torch.pi * t / 3) + 
            0.5 * torch.cos(2 * torch.pi * t / 1.5) + 
            0.3 * torch.sin(2 * torch.pi * t / 7))

def generate_abrupt_change_data(t):
    """Data that should favor Wavelet expert - sudden shocks and discontinuities"""
    # Create shock events at different times
    shock1 = torch.where(t > 2, 1.0 * torch.exp(-(t-2)), 0.0)  # Sudden onset at t=2
    shock2 = torch.where(t > -3, -0.8 * torch.exp(-2*(t+3)), 0.0)  # Shock at t=-3
    shock3 = torch.where((t > 5) & (t < 6), 1.5, 0.0)  # Step function
    return shock1 + shock2 + shock3

def generate_localized_event_data(t):
    """Data that should favor RBF expert - localized Gaussian-like events"""
    event1 = 1.2 * torch.exp(-((t - 1)**2) / 0.5)  # Gaussian peak at t=1
    event2 = -0.8 * torch.exp(-((t + 4)**2) / 0.8)  # Negative peak at t=-4
    event3 = 0.6 * torch.exp(-((t - 6)**2) / 0.3)   # Sharp peak at t=6
    return event1 + event2 + event3

def generate_mixed_pattern_data(t):
    """Complex mixed pattern combining all expert domains"""
    smooth_trend = 0.05 * t**2  # B-spline domain
    periodic_part = 0.4 * torch.sin(2 * torch.pi * t / 4)  # Fourier domain
    shock_event = torch.where(t > 3, 1.0 * torch.exp(-(t-3)), 0.0)  # Wavelet domain
    localized_event = 0.8 * torch.exp(-((t + 2)**2) / 0.6)  # RBF domain
    return smooth_trend + periodic_part + shock_event + localized_event


def get_function_set(function_set='A'):
    """Get the specified function set"""
    if function_set == 'A':
        return {
            'y = sin(x)': func_sin,
            'y = (1+sin(x))sin(2x)': func_modulated_sin,
            'y = log(1+e^x)': func_softplus,
            'y = x/(1+e^-x)': func_swish
        }
    elif function_set == 'B':
        return {
            'y = sin(x)': func_sin,
            'y = step(x)': func_step,
            'y = x³': func_cubic,
            'y = sin(x) + step(x-2)': func_mixed
        }
    else: # function_set == 'C'
        return {
            'Smooth Trend': generate_smooth_trend_data,
            'Periodic': generate_periodic_data,
            'Abrupt Changes': generate_abrupt_change_data,
            'Mixed Patterns': generate_mixed_pattern_data
        }


def run_analysis(function_set='A'):
    """Run comprehensive K-MOTE analysis"""
    print(f"K-MOTE Mathematical Function Analysis (Set {function_set})")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if function_set == 'C':
        t = torch.linspace(-8, 8, 500, device=device) # Wider range for new functions
    else:
        t = torch.linspace(-5, 5, 500, device=device)
    
    functions = get_function_set(function_set)
    # Adjust models and plots for Set C which has 4 functions
    num_funcs = len(functions)
    model_names = ['B-SplineKAN', 'FourierKAN', 'WaveletKAN', 'K-MOTE']
    colors = {'B-SplineKAN': 'green', 'FourierKAN': 'purple', 'WaveletKAN': 'orange', 'K-MOTE': 'black'}
    
    # Store results
    all_models = {}
    all_losses = {}
    
    # Train all models on all functions
    for func_name, func in functions.items():
        print(f"\nTraining on {func_name}...")
        y_true = func(t)
        X = t.unsqueeze(-1) # Add feature dimension: (500) -> (500, 1)
        
        # Reshape for batch processing
        if X.dim() == 2:
            X = X.unsqueeze(0) # (500, 1) -> (1, 500, 1)
        if y_true.dim() == 1: 
            y_true = y_true.unsqueeze(0).unsqueeze(-1) # (500) -> (1, 500, 1)

        print("shape and value of X and y_true are as follows:")
        print(X.shape)
        print(y_true.shape)
        
        all_models[func_name] = {}
        all_losses[func_name] = {}
        
        for model_name in model_names:
            print(f"  - {model_name}...", end=' ')
            
            # Create fresh model
            models = create_models()
            model = models[model_name].to(device)
            
            # Train
            loss = train_simple_model(model, X, y_true, epochs=MAX_EPOCHS)
            print(f"Final Loss: {loss:.6f}")
            
            # Store
            all_models[func_name][model_name] = model
            all_losses[func_name][model_name] = loss
    
    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(6, num_funcs, figsize=(4 * num_funcs, 24))
    fig.suptitle(f'K-MOTE vs Individual Experts (Function Set {function_set})', fontsize=16, fontweight='bold')
    
    # Plot results
    for i, model_name in enumerate(model_names):
        for j, (func_name, func) in enumerate(functions.items()):
            ax = axes[i, j]
            
            # Generate data
            y_true = func(t)
            X_plot = t.unsqueeze(-1).unsqueeze(0) # Shape for model: (1, 500, 1)
            model = all_models[func_name][model_name]
            
            # Predict
            with torch.no_grad():
                y_pred = model(X_plot).squeeze()
            
            # Plot
            ax.plot(t.cpu().numpy(), y_true.cpu().numpy(), 'b-', linewidth=2, label='Target', alpha=0.8)
            ax.plot(t.cpu().numpy(), y_pred.cpu().numpy(), color=colors[model_name], 
                   linewidth=2, label='Prediction')
            
            # Format
            loss = all_losses[func_name][model_name]
            ax.set_title(f'{model_name}\n{func_name}\nLoss: {loss:.4f}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    # --- NEW: Plot K-MOTE Expert Contributions ---
    expert_names = ['Spline', 'Fourier', 'Wavelet']
    expert_colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] # Blue, Orange, Green

    for j, (func_name, func) in enumerate(functions.items()):
        ax = axes[4, j]
        k_mote_model = all_models[func_name]['K-MOTE']
        y_true = func(t)
        X_plot = t.unsqueeze(-1).unsqueeze(0)

        with torch.no_grad():
            # Get the base transformed time input
            t_base = k_mote_model.time_base_transform(X_plot)
            
            # Get final K-MOTE prediction
            y_pred_kmote = k_mote_model(X_plot).squeeze()

            # Plot target and final prediction
            ax.plot(t.cpu().numpy(), y_true.cpu().numpy(), 'k--', linewidth=2, label='Target', alpha=0.6)
            ax.plot(t.cpu().numpy(), y_pred_kmote.cpu().numpy(), 'r-', linewidth=2.5, label='K-MOTE Final')

            # Get and plot each expert's individual output
            for i, expert_module in enumerate(k_mote_model.experts):
                if k_mote_model.adapter_type == 'affine':
                    t_adapted = t_base * k_mote_model.expert_scales[i] + k_mote_model.expert_shifts[i]
                else:
                    t_adapted = k_mote_model.expert_adapters[i](t_base)
                
                expert_output = expert_module(t_adapted).squeeze()
                ax.plot(t.cpu().numpy(), expert_output.cpu().numpy(), color=expert_colors[i], 
                        linestyle=':', linewidth=2, label=f'Expert: {expert_names[i]}')

        ax.set_title(f'K-MOTE Expert Contributions\n{func_name}', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    # --- NEW: Plot K-MOTE Gating Weights ---
    for j, (func_name, func) in enumerate(functions.items()):
        ax = axes[5, j]
        k_mote_model = all_models[func_name]['K-MOTE']
        X_plot = t.unsqueeze(-1).unsqueeze(0)

        with torch.no_grad():
            _, gating_weights = k_mote_model(X_plot, return_weights=True)
            gating_weights = gating_weights.squeeze()

        for i in range(k_mote_model.num_experts):
            ax.plot(t.cpu().numpy(), gating_weights[:, i].cpu().numpy(), 
                    color=expert_colors[i], linewidth=2, label=f'Weight: {expert_names[i]}')
        
        ax.set_title(f'K-MOTE Gating Weights\n{func_name}', fontsize=10)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    
    plt.tight_layout(rect=[0, 0, 1, 0.97]) # Adjust layout for main title
    
    # Save
    filename = f'kmote_contributions_analysis_set_{function_set.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved to: {filename}")
    
    # Print summary
    print(f"\n=== Performance Summary (Function Set {function_set}) ===")
    print(f"{'Function':<25} {'B-Spline':<10} {'Fourier':<10} {'Wavelet':<10} {'K-MOTE':<10}")
    print("-" * 75)
    for func_name in functions.keys():
        row = f"{func_name:<25}"
        for model_name in model_names:
            loss = all_losses[func_name][model_name]
            row += f" {loss:<9.3f}"
        print(row)
    
    plt.show()
    return all_models, all_losses

if __name__ == "__main__":
    print("K-MOTE Mathematical Function Analysis")
    print("=====================================")
    print("Available function sets:")
    print("  A: LeTE comparison set (sin, modulated sin, softplus, swish)")
    print("  B: Expert-highlighting set (sin, step, cubic, mixed)")
    print("  C: Expert-specialization set (smooth, periodic, abrupt, mixed)")
    
    # Run analysis on all sets
    print("\nRunning Set A (LeTE Comparison)...")
    run_analysis('A')
    
    print("\nRunning Set B (Expert Highlighting)...")
    run_analysis('B')

    print("\nRunning Set C (Expert Specialization)...")
    run_analysis('C')
