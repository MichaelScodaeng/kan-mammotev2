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

# Import SequentialKMOTE and its experts
from models.time_encoders.sequential_kmote import (
    SequentialKMOTE, LeTESpline, EfficientFourierKAN, EnhancedWaveletKAN
)

# Global training configuration
MAX_EPOCHS = 100

class SimpleWrapper(nn.Module):
    """Simple wrapper for KAN layers, ensuring output is always (B, S, 1)"""
    def __init__(self, kan_layer, input_dim, output_dim=1):
        super().__init__()
        self.kan = kan_layer
        # LeTESpline outputs hidden_dim, so we need a projection layer
        if isinstance(kan_layer, LeTESpline):
            self.projection = nn.Linear(input_dim, output_dim)
        else:
            self.projection = nn.Identity()
        
    def forward(self, x):
        # The experts expect (B, S, D)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        output = self.kan(x)
        output = self.projection(output) # Apply projection
        return output

def create_models():
    """Create simple model instances based on SequentialKMOTE's experts"""
    hidden_dim = 64
    output_dim = 1
    
    # Instantiate a template model to access its time_transform. This ensures
    # all models start with the same initialized projection.
    template_model = SequentialKMOTE(input_dim=1, output_dim=output_dim, hidden_dim=hidden_dim)
    time_transform = template_model.time_transform

    # Define the final projection head that all experts need
    projection_head = nn.Linear(hidden_dim, output_dim)

    # --- FIX: Build each expert as a proper sequence of modules ---
    # The structure is: TimeTransform -> ExpertCore -> ProjectionHead
    models = {
        'LeTE_Spline': nn.Sequential(
            time_transform,
            LeTESpline(dim_spline=hidden_dim, grid_size_spline=5, order_spline=3),
            projection_head
        ),
        'EfficientFourier': nn.Sequential(
            time_transform,
            # This expert's output_dim should be hidden_dim
            EfficientFourierKAN(input_dim=hidden_dim, output_dim=hidden_dim, intermediate_dim=64, n_harmonics=8),
            projection_head
        ),
        'EnhancedWavelet': nn.Sequential(
            time_transform,
            # This expert's output_dim should be hidden_dim
            EnhancedWaveletKAN(input_dim=hidden_dim, output_dim=hidden_dim, n_wavelets=8, wavelet_type='shock'),
            projection_head
        ),
        # The full model already contains all these components internally
        'SequentialK-MOTE': template_model 
    }
        
    return models

def train_simple_model(model, X, y, epochs=50):
    """Train a model with simple training loop"""
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.MSELoss()
    if X.dim() == 2:
        # Assuming input is (Seq, Features), add a batch dimension
        X = X.unsqueeze(0)  # (500, 1) -> (1, 500, 1)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred.squeeze(), y)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        final_pred = model(X)
        final_loss = criterion(final_pred.squeeze(), y)
    
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

def get_function_set(function_set='A'):
    """Get the specified function set"""
    if function_set == 'A':
        return {
            'y = sin(x)': func_sin,
            'y = (1+sin(x))sin(2x)': func_modulated_sin,
            'y = log(1+e^x)': func_softplus,
            'y = x/(1+e^-x)': func_swish
        }
    else:  # function_set == 'B'
        return {
            'y = sin(x)': func_sin,
            'y = step(x)': func_step,
            'y = x³': func_cubic,
            'y = sin(x) + step(x-2)': func_mixed
        }

def run_analysis(function_set='A'):
    """Run comprehensive Sequential K-MOTE analysis"""
    print(f"Sequential K-MOTE Mathematical Function Analysis (Set {function_set})")
    print("=" * 60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = torch.linspace(-2, 2, 500, device=device)
    
    functions = get_function_set(function_set)
    model_names = ['LeTE_Spline', 'EfficientFourier', 'EnhancedWavelet', 'SequentialK-MOTE']
    colors = {'LeTE_Spline': 'green', 'EfficientFourier': 'purple', 'EnhancedWavelet': 'orange', 'SequentialK-MOTE': 'black'}
    
    # Store results
    all_models = {}
    all_losses = {}
    
    # Train all models on all functions
    for func_name, func in functions.items():
        print(f"\nTraining on {func_name}...")
        y_true = func(t)
        X = t.unsqueeze(-1)  # Shape: (500, 1)
        
        all_models[func_name] = {}
        all_losses[func_name] = {}
        
        # Create a fresh set of models for each function to ensure fair comparison
        models = create_models()
        
        for model_name in model_names:
            print(f"  - {model_name}...", end=' ')
            
            model = models[model_name].to(device)
            
            # Train
            loss = train_simple_model(model, X, y_true, epochs=MAX_EPOCHS)
            print(f"Final Loss: {loss:.6f}")
            
            # Store
            all_models[func_name][model_name] = model
            all_losses[func_name][model_name] = loss
    
    # Create visualization
    print("\nCreating visualization...")
    fig, axes = plt.subplots(5, 4, figsize=(16, 20))
    fig.suptitle(f'Sequential K-MOTE Expert Decomposition Analysis (Function Set {function_set})', fontsize=16, fontweight='bold')
    
    # Colors for experts in K-MOTE
    expert_colors = ['green', 'purple', 'orange']
    expert_names = ['LeTE_Spline Expert', 'EfficientFourier Expert', 'EnhancedWavelet Expert']
    
    # Plot results
    for j, (func_name, func) in enumerate(functions.items()):
        y_true = func(t)
        X = t.unsqueeze(-1)
        
        # Plot individual experts (rows 0-2)
        for i, model_name in enumerate(model_names[:3]):
            ax = axes[i, j]
            model = all_models[func_name][model_name]
            
            with torch.no_grad():
                y_pred = model(X).squeeze()
            
            ax.plot(t.cpu().numpy(), y_true.cpu().numpy(), 'b-', linewidth=2, label='Target', alpha=0.8)
            ax.plot(t.cpu().numpy(), y_pred.cpu().numpy(), color=colors[model_name], 
                   linewidth=2, label='Prediction')
            
            loss = all_losses[func_name][model_name]
            ax.set_title(f'{model_name}\n{func_name}\nLoss: {loss:.4f}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Plot K-MOTE with expert decomposition (row 3)
        ax_kmote = axes[3, j]
        kmote_model = all_models[func_name]['SequentialK-MOTE']
        
        with torch.no_grad():
            final_output, expert_outputs, gating_weights = kmote_model(X, return_expert_info=True)
            
            ax_kmote.plot(t.cpu().numpy(), y_true.cpu().numpy(), 'b-', linewidth=3, label='Target', alpha=0.8)
            
            for i, expert_output in enumerate(expert_outputs):
                expert_pred = expert_output.squeeze()
                ax_kmote.plot(t.cpu().numpy(), expert_pred.cpu().numpy(), 
                             color=expert_colors[i], linewidth=1.5, alpha=0.7,
                             linestyle='--', label=expert_names[i])
            
            final_pred = final_output.squeeze()
            ax_kmote.plot(t.cpu().numpy(), final_pred.cpu().numpy(), 
                         color='black', linewidth=3, label='K-MOTE Combined')
        
        loss = all_losses[func_name]['SequentialK-MOTE']
        ax_kmote.set_title(f'Sequential K-MOTE Expert Decomposition\n{func_name}\nLoss: {loss:.4f}', fontsize=10)
        ax_kmote.grid(True, alpha=0.3)
        ax_kmote.legend(fontsize=8)
        
        # Plot actual gating weights from K-MOTE (row 4)
        ax_gating = axes[4, j]
        
        gating_weights_squeezed = gating_weights.squeeze(1)
        for i in range(gating_weights_squeezed.shape[-1]):
            weight = gating_weights_squeezed[:, i].cpu().numpy()
            ax_gating.plot(t.cpu().numpy(), weight, 
                          color=expert_colors[i], linewidth=2, label=expert_names[i])
        
        ax_gating.set_title(f'Sequential K-MOTE Gating Weights\n{func_name}', fontsize=10)
        ax_gating.set_ylabel('Gating Weight', fontsize=9)
        ax_gating.set_xlabel('Input (x)', fontsize=9)
        ax_gating.grid(True, alpha=0.3)
        ax_gating.legend(fontsize=8)
        ax_gating.set_ylim([0, 1])
    
    plt.tight_layout()
    
    filename = f'sequential_kmote_analysis_set_{function_set.lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"\nSaved to: {filename}")
    
    print(f"\n=== Performance Summary (Function Set {function_set}) ===")
    print(f"{'Function':<25} {'Spline':<10} {'Fourier':<10} {'Wavelet':<10} {'SeqK-MOTE':<10}")
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
    print("Sequential K-MOTE Mathematical Function Analysis")
    print("================================================")
    print("Available function sets:")
    print("  A: LeTE comparison set (sin, modulated sin, softplus, swish)")
    print("  B: Expert-highlighting set (sin, step, cubic, mixed)")
    
    # Run analysis on both sets
    print("\nRunning Set A (LeTE Comparison)...")
    run_analysis('A')
    
    print("\nRunning Set B (Expert Highlighting)...")
    run_analysis('B')
