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
MAX_EPOCHS = 2000
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
    """Run comprehensive K-MOTE analysis"""
    print(f"K-MOTE Mathematical Function Analysis (Set {function_set})")
    print("=" * 50)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = torch.linspace(-5, 5, 500, device=device)
    
    functions = get_function_set(function_set)
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
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    fig.suptitle(f'K-MOTE vs Individual Experts (Function Set {function_set})', fontsize=16, fontweight='bold')
    
    # Plot results
    for i, model_name in enumerate(model_names):
        for j, (func_name, func) in enumerate(functions.items()):
            ax = axes[i, j]
            
            # Generate data
            y_true = func(t)
            X = t.unsqueeze(-1)
            model = all_models[func_name][model_name]
            
            # Predict
            with torch.no_grad():
                y_pred = model(X).squeeze()
            
            # Plot
            ax.plot(t.cpu().numpy(), y_true.cpu().numpy(), 'b-', linewidth=2, label='Target', alpha=0.8)
            ax.plot(t.cpu().numpy(), y_pred.cpu().numpy(), color=colors[model_name], 
                   linewidth=2, label='Prediction')
            
            # Format
            loss = all_losses[func_name][model_name]
            ax.set_title(f'{model_name}\n{func_name}\nLoss: {loss:.4f}', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
    
    plt.tight_layout()
    
    # Save
    filename = f'kmote_simple_analysis_set_{function_set.lower()}.png'
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
    
    # Run analysis on both sets
    print("\nRunning Set A (LeTE Comparison)...")
    run_analysis('A')
    
    print("\nRunning Set B (Expert Highlighting)...")
    run_analysis('B')