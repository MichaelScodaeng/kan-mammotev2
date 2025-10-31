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

from models.time_encoders.k_mote import KMOTE
from analysis.LeTE import CombinedLeTE, FourierSeries
from models.time_encoders.optimized_kmote import OptimizedKMOTE

# Global training configuration
MAX_EPOCHS = 100

def create_lete_regressor(embedding_dim=64, p=1.0):
    """Create LeTE regressor wrapper"""
    class LeTERegressor(nn.Module):
        def __init__(self, embedding_dim, p):
            super().__init__()
            self.time_encoder = CombinedLeTE(dim=embedding_dim, p=p)
            self.output_head = nn.Linear(embedding_dim, 1)

        def forward(self, t):
            if t.dim() == 1: 
                t = t.unsqueeze(0)
            embeddings = self.time_encoder(t)
            output = self.output_head(embeddings)
            return output
    
    return LeTERegressor(embedding_dim, p)

def train_model_simple(model, t_data, y_true, epochs=MAX_EPOCHS, lr=0.01):
    """Simple training function with proper input handling"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # Ensure proper input shapes
    if t_data.dim() == 1: 
        t_data = t_data.unsqueeze(0)
    if y_true.dim() == 1: 
        y_true = y_true.unsqueeze(0).unsqueeze(-1)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        y_pred = model(t_data)
        loss = loss_fn(y_pred, y_true)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    WARNING: NaN/Inf loss at epoch {epoch+1}, stopping.")
            return float('inf')
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        y_pred = model(t_data)
        final_loss = loss_fn(y_pred, y_true)
    
    return final_loss.item()

def func_sin(t):
    return torch.sin(t)

def plot_comparison(t, y_true, models, results, title):
    """Plot comparison of all models"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    colors = ['red', 'purple', 'orange', 'green']
    
    for i, (model_name, model) in enumerate(models.items()):
        ax = axes[i]
        
        # Plot target
        ax.plot(t.cpu().numpy(), y_true.cpu().numpy(), 
               'b-', linewidth=2, label='Target', alpha=0.8)
        
        # Plot prediction
        model.eval()
        with torch.no_grad():
            t_input = t.unsqueeze(0) if t.dim() == 1 else t
            y_pred = model(t_input).squeeze()
            
        ax.plot(t.cpu().numpy(), y_pred.cpu().numpy(), 
               color=colors[i], linewidth=2, linestyle='--', label='Prediction')
        
        # Format
        loss = results[model_name]
        ax.set_title(f'{model_name}\nLoss: {loss:.6f}', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def run_performance_comparison():
    """Compare original vs enhanced K-MOTE vs LeTE vs OptimizedKMOTE"""
    print("=== PERFORMANCE COMPARISON: Enhanced K-MOTE vs LeTE ===")
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t = torch.linspace(-2, 2, 500, device=device)
    y_true = func_sin(t)
    
    # Create models
    models = {
        'LeTE Fourier': create_lete_regressor(embedding_dim=64, p=1.0),
        'Original K-MOTE': KMOTE(input_dim=1, output_dim=1, hidden_dim=32),
        'Enhanced K-MOTE': KMOTE(input_dim=1, output_dim=1, hidden_dim=64),  # Larger hidden_dim
        'Optimized K-MOTE': OptimizedKMOTE(input_dim=1, output_dim=1, 
                                         hidden_dim=64, fourier_dim=64,
                                         use_spline_expert=False,  # Only Fourier + Wavelet
                                         enable_checkpointing=False)  # Disable for fair comparison
    }
    
    # Move models to device
    for model in models.values():
        model.to(device)
    
    # Train and evaluate
    results = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,}")
        
        # Train
        loss = train_model_simple(model, t, y_true, epochs=MAX_EPOCHS)
        results[model_name] = loss
        print(f"  Final Loss: {loss:.6f}")
        print()
    
    # Plot results
    fig = plot_comparison(t, y_true, models, results, 
                         'Performance Comparison: Enhanced K-MOTE vs LeTE')
    plt.savefig('enhanced_kmote_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("=== RESULTS SUMMARY ===")
    print(f"{'Model':<20} {'Loss':<12} {'Parameters':<12}")
    print("-" * 50)
    for model_name, model in models.items():
        loss = results[model_name]
        params = sum(p.numel() for p in model.parameters())
        print(f"{model_name:<20} {loss:<12.6f} {params:<12,}")
    
    print()
    print("=== ANALYSIS ===")
    lete_loss = results['LeTE Fourier']
    enhanced_loss = results['Enhanced K-MOTE']
    original_loss = results['Original K-MOTE']
    optimized_loss = results['Optimized K-MOTE']
    
    print(f"Original K-MOTE vs LeTE: {original_loss/lete_loss:.1f}x worse")
    print(f"Enhanced K-MOTE vs LeTE: {enhanced_loss/lete_loss:.1f}x worse")
    print(f"Optimized K-MOTE vs LeTE: {optimized_loss/lete_loss:.1f}x worse")
    print(f"Enhancement improvement: {original_loss/enhanced_loss:.1f}x better")
    
    return results

if __name__ == "__main__":
    run_performance_comparison()