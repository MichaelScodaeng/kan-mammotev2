import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Add the parent directory to Python path to import models
# Note: Ensure 'k_mote.py' is in a 'models/time_encoders' subdirectory relative to this script's parent.
# Or, more simply, ensure 'k_mote.py' is accessible in the same directory structure as before.
try:
    from models.time_encoders.k_mote import KMOTE, SplineKANLayer, FourierKANLayer, WaveletKANLayer
except ImportError:
    print("Error: Could not import K-MOTE. Make sure 'k_mote.py' is in the correct path.")
    sys.exit(1)

# Create output directory for saving figures if it doesn't exist
os.makedirs('analysis_figures_math', exist_ok=True)


# --- Helper Functions (reused from previous analysis) ---

class SingleExpertModel(nn.Module):
    """Wrapper for individual experts to make them compatible with analysis"""
    def __init__(self, expert_class, **kwargs):
        super().__init__()
        self.expert = expert_class(input_dim=1, output_dim=1, **kwargs)

    def forward(self, x):
        return self.expert(x)

def train_model_with_loss_return(model, t_data, y_true, epochs=5000, lr=2e-4):
    """Train model and return final loss"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5) # Added weight decay for stability
    loss_fn = nn.MSELoss()
    
    # Reshape data to (Batch, Seq_len, Dim)
    if t_data.dim() == 1: t_data = t_data.unsqueeze(0).unsqueeze(-1)
    if y_true.dim() == 1: y_true = y_true.unsqueeze(0).unsqueeze(-1)

    for epoch in range(epochs):
        model.train()
        y_pred = model(t_data)
        
        loss = loss_fn(y_pred, y_true)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    WARNING: NaN/Inf loss at epoch {epoch+1}, stopping.")
            return float('inf')
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
    final_loss = loss.item()
    print(f"    Final Loss: {final_loss:.6f}")
    return final_loss

def plot_fit(ax, t, y_true, model, title):
    """Function to plot the results of the model fit"""
    model.eval()
    with torch.no_grad():
        t_input = t.unsqueeze(0).unsqueeze(-1) # Reshape for model
        y_pred = model(t_input).squeeze()
        
    ax.plot(t.cpu().numpy(), y_true.cpu().numpy(), label='Ground Truth', linewidth=3, alpha=0.7, color='blue')
    ax.plot(t.cpu().numpy(), y_pred.cpu().numpy(), label='Model Fit', linestyle='--', color='red', linewidth=2)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# --- Mathematical Function Definitions ---

def func_sin(t):
    return torch.sin(t)

def func_modulated_sin(t):
    return (1 + torch.sin(t)) * torch.sin(2*t)

def func_softplus(t):
    return torch.log(1 + torch.exp(t))

def func_swish(t):
    # y = x / (1 + e^-x) = x * sigmoid(x)
    return t * torch.sigmoid(t)


# --- Main Analysis Script ---

def run_math_function_analysis():
    print("--- Starting K-MOTE Analysis on Mathematical Functions ---")
    
    # 1. Define the input range and target functions
    t = torch.linspace(-5, 5, 500)
    target_functions = {
        "y = sin(x)": func_sin(t),
        "y = (1+sin(x))sin(2x)": func_modulated_sin(t),
        "y = log(1+e^x) [Softplus]": func_softplus(t),
        "y = x / (1+e^-x) [Swish]": func_swish(t)
    }

    # 2. Define the models to be tested, using the corrected SplineKANLayer
    models_to_test = {
        "B-Spline Expert": lambda: SingleExpertModel(SplineKANLayer, basis_function='b_spline'),
        "Fourier Expert": lambda: SingleExpertModel(FourierKANLayer),
        "Wavelet Expert": lambda: SingleExpertModel(WaveletKANLayer, wavelet_type='shock'),
        "RBF Expert": lambda: SingleExpertModel(SplineKANLayer, basis_function='rbf'),
        "Full K-MOTE": lambda: KMOTE(input_dim=1, output_dim=1, wavelet_type='shock')
    }
    
    # 3. Loop through each function, train each model, and plot the results
    for func_name, y_true in target_functions.items():
        print(f"\n[INFO] Testing on function: {func_name}")
        fig, axes = plt.subplots(1, len(models_to_test), figsize=(28, 5))
        fig.suptitle(f'Expert Performance on: {func_name}', fontsize=18, y=1.02)
        
        # Normalize the target function for more stable training if its magnitude is large
        y_mean = y_true.mean()
        y_std = y_true.std()
        y_norm = (y_true - y_mean) / y_std
        
        for i, (model_name, model_factory) in enumerate(models_to_test.items()):
            print(f"  - Training {model_name}...")
            
            # Create a fresh instance of the model
            model = model_factory()
            
            # Train the model on the normalized data
            final_loss = train_model_with_loss_return(model, t, y_norm)
            
            # For plotting, we need to un-normalize the model's predictions
            class UnnormalizedModel(nn.Module):
                def __init__(self, trained_model, mean, std):
                    super().__init__()
                    self.model = trained_model
                    self.mean = mean
                    self.std = std
                def forward(self, x):
                    # Model was trained on normalized data, so we un-normalize its output
                    return self.model(x) * self.std + self.mean

            unnorm_model = UnnormalizedModel(model, y_mean, y_std)
            
            # Plot the fit against the original (un-normalized) data
            plot_fit(axes[i], t, y_true, unnorm_model, f"{model_name}\nLoss: {final_loss:.4f}")
            
        plt.tight_layout()
        
        # Save figure with a safe filename
        safe_filename = func_name.replace(' ', '_').replace('=', '').replace('/', '_').replace('[','').replace(']','')
        plt.savefig(f'analysis_figures_math/math_perf_{safe_filename}.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    # Ensure the corrected k_mote.py is accessible before running
    run_math_function_analysis()
    print("\n✨ Mathematical function analysis complete!")
    print("📁 Figures saved in 'analysis_figures_math' directory.")