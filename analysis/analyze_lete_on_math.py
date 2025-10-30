import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Global training configuration
MAX_EPOCHS = 4000

# Import the CombinedLeTE model from the LeTE.py file
# Ensure LeTE.py is in the same directory or accessible via PYTHONPATH
try:
    
    from LeTE import CombinedLeTE
except ImportError:
    print("Error: Could not import CombinedLeTE. Make sure 'LeTE.py' is in the same directory.")
    sys.exit(1)

# Create output directory for saving figures
os.makedirs('analysis_figures_lete', exist_ok=True)

# --- LeTE Regressor Wrapper ---

class LeTERegressor(nn.Module):
    """
    A wrapper to use CombinedLeTE for a scalar regression task.
    It takes a time value 't' and predicts a single scalar output 'y'.
    """
    def __init__(self, embedding_dim=64, p=0.5):
        super().__init__()
        # LeTE creates a rich, high-dimensional embedding from a scalar time input
        self.time_encoder = CombinedLeTE(dim=embedding_dim, p=p)
        
        # A linear head to project the embedding to a single output value
        self.output_head = nn.Linear(embedding_dim, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): Shape (batch_size, seq_len) - Input time values.
        Returns:
            torch.Tensor: Shape (batch_size, seq_len, 1) - Predicted output values.
        """
        # 1. Get time embeddings from LeTE
        # Input shape: (batch_size, seq_len) -> Output shape: (batch_size, seq_len, embedding_dim)
        embeddings = self.time_encoder(t)
        
        # 2. Project embeddings to the final output
        # Input shape: (batch_size, seq_len, embedding_dim) -> Output shape: (batch_size, seq_len, 1)
        output = self.output_head(embeddings)
        
        return output

# --- Helper Functions (similar to previous analysis) ---

def train_model_with_loss_return(model, t_data, y_true, epochs=MAX_EPOCHS, lr=2e-4):
    """Train model and return final loss"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()
    
    # LeTE expects 2D input: (batch_size, seq_len)
    if t_data.dim() == 1: t_data = t_data.unsqueeze(0)
    if y_true.dim() == 1: y_true = y_true.unsqueeze(0).unsqueeze(-1) # Target shape: (batch, seq_len, 1)

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
        t_input = t.unsqueeze(0) # Reshape for model: (1, seq_len)
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
    return t * torch.sigmoid(t)

# --- Main Analysis Script ---

def run_lete_analysis_on_math():
    print("--- Starting LeTE Analysis on Mathematical Functions ---")
    
    # 1. Define input range and target functions
    t = torch.linspace(-5, 5, 500)
    target_functions = {
        "y = sin(x)": func_sin(t),
        "y = (1+sin(x))sin(2x)": func_modulated_sin(t),
        "y = log(1+e^x) [Softplus]": func_softplus(t),
        "y = x / (1+e^-x) [Swish]": func_swish(t)
    }

    # 2. Define the LeTE configurations to test
    lete_configs = {
        "Pure Spline LeTE (p=0.0)": 0.0,
        "Combined LeTE (p=0.5)": 0.5,
        "Pure Fourier LeTE (p=1.0)": 1.0,
    }
    
    # 3. Loop through each function, train each LeTE config, and plot
    for func_name, y_true in target_functions.items():
        print(f"\n[INFO] Testing LeTE on function: {func_name}")
        fig, axes = plt.subplots(1, len(lete_configs), figsize=(21, 5))
        fig.suptitle(f'LeTE Performance on: {func_name}', fontsize=18, y=1.02)
        
        # Normalize target for stable training
        y_mean, y_std = y_true.mean(), y_true.std()
        y_norm = (y_true - y_mean) / y_std
        
        for i, (model_name, p_value) in enumerate(lete_configs.items()):
            print(f"  - Training {model_name}...")
            
            model = LeTERegressor(embedding_dim=64, p=p_value)
            
            final_loss = train_model_with_loss_return(model, t, y_norm)
            
            # Create a wrapper to un-normalize model output for plotting
            class UnnormalizedModel(nn.Module):
                def __init__(self, trained_model, mean, std):
                    super().__init__()
                    self.model = trained_model
                    self.mean = mean
                    self.std = std
                def forward(self, x):
                    return self.model(x) * self.std + self.mean

            unnorm_model = UnnormalizedModel(model, y_mean, y_std)
            
            plot_fit(axes[i], t, y_true, unnorm_model, f"{model_name}\nLoss: {final_loss:.4f}")
            
        plt.tight_layout()
        
        safe_filename = func_name.replace(' ', '_').replace('=', '').replace('/', '_').replace('[','').replace(']','')
        plt.savefig(f'analysis_figures_lete/lete_perf_{safe_filename}.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == '__main__':
    run_lete_analysis_on_math()
    print("\n✨ LeTE mathematical function analysis complete!")
    print("📁 Figures saved in 'analysis_figures_lete' directory.")