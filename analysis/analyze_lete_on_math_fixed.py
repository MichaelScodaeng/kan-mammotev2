import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Global training configuration
MAX_EPOCHS = 2000

# Import the CombinedLeTE model from the LeTE.py file
try:
    from LeTE import CombinedLeTE
except ImportError:
    print("Error: Could not import CombinedLeTE. Make sure 'LeTE.py' is in the same directory.")
    sys.exit(1)

# Import FTE (Time2Vec) implementation
try:
    # Add the project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    from models.time_encoders.time2vec_encoder import Time2VecEncoder
except ImportError:
    print("Error: Could not import Time2VecEncoder. Make sure the path is correct.")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(__file__)}")
    print(f"Project root: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")
    sys.exit(1)

# Create output directory for saving figures
os.makedirs('analysis_figures_lete', exist_ok=True)

# --- FTE (Time2Vec) Regressor Wrapper ---

class FTERegressor(nn.Module):
    """
    A wrapper to use Time2Vec (FTE) for scalar regression task.
    This represents the "Fixed Time Encoding" baseline that should fail.
    """
    def __init__(self, embedding_dim=64):
        super().__init__()
        # FTE uses fixed non-linear transformation functions
        self.time_encoder = Time2VecEncoder(time_dim=embedding_dim, activation='sin')
        
        # A linear head to project the embedding to a single output value
        self.output_head = nn.Linear(embedding_dim, 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): Shape (batch_size, seq_len) - Input time values.
        Returns:
            torch.Tensor: Shape (batch_size, seq_len, 1) - Predicted output values.
        """
        # 1. Get time embeddings from FTE (Time2Vec)
        # Input shape: (batch_size, seq_len) -> Output shape: (batch_size, seq_len, embedding_dim)
        embeddings = self.time_encoder(t_rel=t.unsqueeze(-1))
        
        # 2. Project embeddings to the final output
        # Input shape: (batch_size, seq_len, embedding_dim) -> Output shape: (batch_size, seq_len, 1)
        output = self.output_head(embeddings)
        
        return output

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

# --- Helper Functions ---

def train_model_with_loss_return(model, t_data, y_true, epochs=MAX_EPOCHS, lr=2e-4):
    """Train model and return final loss"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()
    
    # Ensure proper input shapes
    if t_data.dim() == 1: 
        t_data = t_data.unsqueeze(0)
    if y_true.dim() == 1: 
        y_true = y_true.unsqueeze(0).unsqueeze(-1) # Target shape: (batch, seq_len, 1)

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

def plot_fit(ax, t, y_true, model, title, line_color='red', linestyle='--'):
    """Function to plot the results of the model fit"""
    model.eval()
    with torch.no_grad():
        t_input = t.unsqueeze(0) if t.dim() == 1 else t
        y_pred = model(t_input).squeeze()
        
    # Plot ground truth (always blue solid line)
    ax.plot(t.cpu().numpy(), y_true.cpu().numpy(), 
           label='Target Function', linewidth=2, alpha=0.8, color='blue')
    
    # Plot model prediction with specified style
    ax.plot(t.cpu().numpy(), y_pred.cpu().numpy(), 
           label='Learned Function', linestyle=linestyle, color=line_color, linewidth=2)
    
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# --- Mathematical Function Definitions ---

def func_sin(t):
    """y = sin(x)"""
    return torch.sin(t)

def func_modulated_sin(t):
    """y = (1 + sin(x))sin(2x)"""
    return (1 + torch.sin(t)) * torch.sin(2*t)

def func_softplus(t):
    """y = log(1 + e^x) [Softplus]"""
    return torch.log(1 + torch.exp(t))

def func_swish(t):
    """y = x / (1 + e^-x) [Swish]"""
    return t * torch.sigmoid(t)

# --- Main Analysis Script (Replicating Figure 13) ---

def run_lete_analysis_on_math():
    print("--- Replicating Figure 13: FTE vs Fourier-based LeTE vs Spline-based LeTE ---")
    
    # 1. Define input range and target functions (exactly as in paper)
    t = torch.linspace(-5, 5, 500)
    target_functions = [
        ("y = sin(x)", func_sin(t)),
        ("y = (1+sin(x))sin(2x)", func_modulated_sin(t)),
        ("y = log(1+e^x)", func_softplus(t)),
        ("y = x/(1+e^-x)", func_swish(t))
    ]
    
    # 2. Define the encoding methods (exactly as in paper)
    encoding_methods = [
        ("FTE", "fte", 'red'),
        ("Fourier-based LeTE", "fourier_lete", 'purple'),  
        ("Spline-based LeTE", "spline_lete", 'orange')
    ]
    
    # 3. Create the figure matching paper layout: 3 rows (methods) × 4 columns (functions)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    fig.suptitle('FTE, Fourier-based LeTE and Spline-based LeTE fitting different functions', fontsize=16, y=0.95)
    
    # Add method labels on the left
    method_labels = ["FTE", "Fourier-based\nLeTE", "Spline-based\nLeTE"]
    for i, label in enumerate(method_labels):
        fig.text(0.02, 0.75 - i*0.28, label, rotation=90, va='center', ha='center', fontsize=12, weight='bold')
    
    # 4. Loop through each method and function combination
    for method_idx, (method_name, method_key, line_color) in enumerate(encoding_methods):
        print(f"\n[INFO] Testing {method_name}...")
        
        for func_idx, (func_name, y_true) in enumerate(target_functions):
            print(f"  - Training {method_name} on {func_name}...")
            
            # Normalize target for stable training
            y_mean, y_std = y_true.mean(), y_true.std()
            y_norm = (y_true - y_mean) / y_std
            
            # Create model based on method
            if method_key == "fte":
                model = FTERegressor(embedding_dim=64)
            elif method_key == "fourier_lete":
                model = LeTERegressor(embedding_dim=64, p=1.0)  # Pure Fourier (p=1.0)
            elif method_key == "spline_lete":
                model = LeTERegressor(embedding_dim=64, p=0.0)  # Pure Spline (p=0.0)
            
            # Train the model
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
            
            # Plot in the correct subplot
            ax = axes[method_idx, func_idx]
            
            # Create title - function name on top row, just loss for others
            if method_idx == 0:
                title = f"{func_name}"
            else:
                title = ""
            
            plot_fit(ax, t, y_true, unnorm_model, title, line_color=line_color)
            
            # Remove legend for cleaner look (except first subplot)
            if not (method_idx == 0 and func_idx == 0):
                ax.legend().set_visible(False)
    
    # Add a single legend at the bottom
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Target Function'),
        plt.Line2D([0], [0], color='red', linewidth=2, linestyle='--', label='Learned Function')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12, bbox_to_anchor=(0.5, 0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.08, top=0.92)
    
    # Save the figure
    plt.savefig('analysis_figures_lete/figure13_replication.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("KEY FINDINGS (as per paper):")
    print("="*70)
    print("• FTE (Time2Vec) fails to capture complex patterns due to fixed transformation functions")
    print("• Fourier-based LeTE successfully captures periodic patterns (sin, modulated sin)")
    print("• Spline-based LeTE successfully captures non-periodic patterns (softplus, swish)")
    print("• Both LeTE variants demonstrate superior pattern modeling compared to FTE")
    print("• This shows LeTE's capability to model complex patterns effectively")

if __name__ == '__main__':
    run_lete_analysis_on_math()
    print("\n✨ Figure 13 replication complete!")
    print("📁 Figure saved as 'analysis_figures_lete/figure13_replication.png'")
    print("\n🔍 The key insight is that FTE fails while LeTE variants succeed,")
    print("   demonstrating LeTE's superior pattern modeling capabilities.")