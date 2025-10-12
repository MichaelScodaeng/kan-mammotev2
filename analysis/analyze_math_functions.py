import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm
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

def train_model_with_loss_return(model, t_data, y_true, max_epochs=5000, lr=5e-4, 
                               patience=200, min_delta=1e-6):
    """Train model until convergence and return final loss and training info"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    # Reshape data to (Batch, Seq_len, Dim)
    if t_data.dim() == 1: t_data = t_data.unsqueeze(0).unsqueeze(-1)
    if y_true.dim() == 1: y_true = y_true.unsqueeze(0).unsqueeze(-1)

    best_loss = float('inf')
    patience_counter = 0
    loss_history = []
    
    # Use tqdm for progress bar
    with tqdm(range(max_epochs), desc="Training", leave=False) as pbar:
        for epoch in pbar:
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
            
            current_loss = loss.item()
            loss_history.append(current_loss)
            
            # Check for improvement
            if current_loss < best_loss - min_delta:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{current_loss:.6f}',
                'Best': f'{best_loss:.6f}',
                'Patience': f'{patience_counter}/{patience}'
            })
            
            # Early stopping
            if patience_counter >= patience:
                print(f"    Converged at epoch {epoch+1} (patience reached)")
                break
                
            # Additional convergence check: if loss is very stable
            if epoch > 100:
                recent_losses = loss_history[-50:]
                if max(recent_losses) - min(recent_losses) < min_delta:
                    print(f"    Converged at epoch {epoch+1} (loss stabilized)")
                    break
        
    final_loss = best_loss
    print(f"    Final Loss: {final_loss:.6f} (converged in {epoch+1} epochs)")
    return final_loss

def plot_fit(ax, t, y_true, model, title, show_legend=True):
    """Function to plot the results of the model fit in compact style"""
    model.eval()
    with torch.no_grad():
        t_input = t.unsqueeze(0).unsqueeze(-1) # Reshape for model
        y_pred = model(t_input).squeeze()
        
    # Plot with compact style
    ax.plot(t.cpu().numpy(), y_true.cpu().numpy(), 'b-', linewidth=2, label='Target Function')
    ax.plot(t.cpu().numpy(), y_pred.cpu().numpy(), 'r--', linewidth=2, label='Learned Function')
    
    ax.set_title(title, fontsize=10, pad=5)
    if show_legend:
        ax.legend(fontsize=8, loc='best')
    
    # Clean styling
    ax.grid(True, linestyle=':', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=8)
    
    # Set axis limits for better view
    ax.set_xlim(t.min().item(), t.max().item())
    
    return ax


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

    # 2. Define the models to be tested
    models_to_test = {
        "B-Spline Expert": lambda: SingleExpertModel(SplineKANLayer, basis_function='b_spline'),
        "Fourier Expert": lambda: SingleExpertModel(FourierKANLayer),
        "Wavelet Expert": lambda: SingleExpertModel(WaveletKANLayer, wavelet_type='shock'),
        "Full K-MOTE": lambda: KMOTE(input_dim=1, output_dim=1, wavelet_type='shock')
    }
    
    # Storage for CSV results
    results_data = []
    
    # Create a single figure with subplots
    n_funcs = len(target_functions)
    n_models = len(models_to_test)
    fig, axes = plt.subplots(n_models, n_funcs, figsize=(4*n_funcs, 3*n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)
    if n_funcs == 1:
        axes = axes.reshape(-1, 1)
    
    # Set row labels (model names)
    model_names = list(models_to_test.keys())
    for i, model_name in enumerate(model_names):
        axes[i, 0].set_ylabel(model_name, fontsize=12, fontweight='bold')
    
    # Set column titles (function names)
    func_names = list(target_functions.keys())
    for j, func_name in enumerate(func_names):
        axes[0, j].set_title(func_name, fontsize=12, fontweight='bold', pad=20)
    
    # 3. Loop through each function and model combination
    for j, (func_name, y_true) in enumerate(target_functions.items()):
        print(f"\n[INFO] Testing on function: {func_name}")
        
        # Normalize the target function for more stable training
        y_mean = y_true.mean()
        y_std = y_true.std()
        y_norm = (y_true - y_mean) / y_std
        
        for i, (model_name, model_factory) in enumerate(models_to_test.items()):
            print(f"  - Training {model_name}...")
            
            # Create a fresh instance of the model
            model = model_factory()
            
            # Train the model on the normalized data
            final_loss = train_model_with_loss_return(model, t, y_norm)
            
            # Store results for CSV
            results_data.append({
                'function': func_name,
                'model': model_name,
                'final_loss': final_loss
            })
            
            # For plotting, we need to un-normalize the model's predictions
            class UnnormalizedModel(nn.Module):
                def __init__(self, trained_model, mean, std):
                    super().__init__()
                    self.model = trained_model
                    self.mean = mean
                    self.std = std
                def forward(self, x):
                    return self.model(x) * self.std + self.mean

            unnorm_model = UnnormalizedModel(model, y_mean, y_std)
            
            # Plot the fit - no title needed as we have row/column labels
            plot_fit(axes[i, j], t, y_true, unnorm_model, "", show_legend=(i==0 and j==0))
    
    # Create a custom legend at the bottom
    legend_elements = [
        plt.Line2D([0], [0], color='blue', lw=2, label='Target Function'),
        plt.Line2D([0], [0], color='red', lw=2, linestyle='--', label='Learned Function')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=2, fontsize=12, 
               bbox_to_anchor=(0.5, -0.02))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)  # Make room for legend
    
    # Save the main comparison figure
    plt.savefig('analysis_figures_math/math_functions_comparison_compact.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Save results to CSV
    print("\n[INFO] Saving results to CSV...")
    
    # Convert results to structured format for CSV
    csv_data = []
    header = ['Function', 'B-Spline Expert', 'Fourier Expert', 'Wavelet Expert', 'Full K-MOTE']
    
    for func_name in func_names:
        row = [func_name]
        for model_name in model_names:
            # Find the loss for this function-model combination
            loss = next(r['final_loss'] for r in results_data 
                       if r['function'] == func_name and r['model'] == model_name)
            row.append(f"{loss:.6f}")
        csv_data.append(row)
    
    # Save to CSV using numpy
    csv_file = 'analysis_figures_math/math_functions_results.csv'
    with open(csv_file, 'w') as f:
        # Write header
        f.write(','.join(header) + '\n')
        # Write data rows
        for row in csv_data:
            f.write(','.join(row) + '\n')
    
    print(f"✅ Results saved to: {csv_file}")
    
    # Also save raw results for detailed analysis
    raw_csv_file = 'analysis_figures_math/math_functions_raw_results.csv'
    with open(raw_csv_file, 'w') as f:
        f.write('Function,Model,Final_Loss\n')
        for result in results_data:
            f.write(f"{result['function']},{result['model']},{result['final_loss']:.6f}\n")
    
    print(f"✅ Raw results saved to: {raw_csv_file}")
    
    return results_data

if __name__ == '__main__':
    # Ensure the corrected k_mote.py is accessible before running
    results = run_math_function_analysis()
    print("\n✨ Mathematical function analysis complete!")
    print("📁 Figures saved in 'analysis_figures_math' directory.")
    print("📊 Results exported to CSV files for further analysis.")