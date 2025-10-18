import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
from tqdm import tqdm

# Global training configuration
MAX_EPOCHS = 5000

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

def train_model_with_loss_return(model, t_data, y_true, max_epochs=MAX_EPOCHS, lr=5e-4, 
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

def save_prediction_data(t, y_true, model, func_name, model_name):
    """Save actual vs prediction data to CSV for later visualization"""
    model.eval()
    with torch.no_grad():
        t_input = t.unsqueeze(0).unsqueeze(-1)
        y_pred = model(t_input).squeeze()
    
    # Create safe filenames
    safe_func_name = func_name.replace(' ', '_').replace('=', '').replace('/', '_').replace('[','').replace(']','').replace('(','').replace(')','')
    safe_model_name = model_name.replace(' ', '_').replace('-', '_')
    
    # Create prediction data directory
    pred_dir = 'analysis_figures_math/prediction_data'
    os.makedirs(pred_dir, exist_ok=True)
    
    # Save data
    filename = f'{pred_dir}/{safe_func_name}_{safe_model_name}_predictions.csv'
    
    data = {
        'time': t.cpu().numpy(),
        'actual': y_true.cpu().numpy(),
        'predicted': y_pred.cpu().numpy()
    }
    
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    
    return filename

def analyze_gating_weights(t, model, func_name):
    """Analyze and save gating weights for K-MOTE models"""
    if not hasattr(model, 'gating_network'):
        return None
    
    model.eval()
    with torch.no_grad():
        t_input = t.unsqueeze(0).unsqueeze(-1)
        y_pred, gating_weights = model(t_input, return_weights=True)
        
        if gating_weights.dim() > 2:
            gating_weights = gating_weights.squeeze()
        gating_weights_np = gating_weights.cpu().numpy()
    
    # Analyze expert usage
    expert_names = ['B-Spline', 'Fourier', 'Wavelet', 'RBF']
    avg_weights = np.mean(gating_weights_np, axis=0)
    
    print(f"    🔍 Expert Usage for {func_name}:")
    for i, (name, weight) in enumerate(zip(expert_names, avg_weights)):
        print(f"      {name}: {weight:.3f} ({weight*100:.1f}%)")
    
    # Find dominant expert regions
    dominant_expert_idx = np.argmax(gating_weights_np, axis=1)
    dominant_expert_counts = np.bincount(dominant_expert_idx, minlength=4)
    
    print(f"    📈 Dominant Expert Regions:")
    for i, (name, count) in enumerate(zip(expert_names, dominant_expert_counts)):
        percentage = count / len(t) * 100
        print(f"      {name}: {count}/{len(t)} points ({percentage:.1f}%)")
    
    # Save gating weights to CSV
    safe_func_name = func_name.replace(' ', '_').replace('=', '').replace('/', '_').replace('[','').replace(']','').replace('(','').replace(')','')
    gating_dir = 'analysis_figures_math/gating_data'
    os.makedirs(gating_dir, exist_ok=True)
    
    gating_filename = f'{gating_dir}/{safe_func_name}_gating_weights.csv'
    gating_data = {
        'time': t.cpu().numpy(),
        'B_Spline_weight': gating_weights_np[:, 0],
        'Fourier_weight': gating_weights_np[:, 1], 
        'Wavelet_weight': gating_weights_np[:, 2],
        'RBF_weight': gating_weights_np[:, 3]
    }
    
    df_gating = pd.DataFrame(gating_data)
    df_gating.to_csv(gating_filename, index=False)
    
    return gating_weights_np, avg_weights


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
        "RBF Expert": lambda: SingleExpertModel(SplineKANLayer, basis_function='rbf'),
        "K-MOTE (4 Experts)": lambda: KMOTE(input_dim=1, output_dim=1, wavelet_type='shock')
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
                    normalized_pred = self.model(x)
                    return normalized_pred * self.std + self.mean
                def __getattr__(self, name):
                    # Delegate attribute access to the wrapped model for gating analysis
                    if name in ['model', 'mean', 'std']:
                        return super().__getattr__(name)
                    return getattr(self.model, name)

            unnorm_model = UnnormalizedModel(model, y_mean, y_std)
            
            # Plot the fit - no title needed as we have row/column labels
            plot_fit(axes[i, j], t, y_true, unnorm_model, "", show_legend=(i==0 and j==0))
            
            # Save actual vs prediction data for this function/model combination
            save_prediction_data(t, y_true, unnorm_model, func_name, model_name)
            
            # Analyze gating weights for K-MOTE models
            if "K-MOTE" in model_name:
                analyze_gating_weights(t, model, func_name)
    
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
    
    # 5. Create separate gating analysis plots for K-MOTE
    print("\n[INFO] Creating K-MOTE gating analysis plots...")
    create_gating_analysis_plots(t, target_functions, models_to_test)
    
    return results_data

def create_gating_analysis_plots(t, target_functions, models_to_test):
    """Create detailed gating analysis plots for K-MOTE models"""
    kmote_models = {k: v for k, v in models_to_test.items() if "K-MOTE" in k}
    
    if not kmote_models:
        print("  No K-MOTE models found for gating analysis.")
        return
    
    func_names = list(target_functions.keys())
    n_funcs = len(func_names)
    
    # Create a comprehensive gating analysis figure
    fig, axes = plt.subplots(n_funcs, 2, figsize=(16, 4*n_funcs))
    if n_funcs == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('K-MOTE Expert Gating Analysis Across Mathematical Functions', fontsize=16)
    
    for i, (func_name, y_true) in enumerate(target_functions.items()):
        print(f"  - Analyzing gating for: {func_name}")
        
        # Normalize the target function
        y_mean = y_true.mean()
        y_std = y_true.std()
        y_norm = (y_true - y_mean) / y_std
        
        # Train K-MOTE model
        for model_name, model_factory in kmote_models.items():
            model = model_factory()
            train_model_with_loss_return(model, t, y_norm)
            
            # Get predictions and gating weights
            model.eval()
            with torch.no_grad():
                t_input = t.unsqueeze(0).unsqueeze(-1)
                y_pred_norm, gating_weights = model(t_input, return_weights=True)
                
                # Un-normalize predictions
                y_pred = y_pred_norm.squeeze() * y_std + y_mean
                
                if gating_weights.dim() > 2:
                    gating_weights = gating_weights.squeeze()
                gating_weights_np = gating_weights.cpu().numpy()
            
            # Plot function fit
            axes[i, 0].plot(t.cpu().numpy(), y_true.cpu().numpy(), 'b-', linewidth=2, label='Target')
            axes[i, 0].plot(t.cpu().numpy(), y_pred.cpu().numpy(), 'r--', linewidth=2, label='K-MOTE')
            axes[i, 0].set_title(f'{func_name} - Function Fit', fontsize=11)
            axes[i, 0].legend(fontsize=9)
            axes[i, 0].grid(True, alpha=0.3)
            
            # Plot gating weights
            expert_names = ['B-Spline', 'Fourier', 'Wavelet', 'RBF']
            colors = ['green', 'blue', 'red', 'magenta']
            
            for j in range(4):
                axes[i, 1].plot(t.cpu().numpy(), gating_weights_np[:, j], 
                              color=colors[j], linewidth=2, label=f'{expert_names[j]}')
            
            axes[i, 1].set_title(f'{func_name} - Expert Weights', fontsize=11)
            axes[i, 1].set_xlabel('Input Value')
            axes[i, 1].set_ylabel('Expert Weight')
            axes[i, 1].legend(fontsize=9)
            axes[i, 1].grid(True, alpha=0.3)
            axes[i, 1].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('analysis_figures_math/kmote_gating_analysis_comprehensive.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("  ✅ K-MOTE gating analysis plots saved!")

if __name__ == '__main__':
    # Ensure the corrected k_mote.py is accessible before running
    results = run_math_function_analysis()
    print("\n✨ Mathematical function analysis complete!")
    print("📁 Figures saved in 'analysis_figures_math' directory.")
    print("📊 Results exported to CSV files for further analysis.")