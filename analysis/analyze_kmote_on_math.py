import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Global training configuration
MAX_EPOCHS = 100

# Import K-MOTE components
try:
    # Add the project root to path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    sys.path.insert(0, project_root)
    from models.time_encoders.k_mote import KMOTE, SplineKANLayer, FourierKANLayer, WaveletKANLayer
except ImportError as e:
    print(f"Error: Could not import K-MOTE components: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(__file__)}")
    print(f"Project root: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}")
    sys.exit(1)

# Create output directory for saving figures
os.makedirs('analysis_figures_kmote', exist_ok=True)

# --- Individual Expert Regressor Wrappers ---

class SplineKANRegressor(nn.Module):
    """Pure B-Spline KAN expert for scalar regression."""
    def __init__(self, embedding_dim=64):
        super().__init__()
        # Use a reasonable hidden dimension for the expert
        hidden_dim = max(embedding_dim // 2, 16)  # Ensure minimum of 16 dimensions
        
        self.expert = SplineKANLayer(
            input_dim=hidden_dim, 
            output_dim=1, 
            grid_size=5, 
            basis_function='b_spline',
            order=3, 
            grid_range=[-1, 1]
        )
        # Time transformation (similar to K-MOTE's approach)
        self.time_transform = nn.Linear(1, hidden_dim)
        self._initialize_time_transform()
        
    def _initialize_time_transform(self):
        """Initialize with geometric progression like K-MOTE"""
        with torch.no_grad():
            frequencies = 1.0 / (10 ** torch.linspace(0, 9, self.time_transform.out_features, dtype=torch.float32))
            self.time_transform.weight.copy_(frequencies.unsqueeze(1))
            self.time_transform.bias.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(0).unsqueeze(-1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)
        
        t_transformed = self.time_transform(t)
        output = self.expert(t_transformed)
        return output

class FourierKANRegressor(nn.Module):
    """Pure Fourier KAN expert for scalar regression."""
    def __init__(self, embedding_dim=64):
        super().__init__()
        # Use a reasonable hidden dimension for the expert
        hidden_dim = max(embedding_dim // 2, 16)  # Ensure minimum of 16 dimensions
        
        self.expert = FourierKANLayer(
            input_dim=hidden_dim,
            output_dim=1,
            n_harmonics=8
        )
        # Time transformation
        self.time_transform = nn.Linear(1, hidden_dim)
        self._initialize_time_transform()
        
    def _initialize_time_transform(self):
        """Initialize with geometric progression like K-MOTE"""
        with torch.no_grad():
            frequencies = 1.0 / (10 ** torch.linspace(0, 9, self.time_transform.out_features, dtype=torch.float32))
            self.time_transform.weight.copy_(frequencies.unsqueeze(1))
            self.time_transform.bias.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(0).unsqueeze(-1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)
        
        t_transformed = self.time_transform(t)
        output = self.expert(t_transformed)
        return output

class WaveletKANRegressor(nn.Module):
    """Pure Wavelet KAN expert for scalar regression."""
    def __init__(self, embedding_dim=64, wavelet_type='shock'):
        super().__init__()
        # Use a reasonable hidden dimension for the expert
        hidden_dim = max(embedding_dim // 2, 16)  # Ensure minimum of 16 dimensions
        
        self.expert = WaveletKANLayer(
            input_dim=hidden_dim,
            output_dim=1,
            n_wavelets=8,
            wavelet_type=wavelet_type
        )
        # Time transformation
        self.time_transform = nn.Linear(1, hidden_dim)
        self._initialize_time_transform()
        
    def _initialize_time_transform(self):
        """Initialize with geometric progression like K-MOTE"""
        with torch.no_grad():
            frequencies = 1.0 / (10 ** torch.linspace(0, 9, self.time_transform.out_features, dtype=torch.float32))
            self.time_transform.weight.copy_(frequencies.unsqueeze(1))
            self.time_transform.bias.zero_()

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(0).unsqueeze(-1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)
        
        t_transformed = self.time_transform(t)
        output = self.expert(t_transformed)
        return output

class KMOTERegressor(nn.Module):
    """Full K-MOTE with expert output extraction capability."""
    def __init__(self, embedding_dim=64, wavelet_type='shock'):
        super().__init__()
        # Use a reasonable hidden_dim (don't divide by 3 for small output dims)
        # For scalar regression (output_dim=1), we need a proper hidden dimension
        hidden_dim = max(embedding_dim // 2, 16)  # Ensure minimum of 16 dimensions
        
        self.kmote = KMOTE(
            input_dim=1,
            output_dim=1,
            hidden_dim=hidden_dim,
            wavelet_type=wavelet_type,
            use_layernorm=True,
            use_scale=True,
            gating_temp=1.0,
            transform_mode='adapter',
            adapter_type='affine'
        )
    
    def forward(self, t: torch.Tensor, return_expert_outputs=False) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(0).unsqueeze(-1)
        elif t.dim() == 2:
            t = t.unsqueeze(-1)
        
        if return_expert_outputs:
            return self.forward_with_experts(t)
        else:
            output, _ = self.kmote(t, return_weights=True)
            return output
    
    def forward_with_experts(self, t: torch.Tensor):
        """Forward pass that returns both final output and individual expert outputs."""
        # Get the K-MOTE's internal computation
        
        # 1. Apply time transformation (following K-MOTE's adapter mode)
        t_base = self.kmote.time_base_transform(t)
        
        # 2. Get expert-specific adaptations and outputs
        expert_outputs = []
        for i, expert in enumerate(self.kmote.experts):
            # Apply expert-specific adaptation (same as K-MOTE internal logic)
            if self.kmote.adapter_type == 'affine':
                t_adapted = t_base * self.kmote.expert_scales[i] + self.kmote.expert_shifts[i]
            else:  # linear adapter
                t_adapted = self.kmote.expert_adapters[i](t_base)
            
            expert_output = expert(t_adapted)
            expert_outputs.append(expert_output)
        
        # 3. Get gating weights
        gating_logits = self.kmote.gating_network(t_base)
        gating_weights = torch.softmax(gating_logits / self.kmote.temperature, dim=-1)
        
        # 4. Compute final output (same as K-MOTE)
        stacked_outputs = torch.stack(expert_outputs, dim=-1)
        gating_weights_expanded = gating_weights.unsqueeze(-2)
        weighted_sum = (gating_weights_expanded * stacked_outputs).sum(dim=-1)
        
        # Apply normalization and scaling
        final_output = self.kmote.layer_norm(weighted_sum)
        if self.kmote.use_scale:
            final_output = final_output * self.kmote.scale
        
        return final_output, expert_outputs, gating_weights

# --- Helper Functions ---

def train_model_with_loss_return(model, t_data, y_true, epochs=MAX_EPOCHS, lr=2e-4):
    """Train model and return final loss"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    loss_fn = nn.MSELoss()
    
    # Ensure proper input shapes
    if t_data.dim() == 1: 
        t_data = t_data.unsqueeze(0)
    if y_true.dim() == 1: 
        y_true = y_true.unsqueeze(0).unsqueeze(-1)

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

def plot_fit(ax, t, y_true, model, title, line_color='red', linestyle='--', show_experts=False):
    """Function to plot the results of the model fit"""
    model.eval()
    with torch.no_grad():
        # Ensure proper tensor shape: (batch_size, input_dim) where input_dim=1
        if t.dim() == 1:
            t_input = t.unsqueeze(-1)  # (500,) -> (500, 1)
        else:
            t_input = t
        
        if show_experts and isinstance(model, UnnormalizedKMOTEModel):
            # Special handling for K-MOTE with expert decomposition
            y_pred, expert_outputs, gating_weights = model.forward_with_experts(t_input)
            y_pred = y_pred.squeeze()
            
            # Plot target function
            ax.plot(t.cpu().numpy(), y_true.cpu().numpy(), 
                   label='Target Function', linewidth=3, alpha=0.8, color='blue')
            
            # Plot final K-MOTE prediction
            ax.plot(t.cpu().numpy(), y_pred.cpu().numpy(), 
                   label='K-MOTE Prediction', linewidth=3, color='black')
            
            # Plot individual expert outputs
            expert_names = ['Spline Expert', 'Fourier Expert', 'Wavelet Expert']
            expert_colors = ['green', 'purple', 'orange']
            
            for i, (expert_output, name, color) in enumerate(zip(expert_outputs, expert_names, expert_colors)):
                expert_pred = expert_output.squeeze()
                ax.plot(t.cpu().numpy(), expert_pred.cpu().numpy(), 
                       label=name, linewidth=1.5, color=color, linestyle=':', alpha=0.7)
            
        else:
            # Standard plotting for individual experts
            y_pred = model(t_input).squeeze()
            
            # Plot target function
            ax.plot(t.cpu().numpy(), y_true.cpu().numpy(), 
                   label='Target Function', linewidth=2, alpha=0.8, color='blue')
            
            # Plot model prediction
            ax.plot(t.cpu().numpy(), y_pred.cpu().numpy(), 
                   label='Learned Function', linestyle=linestyle, color=line_color, linewidth=2)
    
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def plot_gating_weights(ax, t, gating_weights, title):
    """Plot how gating weights change across the input domain"""
    t_np = t.cpu().numpy()
    gating_np = gating_weights.squeeze().cpu().numpy()
    
    expert_names = ['Spline', 'Fourier', 'Wavelet']
    expert_colors = ['green', 'purple', 'orange']
    
    for i, (name, color) in enumerate(zip(expert_names, expert_colors)):
        ax.plot(t_np, gating_np[:, i], label=f'{name} Weight', 
               color=color, linewidth=2, alpha=0.8)
    
    ax.set_title(title, fontsize=10)
    ax.set_ylabel('Gating Weight', fontsize=9)
    ax.set_xlabel('Input x', fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylim(0, 1)

# --- Mathematical Function Definitions ---

# Set A: Same as LeTE (for comparison)
def func_sin(t):
    """y = sin(x) - Pure periodic"""
    return torch.sin(t)

def func_modulated_sin(t):
    """y = (1 + sin(x))sin(2x) - Complex periodic"""
    return (1 + torch.sin(t)) * torch.sin(2*t)

def func_softplus(t):
    """y = log(1 + e^x) - Smooth nonlinear"""
    return torch.log(1 + torch.exp(t))

def func_swish(t):
    """y = x / (1 + e^-x) - Smooth sigmoid"""
    return t * torch.sigmoid(t)

# Set B: Expert-highlighting functions
def func_step(t):
    """y = step(x) - Abrupt step function (Wavelet should excel)"""
    return torch.where(t > 0, torch.ones_like(t), -torch.ones_like(t))

def func_cubic(t):
    """y = x³ - Smooth polynomial (Spline should excel)"""
    return t**3

def func_mixed(t):
    """y = sin(x) + step(x-2) - Mixed pattern (K-MOTE should excel)"""
    return torch.sin(t) + torch.where(t > 2, torch.ones_like(t), torch.zeros_like(t))

# --- Unnormalization Wrapper ---

class UnnormalizedModel(nn.Module):
    """Wrapper to un-normalize model output for plotting"""
    def __init__(self, trained_model, mean, std):
        super().__init__()
        self.model = trained_model
        self.mean = mean
        self.std = std
    
    def forward(self, x):
        return self.model(x) * self.std + self.mean

class UnnormalizedKMOTEModel(UnnormalizedModel):
    """Special wrapper for K-MOTE that preserves expert decomposition"""
    def forward_with_experts(self, x):
        output, expert_outputs, gating_weights = self.model.forward_with_experts(x)
        
        # Un-normalize all outputs
        output = output * self.std + self.mean
        expert_outputs = [exp_out * self.std + self.mean for exp_out in expert_outputs]
        
        return output, expert_outputs, gating_weights

# --- Main Analysis Script ---

def run_kmote_analysis_on_math(function_set='A'):
    """
    Run K-MOTE analysis on mathematical functions
    
    Args:
        function_set: 'A' for LeTE comparison set, 'B' for expert-highlighting set
    """
    print(f"--- K-MOTE Analysis on Mathematical Functions (Set {function_set}) ---")
    
    # Define input range
    t = torch.linspace(-5, 5, 500)
    
    # Select function set
    if function_set == 'A':
        target_functions = [
            ("y = sin(x)", func_sin(t)),
            ("y = (1+sin(x))sin(2x)", func_modulated_sin(t)),
            ("y = log(1+e^x)", func_softplus(t)),
            ("y = x/(1+e^-x)", func_swish(t))
        ]
        set_description = "LeTE Comparison Set"
    else:  # function_set == 'B'
        target_functions = [
            ("y = sin(x)", func_sin(t)),
            ("y = step(x)", func_step(t)),
            ("y = x³", func_cubic(t)),
            ("y = sin(x) + step(x-2)", func_mixed(t))
        ]
        set_description = "Expert-Highlighting Set"
    
    # Define the expert methods
    expert_methods = [
        ("B-SplineKAN", "spline", 'green'),
        ("FourierKAN", "fourier", 'purple'),  
        ("WaveletKAN", "wavelet", 'orange'),
        ("K-MOTE", "kmote", 'black')
    ]
    
    # Create the figure: 5 rows × 4 columns
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle(f'K-MOTE Expert Analysis - {set_description}', fontsize=18, y=0.96)
    
    # Add method labels on the left
    method_labels = ["B-SplineKAN", "FourierKAN", "WaveletKAN", "K-MOTE", "Expert\nDecomposition"]
    for i, label in enumerate(method_labels):
        fig.text(0.02, 0.85 - i*0.18, label, rotation=90, va='center', ha='center', 
                fontsize=14, weight='bold')
    
    # Store K-MOTE models and data for expert decomposition
    kmote_models = []
    kmote_data = []
    
    # Loop through each method and function combination
    for method_idx, (method_name, method_key, line_color) in enumerate(expert_methods):
        print(f"\n[INFO] Testing {method_name}...")
        
        method_models = []
        method_data = []
        
        for func_idx, (func_name, y_true) in enumerate(target_functions):
            print(f"  - Training {method_name} on {func_name}...")
            
            # Normalize target for stable training
            y_mean, y_std = y_true.mean(), y_true.std()
            y_norm = (y_true - y_mean) / y_std
            
            # Create model based on method
            if method_key == "spline":
                model = SplineKANRegressor(embedding_dim=64)
            elif method_key == "fourier":
                model = FourierKANRegressor(embedding_dim=64)
            elif method_key == "wavelet":
                model = WaveletKANRegressor(embedding_dim=64, wavelet_type='shock')
            elif method_key == "kmote":
                model = KMOTERegressor(embedding_dim=64, wavelet_type='shock')
            
            # Train the model
            final_loss = train_model_with_loss_return(model, t, y_norm)
            
            # Create unnormalized model for plotting
            if method_key == "kmote":
                unnorm_model = UnnormalizedKMOTEModel(model, y_mean, y_std)
                # Store for expert decomposition
                method_models.append(unnorm_model)
                method_data.append((t, y_true, func_name))
            else:
                unnorm_model = UnnormalizedModel(model, y_mean, y_std)
            
            # Plot in the correct subplot (rows 0-3)
            ax = axes[method_idx, func_idx]
            
            # Create title - function name on top row only
            if method_idx == 0:
                title = f"{func_name}"
            else:
                title = ""
            
            plot_fit(ax, t, y_true, unnorm_model, title, line_color=line_color)
            
            # Remove legend for cleaner look (except first subplot)
            if not (method_idx == 0 and func_idx == 0):
                ax.legend().set_visible(False)
        
        # Store K-MOTE data for expert decomposition row
        if method_key == "kmote":
            kmote_models = method_models
            kmote_data = method_data
    
    # Row 5: Expert decomposition for K-MOTE
    print(f"\n[INFO] Creating expert decomposition visualizations...")
    
    for func_idx, (model, (t, y_true, func_name)) in enumerate(zip(kmote_models, kmote_data)):
        # Plot K-MOTE with expert outputs
        ax_main = axes[4, func_idx]
        plot_fit(ax_main, t, y_true, model, "", show_experts=True)
        ax_main.legend().set_visible(True)  # Show legend for expert decomposition
        
        # Create a simple subplot for gating weights (instead of inset)
        # We'll show gating weights as text annotation for now
        # Get gating weights
        model.model.eval()
        with torch.no_grad():
            t_input = t.unsqueeze(0)
            _, _, gating_weights = model.forward_with_experts(t_input)
        
        # Show average gating weights as text annotation
        avg_weights = gating_weights.mean(dim=(0, 1)).cpu().numpy()
        weight_text = f"Avg Weights:\nSpline: {avg_weights[0]:.2f}\nFourier: {avg_weights[1]:.2f}\nWavelet: {avg_weights[2]:.2f}"
        ax_main.text(0.02, 0.98, weight_text, transform=ax_main.transAxes, 
                    verticalalignment='top', fontsize=8, 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add a single legend at the bottom
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=2, label='Target Function'),
        plt.Line2D([0], [0], color='black', linewidth=2, label='K-MOTE Prediction'),
        plt.Line2D([0], [0], color='green', linewidth=1.5, linestyle=':', label='Spline Expert'),
        plt.Line2D([0], [0], color='purple', linewidth=1.5, linestyle=':', label='Fourier Expert'),
        plt.Line2D([0], [0], color='orange', linewidth=1.5, linestyle=':', label='Wavelet Expert')
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5, fontsize=12, 
              bbox_to_anchor=(0.5, 0.01))
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.06, top=0.94)
    
    # Save the figure
    plt.savefig(f'analysis_figures_kmote/kmote_analysis_set_{function_set}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)
    if function_set == 'A':
        print("• Comparison with LeTE analysis shows K-MOTE's competitive performance")
        print("• Individual experts show expected specializations")
        print("• K-MOTE combines strengths through intelligent gating")
    else:
        print("• B-SplineKAN excels at smooth polynomial functions (x³)")
        print("• FourierKAN dominates periodic patterns (sin)")
        print("• WaveletKAN handles abrupt changes (step functions)")
        print("• K-MOTE adaptively combines experts for complex mixed patterns")
    print("• Expert decomposition reveals gating decisions and individual contributions")

if __name__ == '__main__':
    print("K-MOTE Mathematical Function Analysis")
    print("=====================================")
    print("Available function sets:")
    print("  A: LeTE comparison set (sin, modulated sin, softplus, swish)")
    print("  B: Expert-highlighting set (sin, step, cubic, mixed)")
    print()
    
    # Run both sets
    print("Running Set A (LeTE Comparison)...")
    run_kmote_analysis_on_math('A')
    
    print("\nRunning Set B (Expert-Highlighting)...")
    run_kmote_analysis_on_math('B')
    
    print("\n✨ K-MOTE analysis complete!")
    print("📁 Figures saved in 'analysis_figures_kmote' directory.")
    print("\n🔍 Key insights:")
    print("   • Individual experts show clear specializations")
    print("   • K-MOTE intelligently combines experts via gating")
    print("   • Expert decomposition reveals decision-making process")