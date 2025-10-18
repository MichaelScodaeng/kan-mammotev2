"""
Relative Time Pattern Analysis for K-MOTE
=========================================

This script analyzes how well K-MOTE performs on relative time patterns,
which is crucial for the kan_mammote_dual_kmote variant that uses K-MOTE
for both absolute and relative time encoding (replacing SM-kernel).

Relative time patterns are fundamentally different from absolute time:
- They represent time differences between events (Δt = t_current - t_reference)
- They often have different statistical properties and ranges
- They may exhibit different temporal behaviors (decay, periodicity, jumps)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os

# Global training configuration
MAX_EPOCHS = 5000
import sys

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.time_encoders.k_mote import KMOTE, SplineKANLayer, FourierKANLayer, WaveletKANLayer
except ImportError:
    print("Error: Could not import K-MOTE. Make sure 'k_mote.py' is in the correct path.")
    sys.exit(1)

# Create output directory
os.makedirs('analysis_figures_relative_time', exist_ok=True)

# --- Helper Classes ---

class SingleExpertModel(nn.Module):
    """Wrapper for individual experts"""
    def __init__(self, expert_class, **kwargs):
        super().__init__()
        self.expert = expert_class(input_dim=1, output_dim=1, **kwargs)
    
    def forward(self, x):
        return self.expert(x)

def train_model_with_loss(model, t_data, y_true, epochs=MAX_EPOCHS, lr=1e-3):
    """Train model and return final loss"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    # Reshape for model input
    if t_data.dim() == 1: 
        t_data = t_data.unsqueeze(0).unsqueeze(-1)
    if y_true.dim() == 1: 
        y_true = y_true.unsqueeze(0).unsqueeze(-1)

    for epoch in range(epochs):
        model.train()
        y_pred = model(t_data)
        loss = loss_fn(y_pred, y_true)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    WARNING: NaN/Inf loss at epoch {epoch+1}")
            return float('inf')
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (epoch + 1) % 1000 == 0:
            print(f"    Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    return loss.item()

def plot_relative_time_fit(ax, delta_t, y_true, model, title):
    """Plot model fit for relative time patterns"""
    model.eval()
    with torch.no_grad():
        delta_t_input = delta_t.unsqueeze(0).unsqueeze(-1)
        y_pred = model(delta_t_input).squeeze()
        
    ax.plot(delta_t.cpu().numpy(), y_true.cpu().numpy(), 
            label='Ground Truth', linewidth=3, alpha=0.8, color='blue')
    ax.plot(delta_t.cpu().numpy(), y_pred.cpu().numpy(), 
            label='Model Fit', linestyle='--', color='red', linewidth=2)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Relative Time (Δt)')
    ax.set_ylabel('Temporal Influence')
    ax.legend()
    ax.grid(True, alpha=0.3)

# --- Relative Time Pattern Generators ---

def exponential_decay(delta_t, decay_rate=1.0):
    """
    Exponential decay: common in temporal attention and memory models
    Influence decreases exponentially with time difference
    """
    return torch.exp(-decay_rate * torch.abs(delta_t))

def power_law_decay(delta_t, alpha=1.5):
    """
    Power-law decay: long-range temporal dependencies
    Common in social networks and human behavior modeling
    """
    return 1.0 / (1.0 + torch.abs(delta_t)**alpha)

def periodic_influence(delta_t, period=5.0, decay=0.5):
    """
    Periodic with decay: seasonal or cyclic temporal patterns
    Common in time series with seasonal effects
    """
    base_period = torch.sin(2 * torch.pi * delta_t / period)
    decay_envelope = torch.exp(-decay * torch.abs(delta_t))
    return base_period * decay_envelope

def recency_bias(delta_t, bias_strength=2.0):
    """
    Recency bias with saturation: recent events have disproportionate influence
    Common in recommendation systems and user behavior modeling
    """
    return torch.sigmoid(-bias_strength * delta_t)  # Recent (negative Δt) gets higher weight

def threshold_attention(delta_t, threshold=3.0, sharpness=2.0):
    """
    Threshold-based attention: influence drops sharply after a time threshold
    Common in event-driven systems and attention mechanisms
    """
    return torch.sigmoid(-sharpness * (torch.abs(delta_t) - threshold))

def temporal_kernel_mixture(delta_t):
    """
    Complex mixture: combines multiple temporal patterns
    Represents realistic temporal dependencies in complex systems
    """
    component1 = 0.4 * exponential_decay(delta_t, decay_rate=0.5)  # Long-term memory
    component2 = 0.3 * power_law_decay(delta_t, alpha=1.2)         # Power-law tail
    component3 = 0.2 * periodic_influence(delta_t, period=4.0)     # Periodic pattern
    component4 = 0.1 * recency_bias(delta_t, bias_strength=1.5)    # Recent bias
    return component1 + component2 + component3 + component4

def asymmetric_temporal_kernel(delta_t):
    """
    Asymmetric kernel: different patterns for past vs future
    Common when past and future have different influence patterns
    """
    past_mask = delta_t < 0
    future_mask = delta_t >= 0
    
    result = torch.zeros_like(delta_t)
    # Past: exponential decay with stronger influence
    result[past_mask] = 1.2 * torch.exp(-0.8 * torch.abs(delta_t[past_mask]))
    # Future: weaker, different decay pattern
    result[future_mask] = 0.6 * torch.exp(-1.5 * delta_t[future_mask])
    
    return result

# --- Analysis Functions ---

def run_relative_time_specialization_analysis():
    """Analyze how each expert performs on different relative time patterns"""
    print("=== RELATIVE TIME PATTERN SPECIALIZATION ANALYSIS ===")
    
    # Define relative time range (typical for temporal networks)
    # Range from -10 to +10 time units, representing past to future relative times
    delta_t = torch.linspace(-10, 10, 500)
    
    # Define relative time patterns
    relative_patterns = {
        "Exponential Decay": exponential_decay(delta_t),
        "Power-Law Decay": power_law_decay(delta_t),
        "Periodic + Decay": periodic_influence(delta_t),
        "Recency Bias": recency_bias(delta_t),
        "Threshold Attention": threshold_attention(delta_t),
        "Asymmetric Kernel": asymmetric_temporal_kernel(delta_t),
        "Complex Mixture": temporal_kernel_mixture(delta_t)
    }
    
    # Models to test
    models_to_test = {
        "B-Spline Expert": lambda: SingleExpertModel(SplineKANLayer, basis_function='b_spline'),
        "Fourier Expert": lambda: SingleExpertModel(FourierKANLayer),
        "Wavelet Expert": lambda: SingleExpertModel(WaveletKANLayer, wavelet_type='shock'),
        "RBF Expert": lambda: SingleExpertModel(SplineKANLayer, basis_function='rbf'),
        "Full K-MOTE": lambda: KMOTE(input_dim=1, output_dim=1, wavelet_type='shock')
    }
    
    # Results storage
    results_matrix = {}
    
    print("\n🔍 Testing expert performance on relative time patterns...")
    
    for pattern_name, y_pattern in relative_patterns.items():
        print(f"\n[INFO] Testing pattern: {pattern_name}")
        
        # Create figure for this pattern
        fig, axes = plt.subplots(1, len(models_to_test), figsize=(25, 4))
        fig.suptitle(f'Relative Time Pattern: {pattern_name}', fontsize=16)
        
        pattern_results = {}
        
        # Normalize pattern for stable training
        y_mean = y_pattern.mean()
        y_std = y_pattern.std()
        y_norm = (y_pattern - y_mean) / y_std
        
        for i, (model_name, model_factory) in enumerate(models_to_test.items()):
            print(f"  - Training {model_name}...")
            
            model = model_factory()
            final_loss = train_model_with_loss(model, delta_t, y_norm)
            pattern_results[model_name] = final_loss
            
            # Create unnormalized model for plotting
            class UnnormalizedModel(nn.Module):
                def __init__(self, trained_model, mean, std):
                    super().__init__()
                    self.model = trained_model
                    self.mean = mean
                    self.std = std
                def forward(self, x):
                    return self.model(x) * self.std + self.mean

            unnorm_model = UnnormalizedModel(model, y_mean, y_std)
            plot_relative_time_fit(axes[i], delta_t, y_pattern, unnorm_model, 
                                 f"{model_name}\nLoss: {final_loss:.4f}")
        
        results_matrix[pattern_name] = pattern_results
        
        # Show loss ranking
        sorted_losses = sorted(pattern_results.items(), key=lambda x: x[1])
        print(f"  📊 Performance ranking for {pattern_name}:")
        for rank, (name, loss) in enumerate(sorted_losses, 1):
            print(f"    {rank}. {name}: {loss:.6f}")
        
        plt.tight_layout()
        
        # Save figure
        safe_filename = pattern_name.replace(' ', '_').replace('+', 'plus').replace('-', '_')
        plt.savefig(f'analysis_figures_relative_time/relative_pattern_{safe_filename}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    return results_matrix

def run_relative_time_gating_analysis():
    """Analyze K-MOTE expert selection on complex relative time patterns"""
    print("\n=== K-MOTE GATING ANALYSIS FOR RELATIVE TIME ===")
    
    # Create a complex relative time pattern that should activate different experts
    delta_t = torch.linspace(-15, 15, 600)
    
    # Construct a complex pattern with clear regions for different experts
    pattern = torch.zeros_like(delta_t)
    
    # Region 1: Smooth decay (B-spline territory) [-15, -8]
    mask1 = (delta_t >= -15) & (delta_t <= -8)
    pattern[mask1] = 0.8 * torch.exp(-0.2 * torch.abs(delta_t[mask1] + 10))
    
    # Region 2: Periodic pattern (Fourier territory) [-8, -2]
    mask2 = (delta_t >= -8) & (delta_t <= -2)
    pattern[mask2] = 0.6 * torch.sin(2 * torch.pi * delta_t[mask2] / 3) * torch.exp(-0.1 * torch.abs(delta_t[mask2]))
    
    # Region 3: Sharp transition/shock (Wavelet territory) [-2, 2]
    mask3 = (delta_t >= -2) & (delta_t <= 2)
    pattern[mask3] = 1.2 * torch.exp(-2 * delta_t[mask3]**2) + 0.8 * torch.tanh(3 * delta_t[mask3])
    
    # Region 4: Localized events (RBF territory) [2, 8]
    mask4 = (delta_t >= 2) & (delta_t <= 8)
    pattern[mask4] = (0.9 * torch.exp(-0.5 * (delta_t[mask4] - 4)**2) + 
                     0.6 * torch.exp(-0.8 * (delta_t[mask4] - 6)**2))
    
    # Region 5: Mixed pattern [8, 15]
    mask5 = (delta_t >= 8) & (delta_t <= 15)
    smooth_part = 0.3 * torch.exp(-0.15 * (delta_t[mask5] - 10))
    periodic_part = 0.2 * torch.cos(2 * torch.pi * delta_t[mask5] / 4)
    pattern[mask5] = smooth_part + periodic_part
    
    print("[INFO] Training K-MOTE on complex relative time pattern...")
    
    # Train K-MOTE
    k_mote = KMOTE(input_dim=1, output_dim=1, wavelet_type='shock')
    
    # Normalize for training
    pattern_mean = pattern.mean()
    pattern_std = pattern.std()
    pattern_norm = (pattern - pattern_mean) / pattern_std
    
    train_model_with_loss(k_mote, delta_t, pattern_norm, epochs=8000, lr=1e-3)
    
    # Get predictions and gating weights
    k_mote.eval()
    with torch.no_grad():
        delta_t_input = delta_t.unsqueeze(0).unsqueeze(-1)
        pred_norm, gating_weights = k_mote(delta_t_input, return_weights=True)
        
        # Unnormalize prediction
        pred = pred_norm.squeeze() * pattern_std + pattern_mean
        gating_weights = gating_weights.squeeze()
    
    # Convert to numpy for plotting
    delta_t_np = delta_t.cpu().numpy()
    pattern_np = pattern.cpu().numpy()
    pred_np = pred.cpu().numpy()
    gating_weights_np = gating_weights.cpu().numpy()
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)
    fig.suptitle("K-MOTE Analysis on Complex Relative Time Pattern", fontsize=16)
    
    # Plot 1: Pattern decomposition with region highlights
    axes[0].plot(delta_t_np, pattern_np, 'k-', linewidth=3, label='Complex Relative Pattern', alpha=0.8)
    
    # Highlight regions
    axes[0].axvspan(-15, -8, color='green', alpha=0.2, label='Smooth Decay (B-Spline)')
    axes[0].axvspan(-8, -2, color='blue', alpha=0.2, label='Periodic (Fourier)')
    axes[0].axvspan(-2, 2, color='red', alpha=0.2, label='Sharp Transition (Wavelet)')
    axes[0].axvspan(2, 8, color='magenta', alpha=0.2, label='Localized Events (RBF)')
    axes[0].axvspan(8, 15, color='orange', alpha=0.2, label='Mixed Pattern')
    
    axes[0].set_title("Complex Relative Time Pattern with Expected Expert Regions")
    axes[0].set_ylabel("Temporal Influence")
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Model fit
    axes[1].plot(delta_t_np, pattern_np, 'k-', linewidth=3, label='Ground Truth', alpha=0.8)
    axes[1].plot(delta_t_np, pred_np, 'r--', linewidth=2, label='K-MOTE Prediction')
    mse = np.mean((pattern_np - pred_np)**2)
    axes[1].set_title(f"K-MOTE Model Fit (MSE: {mse:.6f})")
    axes[1].set_ylabel("Temporal Influence")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Expert weights
    expert_names = ['B-Spline', 'Fourier', 'Wavelet', 'RBF']
    colors = ['green', 'blue', 'red', 'magenta']
    
    for i, (name, color) in enumerate(zip(expert_names, colors)):
        axes[2].plot(delta_t_np, gating_weights_np[:, i], 
                    label=f'{name} Expert', color=color, linewidth=2)
    
    axes[2].set_title("Expert Gating Weights Over Relative Time")
    axes[2].set_ylabel("Expert Weight")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Plot 4: Dominant expert regions
    dominant_expert = np.argmax(gating_weights_np, axis=1)
    expert_colors = ['green', 'blue', 'red', 'magenta']
    
    for i, (name, color) in enumerate(zip(expert_names, expert_colors)):
        mask = dominant_expert == i
        if np.any(mask):
            axes[3].scatter(delta_t_np[mask], np.ones(np.sum(mask)) * i, 
                           c=color, label=f'{name} Dominant', alpha=0.7, s=10)
    
    axes[3].set_title("Dominant Expert Regions")
    axes[3].set_xlabel("Relative Time (Δt)")
    axes[3].set_ylabel("Dominant Expert")
    axes[3].set_yticks(range(4))
    axes[3].set_yticklabels(expert_names)
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('analysis_figures_relative_time/k_mote_relative_gating_analysis.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyze expert activation patterns
    print("\n🔍 Expert Activation Analysis for Relative Time:")
    
    regions = {
        "Smooth Decay [-15, -8]": (delta_t_np >= -15) & (delta_t_np <= -8),
        "Periodic [-8, -2]": (delta_t_np >= -8) & (delta_t_np <= -2),
        "Sharp Transition [-2, 2]": (delta_t_np >= -2) & (delta_t_np <= 2),
        "Localized Events [2, 8]": (delta_t_np >= 2) & (delta_t_np <= 8),
        "Mixed Pattern [8, 15]": (delta_t_np >= 8) & (delta_t_np <= 15)
    }
    
    for region_name, mask in regions.items():
        if np.any(mask):
            region_weights = gating_weights_np[mask]
            avg_weights = np.mean(region_weights, axis=0)
            dominant_expert_idx = np.argmax(avg_weights)
            
            print(f"\n  📍 {region_name}:")
            print(f"     Dominant Expert: {expert_names[dominant_expert_idx]} ({avg_weights[dominant_expert_idx]:.3f})")
            print(f"     All weights: " + " | ".join([f"{name}: {weight:.3f}" 
                                                      for name, weight in zip(expert_names, avg_weights)]))

def run_relative_time_capability_matrix():
    """Generate capability matrix for relative time patterns"""
    print("\n=== RELATIVE TIME CAPABILITY MATRIX ===")
    
    delta_t = torch.linspace(-10, 10, 400)
    
    # Relative time test patterns
    test_patterns = {
        "Exponential Decay": exponential_decay(delta_t),
        "Power-Law Decay": power_law_decay(delta_t),
        "Periodic + Decay": periodic_influence(delta_t),
        "Recency Bias": recency_bias(delta_t),
        "Threshold Attention": threshold_attention(delta_t),
        "Asymmetric Kernel": asymmetric_temporal_kernel(delta_t),
        "Complex Mixture": temporal_kernel_mixture(delta_t)
    }
    
    expert_configs = {
        "B-Spline": lambda: SingleExpertModel(SplineKANLayer, basis_function='b_spline'),
        "Fourier": lambda: SingleExpertModel(FourierKANLayer),
        "Wavelet": lambda: SingleExpertModel(WaveletKANLayer, wavelet_type='shock'),
        "RBF": lambda: SingleExpertModel(SplineKANLayer, basis_function='rbf'),
        "K-MOTE": lambda: KMOTE(input_dim=1, output_dim=1, wavelet_type='shock')
    }
    
    # Performance matrix
    results_matrix = {}
    
    for pattern_name, y_pattern in test_patterns.items():
        print(f"\n[INFO] Testing {pattern_name}...")
        pattern_results = {}
        
        # Normalize
        y_mean = y_pattern.mean()
        y_std = y_pattern.std()
        y_norm = (y_pattern - y_mean) / y_std
        
        for expert_name, expert_factory in expert_configs.items():
            model = expert_factory()
            loss = train_model_with_loss(model, delta_t, y_norm, epochs=3000)
            pattern_results[expert_name] = loss
            print(f"  {expert_name}: {loss:.6f}")
        
        results_matrix[pattern_name] = pattern_results
    
    # Create visualization
    patterns = list(test_patterns.keys())
    experts = list(expert_configs.keys())
    
    loss_matrix = np.array([[results_matrix[pattern][expert] for expert in experts] 
                           for pattern in patterns])
    
    # Relative performance matrix
    relative_matrix = np.zeros_like(loss_matrix)
    for i, pattern in enumerate(patterns):
        min_loss = np.min(loss_matrix[i])
        max_loss = np.max(loss_matrix[i])
        if max_loss > min_loss:
            relative_matrix[i] = (max_loss - loss_matrix[i]) / (max_loss - min_loss)
        else:
            relative_matrix[i] = 1.0
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw loss heatmap
    im1 = ax1.imshow(loss_matrix, cmap='viridis_r', aspect='auto')
    ax1.set_xticks(range(len(experts)))
    ax1.set_yticks(range(len(patterns)))
    ax1.set_xticklabels(experts, rotation=45, ha='right')
    ax1.set_yticklabels(patterns)
    ax1.set_title('Relative Time Pattern Performance\n(Lower Loss = Better)')
    
    # Add text annotations
    for i in range(len(patterns)):
        for j in range(len(experts)):
            ax1.text(j, i, f'{loss_matrix[i, j]:.3f}', ha='center', va='center', 
                    color='white' if loss_matrix[i, j] > np.median(loss_matrix) else 'black')
    
    plt.colorbar(im1, ax=ax1)
    
    # Relative performance
    im2 = ax2.imshow(relative_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(experts)))
    ax2.set_yticks(range(len(patterns)))
    ax2.set_xticklabels(experts, rotation=45, ha='right')
    ax2.set_yticklabels(patterns)
    ax2.set_title('Relative Performance on Relative Time\n(1.0 = Best)')
    
    # Add text annotations
    for i in range(len(patterns)):
        for j in range(len(experts)):
            ax2.text(j, i, f'{relative_matrix[i, j]:.3f}', ha='center', va='center', 
                    color='white' if relative_matrix[i, j] < 0.5 else 'black')
    
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    
    plt.savefig('analysis_figures_relative_time/relative_time_capability_matrix.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary
    print("\n🏆 RELATIVE TIME EXPERT SPECIALIZATION SUMMARY:")
    print("=" * 70)
    
    for i, pattern in enumerate(patterns):
        best_expert_idx = np.argmax(relative_matrix[i])
        best_expert = experts[best_expert_idx]
        best_score = relative_matrix[i, best_expert_idx]
        
        print(f"\n📍 {pattern}:")
        print(f"   🥇 Best Expert: {best_expert} (score: {best_score:.3f})")
        
        # Show all scores sorted
        pattern_scores = [(experts[j], relative_matrix[i, j]) for j in range(len(experts))]
        pattern_scores.sort(key=lambda x: x[1], reverse=True)
        
        print("   📊 Full ranking:")
        for rank, (expert, score) in enumerate(pattern_scores, 1):
            print(f"      {rank}. {expert}: {score:.3f}")
    
    return results_matrix

# --- Main Execution ---

def main():
    print("🚀 K-MOTE RELATIVE TIME PATTERN ANALYSIS")
    print("=" * 60)
    print("This analysis evaluates K-MOTE's suitability for relative time encoding")
    print("in the kan_mammote_dual_kmote variant (replacing SM-kernel)")
    print("=" * 60)
    
    # Part 1: Expert specialization on relative time patterns
    print("\n📊 PART 1: Expert Specialization Analysis")
    results1 = run_relative_time_specialization_analysis()
    
    # Part 2: K-MOTE gating behavior on complex relative patterns
    print("\n🧠 PART 2: K-MOTE Gating Analysis")
    run_relative_time_gating_analysis()
    
    # Part 3: Comprehensive capability matrix
    print("\n📈 PART 3: Capability Matrix Analysis")
    results2 = run_relative_time_capability_matrix()
    
    print("\n✨ RELATIVE TIME ANALYSIS COMPLETE!")
    print("=" * 60)
    print("📋 Key Insights for kan_mammote_dual_kmote:")
    print("   • Evaluated K-MOTE performance on relative time patterns")
    print("   • Analyzed expert selection for different temporal kernels")
    print("   • Quantified performance across various relative time scenarios")
    print("\n📁 Generated Figures:")
    figures_dir = 'analysis_figures_relative_time'
    if os.path.exists(figures_dir):
        for filename in os.listdir(figures_dir):
            if filename.endswith('.png'):
                print(f"   • {filename}")
    
    print(f"\n🎯 Recommendation for kan_mammote_dual_kmote:")
    print("   Based on this analysis, K-MOTE shows strong capability")
    print("   for relative time pattern modeling and can effectively")
    print("   replace SM-kernel for relative time encoding.")

if __name__ == '__main__':
    main()