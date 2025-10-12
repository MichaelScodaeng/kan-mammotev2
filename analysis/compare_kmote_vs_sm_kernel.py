"""
K-MOTE vs SM-Kernel Comparison for Relative Time Encoding
========================================================

This script directly compares K-MOTE with SM-kernel (Spectral Mixture) 
performance on relative time encoding tasks, helping you decide whether
to replace SM-kernel with K-MOTE in the kan_mammote_dual_kmote variant.

The comparison focuses on:
1. Modeling accuracy on various relative time patterns
2. Training stability and convergence
3. Computational efficiency
4. Interpretability and expert utilization
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from models.time_encoders.k_mote import KMOTE
    # Try to import SM-kernel if available
    try:
        from models.time_encoders.spectral_mixture import SpectralMixtureKernel
        SM_KERNEL_AVAILABLE = True
    except ImportError:
        print("⚠️  SM-Kernel not found. Creating a simplified reference implementation.")
        SM_KERNEL_AVAILABLE = False
except ImportError:
    print("Error: Could not import K-MOTE. Make sure 'k_mote.py' is in the correct path.")
    sys.exit(1)

# Create output directory
os.makedirs('analysis_figures_kmote_vs_sm', exist_ok=True)

# --- Reference SM-Kernel Implementation (if not available) ---

class SimplifiedSMKernel(nn.Module):
    """
    Simplified Spectral Mixture Kernel for comparison
    Based on the spectral mixture approach for temporal modeling
    """
    def __init__(self, input_dim: int, output_dim: int, n_mixtures: int = 8):
        super().__init__()
        self.n_mixtures = n_mixtures
        self.output_dim = output_dim
        
        # Spectral mixture parameters
        self.mixture_weights = nn.Parameter(torch.randn(n_mixtures))
        self.mixture_means = nn.Parameter(torch.randn(n_mixtures))
        self.mixture_scales = nn.Parameter(torch.randn(n_mixtures))
        
        # Output projection
        self.output_projection = nn.Linear(n_mixtures, output_dim)
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.unsqueeze(-1)
        
        batch_size, seq_len, _ = t.shape
        
        # Expand dimensions for mixture computation
        t_expanded = t.unsqueeze(-1)  # [batch, seq, 1, 1]
        
        # Compute spectral mixture kernel
        weights = F.softmax(self.mixture_weights, dim=0)  # [n_mixtures]
        means = self.mixture_means.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, n_mixtures]
        scales = F.softplus(self.mixture_scales).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, n_mixtures]
        
        # Spectral mixture computation
        cos_component = torch.cos(2 * torch.pi * means * t_expanded)
        exp_component = torch.exp(-2 * torch.pi**2 * scales * t_expanded**2)
        
        mixture_output = weights * cos_component * exp_component  # [batch, seq, 1, n_mixtures]
        mixture_sum = mixture_output.sum(dim=-1)  # [batch, seq, 1]
        
        # Project to output dimension
        output = self.output_projection(mixture_output.squeeze(2))  # [batch, seq, output_dim]
        
        return output

# Use either imported or simplified SM-kernel
if not SM_KERNEL_AVAILABLE:
    SpectralMixtureKernel = SimplifiedSMKernel

# --- Helper Functions ---

def train_model_with_metrics(model, t_data, y_true, epochs=5000, lr=1e-3, model_name="Model"):
    """Train model and return comprehensive metrics"""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.MSELoss()
    
    # Reshape for model input
    if t_data.dim() == 1: 
        t_data = t_data.unsqueeze(0).unsqueeze(-1)
    if y_true.dim() == 1: 
        y_true = y_true.unsqueeze(0).unsqueeze(-1)

    losses = []
    start_time = time.time()
    converged_epoch = epochs
    
    for epoch in range(epochs):
        model.train()
        y_pred = model(t_data)
        loss = loss_fn(y_pred, y_true)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"    ⚠️  {model_name}: NaN/Inf loss at epoch {epoch+1}")
            return {
                'final_loss': float('inf'),
                'training_time': time.time() - start_time,
                'converged_epoch': epoch,
                'loss_curve': losses,
                'converged': False
            }
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        # Check for convergence (loss improvement < 1e-6 for 100 consecutive epochs)
        if epoch > 100:
            recent_improvement = losses[-100] - losses[-1]
            if recent_improvement < 1e-6:
                converged_epoch = epoch
                break
        
        if (epoch + 1) % 1000 == 0:
            print(f"    {model_name} - Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")
    
    training_time = time.time() - start_time
    
    return {
        'final_loss': losses[-1] if losses else float('inf'),
        'training_time': training_time,
        'converged_epoch': converged_epoch,
        'loss_curve': losses,
        'converged': converged_epoch < epochs
    }

def compute_model_complexity(model):
    """Compute model complexity metrics"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
    }

# --- Relative Time Pattern Generators (from previous analysis) ---

def exponential_decay(delta_t, decay_rate=1.0):
    """Exponential decay pattern"""
    return torch.exp(-decay_rate * torch.abs(delta_t))

def power_law_decay(delta_t, alpha=1.5):
    """Power-law decay pattern"""
    return 1.0 / (1.0 + torch.abs(delta_t)**alpha)

def periodic_influence(delta_t, period=5.0, decay=0.5):
    """Periodic with decay pattern"""
    base_period = torch.sin(2 * torch.pi * delta_t / period)
    decay_envelope = torch.exp(-decay * torch.abs(delta_t))
    return base_period * decay_envelope

def temporal_kernel_mixture(delta_t):
    """Complex mixture pattern"""
    component1 = 0.4 * exponential_decay(delta_t, decay_rate=0.5)
    component2 = 0.3 * power_law_decay(delta_t, alpha=1.2)
    component3 = 0.2 * periodic_influence(delta_t, period=4.0)
    component4 = 0.1 * torch.sigmoid(-2.0 * delta_t)  # Recency bias
    return component1 + component2 + component3 + component4

# --- Comparison Analysis Functions ---

def run_comprehensive_comparison():
    """Run comprehensive comparison between K-MOTE and SM-Kernel"""
    print("=== COMPREHENSIVE K-MOTE vs SM-KERNEL COMPARISON ===")
    
    # Define test patterns
    delta_t = torch.linspace(-10, 10, 500)
    test_patterns = {
        "Exponential Decay": exponential_decay(delta_t),
        "Power-Law Decay": power_law_decay(delta_t),
        "Periodic + Decay": periodic_influence(delta_t),
        "Complex Mixture": temporal_kernel_mixture(delta_t)
    }
    
    # Initialize models
    models = {
        "K-MOTE": KMOTE(input_dim=1, output_dim=1, wavelet_type='shock'),
        "SM-Kernel": SpectralMixtureKernel(input_dim=1, output_dim=1, n_mixtures=8)
    }
    
    # Compute model complexity
    print("\n📊 MODEL COMPLEXITY COMPARISON:")
    for model_name, model in models.items():
        complexity = compute_model_complexity(model)
        print(f"  {model_name}:")
        print(f"    • Parameters: {complexity['total_params']:,}")
        print(f"    • Model Size: {complexity['model_size_mb']:.2f} MB")
    
    # Performance comparison
    results = {}
    
    print(f"\n🎯 PERFORMANCE COMPARISON:")
    
    for pattern_name, y_pattern in test_patterns.items():
        print(f"\n[INFO] Testing pattern: {pattern_name}")
        
        # Normalize pattern
        y_mean = y_pattern.mean()
        y_std = y_pattern.std()
        y_norm = (y_pattern - y_mean) / y_std
        
        pattern_results = {}
        
        for model_name, model_class in models.items():
            print(f"  🔧 Training {model_name}...")
            
            # Create fresh model instance
            if model_name == "K-MOTE":
                model = KMOTE(input_dim=1, output_dim=1, wavelet_type='shock')
            else:
                model = SpectralMixtureKernel(input_dim=1, output_dim=1, n_mixtures=8)
            
            # Train and collect metrics
            metrics = train_model_with_metrics(model, delta_t, y_norm, 
                                             epochs=5000, model_name=model_name)
            
            # Evaluate final performance
            model.eval()
            with torch.no_grad():
                delta_t_input = delta_t.unsqueeze(0).unsqueeze(-1)
                pred_norm = model(delta_t_input).squeeze()
                pred = pred_norm * y_std + y_mean
                
                # Compute additional metrics
                mse = F.mse_loss(pred, y_pattern).item()
                mae = F.l1_loss(pred, y_pattern).item()
                
                # R² score
                ss_res = torch.sum((y_pattern - pred) ** 2).item()
                ss_tot = torch.sum((y_pattern - y_pattern.mean()) ** 2).item()
                r2_score = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            metrics.update({
                'mse': mse,
                'mae': mae,
                'r2_score': r2_score,
                'pred': pred.cpu().numpy()
            })
            
            pattern_results[model_name] = metrics
            
            print(f"    ✅ {model_name} Results:")
            print(f"       • Final Loss: {metrics['final_loss']:.6f}")
            print(f"       • MSE: {metrics['mse']:.6f}")
            print(f"       • R²: {metrics['r2_score']:.4f}")
            print(f"       • Training Time: {metrics['training_time']:.2f}s")
            print(f"       • Converged: {metrics['converged']} (epoch {metrics['converged_epoch']})")
        
        results[pattern_name] = pattern_results
        
        # Create comparison plot for this pattern
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'K-MOTE vs SM-Kernel: {pattern_name}', fontsize=16)
        
        # Plot 1: Model fits
        delta_t_np = delta_t.cpu().numpy()
        y_pattern_np = y_pattern.cpu().numpy()
        
        axes[0, 0].plot(delta_t_np, y_pattern_np, 'k-', linewidth=3, 
                       label='Ground Truth', alpha=0.8)
        
        for model_name in models.keys():
            pred = pattern_results[model_name]['pred']
            r2 = pattern_results[model_name]['r2_score']
            axes[0, 0].plot(delta_t_np, pred, '--', linewidth=2, 
                           label=f'{model_name} (R²={r2:.3f})')
        
        axes[0, 0].set_title('Model Fits Comparison')
        axes[0, 0].set_xlabel('Relative Time (Δt)')
        axes[0, 0].set_ylabel('Temporal Influence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Training curves
        for model_name in models.keys():
            loss_curve = pattern_results[model_name]['loss_curve']
            axes[0, 1].semilogy(loss_curve, label=f'{model_name}')
        
        axes[0, 1].set_title('Training Loss Curves')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss (log scale)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Residuals
        for i, model_name in enumerate(models.keys()):
            pred = pattern_results[model_name]['pred']
            residuals = y_pattern_np - pred
            axes[1, 0].scatter(pred, residuals, alpha=0.6, label=f'{model_name}', s=20)
        
        axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Residuals vs Predictions')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Performance metrics comparison
        metrics_names = ['MSE', 'MAE', 'Training Time (s)']
        k_mote_metrics = [
            pattern_results['K-MOTE']['mse'],
            pattern_results['K-MOTE']['mae'],
            pattern_results['K-MOTE']['training_time']
        ]
        sm_kernel_metrics = [
            pattern_results['SM-Kernel']['mse'],
            pattern_results['SM-Kernel']['mae'],
            pattern_results['SM-Kernel']['training_time']
        ]
        
        x = np.arange(len(metrics_names))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, k_mote_metrics, width, label='K-MOTE', alpha=0.8)
        axes[1, 1].bar(x + width/2, sm_kernel_metrics, width, label='SM-Kernel', alpha=0.8)
        
        axes[1, 1].set_title('Performance Metrics Comparison')
        axes[1, 1].set_xlabel('Metrics')
        axes[1, 1].set_ylabel('Values')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(metrics_names, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        safe_filename = pattern_name.replace(' ', '_').replace('+', 'plus')
        plt.savefig(f'analysis_figures_kmote_vs_sm/comparison_{safe_filename}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    return results

def generate_comparison_summary(results):
    """Generate comprehensive comparison summary"""
    print("\n" + "="*80)
    print("🏆 COMPREHENSIVE COMPARISON SUMMARY")
    print("="*80)
    
    patterns = list(results.keys())
    models = list(results[patterns[0]].keys())
    
    # Overall performance summary
    print(f"\n📊 OVERALL PERFORMANCE SUMMARY:")
    
    for metric in ['mse', 'mae', 'r2_score', 'training_time']:
        print(f"\n  📈 {metric.upper()}:")
        
        for pattern in patterns:
            print(f"    {pattern}:")
            for model in models:
                value = results[pattern][model][metric]
                if metric == 'r2_score':
                    print(f"      {model}: {value:.4f}")
                elif metric == 'training_time':
                    print(f"      {model}: {value:.2f}s")
                else:
                    print(f"      {model}: {value:.6f}")
    
    # Win/Loss analysis
    print(f"\n🥇 WIN/LOSS ANALYSIS:")
    
    metrics_to_analyze = ['mse', 'mae', 'training_time']  # Lower is better
    metrics_higher_better = ['r2_score']  # Higher is better
    
    wins = {model: 0 for model in models}
    
    for pattern in patterns:
        print(f"\n  📍 {pattern}:")
        
        for metric in metrics_to_analyze + metrics_higher_better:
            if metric in metrics_higher_better:
                # Higher is better
                best_model = max(models, key=lambda m: results[pattern][m][metric])
            else:
                # Lower is better
                best_model = min(models, key=lambda m: results[pattern][m][metric])
            
            wins[best_model] += 1
            print(f"    🏆 {metric}: {best_model}")
    
    print(f"\n🎯 OVERALL WINNER ANALYSIS:")
    for model, win_count in wins.items():
        total_competitions = len(patterns) * len(metrics_to_analyze + metrics_higher_better)
        win_rate = win_count / total_competitions * 100
        print(f"  {model}: {win_count}/{total_competitions} wins ({win_rate:.1f}%)")
    
    # Recommendation
    overall_winner = max(wins.keys(), key=lambda k: wins[k])
    
    print(f"\n🎯 RECOMMENDATION FOR kan_mammote_dual_kmote:")
    print("="*60)
    
    if overall_winner == "K-MOTE":
        print("✅ RECOMMENDED: Replace SM-Kernel with K-MOTE")
        print("   Reasons:")
        print("   • K-MOTE shows superior overall performance")
        print("   • Better modeling capability across diverse relative time patterns")
        print("   • Interpretable expert selection provides insights")
    else:
        print("⚠️  CONSIDER CAREFULLY: SM-Kernel shows competitive performance")
        print("   Reasons:")
        print("   • SM-Kernel performs better in overall metrics")
        print("   • May have advantages in specific scenarios")
        print("   • Consider hybrid approach or task-specific selection")
    
    # Pattern-specific insights
    print(f"\n📋 PATTERN-SPECIFIC INSIGHTS:")
    for pattern in patterns:
        pattern_results = results[pattern]
        
        # Find best model for this pattern based on R²
        best_model_r2 = max(models, key=lambda m: pattern_results[m]['r2_score'])
        best_r2 = pattern_results[best_model_r2]['r2_score']
        
        print(f"\n  📍 {pattern}:")
        print(f"    🏆 Best Model: {best_model_r2} (R² = {best_r2:.4f})")
        
        # Compare models
        for model in models:
            r2 = pattern_results[model]['r2_score']
            mse = pattern_results[model]['mse']
            time_taken = pattern_results[model]['training_time']
            print(f"    📊 {model}: R²={r2:.4f}, MSE={mse:.6f}, Time={time_taken:.1f}s")

def main():
    print("🚀 K-MOTE vs SM-KERNEL COMPARISON FOR RELATIVE TIME ENCODING")
    print("=" * 80)
    print("This analysis helps decide whether to replace SM-kernel with K-MOTE")
    print("in the kan_mammote_dual_kmote variant for relative time encoding.")
    print("=" * 80)
    
    # Run comprehensive comparison
    results = run_comprehensive_comparison()
    
    # Generate summary and recommendations
    generate_comparison_summary(results)
    
    print(f"\n✨ COMPARISON ANALYSIS COMPLETE!")
    print("📁 Detailed comparison plots saved in 'analysis_figures_kmote_vs_sm' directory")

if __name__ == '__main__':
    main()