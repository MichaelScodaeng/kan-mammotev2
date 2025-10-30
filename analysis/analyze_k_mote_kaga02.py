import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

# Global training configuration
MAX_EPOCHS = 20000

# Add the parent directory to Python path to import models
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# นำเข้าโมเดลจากไฟล์ k_mote.py
from models.time_encoders.k_mote import KMOTE, SplineKANLayer, FourierKANLayer, WaveletKANLayer

# Create output directory for saving figures
os.makedirs('analysis_figures', exist_ok=True)

# --- PART 0: Helper Functions & Setup ---

class SingleExpertModel(nn.Module):
    """Wrapper for individual experts to make them compatible with analysis"""
    def __init__(self, expert_class, **kwargs):
        super().__init__()
        self.expert = expert_class(input_dim=1, output_dim=1, **kwargs)
    def forward(self, x):
        return self.expert(x)

def train_model(model, t_data, y_true, max_epochs=MAX_EPOCHS, lr=2e-4, patience=500, min_delta=1e-6):
    """ฟังก์ชันสำหรับเทรนโมเดลเพื่อ fit ข้อมูล"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # เพิ่มมิติสุดท้ายเพื่อให้เข้ากับโมเดล (Batch, Seq_len, Dim)
    if t_data.dim() == 1:
        t_data = t_data.unsqueeze(-1)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(-1)

    best_loss = float('inf')
    patience_counter = 0
    loss_history = []

    with tqdm(range(max_epochs), desc="Training Model", leave=False) as pbar:
        for epoch in pbar:
            model.train()
            y_pred = model(t_data)
            
            # Check for NaN/Inf in predictions
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                print(f"WARNING: NaN/Inf detected in predictions at epoch {epoch+1}")
                # Try to recover
                for param in model.parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        param.data.normal_(0, 0.01)
                continue
            
            # Ensure consistent dimensions - squeeze extra dimensions from model output
            if y_pred.dim() > y_true.dim():
                y_pred = y_pred.squeeze(-1)
            
            loss = loss_fn(y_pred, y_true)
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: NaN/Inf loss at epoch {epoch+1}, skipping update")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
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
                print(f"Converged at epoch {epoch+1} (patience reached)")
                break
                
            # Additional convergence check
            if epoch > 100:
                recent_losses = loss_history[-50:]
                if max(recent_losses) - min(recent_losses) < min_delta:
                    print(f"Converged at epoch {epoch+1} (loss stabilized)")
                    break
            
            # Early stopping if loss becomes too large
            if current_loss > 1e6:
                print(f"Loss exploded at epoch {epoch+1}, stopping training")
                break

def plot_fit(ax, t, y_true, model, title):
    """ฟังก์ชันสำหรับพล็อตผลลัพธ์การ fit"""
    model.eval()
    with torch.no_grad():
        t_input = t.unsqueeze(-1)
        y_pred = model(t_input)
        
        # Ensure consistent dimensions for plotting
        if y_pred.dim() > 1:
            y_pred = y_pred.squeeze()
        y_pred = y_pred.cpu().numpy()
        
    ax.plot(t.cpu().numpy(), y_true.cpu().numpy(), label='Ground Truth', linewidth=3, alpha=0.7)
    ax.plot(t.cpu().numpy(), y_pred, label='Model Fit', linestyle='--', color='red')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_fit_with_gating(axes, t, y_true, model, title):
    """ฟังก์ชันสำหรับพล็อตผลลัพธ์การ fit พร้อมกับ gating weights สำหรับ K-MOTE"""
    model.eval()
    with torch.no_grad():
        t_input = t.unsqueeze(-1)
        
        # Check if model supports gating weights (K-MOTE)
        if hasattr(model, 'gating_network'):
            y_pred, gating_weights = model(t_input, return_weights=True)
            
            # Plot main fit
            if y_pred.dim() > 1:
                y_pred = y_pred.squeeze()
            y_pred_np = y_pred.cpu().numpy()
            
            axes[0].plot(t.cpu().numpy(), y_true.cpu().numpy(), label='Ground Truth', linewidth=3, alpha=0.7)
            axes[0].plot(t.cpu().numpy(), y_pred_np, label='K-MOTE Fit', linestyle='--', color='red')
            axes[0].set_title(f"{title} - Model Fit")
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)
            
            # Plot gating weights
            if gating_weights.dim() > 2:
                gating_weights = gating_weights.squeeze()
            gating_weights_np = gating_weights.cpu().numpy()
            
            expert_names = ['B-Spline', 'Fourier', 'Wavelet', 'RBF']
            colors = ['green', 'blue', 'red', 'magenta']
            
            for i in range(gating_weights_np.shape[1]):
                axes[1].plot(t.cpu().numpy(), gating_weights_np[:, i], 
                           color=colors[i], linewidth=2, label=f'{expert_names[i]} Weight')
            
            axes[1].set_title(f"{title} - Expert Gating Weights")
            axes[1].set_xlabel("Time (t)")
            axes[1].set_ylabel("Expert Weight")
            axes[1].legend()
            axes[1].grid(True, linestyle='--', alpha=0.6)
            
            return gating_weights_np
        else:
            # Regular single expert plot
            y_pred = model(t_input)
            if y_pred.dim() > 1:
                y_pred = y_pred.squeeze()
            y_pred_np = y_pred.cpu().numpy()
            
            axes[0].plot(t.cpu().numpy(), y_true.cpu().numpy(), label='Ground Truth', linewidth=3, alpha=0.7)
            axes[0].plot(t.cpu().numpy(), y_pred_np, label='Model Fit', linestyle='--', color='red')
            axes[0].set_title(f"{title} - Model Fit")
            axes[0].legend()
            axes[0].grid(True, linestyle='--', alpha=0.6)
            
            # Empty second plot for consistency
            axes[1].text(0.5, 0.5, 'No Gating\n(Single Expert)', ha='center', va='center', 
                        transform=axes[1].transAxes, fontsize=12)
            axes[1].set_title(f"{title} - No Gating Available")
            
            return None

# สร้างข้อมูลสังเคราะห์ที่เหมาะกับแต่ละ expert
def generate_smooth_trend_data(t):
    """Data that should favor B-Spline expert - smooth polynomial trends"""
    return 0.1 * t**3 - 0.5 * t**2 + 0.3 * t + 0.2

def generate_periodic_data(t):
    """Data that should favor Fourier expert - complex periodic patterns"""
    return (torch.sin(2 * torch.pi * t / 3) + 
            0.5 * torch.cos(2 * torch.pi * t / 1.5) + 
            0.3 * torch.sin(2 * torch.pi * t / 7))

def generate_abrupt_change_data(t):
    """Data that should favor Wavelet expert - sudden shocks and discontinuities"""
    # Create shock events at different times
    shock1 = torch.where(t > 2, 1.0 * torch.exp(-(t-2)), 0.0)  # Sudden onset at t=2
    shock2 = torch.where(t > -3, -0.8 * torch.exp(-2*(t+3)), 0.0)  # Shock at t=-3
    shock3 = torch.where((t > 5) & (t < 6), 1.5, 0.0)  # Step function
    return shock1 + shock2 + shock3

def generate_localized_event_data(t):
    """Data that should favor RBF expert - localized Gaussian-like events"""
    event1 = 1.2 * torch.exp(-((t - 1)**2) / 0.5)  # Gaussian peak at t=1
    event2 = -0.8 * torch.exp(-((t + 4)**2) / 0.8)  # Negative peak at t=-4
    event3 = 0.6 * torch.exp(-((t - 6)**2) / 0.3)   # Sharp peak at t=6
    return event1 + event2 + event3

def generate_mixed_pattern_data(t):
    """Complex mixed pattern combining all expert domains"""
    smooth_trend = 0.05 * t**2  # B-spline domain
    periodic_part = 0.4 * torch.sin(2 * torch.pi * t / 4)  # Fourier domain
    shock_event = torch.where(t > 3, 1.0 * torch.exp(-(t-3)), 0.0)  # Wavelet domain
    localized_event = 0.8 * torch.exp(-((t + 2)**2) / 0.6)  # RBF domain
    return smooth_trend + periodic_part + shock_event + localized_event

# --- PART 1: Fitting Ability Analysis ---

def run_fitting_analysis():
    print("--- Starting Part 1: Expert Specialization Analysis ---")
    
    # 1. สร้างชุดข้อมูลที่เหมาะกับแต่ละ expert
    t = torch.linspace(-8, 8, 400)
    target_functions = {
        "Smooth Trends (B-Spline Domain)": generate_smooth_trend_data(t),
        "Periodic Patterns (Fourier Domain)": generate_periodic_data(t), 
        "Abrupt Changes (Wavelet Domain)": generate_abrupt_change_data(t),
        "Localized Events (RBF Domain)": generate_localized_event_data(t)
    }

    # 2. สร้างโมเดลที่จะทดสอบ - แต่ละ expert แยกต่างหาก + K-MOTE รวม
    models_to_test = {
        "B-Spline Expert": SingleExpertModel(SplineKANLayer, basis_function='b_spline'),
        "Fourier Expert": SingleExpertModel(FourierKANLayer),
        "Wavelet Expert": SingleExpertModel(WaveletKANLayer, wavelet_type='shock'),  # Use more stable wavelet
        "Full K-MOTE (3 Experts)": KMOTE(input_dim=1, output_dim=1, wavelet_type='shock')
    }
    
    # 3. เทรนและพล็อตผลลัพธ์แต่ละแบบ - 5 columns in one row
    for func_name, y_true in target_functions.items():
        print(f"\n[INFO] Testing on: {func_name}")
        
        # Create single figure with 5 columns: B-Spline, Fourier, Wavelet, Full K-MOTE, Gating
        fig, axes = plt.subplots(1, 5, figsize=(25, 5))
        fig.suptitle(f'Expert Analysis on: {func_name}', fontsize=16)
        
        model_losses = {}
        trained_models = {}
        
        # Define expert configurations
        expert_configs = [
            ("B-Spline Expert", lambda: SingleExpertModel(SplineKANLayer, basis_function='b_spline')),
            ("Fourier Expert", lambda: SingleExpertModel(FourierKANLayer)),
            ("Wavelet Expert", lambda: SingleExpertModel(WaveletKANLayer, wavelet_type='shock')),
        ]
        
        # Train individual experts (columns 0-2)
        for i, (model_name, model_factory) in enumerate(expert_configs):
            print(f"  - Training {model_name}...")
            
            model = model_factory()
            final_loss = train_model_with_loss_return(model, t, y_true)
            model_losses[model_name] = final_loss
            trained_models[model_name] = model
            
            plot_fit(axes[i], t, y_true, model, f"{model_name}\nLoss: {final_loss:.4f}")
        
        # Train K-MOTE (column 3)
        print(f"  - Training Full K-MOTE...")
        kmote_model = KMOTE(input_dim=1, output_dim=1, wavelet_type='shock')
        kmote_loss = train_model_with_loss_return(kmote_model, t, y_true)
        model_losses["Full K-MOTE"] = kmote_loss
        trained_models["Full K-MOTE"] = kmote_model
        
        plot_fit(axes[3], t, y_true, kmote_model, f"Full K-MOTE\nLoss: {kmote_loss:.4f}")
        
        # Plot gating weights (column 4)
        print(f"  - Analyzing gating weights...")
        kmote_model.eval()
        with torch.no_grad():
            t_input = t.unsqueeze(-1)
            y_pred, gating_weights = kmote_model(t_input, return_weights=True)
            
            # Ensure consistent dimensions
            if gating_weights.dim() > 2:
                gating_weights = gating_weights.squeeze()
            gating_weights_np = gating_weights.cpu().numpy()
            
            expert_names = ['B-Spline', 'Fourier', 'Wavelet', 'RBF']
            colors = ['green', 'blue', 'red', 'magenta']
            
            for j in range(gating_weights_np.shape[1]):
                axes[4].plot(t.cpu().numpy(), gating_weights_np[:, j], 
                           color=colors[j], linewidth=2, label=f'{expert_names[j]}')
            
            axes[4].set_title("Expert Gating Weights")
            axes[4].set_xlabel("Time (t)")
            axes[4].set_ylabel("Weight")
            axes[4].legend()
            axes[4].grid(True, linestyle='--', alpha=0.6)
            
            # Analyze gating patterns
            print(f"    📊 Expert Usage Analysis:")
            avg_weights = np.mean(gating_weights_np, axis=0)
            for j, (name, weight) in enumerate(zip(expert_names, avg_weights)):
                print(f"      {name}: {weight:.3f} ({weight*100:.1f}%)")
            
            # Find dominant expert at key regions
            dominant_expert_idx = np.argmax(gating_weights_np, axis=1)
            dominant_expert_counts = np.bincount(dominant_expert_idx, minlength=4)
            print(f"    📈 Dominant Expert Regions:")
            for j, (name, count) in enumerate(zip(expert_names, dominant_expert_counts)):
                percentage = count / len(t) * 100
                print(f"      {name}: {count}/{len(t)} points ({percentage:.1f}%)")
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        safe_filename = func_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        plt.savefig(f'analysis_figures/comprehensive_analysis_{safe_filename}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # แสดงผลการเปรียบเทียบ loss
        print(f"  📊 Loss Comparison for {func_name}:")
        sorted_losses = sorted(model_losses.items(), key=lambda x: x[1])
        for rank, (name, loss) in enumerate(sorted_losses, 1):
            print(f"    {rank}. {name}: {loss:.6f}")
        print()

def train_model_with_loss_return(model, t_data, y_true, max_epochs=MAX_EPOCHS*2, lr=1e-4, 
                               patience=300, min_delta=1e-7):
    """Train model until convergence and return final loss and training info"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    if t_data.dim() == 1:
        t_data = t_data.unsqueeze(-1)
    if y_true.dim() == 1:
        y_true = y_true.unsqueeze(-1)

    best_loss = float('inf')
    patience_counter = 0
    loss_history = []
    
    # Use tqdm for progress bar
    with tqdm(range(max_epochs), desc="Training", leave=False) as pbar:
        for epoch in pbar:
            model.train()
            y_pred = model(t_data)
            
            # Check for NaN/Inf in predictions
            if torch.isnan(y_pred).any() or torch.isinf(y_pred).any():
                print(f"    WARNING: NaN/Inf detected in predictions at epoch {epoch+1}")
                # Try to recover by resetting problematic parameters
                for param in model.parameters():
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        param.data.normal_(0, 0.01)
                continue
            
            # Ensure consistent dimensions - squeeze extra dimensions from model output
            if y_pred.dim() > y_true.dim():
                y_pred = y_pred.squeeze(-1)
                
            loss = loss_fn(y_pred, y_true)
            
            # Check for NaN/Inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"    WARNING: NaN/Inf loss at epoch {epoch+1}, skipping update")
                continue
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent explosion
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
    
    # Final safety check
    final_loss = best_loss if best_loss != float('inf') else float('inf')
    print(f"    Final Loss: {final_loss:.6f} (converged in {epoch+1} epochs)")
    return final_loss


def generate_mixed_pattern_data(t):
    """Generate data with multiple temporal patterns"""
    smooth_trend = 0.3 * t**2 - 0.1 * t
    periodic = 0.5 * torch.sin(4 * t) + 0.3 * torch.sin(8 * t)
    shock_events = torch.zeros_like(t)
    shock_events[torch.abs(t + 2) < 0.5] = 2.0  # Shock at t=-2
    shock_events[torch.abs(t - 3) < 0.3] = -1.5  # Shock at t=3
    localized = 1.2 * torch.exp(-((t - 1)**2) / 0.5)  # Peak at t=1
    return smooth_trend + periodic + shock_events + localized

# --- PART 2: Interpretability Analysis (Gating Network) ---

def run_interpretability_analysis():
    print("\n--- Starting Part 2: K-MOTE Gating Analysis on Mixed Patterns ---")
    
    # 1. ใช้ข้อมูลแบบผสมที่มีองค์ประกอบของทุก expert
    t_mixed = torch.linspace(-8, 8, 400)
    y_mixed = generate_mixed_pattern_data(t_mixed)
    
    print("[INFO] Training K-MOTE on mixed pattern data...")
    k_mote_model = KMOTE(input_dim=1, output_dim=1, wavelet_type='shock')  # Use stable wavelet
    train_model(k_mote_model, t_mixed, y_mixed, max_epochs=MAX_EPOCHS, lr=2e-4)
    
    # 2. ดึงค่า prediction และ gating weights ออกมา
    k_mote_model.eval()
    with torch.no_grad():
        t_input = t_mixed.unsqueeze(-1)
        y_pred, gating_weights = k_mote_model(t_input, return_weights=True)
        
        # Ensure consistent dimensions
        if y_pred.dim() > 1:
            y_pred = y_pred.squeeze()
        if gating_weights.dim() > 2:
            gating_weights = gating_weights.squeeze()
            
        y_pred_np = y_pred.cpu().numpy()
        gating_weights_np = gating_weights.cpu().numpy()
        t_mixed_np = t_mixed.cpu().numpy()

    # 3. Decompose the mixed signal to show which parts should activate which experts
    smooth_component = 0.05 * t_mixed**2
    periodic_component = 0.4 * torch.sin(2 * torch.pi * t_mixed / 4)
    shock_component = torch.where(t_mixed > 3, 1.0 * torch.exp(-(t_mixed-3)), 0.0)
    localized_component = 0.8 * torch.exp(-((t_mixed + 2)**2) / 0.6)

    # 4. พล็อตกราฟวิเคราะห์แบบละเอียด
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("K-MOTE Expert Specialization Analysis", fontsize=16)

    # กราฟที่ 1: Signal decomposition
    axes[0].plot(t_mixed_np, y_mixed.cpu().numpy(), 'k-', linewidth=3, label='Mixed Signal', alpha=0.8)
    axes[0].plot(t_mixed_np, smooth_component.cpu().numpy(), 'g--', label='Smooth Trend (B-Spline)', alpha=0.7)
    axes[0].plot(t_mixed_np, periodic_component.cpu().numpy(), 'b--', label='Periodic (Fourier)', alpha=0.7)
    axes[0].plot(t_mixed_np, shock_component.cpu().numpy(), 'r--', label='Shock Event (Wavelet)', alpha=0.7)
    axes[0].plot(t_mixed_np, localized_component.cpu().numpy(), 'm--', label='Localized Event (RBF)', alpha=0.7)
    axes[0].set_title("Mixed Signal Decomposition")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # เพิ่มแถบสีเพื่อไฮไลท์ช่วงสำคัญ
    axes[0].axvspan(-3, -1, color='magenta', alpha=0.2, label='RBF Region')
    axes[0].axvspan(2.5, 4.5, color='red', alpha=0.2, label='Shock Region')

    # กราฟที่ 2: Model fit
    axes[1].plot(t_mixed_np, y_mixed.cpu().numpy(), 'k-', linewidth=3, label='Ground Truth', alpha=0.8)
    axes[1].plot(t_mixed_np, y_pred_np, 'r--', linewidth=2, label='K-MOTE Prediction')
    axes[1].set_title("K-MOTE Model Fit")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # เพิ่มแถบสีในกราฟที่ 2 ด้วย
    axes[1].axvspan(-3, -1, color='magenta', alpha=0.2)
    axes[1].axvspan(2.5, 4.5, color='red', alpha=0.2)

    # กราฟที่ 3: Expert weights with expected activations
    expert_names = ['B-Spline', 'Fourier', 'Shock Wavelet', 'RBF']
    colors = ['green', 'blue', 'red', 'magenta']
    
    for i in range(gating_weights_np.shape[1]):
        axes[2].plot(t_mixed_np, gating_weights_np[:, i], 
                    color=colors[i], linewidth=2, label=f'{expert_names[i]} Weight')
    
    axes[2].set_title("Expert Gating Weights Over Time")
    axes[2].set_xlabel("Time (t)")
    axes[2].set_ylabel("Expert Weight (Softmax)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # เพิ่มแถบสีในกราฟที่ 3 พร้อม annotation
    axes[2].axvspan(-3, -1, color='magenta', alpha=0.2)
    axes[2].axvspan(2.5, 4.5, color='red', alpha=0.2)
    axes[2].text(-2, 0.8, 'Expected:\nRBF Expert\nActivation', ha='center', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="magenta", alpha=0.3))
    axes[2].text(3.5, 0.8, 'Expected:\nWavelet Expert\nActivation', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig('analysis_figures/k_mote_gating_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. วิเคราะห์ผลลัพธ์
    print("\n🔍 Expert Activation Analysis:")
    
    # หาช่วงที่แต่ละ expert มี weight สูงสุด
    dominant_expert_per_time = np.argmax(gating_weights_np, axis=1)
    
    # วิเคราะห์ในช่วงต่างๆ
    regions = {
        "Smooth Region (-8 to -4)": (t_mixed_np >= -8) & (t_mixed_np <= -4),
        "RBF Region (-3 to -1)": (t_mixed_np >= -3) & (t_mixed_np <= -1), 
        "Periodic Region (-1 to 2.5)": (t_mixed_np >= -1) & (t_mixed_np <= 2.5),
        "Shock Region (2.5 to 4.5)": (t_mixed_np >= 2.5) & (t_mixed_np <= 4.5),
        "Mixed Region (4.5 to 8)": (t_mixed_np >= 4.5) & (t_mixed_np <= 8)
    }
    
    for region_name, mask in regions.items():
        if np.any(mask):
            region_weights = gating_weights_np[mask]
            avg_weights = np.mean(region_weights, axis=0)
            dominant_expert = np.argmax(avg_weights)
            
            print(f"  📊 {region_name}:")
            print(f"     Dominant Expert: {expert_names[dominant_expert]} ({avg_weights[dominant_expert]:.3f})")
            print(f"     All weights: {', '.join([f'{name}: {w:.3f}' for name, w in zip(expert_names, avg_weights)])}")
            print()
    
    return k_mote_model, gating_weights_np


# --- PART 3: Expert Capability Matrix Analysis ---

def run_expert_capability_matrix():
    """Generate a capability matrix showing which expert performs best on which type of data"""
    print("\n--- Starting Part 3: Expert Capability Matrix Analysis ---")
    
    t = torch.linspace(-8, 8, 400)
    
    # Define test datasets
    test_datasets = {
        "Smooth Polynomial": generate_smooth_trend_data(t),
        "Complex Periodic": generate_periodic_data(t),
        "Shock Events": generate_abrupt_change_data(t),
        "Localized Peaks": generate_localized_event_data(t),
        "Mixed Pattern": generate_mixed_pattern_data(t)
    }
    
    # Define experts to test
    expert_configs = {
        "B-Spline": lambda: SingleExpertModel(SplineKANLayer, basis_function='b_spline'),
        "Fourier": lambda: SingleExpertModel(FourierKANLayer),
        "Shock Wavelet": lambda: SingleExpertModel(WaveletKANLayer, wavelet_type='shock'),  # Use stable wavelet
        "RBF": lambda: SingleExpertModel(SplineKANLayer, basis_function='rbf'),
        "K-MOTE (4 Experts)": lambda: KMOTE(input_dim=1, output_dim=1, wavelet_type='shock')
    }
    
    # Performance matrix
    results_matrix = {}
    
    print("🧪 Testing expert performance across different data types...")
    print("=" * 70)
    
    for dataset_name, y_data in test_datasets.items():
        print(f"\n📊 Testing on: {dataset_name}")
        results_matrix[dataset_name] = {}
        
        for expert_name, expert_factory in expert_configs.items():
            print(f"  Training {expert_name}...", end=" ")
            
            # Create fresh model instance
            model = expert_factory()
            
            # Train and get final loss
            final_loss = train_model_with_loss_return(model, t, y_data, max_epochs=MAX_EPOCHS, lr=2e-4)
            results_matrix[dataset_name][expert_name] = final_loss
            
            print(f"Loss: {final_loss:.6f}")
    
    # Create performance visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Convert to matrices for heatmap
    datasets = list(test_datasets.keys())
    experts = list(expert_configs.keys())
    
    loss_matrix = np.array([[results_matrix[dataset][expert] for expert in experts] 
                           for dataset in datasets])
    
    # Plot 1: Raw loss heatmap
    im1 = ax1.imshow(loss_matrix, cmap='viridis_r', aspect='auto')
    ax1.set_xticks(range(len(experts)))
    ax1.set_yticks(range(len(datasets)))
    ax1.set_xticklabels(experts, rotation=45, ha='right')
    ax1.set_yticklabels(datasets)
    ax1.set_title('Performance Matrix (Lower Loss = Better)')
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(experts)):
            text = ax1.text(j, i, f'{loss_matrix[i, j]:.4f}',
                           ha="center", va="center", color="white", fontweight='bold')
    
    plt.colorbar(im1, ax=ax1)
    
    # Plot 2: Relative performance (best expert = 1.0, others scaled)
    relative_matrix = np.zeros_like(loss_matrix)
    for i, dataset in enumerate(datasets):
        best_loss = np.min(loss_matrix[i])
        relative_matrix[i] = best_loss / loss_matrix[i]  # Higher = better
    
    im2 = ax2.imshow(relative_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax2.set_xticks(range(len(experts)))
    ax2.set_yticks(range(len(datasets)))
    ax2.set_xticklabels(experts, rotation=45, ha='right')
    ax2.set_yticklabels(datasets)
    ax2.set_title('Relative Performance (1.0 = Best)')
    
    # Add text annotations
    for i in range(len(datasets)):
        for j in range(len(experts)):
            color = "white" if relative_matrix[i, j] < 0.5 else "black"
            text = ax2.text(j, i, f'{relative_matrix[i, j]:.3f}',
                           ha="center", va="center", color=color, fontweight='bold')
    
    plt.colorbar(im2, ax=ax2)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('analysis_figures/expert_capability_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary analysis
    print("\n🏆 EXPERT SPECIALIZATION SUMMARY:")
    print("=" * 70)
    
    for i, dataset in enumerate(datasets):
        best_expert_idx = np.argmin(loss_matrix[i])
        best_expert = experts[best_expert_idx]
        best_loss = loss_matrix[i, best_expert_idx]
        
        print(f"📈 {dataset}:")
        print(f"   🥇 Best Expert: {best_expert} (Loss: {best_loss:.6f})")
        
        # Show top 3 performers
        sorted_indices = np.argsort(loss_matrix[i])
        for rank, idx in enumerate(sorted_indices[:3], 2):
            expert_name = experts[idx]
            loss_val = loss_matrix[i, idx]
            relative_perf = relative_matrix[i, idx]
            print(f"   🥈 #{rank}: {expert_name} (Loss: {loss_val:.6f}, Relative: {relative_perf:.3f})")
        print()
    
    return results_matrix

# --- Main Execution ---
if __name__ == '__main__':
    print("🚀 K-MOTE Comprehensive Expert Analysis")
    print("=" * 60)
    
    # Part 1: Individual expert specialization testing
    run_fitting_analysis()
    
    # Part 2: Mixed pattern gating analysis  
    run_interpretability_analysis()
    
    # Part 3: Comprehensive capability matrix
    run_expert_capability_matrix()
    
    print("\n✨ Analysis Complete! ")
    print("📋 Summary:")
    print("   - Part 1: Shows each expert's strength on their specialized data")
    print("   - Part 2: Demonstrates K-MOTE's intelligent expert selection")
    print("   - Part 3: Provides quantitative performance comparison matrix")
    
    print("\n📁 Generated Figures:")
    figures_dir = 'analysis_figures'
    if os.path.exists(figures_dir):
        for filename in sorted(os.listdir(figures_dir)):
            if filename.endswith('.png'):
                print(f"   - {os.path.join(figures_dir, filename)}")
    else:
        print("   - No figures directory found")