#!/usr/bin/env python3
"""
K-MOTE Expert Visualization Script

This script generates clear visualizations for each type of K-MOTE expert:
1. B-Spline Expert - Smooth piecewise polynomial basis functions
2. Fourier Expert - Harmonic cosine/sine basis functions  
3. Wavelet Expert - Various wavelet types (Shock, Morlet, Mexican Hat, Haar)

Each visualization shows the characteristic shape and behavior of the expert type.
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from matplotlib.patches import Rectangle

# Set style for professional plots
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

def create_expert_visualizations():
    """Create a comprehensive visualization of all K-MOTE experts"""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    fig.suptitle('K-MOTE Expert Types: Basis Function Characteristics', fontsize=20, fontweight='bold')
    
    # Define common time range
    t = np.linspace(-3, 3, 1000)
    
    # ================================
    # 1. B-SPLINE EXPERT
    # ================================
    ax1 = plt.subplot(2, 4, 1)
    
    # B-spline basis functions of different orders
    def b_spline_basis(t, knots, order):
        """Compute B-spline basis function"""
        n = len(knots) - order - 1
        basis = np.zeros((len(t), n))
        
        for i in range(n):
            # Start with order 0 (indicator function)
            if order == 0:
                basis[:, i] = ((t >= knots[i]) & (t < knots[i+1])).astype(float)
            else:
                # Recursive formula for higher orders
                for k in range(1, order + 1):
                    if k == 1:
                        prev_basis = ((t >= knots[i]) & (t < knots[i+1])).astype(float)
                    
                    # Cox-de Boor recursion
                    left_term = np.zeros_like(t)
                    right_term = np.zeros_like(t)
                    
                    if knots[i+k] != knots[i]:
                        left_term = (t - knots[i]) / (knots[i+k] - knots[i]) * prev_basis
                    
                    if i+k+1 < len(knots) and knots[i+k+1] != knots[i+1]:
                        next_basis = ((t >= knots[i+1]) & (t < knots[i+2])).astype(float) if k == 1 else basis[:, i+1] if i+1 < n else np.zeros_like(t)
                        right_term = (knots[i+k+1] - t) / (knots[i+k+1] - knots[i+1]) * next_basis
                    
                    prev_basis = left_term + right_term
                
                basis[:, i] = prev_basis
        
        return basis
    
    # Create knot vector for cubic B-splines
    knots = np.array([-3, -2, -1, 0, 1, 2, 3, 4, 5])  # Extended knot vector
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Plot several B-spline basis functions
    for i in range(min(5, len(knots)-4)):
        knot_subset = knots[i:i+5]  # Local knot vector for each basis function
        t_local = np.linspace(knot_subset[0], knot_subset[-1], 200)
        
        # Simple cubic B-spline shape (approximation)
        center = (knot_subset[1] + knot_subset[2]) / 2
        width = knot_subset[2] - knot_subset[1]
        
        # Create smooth B-spline-like curve
        normalized_t = (t_local - center) / (width * 1.5)
        basis_val = np.maximum(0, (1 - np.abs(normalized_t))**3)  # Approximation
        
        mask = (t_local >= knot_subset[0]) & (t_local <= knot_subset[-1])
        basis_val = basis_val * mask
        
        ax1.plot(t_local, basis_val, color=colors[i % len(colors)], linewidth=2.5, 
                label=f'B-spline {i+1}')
    
    ax1.set_title('B-Spline Expert\n(Smooth Piecewise Polynomials)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Time (t)', fontsize=12)
    ax1.set_ylabel('Basis Value', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-3, 3)
    
    # Add text annotation
    ax1.text(0.05, 0.95, '• Smooth transitions\n• Local support\n• Piecewise polynomial\n• Controllable via knots', 
             transform=ax1.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # ================================
    # 2. FOURIER EXPERT - COSINE
    # ================================
    ax2 = plt.subplot(2, 4, 2)
    
    # Fourier cosine basis functions
    harmonics = [1, 2, 3, 4, 5]
    colors_cos = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i, k in enumerate(harmonics):
        cosine_basis = np.cos(k * t)
        ax2.plot(t, cosine_basis, color=colors_cos[i], linewidth=2.5, 
                label=f'cos({k}t)')
    
    ax2.set_title('Fourier Expert - Cosine\n(Harmonic Components)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time (t)', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-3, 3)
    ax2.set_ylim(-1.2, 1.2)
    
    # Add text annotation
    ax2.text(0.05, 0.95, '• Periodic patterns\n• Global support\n• Frequency domain\n• Harmonic analysis', 
             transform=ax2.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    
    # ================================
    # 3. FOURIER EXPERT - SINE
    # ================================
    ax3 = plt.subplot(2, 4, 3)
    
    # Fourier sine basis functions
    colors_sin = ['#e67e22', '#1abc9c', '#8e44ad', '#34495e', '#c0392b']
    
    for i, k in enumerate(harmonics):
        sine_basis = np.sin(k * t)
        ax3.plot(t, sine_basis, color=colors_sin[i], linewidth=2.5, 
                label=f'sin({k}t)')
    
    ax3.set_title('Fourier Expert - Sine\n(Harmonic Components)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Time (t)', fontsize=12)
    ax3.set_ylabel('Amplitude', fontsize=12)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(-3, 3)
    ax3.set_ylim(-1.2, 1.2)
    
    # Add text annotation
    ax3.text(0.05, 0.95, '• Periodic patterns\n• Global support\n• Phase-shifted\n• Orthogonal to cosine', 
             transform=ax3.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    # ================================
    # 4. COMBINED FOURIER EXPERT
    # ================================
    ax4 = plt.subplot(2, 4, 4)
    
    # Combined Fourier representation
    base = 0.3 * np.ones_like(t)  # Base/bias term
    fourier_sum = base + 0.5*np.cos(t) + 0.3*np.sin(2*t) + 0.2*np.cos(3*t) + 0.1*np.sin(4*t)
    
    ax4.plot(t, base, '--', color='gray', linewidth=2, label='Base', alpha=0.7)
    ax4.plot(t, fourier_sum, color='#2c3e50', linewidth=3, label='Combined Fourier')
    ax4.fill_between(t, base, fourier_sum, alpha=0.3, color='#3498db')
    
    ax4.set_title('Combined Fourier Expert\n(Base + Harmonics)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time (t)', fontsize=12)
    ax4.set_ylabel('Output', fontsize=12)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(-3, 3)
    
    # Add text annotation
    ax4.text(0.05, 0.95, '• Learnable weights\n• Multiple frequencies\n• Complex patterns\n• Global representation', 
             transform=ax4.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    # ================================
    # 5. MORLET WAVELET
    # ================================
    ax5 = plt.subplot(2, 4, 5)
    
    # Morlet wavelet family
    def morlet_wavelet(t, sigma=1.0, frequency=5.0):
        """Morlet wavelet"""
        c = math.pi**(-0.25)
        return c * np.exp(-0.5 * (t/sigma)**2) * np.cos(frequency * t)
    
    # Different scales of Morlet wavelets
    scales = [0.5, 0.8, 1.0, 1.5, 2.0]
    colors_morlet = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db', '#9b59b6']
    
    for i, scale in enumerate(scales):
        morlet_vals = morlet_wavelet(t, sigma=scale, frequency=5.0)
        ax5.plot(t, morlet_vals, color=colors_morlet[i], linewidth=2.5, 
                label=f'σ={scale}')
    
    ax5.set_title('Morlet Wavelet Expert\n(Localized Oscillations)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Time (t)', fontsize=12)
    ax5.set_ylabel('Amplitude', fontsize=12)
    ax5.legend(fontsize=10)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim(-3, 3)
    
    # Add text annotation
    ax5.text(0.05, 0.95, '• Time-frequency\n• Gaussian envelope\n• Oscillatory core\n• Multi-scale analysis', 
             transform=ax5.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.7))
    
    # ================================
    # 6. SHOCK WAVELET
    # ================================
    ax6 = plt.subplot(2, 4, 6)
    
    # Shock wavelet (asymmetric, good for abrupt changes)
    def shock_wavelet(t, asymmetry=0.0, steepness=2.0):
        """Shock wavelet optimized for abrupt changes"""
        asym = np.tanh(asymmetry)
        steep = steepness + 0.1
        
        left_exponent = np.clip(steep * t * (1 + asym), -10, 10)
        right_exponent = np.clip(-steep * t * (1 - asym), -10, 10)
        
        shock_profile = np.where(t < 0, np.exp(left_exponent), np.exp(right_exponent))
        
        freq = np.clip(steep, None, 3.0)
        oscillation = np.cos(freq * t)
        
        return np.clip(shock_profile * oscillation, -100, 100)
    
    # Different shock wavelet configurations
    configs = [(0.0, 1.5), (0.5, 2.0), (-0.5, 2.0), (0.0, 3.0)]
    colors_shock = ['#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    labels_shock = ['Symmetric', 'Right-skewed', 'Left-skewed', 'High frequency']
    
    for i, (asym, steep) in enumerate(configs):
        shock_vals = shock_wavelet(t, asymmetry=asym, steepness=steep)
        ax6.plot(t, shock_vals, color=colors_shock[i], linewidth=2.5, 
                label=labels_shock[i])
    
    ax6.set_title('Shock Wavelet Expert\n(Abrupt Changes)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Time (t)', fontsize=12)
    ax6.set_ylabel('Amplitude', fontsize=12)
    ax6.legend(fontsize=10)
    ax6.grid(True, alpha=0.3)
    ax6.set_xlim(-3, 3)
    
    # Add text annotation
    ax6.text(0.05, 0.95, '• Asymmetric shape\n• Sharp transitions\n• Configurable skew\n• Discontinuity modeling', 
             transform=ax6.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightpink', alpha=0.7))
    
    # ================================
    # 7. MEXICAN HAT WAVELET
    # ================================
    ax7 = plt.subplot(2, 4, 7)
    
    # Mexican hat wavelet family
    def mexican_hat_wavelet(t, sigma=1.0):
        """Mexican hat (Ricker) wavelet"""
        c = 2 / (np.sqrt(3) * np.pi**(1/4))
        normalized_t = t / sigma
        return c * (1 - normalized_t**2) * np.exp(-normalized_t**2 / 2) / np.sqrt(sigma)
    
    # Different scales
    scales_mex = [0.5, 0.8, 1.0, 1.5, 2.0]
    colors_mex = ['#e67e22', '#1abc9c', '#8e44ad', '#34495e', '#c0392b']
    
    for i, scale in enumerate(scales_mex):
        mex_vals = mexican_hat_wavelet(t, sigma=scale)
        ax7.plot(t, mex_vals, color=colors_mex[i], linewidth=2.5, 
                label=f'σ={scale}')
    
    ax7.set_title('Mexican Hat Wavelet\n(Edge Detection)', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Time (t)', fontsize=12)
    ax7.set_ylabel('Amplitude', fontsize=12)
    ax7.legend(fontsize=10)
    ax7.grid(True, alpha=0.3)
    ax7.set_xlim(-3, 3)
    
    # Add text annotation
    ax7.text(0.05, 0.95, '• Zero mean\n• Symmetric\n• Edge detection\n• Second derivative', 
             transform=ax7.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7))
    
    # ================================
    # 8. HAAR WAVELET
    # ================================
    ax8 = plt.subplot(2, 4, 8)
    
    # Haar wavelet family
    def haar_wavelet(t, scale=1.0, shift=0.0):
        """Haar wavelet - simplest orthogonal wavelet"""
        normalized_t = (t - shift) / scale
        result = np.zeros_like(t)
        
        # Haar wavelet definition
        mask1 = (normalized_t >= 0) & (normalized_t < 0.5)
        mask2 = (normalized_t >= 0.5) & (normalized_t < 1.0)
        
        result[mask1] = 1.0 / np.sqrt(scale)
        result[mask2] = -1.0 / np.sqrt(scale)
        
        return result
    
    # Different scales and shifts
    configs_haar = [(1.0, 0), (0.5, 0), (0.25, -0.5), (0.25, 0), (0.25, 0.5)]
    colors_haar = ['#2c3e50', '#e74c3c', '#f39c12', '#2ecc71', '#3498db']
    labels_haar = ['Scale 1.0', 'Scale 0.5', 'Scale 0.25 (L)', 'Scale 0.25 (M)', 'Scale 0.25 (R)']
    
    for i, (scale, shift) in enumerate(configs_haar):
        haar_vals = haar_wavelet(t, scale=scale, shift=shift)
        ax8.plot(t, haar_vals, color=colors_haar[i], linewidth=2.5, 
                label=labels_haar[i], drawstyle='steps-mid')
    
    ax8.set_title('Haar Wavelet Expert\n(Piecewise Constant)', fontsize=14, fontweight='bold')
    ax8.set_xlabel('Time (t)', fontsize=12)
    ax8.set_ylabel('Amplitude', fontsize=12)
    ax8.legend(fontsize=10)
    ax8.grid(True, alpha=0.3)
    ax8.set_xlim(-2, 2)
    ax8.set_ylim(-2.5, 2.5)
    
    # Add text annotation
    ax8.text(0.05, 0.95, '• Piecewise constant\n• Orthogonal\n• Compact support\n• Multi-resolution', 
             transform=ax8.transAxes, fontsize=9, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    
    return fig

def create_kmote_architecture_diagram():
    """Create a diagram showing K-MOTE architecture"""
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    
    # Hide axes
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'K-MOTE Architecture: Expert Mixture System', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Input
    input_box = Rectangle((0.5, 6), 1.5, 0.8, 
                         facecolor='lightblue', edgecolor='black', linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.25, 6.4, 'Time Input\nt', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Expert boxes
    expert_positions = [(2.5, 5), (2.5, 3.5), (2.5, 2)]
    expert_colors = ['lightgreen', 'lightcoral', 'lightyellow']
    expert_labels = ['B-Spline Expert\n(Smooth Functions)', 'Fourier Expert\n(Periodic Patterns)', 'Wavelet Expert\n(Local Features)']
    
    for i, (pos, color, label) in enumerate(zip(expert_positions, expert_colors, expert_labels)):
        expert_box = Rectangle(pos, 2.5, 1.2, 
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(expert_box)
        ax.text(pos[0] + 1.25, pos[1] + 0.6, label, 
                ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Gating network
    gate_box = Rectangle((6, 4), 2, 1.5, 
                        facecolor='lightpink', edgecolor='black', linewidth=2)
    ax.add_patch(gate_box)
    ax.text(7, 4.75, 'Gating Network\n(Mixture Weights)', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Output
    output_box = Rectangle((8.5, 3.5), 1, 2.5, 
                          facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(output_box)
    ax.text(9, 4.75, 'Weighted\nSum', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrows
    # Input to experts
    for pos in expert_positions:
        ax.annotate('', xy=(pos[0], pos[1] + 0.6), xytext=(2, 6.4),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Input to gating
    ax.annotate('', xy=(6, 4.75), xytext=(2, 6.4),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Experts to output
    for pos in expert_positions:
        ax.annotate('', xy=(8.5, 4.75), xytext=(pos[0] + 2.5, pos[1] + 0.6),
                   arrowprops=dict(arrowstyle='->', lw=2, color='green'))
    
    # Gating to output
    ax.annotate('', xy=(8.5, 4.75), xytext=(8, 4.75),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    # Add mathematical formula
    ax.text(5, 1, r'$y = \sum_{i=1}^{3} g_i(t) \cdot f_i(t)$', 
            fontsize=16, ha='center', 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='black'))
    
    ax.text(5, 0.4, r'where $g_i(t)$ are gating weights and $f_i(t)$ are expert outputs', 
            fontsize=12, ha='center', style='italic')
    
    return fig

def main():
    """Main function to generate all visualizations"""
    
    print("🎨 Generating K-MOTE Expert Visualizations...")
    
    # Create expert basis function visualizations
    fig1 = create_expert_visualizations()
    fig1.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/kmote_experts_visualization.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: kmote_experts_visualization.png")
    
    # Create architecture diagram
    fig2 = create_kmote_architecture_diagram()
    fig2.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/kmote_architecture_diagram.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: kmote_architecture_diagram.png")
    
    # Show plots
    plt.show()
    
    print("\n🎯 Visualization Summary:")
    print("=" * 50)
    print("📊 Generated 2 comprehensive visualizations:")
    print("   1. Expert Basis Functions - Shows mathematical characteristics of each expert type")
    print("   2. K-MOTE Architecture - Shows system-level mixture architecture") 
    print("\n🔍 Expert Types Visualized:")
    print("   • B-Spline Expert: Smooth piecewise polynomials with local support")
    print("   • Fourier Expert: Harmonic cosine/sine components for periodic patterns")  
    print("   • Wavelet Expert: Multiple wavelet types (Morlet, Shock, Mexican Hat, Haar)")
    print("\n💡 Key Insights:")
    print("   • Each expert captures different temporal patterns")
    print("   • Gating network learns optimal mixture weights")
    print("   • Combined system provides rich representational capacity")

if __name__ == "__main__":
    main()