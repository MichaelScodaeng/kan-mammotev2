#!/usr/bin/env python3
"""
Simplified K-MOTE Expert Visualization

This script creates a clean, focused visualization showing the key characteristics
of each K-MOTE expert type with representative functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import math

def create_simple_expert_comparison():
    """Create a simple 2x2 grid showing each expert type clearly"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('K-MOTE Expert Types: Function Characteristics', fontsize=18, fontweight='bold')
    
    # Define time range
    t = np.linspace(-2, 2, 1000)
    
    # ================================
    # 1. B-SPLINE EXPERT (Top Left)
    # ================================
    ax = axes[0, 0]
    
    # Create smooth B-spline-like functions
    def smooth_basis(t, center, width):
        normalized_t = (t - center) / width
        return np.maximum(0, (1 - np.abs(normalized_t))**3)
    
    centers = [-1.0, -0.3, 0.4, 1.1]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    
    for i, (center, color) in enumerate(zip(centers, colors)):
        basis = smooth_basis(t, center, 0.8)
        ax.plot(t, basis, color=color, linewidth=3, label=f'Basis {i+1}')
    
    ax.set_title('B-Spline Expert\n(Smooth Local Functions)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, '• Smooth curves\n• Local support\n• Piecewise polynomial', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    # ================================
    # 2. FOURIER EXPERT (Top Right)
    # ================================
    ax = axes[0, 1]
    
    # Fourier components
    harmonics = [1, 2, 3, 4]
    fourier_colors = ['#e67e22', '#1abc9c', '#8e44ad', '#34495e']
    
    for k, color in zip(harmonics, fourier_colors):
        cosine = np.cos(k * t)
        ax.plot(t, cosine, color=color, linewidth=3, label=f'cos({k}t)')
    
    # Show combined result
    combined = 0.4*np.cos(t) + 0.3*np.cos(2*t) + 0.2*np.cos(3*t) + 0.1*np.cos(4*t)
    ax.plot(t, combined, 'k-', linewidth=4, alpha=0.8, label='Combined')
    
    ax.set_title('Fourier Expert\n(Harmonic Components)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, '• Periodic patterns\n• Global support\n• Frequency domain', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # ================================
    # 3. MORLET WAVELET (Bottom Left)
    # ================================
    ax = axes[1, 0]
    
    def morlet_wavelet(t, scale=1.0, frequency=5.0):
        """Morlet wavelet"""
        c = math.pi**(-0.25)
        return c * np.exp(-0.5 * (t/scale)**2) * np.cos(frequency * t)
    
    scales = [0.3, 0.5, 0.8, 1.2]
    morlet_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#9b59b6']
    
    for scale, color in zip(scales, morlet_colors):
        morlet = morlet_wavelet(t, scale=scale)
        ax.plot(t, morlet, color=color, linewidth=3, label=f'σ={scale}')
    
    ax.set_title('Morlet Wavelet Expert\n(Time-Frequency Localized)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, '• Gaussian envelope\n• Oscillatory core\n• Multi-scale', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # ================================
    # 4. SHOCK WAVELET (Bottom Right)
    # ================================
    ax = axes[1, 1]
    
    def shock_wavelet(t, asymmetry=0.0, steepness=2.0):
        """Shock wavelet for abrupt changes"""
        asym = np.tanh(asymmetry)
        steep = steepness + 0.1
        
        left_exponent = np.clip(steep * t * (1 + asym), -10, 10)
        right_exponent = np.clip(-steep * t * (1 - asym), -10, 10)
        
        shock_profile = np.where(t < 0, np.exp(left_exponent), np.exp(right_exponent))
        oscillation = np.cos(steep * t)
        
        return np.clip(shock_profile * oscillation, -10, 10)
    
    # Different shock configurations
    configs = [(0.0, 1.5), (0.8, 2.0), (-0.8, 2.0)]
    shock_colors = ['#3498db', '#e74c3c', '#2ecc71']
    labels = ['Symmetric', 'Right-skewed', 'Left-skewed']
    
    for (asym, steep), color, label in zip(configs, shock_colors, labels):
        shock = shock_wavelet(t, asymmetry=asym, steepness=steep)
        ax.plot(t, shock, color=color, linewidth=3, label=label)
    
    ax.set_title('Shock Wavelet Expert\n(Abrupt Change Detection)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 0.95, '• Asymmetric shape\n• Sharp transitions\n• Edge detection', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_kmote_concept_diagram():
    """Create a conceptual diagram showing how K-MOTE works"""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Title
    ax.text(5, 7.5, 'K-MOTE: Mixture of Time Experts', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Input time signal
    t_input = np.linspace(0.5, 3.5, 100)
    signal = 0.5 * np.sin(3*t_input) + 0.3 * np.cos(7*t_input) + 0.2 * np.exp(-(t_input-2)**2)
    ax.plot(t_input, 6.5 + 0.5*signal, 'k-', linewidth=3, label='Input Time Signal')
    ax.text(2, 6, 'Input: Time Signal t', ha='center', fontsize=12, fontweight='bold')
    
    # Expert outputs (simplified representations)
    expert_x = [1, 3, 5]
    expert_y = 4
    expert_labels = ['B-Spline\nExpert', 'Fourier\nExpert', 'Wavelet\nExpert']
    expert_colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    for i, (x, label, color) in enumerate(zip(expert_x, expert_labels, expert_colors)):
        # Expert box
        rect = plt.Rectangle((x-0.4, expert_y-0.4), 0.8, 0.8, 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x, expert_y, label, ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Show expert output
        t_expert = np.linspace(x-0.3, x+0.3, 30)
        if i == 0:  # B-spline (smooth)
            output = 0.3 * np.exp(-(t_expert-x)**2/0.1)
        elif i == 1:  # Fourier (periodic)
            output = 0.2 * np.cos(20*(t_expert-x))
        else:  # Wavelet (localized oscillation)
            output = 0.25 * np.exp(-(t_expert-x)**2/0.05) * np.cos(30*(t_expert-x))
        
        ax.plot(t_expert, expert_y - 1 + output, linewidth=2, color='darkblue')
        
        # Arrow from input to expert
        ax.annotate('', xy=(x, expert_y+0.4), xytext=(2, 6),
                   arrowprops=dict(arrowstyle='->', lw=2, color='blue'))
    
    # Gating network
    ax.text(7, 4.5, 'Gating\nNetwork', ha='center', va='center', fontsize=12, fontweight='bold',
           bbox=dict(boxstyle='round', facecolor='lightpink', edgecolor='black'))
    
    # Weights
    ax.text(7, 3.5, 'w₁, w₂, w₃', ha='center', va='center', fontsize=11, style='italic')
    
    # Final output
    ax.text(5, 1.5, 'Final Output = w₁·f₁(t) + w₂·f₂(t) + w₃·f₃(t)', 
            ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # Arrows to gating
    for x in expert_x:
        ax.annotate('', xy=(6.5, 4.3), xytext=(x+0.4, expert_y),
                   arrowprops=dict(arrowstyle='->', lw=1.5, color='green'))
    
    # Arrow from input to gating
    ax.annotate('', xy=(7, 5), xytext=(2, 6),
               arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    return fig

def main():
    """Generate simplified K-MOTE visualizations"""
    
    print("🎨 Creating Simple K-MOTE Expert Visualizations...")
    
    # Create simple comparison
    fig1 = create_simple_expert_comparison()
    fig1.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/kmote_simple_comparison.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: kmote_simple_comparison.png")
    
    # Create concept diagram
    fig2 = create_kmote_concept_diagram()
    fig2.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/kmote_concept_diagram.png', 
                 dpi=300, bbox_inches='tight', facecolor='white')
    print("✅ Saved: kmote_concept_diagram.png")
    
    plt.show()
    
    print("\n🎯 Simple Visualization Complete!")
    print("=" * 40)
    print("Generated 2 focused visualizations:")
    print("1. Expert Comparison - Shows function shapes side-by-side") 
    print("2. Concept Diagram - Shows mixture architecture")

if __name__ == "__main__":
    main()