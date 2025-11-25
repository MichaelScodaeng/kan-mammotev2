#!/usr/bin/env python3
"""
Fourier Function Visualization

This script plots Fourier series components (cosine and sine harmonics)
used in the K-MOTE Fourier expert for periodic pattern modeling.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_fourier():
    """Plot Fourier series components"""
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
    
    # Define time range
    t = np.linspace(-2*np.pi, 2*np.pi, 1000)
    
    # ================================
    # 1. Cosine harmonics
    # ================================
    harmonics = [1, 2, 3, 4, 5]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for k, color in zip(harmonics, colors):
        cosine = np.cos(k * t)
        ax1.plot(t, cosine, color=color, linewidth=2.5)
    
    ax1.set_xlim(-2*np.pi, 2*np.pi)
    ax1.set_ylim(-1.2, 1.2)
    ax1.axis('off')
    
    # ================================
    # 2. Sine harmonics
    # ================================
    for k, color in zip(harmonics, colors):
        sine = np.sin(k * t)
        ax2.plot(t, sine, color=color, linewidth=2.5, linestyle='--')
    
    ax2.set_xlim(-2*np.pi, 2*np.pi)
    ax2.set_ylim(-1.2, 1.2)
    ax2.axis('off')
    
    # ================================
    # 3. Combined Fourier series
    # ================================
    # Base/bias term
    base = 0.2
    
    # Fourier series combination
    fourier_sum = base
    weights_cos = [0.5, 0.3, 0.2, 0.1, 0.05]
    weights_sin = [0.3, 0.2, 0.15, 0.08, 0.03]
    
    for k, w_cos, w_sin in zip(harmonics, weights_cos, weights_sin):
        fourier_sum += w_cos * np.cos(k * t) + w_sin * np.sin(k * t)
    
    # Plot combined result
    ax3.plot(t, fourier_sum, 'k-', linewidth=3)
    ax3.fill_between(t, base, fourier_sum, alpha=0.3, color='lightblue')
    
    ax3.set_xlim(-2*np.pi, 2*np.pi)
    ax3.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/fourier_function.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("✅ Fourier visualization saved as: fourier_function.png")

if __name__ == "__main__":
    plot_fourier()