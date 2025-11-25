#!/usr/bin/env python3
"""
Fourier KAN Expert Visualization
From K-MOTE FourierKANLayer
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_fourier_kan():
    """Plot Fourier KAN expert harmonics"""
    plt.figure(figsize=(8, 6))
    
    t = np.linspace(-2*np.pi, 2*np.pi, 1000)
    
    # Fourier KAN implementation based on FourierKANLayer
    n_harmonics = 5
    
    # Base component (learnable bias)
    base = 0.3
    
    # Harmonic components
    fourier_sum = np.full_like(t, base)
    
    # Cosine and sine weights (mimicking learnable parameters)
    cos_weights = [0.6, 0.4, 0.3, 0.2, 0.1]
    sin_weights = [0.4, 0.3, 0.25, 0.15, 0.08]
    
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Plot individual harmonic components
    for k in range(1, n_harmonics + 1):
        # Cosine component
        cos_component = cos_weights[k-1] * np.cos(k * t)
        plt.plot(t, cos_component, color=colors[(k-1) % len(colors)], 
                linewidth=2.5, alpha=0.7)
        
        # Sine component
        sin_component = sin_weights[k-1] * np.sin(k * t)
        plt.plot(t, sin_component, color=colors[(k-1) % len(colors)], 
                linewidth=2.5, alpha=0.5, linestyle='--')
        
        # Add to combined sum
        fourier_sum += cos_component + sin_component
    
    # Plot the combined Fourier series
    plt.plot(t, fourier_sum, 'k-', linewidth=4, alpha=0.9)
    
    # Fill area to show the contribution above base
    plt.fill_between(t, base, fourier_sum, alpha=0.2, color='lightblue')
    
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/fourier_kan_expert.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Fourier KAN expert saved: fourier_kan_expert.png")

if __name__ == "__main__":
    plot_fourier_kan()