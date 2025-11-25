#!/usr/bin/env python3
"""
Individual Fourier Expert Visualization
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_single_fourier():
    """Plot clean Fourier function"""
    plt.figure(figsize=(8, 6))
    
    t = np.linspace(-2*np.pi, 2*np.pi, 1000)
    
    # Combined Fourier series
    base = 0.2
    fourier_sum = base
    harmonics = [1, 2, 3, 4, 5]
    weights_cos = [0.5, 0.3, 0.2, 0.1, 0.05]
    weights_sin = [0.3, 0.2, 0.15, 0.08, 0.03]
    
    for k, w_cos, w_sin in zip(harmonics, weights_cos, weights_sin):
        fourier_sum += w_cos * np.cos(k * t) + w_sin * np.sin(k * t)
    
    plt.plot(t, fourier_sum, 'k-', linewidth=4)
    plt.fill_between(t, base, fourier_sum, alpha=0.3, color='lightblue')
    
    plt.xlim(-2*np.pi, 2*np.pi)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/fourier_expert.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Fourier expert saved: fourier_expert.png")

if __name__ == "__main__":
    plot_single_fourier()