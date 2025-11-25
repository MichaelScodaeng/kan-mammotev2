#!/usr/bin/env python3
"""
Individual Shock Wavelet Expert Visualization
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_single_shock():
    """Plot clean shock wavelet function"""
    plt.figure(figsize=(8, 6))
    
    t = np.linspace(-3, 3, 1000)
    
    def shock_wavelet(t, asymmetry=0.0, steepness=2.0):
        asym = np.tanh(asymmetry)
        steep = steepness + 0.1
        
        left_exponent = np.clip(steep * t * (1 + asym), -10, 10)
        right_exponent = np.clip(-steep * t * (1 - asym), -10, 10)
        
        shock_profile = np.where(t < 0, 
                               np.exp(left_exponent), 
                               np.exp(right_exponent))
        
        freq = np.clip(steep, 0.1, 3.0)
        oscillation = np.cos(freq * t)
        
        return np.clip(shock_profile * oscillation, -100, 100)
    
    configs = [
        (0.0, 1.5),   # Symmetric
        (0.8, 2.0),   # Right-skewed
        (-0.8, 2.0),  # Left-skewed
        (0.0, 3.0),   # High frequency
        (0.5, 1.0)    # Moderate right-skew
    ]
    
    colors = ['#2c3e50', '#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    
    for (asym, steep), color in zip(configs, colors):
        shock = shock_wavelet(t, asymmetry=asym, steepness=steep)
        plt.plot(t, shock, color=color, linewidth=3)
    
    plt.xlim(-3, 3)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/shock_wavelet_expert.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Shock wavelet expert saved: shock_wavelet_expert.png")

if __name__ == "__main__":
    plot_single_shock()