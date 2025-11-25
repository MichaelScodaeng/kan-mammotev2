#!/usr/bin/env python3
"""
Shock Wavelet Visualization

This script plots shock wavelets used in the K-MOTE wavelet expert
for modeling abrupt changes and asymmetric patterns.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_shock_wavelet():
    """Plot shock wavelet functions"""
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Define time range
    t = np.linspace(-3, 3, 1000)
    
    def shock_wavelet(t, asymmetry=0.0, steepness=2.0):
        """
        Shock wavelet function optimized for abrupt changes
        
        Args:
            t: time array
            asymmetry: asymmetry parameter (-1 to 1)
            steepness: steepness parameter (controls decay and frequency)
        """
        # Apply tanh to asymmetry to keep it bounded
        asym = np.tanh(asymmetry)
        steep = steepness + 0.1  # Ensure positive steepness
        
        # Clamp exponents to prevent overflow
        left_exponent = np.clip(steep * t * (1 + asym), -10, 10)
        right_exponent = np.clip(-steep * t * (1 - asym), -10, 10)
        
        # Asymmetric exponential decay
        shock_profile = np.where(t < 0, 
                               np.exp(left_exponent), 
                               np.exp(right_exponent))
        
        # Oscillatory component
        freq = np.clip(steep, 0.1, 3.0)  # Limit frequency
        oscillation = np.cos(freq * t)
        
        # Combined shock wavelet (clamp to prevent extreme values)
        result = np.clip(shock_profile * oscillation, -100, 100)
        return result
    
    # Plot different shock wavelet configurations
    configs = [
        (0.0, 1.5, 'Symmetric (asym=0.0, steep=1.5)'),
        (0.8, 2.0, 'Right-skewed (asym=0.8, steep=2.0)'),
        (-0.8, 2.0, 'Left-skewed (asym=-0.8, steep=2.0)'),
        (0.0, 3.0, 'High frequency (asym=0.0, steep=3.0)'),
        (0.5, 1.0, 'Moderate right-skew (asym=0.5, steep=1.0)')
    ]
    
    colors = ['#2c3e50', '#e74c3c', '#2ecc71', '#3498db', '#f39c12']
    
    for i, ((asym, steep, label), color) in enumerate(zip(configs, colors)):
        shock = shock_wavelet(t, asymmetry=asym, steepness=steep)
        plt.plot(t, shock, color=color, linewidth=2.5)
    
    plt.xlim(-3, 3)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/shock_wavelet_function.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("✅ Shock wavelet visualization saved as: shock_wavelet_function.png")

if __name__ == "__main__":
    plot_shock_wavelet()