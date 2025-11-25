#!/usr/bin/env python3
"""
Wavelet KAN Expert Visualization  
From K-MOTE WaveletKANLayer (EnhancedWaveletKAN)
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_wavelet_kan():
    """Plot Wavelet KAN expert functions"""
    plt.figure(figsize=(8, 6))
    
    t = np.linspace(-3, 3, 1000)
    
    # Shock wavelet implementation (from WaveletKANLayer)
    def shock_wavelet(t, asymmetry=0.0, steepness=2.0):
        """
        Shock wavelet from K-MOTE WaveletKANLayer
        Optimized for abrupt changes and asymmetric patterns
        """
        # Apply tanh to bound asymmetry
        asym = np.tanh(asymmetry)
        steep = steepness + 0.1  # Ensure positive steepness
        steep = np.clip(steep, None, 5.0)  # Limit max steepness
        
        # Asymmetric exponential decay profile
        left_exponent = np.clip(steep * t * (1 + asym), -10, 10)
        right_exponent = np.clip(-steep * t * (1 - asym), -10, 10)
        
        shock_profile = np.where(t < 0, 
                               np.exp(left_exponent), 
                               np.exp(right_exponent))
        
        # Oscillatory component
        freq = np.clip(steep, None, 3.0)
        oscillation = np.cos(freq * t)
        
        # Combined result (clamp to prevent extreme values)
        result = np.clip(shock_profile * oscillation, -100, 100)
        return result
    
    # Morlet wavelet implementation (standard reference)
    def morlet_wavelet(t, scale=1.0):
        """Standard Morlet wavelet"""
        c = np.pi**(-0.25)
        normalized_t = t / scale
        return c * np.exp(-0.5 * normalized_t**2) * np.cos(5.0 * normalized_t) / np.sqrt(scale)
    
    colors = ['#2c3e50', '#e74c3c', '#2ecc71', '#3498db', '#f39c12', '#9b59b6']
    
    # Plot different shock wavelets (multiple scales/parameters)
    shock_configs = [
        (0.0, 1.5),    # Symmetric
        (0.6, 2.0),    # Right-skewed  
        (-0.6, 2.0),   # Left-skewed
        (0.0, 2.8),    # Higher frequency
    ]
    
    for i, (asym, steep) in enumerate(shock_configs):
        shock = shock_wavelet(t, asymmetry=asym, steepness=steep)
        plt.plot(t, shock, color=colors[i], linewidth=3, alpha=0.8)
    
    # Add a couple of Morlet wavelets for comparison
    morlet_scales = [0.8, 1.2]
    for i, scale in enumerate(morlet_scales):
        morlet = morlet_wavelet(t, scale=scale)
        plt.plot(t, morlet, color=colors[4+i], linewidth=2.5, 
                linestyle=':', alpha=0.7)
    
    plt.xlim(-3, 3)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/wavelet_kan_expert.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Wavelet KAN expert saved: wavelet_kan_expert.png")

if __name__ == "__main__":
    plot_wavelet_kan()