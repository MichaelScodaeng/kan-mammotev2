#!/usr/bin/env python3
"""
Plot individual function types from K-MOTE analysis
Clean plots without titles, legends, or axis labels
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_smooth_trend_data(t):
    """Data that should favor B-Spline expert - smooth polynomial trends"""
    return 0.1 * t**3 - 0.5 * t**2 + 0.3 * t + 0.2

def generate_periodic_data(t):
    """Data that should favor Fourier expert - complex periodic patterns"""
    return (np.sin(2 * np.pi * t / 3) + 
            0.5 * np.cos(2 * np.pi * t / 1.5) + 
            0.3 * np.sin(2 * np.pi * t / 7))

def generate_abrupt_change_data(t):
    """Data that should favor Wavelet expert - sudden shocks and discontinuities"""
    # Create shock events at different times
    shock1 = np.where(t > 2, 1.0 * np.exp(-(t-2)), 0.0)  # Sudden onset at t=2
    shock2 = np.where(t > -3, -0.8 * np.exp(-2*(t+3)), 0.0)  # Shock at t=-3
    shock3 = np.where((t > 5) & (t < 6), 1.5, 0.0)  # Step function
    return shock1 + shock2 + shock3

def generate_mixed_pattern_data(t):
    """Complex mixed pattern combining all expert domains"""
    smooth_trend = 0.05 * t**2  # B-spline domain
    periodic_part = 0.4 * np.sin(2 * np.pi * t / 4)  # Fourier domain
    shock_event = np.where(t > 3, 1.0 * np.exp(-(t-3)), 0.0)  # Wavelet domain
    localized_event = 0.8 * np.exp(-((t + 2)**2) / 0.6)  # RBF domain
    return smooth_trend + periodic_part + shock_event + localized_event

def plot_smooth_trend():
    """Plot smooth trend function"""
    plt.figure(figsize=(8, 6))
    
    t = np.linspace(-8, 8, 1000)
    y = generate_smooth_trend_data(t)
    
    plt.plot(t, y, 'b-', linewidth=4)
    plt.xlim(-8, 8)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/function_smooth_trend.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Smooth trend function saved: function_smooth_trend.png")

def plot_periodic():
    """Plot periodic function"""
    plt.figure(figsize=(8, 6))
    
    t = np.linspace(-8, 8, 1000)
    y = generate_periodic_data(t)
    
    plt.plot(t, y, 'r-', linewidth=4)
    plt.xlim(-8, 8)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/function_periodic.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Periodic function saved: function_periodic.png")

def plot_abrupt_changes():
    """Plot abrupt changes function"""
    plt.figure(figsize=(8, 6))
    
    t = np.linspace(-8, 8, 1000)
    y = generate_abrupt_change_data(t)
    
    plt.plot(t, y, 'g-', linewidth=4)
    plt.xlim(-8, 8)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/function_abrupt_changes.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Abrupt changes function saved: function_abrupt_changes.png")

def plot_mixed_patterns():
    """Plot mixed patterns function"""
    plt.figure(figsize=(8, 6))
    
    t = np.linspace(-8, 8, 1000)
    y = generate_mixed_pattern_data(t)
    
    plt.plot(t, y, 'm-', linewidth=4)
    plt.xlim(-8, 8)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/function_mixed_patterns.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ Mixed patterns function saved: function_mixed_patterns.png")

def plot_all_functions():
    """Plot all four functions"""
    print("🎨 Creating clean function plots...")
    
    plot_smooth_trend()
    plot_periodic()
    plot_abrupt_changes()
    plot_mixed_patterns()
    
    print("\n✅ All function plots created successfully!")
    print("Generated files:")
    print("  - function_smooth_trend.png")
    print("  - function_periodic.png")
    print("  - function_abrupt_changes.png")
    print("  - function_mixed_patterns.png")

if __name__ == "__main__":
    plot_all_functions()