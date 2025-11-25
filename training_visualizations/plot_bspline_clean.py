#!/usr/bin/env python3
"""
Individual B-Spline Expert Visualization
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_single_bspline():
    """Plot clean B-spline function"""
    plt.figure(figsize=(8, 6))
    
    t = np.linspace(-3, 3, 1000)
    
    def bspline_basis(t, center, width):
        normalized_t = (t - center) / width
        return np.maximum(0, (1 - np.abs(normalized_t))**3)
    
    centers = [-1.5, -0.5, 0.5, 1.5]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    width = 1.0
    
    for center, color in zip(centers, colors):
        basis = bspline_basis(t, center, width)
        plt.plot(t, basis, color=color, linewidth=4)
    
    plt.xlim(-3, 3)
    plt.ylim(-0.1, 1.1)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/bspline_expert.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ B-spline expert saved: bspline_expert.png")

if __name__ == "__main__":
    plot_single_bspline()