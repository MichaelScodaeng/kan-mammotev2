#!/usr/bin/env python3
"""
B-Spline Function Visualization

This script plots B-spline basis functions showing their smooth, 
local support characteristics used in the K-MOTE B-spline expert.
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_bspline():
    """Plot B-spline basis functions"""
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Define time range
    t = np.linspace(-3, 3, 1000)
    
    # B-spline basis function (cubic approximation)
    def bspline_basis(t, center, width):
        """Cubic B-spline basis function approximation"""
        normalized_t = (t - center) / width
        # Cubic B-spline approximation using max(0, (1-|x|)^3)
        return np.maximum(0, (1 - np.abs(normalized_t))**3)
    
    # Plot multiple B-spline basis functions
    centers = [-1.5, -0.5, 0.5, 1.5]
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    width = 1.0
    
    for i, (center, color) in enumerate(zip(centers, colors)):
        basis = bspline_basis(t, center, width)
        plt.plot(t, basis, color=color, linewidth=3)
    
    # Add a combined weighted sum to show flexibility
    combined = (0.3 * bspline_basis(t, -1.5, width) + 
                0.5 * bspline_basis(t, -0.5, width) + 
                0.7 * bspline_basis(t, 0.5, width) + 
                0.2 * bspline_basis(t, 1.5, width))
    
    plt.plot(t, combined, 'k--', linewidth=3, alpha=0.8)
    
    plt.xlim(-3, 3)
    plt.ylim(-0.1, 1.1)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/bspline_function.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print("✅ B-spline visualization saved as: bspline_function.png")

if __name__ == "__main__":
    plot_bspline()