#!/usr/bin/env python3
"""
B-Spline KAN Expert Visualization
From K-MOTE SplineKANLayer
"""

import numpy as np
import matplotlib.pyplot as plt

def plot_bspline_kan():
    """Plot B-spline KAN expert basis functions"""
    plt.figure(figsize=(8, 6))
    
    t = np.linspace(-2, 2, 1000)
    
    # B-spline basis function implementation (cubic B-splines)
    def cox_de_boor_bspline(t, knots, order=3):
        """Cox-de Boor B-spline basis function"""
        n = len(knots) - order - 1
        if n <= 0:
            return np.zeros_like(t)
        
        # Initialize with order 0 (piecewise constant)
        basis = np.zeros((len(t), n))
        
        # Order 0: indicator functions
        for i in range(n + order):
            if i < n:
                mask = (t >= knots[i]) & (t < knots[i+1])
                if i == n-1:  # Include right endpoint for last interval
                    mask = (t >= knots[i]) & (t <= knots[i+1])
                basis[mask, i] = 1.0
        
        # Recursively build higher order basis functions
        for k in range(1, order + 1):
            basis_new = np.zeros((len(t), n))
            for i in range(n):
                # Left term
                if i + k < len(knots) and knots[i + k] != knots[i]:
                    basis_new[:, i] += (t - knots[i]) / (knots[i + k] - knots[i]) * basis[:, i]
                
                # Right term
                if i + 1 < n and i + k + 1 < len(knots) and knots[i + k + 1] != knots[i + 1]:
                    basis_new[:, i] += (knots[i + k + 1] - t) / (knots[i + k + 1] - knots[i + 1]) * basis[:, i + 1]
            
            basis = basis_new
        
        return basis
    
    # Create knot vector for cubic B-splines (order=3)
    order = 3
    grid_size = 5
    grid_range = [-2, 2]
    
    # Create uniform knot vector
    knots = np.linspace(grid_range[0], grid_range[1], grid_size + 1)
    # Extend with repeated knots at boundaries for cubic splines
    knots = np.concatenate([
        np.repeat(knots[0], order),
        knots,
        np.repeat(knots[-1], order)
    ])
    
    # Compute B-spline basis functions
    basis = cox_de_boor_bspline(t, knots, order=order)
    
    # Plot multiple basis functions
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    for i in range(min(5, basis.shape[1])):
        if np.max(np.abs(basis[:, i])) > 1e-10:  # Only plot non-zero basis functions
            plt.plot(t, basis[:, i], color=colors[i % len(colors)], linewidth=4)
    
    # Add a weighted combination to show expressiveness
    if basis.shape[1] >= 3:
        weights = [0.8, 1.2, 0.6]
        combined = np.zeros_like(t)
        for i, w in enumerate(weights[:min(basis.shape[1], len(weights))]):
            combined += w * basis[:, i]
        
        plt.plot(t, combined, 'k--', linewidth=3, alpha=0.7)
    
    plt.xlim(-2, 2)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/bspline_kan_expert.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✅ B-spline KAN expert saved: bspline_kan_expert.png")

if __name__ == "__main__":
    plot_bspline_kan()