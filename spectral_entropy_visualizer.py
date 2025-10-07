#!/usr/bin/env python3
"""
Spectral Entropy Visualization Module

This module creates publication-quality density plots for spectral entropy analysis,
replicating and extending Figure 8 to include all 13 datasets.

Author: KAN-MAMMOTE Research Team
Date: October 7, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import pickle
from pathlib import Path
import matplotlib.patches as patches
from matplotlib.colors import to_rgba

class SpectralEntropyVisualizer:
    """
    Creates visualizations for spectral entropy analysis results.
    """
    
    def __init__(self, results_file=None, output_dir="./spectral_entropy_results"):
        """
        Initialize the visualizer.
        
        Args:
            results_file (str): Path to pickled results file
            output_dir (str): Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if results_file:
            self.load_results(results_file)
        else:
            self.entropy_results = {}
    
    def load_results(self, results_file):
        """Load analysis results from pickle file."""
        with open(results_file, 'rb') as f:
            self.entropy_results = pickle.load(f)
        print(f"Loaded results for {len(self.entropy_results)} datasets")
    
    def create_density_plots(self, figsize=(15, 6), save_format='pdf'):
        """
        Create density plots similar to Figure 8, showing all datasets.
        
        Args:
            figsize (tuple): Figure size (width, height)
            save_format (str): Format to save plots ('pdf', 'png', 'svg')
        """
        if not self.entropy_results:
            print("No results to plot. Run analysis first.")
            return
        
        # Set up the plotting style
        plt.style.use('default')
        sns.set_palette("husl", n_colors=len(self.entropy_results))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Colors for different datasets
        colors = plt.cm.Set3(np.linspace(0, 1, len(self.entropy_results)))
        
        # Plot timestamp entropies (left plot)
        ax1.set_title('Density Plot of Spectral Entropy (Timestamp of Interactions)', 
                     fontsize=14, pad=20)
        ax1.set_xlabel('Spectral Entropy', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        
        # Plot time difference entropies (right plot)
        ax2.set_title('Density Plot of Spectral Entropy (Time Difference between Interactions)', 
                     fontsize=14, pad=20)
        ax2.set_xlabel('Spectral Entropy', fontsize=12)
        ax2.set_ylabel('Density', fontsize=12)
        
        # Process each dataset
        max_density = 0
        dataset_labels = []
        
        for i, (dataset_name, results) in enumerate(self.entropy_results.items()):
            if results is None:
                continue
                
            color = colors[i]
            dataset_labels.append(dataset_name.title())
            
            # Plot timestamp entropy density
            ts_entropies = results['timestamp_entropies']
            if len(ts_entropies) > 0:
                # Use KDE for smooth density estimation
                try:
                    kde_ts = gaussian_kde(ts_entropies)
                    x_ts = np.linspace(max(0, ts_entropies.min()), ts_entropies.max(), 200)
                    density_ts = kde_ts(x_ts)
                    
                    ax1.fill_between(x_ts, density_ts, alpha=0.6, color=color, 
                                   label=dataset_name.title())
                    ax1.plot(x_ts, density_ts, color=color, linewidth=1.5)
                    
                    max_density = max(max_density, density_ts.max())
                except:
                    # Fallback to histogram if KDE fails
                    ax1.hist(ts_entropies, bins=30, alpha=0.6, color=color, 
                           density=True, label=dataset_name.title())
            
            # Plot time difference entropy density
            td_entropies = results['time_diff_entropies']
            if len(td_entropies) > 0:
                try:
                    kde_td = gaussian_kde(td_entropies)
                    x_td = np.linspace(max(0, td_entropies.min()), td_entropies.max(), 200)
                    density_td = kde_td(x_td)
                    
                    ax2.fill_between(x_td, density_td, alpha=0.6, color=color)
                    ax2.plot(x_td, density_td, color=color, linewidth=1.5)
                    
                    max_density = max(max_density, density_td.max())
                except:
                    # Fallback to histogram if KDE fails
                    ax2.hist(td_entropies, bins=30, alpha=0.6, color=color, density=True)
        
        # Set consistent y-axis limits
        y_max = max_density * 1.1
        ax1.set_ylim(0, y_max)
        ax2.set_ylim(0, y_max)
        
        # Add legend (only on the first plot to avoid duplication)
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Grid and styling
        ax1.grid(True, alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # Tight layout
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / f'spectral_entropy_density_plots.{save_format}'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Density plots saved to: {output_file}")
        
        # Also save as PNG for easy viewing
        if save_format != 'png':
            png_file = self.output_dir / 'spectral_entropy_density_plots.png'
            plt.savefig(png_file, dpi=300, bbox_inches='tight')
            print(f"PNG version saved to: {png_file}")
        
        plt.show()
    
    def create_summary_statistics_plot(self, figsize=(12, 8)):
        """
        Create a summary plot showing statistics across datasets.
        
        Args:
            figsize (tuple): Figure size
        """
        if not self.entropy_results:
            print("No results to plot. Run analysis first.")
            return
        
        # Prepare data for plotting
        data = []
        for dataset_name, results in self.entropy_results.items():
            if results is None:
                continue
            
            data.append({
                'dataset': dataset_name.title(),
                'nodes_analyzed': results['total_nodes'],
                'total_interactions': results['total_interactions'],
                'avg_timestamp_entropy': np.mean(results['timestamp_entropies']),
                'avg_time_diff_entropy': np.mean(results['time_diff_entropies']),
                'std_timestamp_entropy': np.std(results['timestamp_entropies']),
                'std_time_diff_entropy': np.std(results['time_diff_entropies'])
            })
        
        df = pd.DataFrame(data)
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Number of nodes analyzed
        bars1 = ax1.bar(range(len(df)), df['nodes_analyzed'], color=plt.cm.Set3(np.linspace(0, 1, len(df))))
        ax1.set_title('Nodes Analyzed per Dataset', fontsize=12)
        ax1.set_ylabel('Number of Nodes')
        ax1.set_xticks(range(len(df)))
        ax1.set_xticklabels(df['dataset'], rotation=45, ha='right')
        
        # Plot 2: Total interactions
        bars2 = ax2.bar(range(len(df)), df['total_interactions'], color=plt.cm.Set3(np.linspace(0, 1, len(df))))
        ax2.set_title('Total Interactions per Dataset', fontsize=12)
        ax2.set_ylabel('Number of Interactions')
        ax2.set_xticks(range(len(df)))
        ax2.set_xticklabels(df['dataset'], rotation=45, ha='right')
        ax2.set_yscale('log')  # Log scale for better visibility
        
        # Plot 3: Average timestamp entropy
        bars3 = ax3.bar(range(len(df)), df['avg_timestamp_entropy'], 
                       yerr=df['std_timestamp_entropy'], capsize=5,
                       color=plt.cm.Set3(np.linspace(0, 1, len(df))))
        ax3.set_title('Average Timestamp Entropy', fontsize=12)
        ax3.set_ylabel('Spectral Entropy')
        ax3.set_xticks(range(len(df)))
        ax3.set_xticklabels(df['dataset'], rotation=45, ha='right')
        
        # Plot 4: Average time difference entropy
        bars4 = ax4.bar(range(len(df)), df['avg_time_diff_entropy'], 
                       yerr=df['std_time_diff_entropy'], capsize=5,
                       color=plt.cm.Set3(np.linspace(0, 1, len(df))))
        ax4.set_title('Average Time Difference Entropy', fontsize=12)
        ax4.set_ylabel('Spectral Entropy')
        ax4.set_xticks(range(len(df)))
        ax4.set_xticklabels(df['dataset'], rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / 'spectral_entropy_summary_statistics.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Summary statistics plot saved to: {output_file}")
        
        plt.show()
        
        # Save statistics as CSV
        csv_file = self.output_dir / 'spectral_entropy_statistics.csv'
        df.to_csv(csv_file, index=False)
        print(f"Statistics saved to: {csv_file}")
    
    def create_comparison_heatmap(self, figsize=(10, 8)):
        """
        Create a heatmap comparing entropy values across datasets.
        
        Args:
            figsize (tuple): Figure size
        """
        if not self.entropy_results:
            print("No results to plot. Run analysis first.")
            return
        
        # Prepare data matrix
        datasets = []
        timestamp_means = []
        timestamp_stds = []
        timediff_means = []
        timediff_stds = []
        
        for dataset_name, results in self.entropy_results.items():
            if results is None:
                continue
            
            datasets.append(dataset_name.title())
            timestamp_means.append(np.mean(results['timestamp_entropies']))
            timestamp_stds.append(np.std(results['timestamp_entropies']))
            timediff_means.append(np.mean(results['time_diff_entropies']))
            timediff_stds.append(np.std(results['time_diff_entropies']))
        
        # Create data matrix
        data_matrix = np.array([
            timestamp_means,
            timestamp_stds,
            timediff_means,
            timediff_stds
        ])
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=figsize)
        
        im = ax.imshow(data_matrix, cmap='viridis', aspect='auto')
        
        # Set labels
        ax.set_xticks(range(len(datasets)))
        ax.set_xticklabels(datasets, rotation=45, ha='right')
        ax.set_yticks(range(4))
        ax.set_yticklabels([
            'Timestamp Entropy (Mean)',
            'Timestamp Entropy (Std)',
            'Time Diff Entropy (Mean)',
            'Time Diff Entropy (Std)'
        ])
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Entropy Value')
        
        # Add text annotations
        for i in range(4):
            for j in range(len(datasets)):
                text = ax.text(j, i, f'{data_matrix[i, j]:.2f}',
                             ha="center", va="center", color="white" if data_matrix[i, j] > np.median(data_matrix) else "black")
        
        ax.set_title('Spectral Entropy Comparison Across Datasets', fontsize=14, pad=20)
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / 'spectral_entropy_heatmap.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {output_file}")
        
        plt.show()


def create_all_visualizations(results_file, output_dir="./spectral_entropy_results"):
    """
    Create all visualization plots from analysis results.
    
    Args:
        results_file (str): Path to the results pickle file
        output_dir (str): Directory to save plots
    """
    visualizer = SpectralEntropyVisualizer(results_file, output_dir)
    
    print("Creating density plots...")
    visualizer.create_density_plots()
    
    print("Creating summary statistics...")
    visualizer.create_summary_statistics_plot()
    
    print("Creating comparison heatmap...")
    visualizer.create_comparison_heatmap()
    
    print("All visualizations complete!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create spectral entropy visualizations')
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to the analysis results pickle file')
    parser.add_argument('--output_dir', type=str, default='./spectral_entropy_results',
                        help='Directory to save plots')
    
    args = parser.parse_args()
    create_all_visualizations(args.results_file, args.output_dir)