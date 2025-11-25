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
    
    def create_density_plots(self, figsize=(16, 6), save_format='eps'):
        """
        Create density plots similar to Figure 8, showing all datasets with improved styling.
        
        Args:
            figsize (tuple): Figure size (width, height)
            save_format (str): Format to save plots ('pdf', 'png', 'svg')
        """
        if not self.entropy_results:
            print("No results to plot. Run analysis first.")
            return
        
        # Set up the plotting style to match the reference
        plt.style.use('default')
        
        # Create figure with proper sizing and layout
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor('white')
        
        # Define beautiful colors for each dataset (similar to reference)
        colors = [
            '#87CEEB',  # Light blue (Wikipedia)
            '#FFA07A',  # Light salmon (Reddit) 
            '#98D982',  # Light green (Mooc)
            '#FFB6C1',  # Light pink (Lastfm)
            '#DDA0DD',  # Plum (Enron)
            '#F0E68C',  # Khaki (SocialEvo)
            '#20B2AA',  # Light sea green (UCI)
            '#FFE4B5',  # Moccasin (CanParl)
            '#E6E6FA',  # Lavender (Contacts)
            '#F5DEB3',  # Wheat (Flights)
            '#D3D3D3',  # Light gray (UNtrade)
            '#FFEFD5',  # Papaya whip (UNvote)
            '#FFFACD'   # Lemon chiffon (USLegis)
        ]
        
        # Collect all valid datasets and their data
        valid_datasets = []
        timestamp_data = []
        time_diff_data = []
        dataset_labels = []
        
        for i, (dataset_name, results) in enumerate(self.entropy_results.items()):
            if results is None or len(results.get('timestamp_entropies', [])) == 0:
                continue
                
            valid_datasets.append(dataset_name)
            timestamp_data.append(results['timestamp_entropies'])
            time_diff_data.append(results['time_diff_entropies'])
            
            # Create proper labels (capitalize first letter)
            label = dataset_name.replace('_', ' ').title()
            if dataset_name.lower() == 'uci':
                label = 'UCI'
            elif dataset_name.lower() == 'socialevo':
                label = 'SocialEvo'
            dataset_labels.append(label)
        
        # Plot 1: Timestamp Entropies
        for i, (data, label) in enumerate(zip(timestamp_data, dataset_labels)):
            if len(data) > 0:
                # Use seaborn KDE for smooth density curves
                sns.kdeplot(data=data, ax=ax1, 
                           color=colors[i % len(colors)], 
                           alpha=0.7, 
                           fill=True, 
                           label=label,
                           linewidth=1.5)
        
        ax1.set_xlabel('Spectral Entropy', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Density', fontsize=16, fontweight='bold')
        ax1.set_title('Timestamp of Interactions', 
                     fontsize=18, fontweight='bold', pad=20)
        ax1.set_xlim(0, 12)
        ax1.set_ylim(0, None)
        
        # Plot 2: Time Difference Entropies  
        for i, (data, label) in enumerate(zip(time_diff_data, dataset_labels)):
            if len(data) > 0:
                sns.kdeplot(data=data, ax=ax2, 
                           color=colors[i % len(colors)], 
                           alpha=0.7, 
                           fill=True, 
                           linewidth=1.5)
        
        ax2.set_xlabel('Spectral Entropy', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Density', fontsize=16, fontweight='bold')
        ax2.set_title('Time Difference between Interactions', 
                     fontsize=18, fontweight='bold', pad=20)
        ax2.set_xlim(0, 12)
        ax2.set_ylim(0, None)
        
        # Remove individual legends from subplots
        ax1.legend().remove()
        ax2.legend().remove()
        
        # Add a single legend below the plots
        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.05), 
                  ncol=len(labels), fontsize=12, columnspacing=0.8, handlelength=1.5)
        
        # Adjust layout to make room for the legend below
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.11)
        
        # Save the plot
        output_file = self.output_dir / f'spectral_entropy_density_plots.{save_format}'
        plt.savefig(output_file, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Density plots saved to: {output_file}")
        
        # Also save as PDF for additional vector format
        if save_format == 'eps':
            pdf_file = self.output_dir / 'spectral_entropy_density_plots.pdf'
            plt.savefig(pdf_file, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            print(f"PDF version saved to: {pdf_file}")
        
        plt.show()
    
    def create_summary_statistics_plot(self, figsize=(15, 6)):
        """
        Create a summary plot showing statistics across datasets with improved styling.
        
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
            
            # Create proper labels
            label = dataset_name.replace('_', ' ').title()
            if dataset_name.lower() == 'uci':
                label = 'UCI'
            elif dataset_name.lower() == 'socialevo':
                label = 'SocialEvo'
            
            data.append({
                'dataset': label,
                'nodes_analyzed': results['total_nodes'],
                'total_interactions': results['total_interactions'],
                'avg_timestamp_entropy': np.mean(results['timestamp_entropies']),
                'avg_time_diff_entropy': np.mean(results['time_diff_entropies']),
                'std_timestamp_entropy': np.std(results['timestamp_entropies']),
                'std_time_diff_entropy': np.std(results['time_diff_entropies'])
            })
        
        if not data:
            print("No valid datasets found for summary statistics.")
            return
        
        df = pd.DataFrame(data)
        
        # Create figure with improved styling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        fig.patch.set_facecolor('white')
        
        x_pos = np.arange(len(df))
        
        # Plot mean entropies with improved colors
        bars1 = ax1.bar(x_pos - 0.2, df['avg_timestamp_entropy'], 0.4, 
                       label='Timestamp Entropy', alpha=0.8, color='#87CEEB')
        bars2 = ax1.bar(x_pos + 0.2, df['avg_time_diff_entropy'], 0.4, 
                       label='Time Diff Entropy', alpha=0.8, color='#FFA07A')
        
        ax1.set_xlabel('Datasets', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Spectral Entropy', fontsize=12, fontweight='bold')
        ax1.set_title('Mean Spectral Entropy by Dataset', fontsize=14, fontweight='bold', pad=20)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df['dataset'], rotation=45, ha='right')
        ax1.legend()
        
        # Plot standard deviations with improved colors
        bars3 = ax2.bar(x_pos - 0.2, df['std_timestamp_entropy'], 0.4, 
                       label='Timestamp Entropy', alpha=0.8, color='#98D982')
        bars4 = ax2.bar(x_pos + 0.2, df['std_time_diff_entropy'], 0.4, 
                       label='Time Diff Entropy', alpha=0.8, color='#FFB6C1')
        
        ax2.set_xlabel('Datasets', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Std Spectral Entropy', fontsize=12, fontweight='bold')
        ax2.set_title('Standard Deviation of Spectral Entropy by Dataset', fontsize=14, fontweight='bold', pad=20)
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(df['dataset'], rotation=45, ha='right')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / 'spectral_entropy_summary_statistics.eps'
        plt.savefig(output_file, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Summary statistics plot saved to: {output_file}")
        
        plt.show()
        
        # Save statistics as CSV
        csv_file = self.output_dir / 'spectral_entropy_statistics.csv'
        df.to_csv(csv_file, index=False)
        print(f"Statistics saved to: {csv_file}")
    
    def create_comparison_heatmap(self, figsize=(12, 8)):
        """
        Create a heatmap comparing entropy values across datasets with improved styling.
        
        Args:
            figsize (tuple): Figure size
        """
        if not self.entropy_results:
            print("No results to plot. Run analysis first.")
            return
        
        # Collect data for heatmap
        data_for_heatmap = []
        dataset_labels = []
        
        for dataset_name, results in self.entropy_results.items():
            if results is None or len(results.get('timestamp_entropies', [])) == 0:
                continue
                
            # Create proper labels
            label = dataset_name.replace('_', ' ').title()
            if dataset_name.lower() == 'uci':
                label = 'UCI'
            elif dataset_name.lower() == 'socialevo':
                label = 'SocialEvo'
            dataset_labels.append(label)
            
            ts_entropies = results['timestamp_entropies']
            td_entropies = results['time_diff_entropies']
            
            row_data = [
                np.mean(ts_entropies),      # Mean timestamp entropy
                np.std(ts_entropies),       # Std timestamp entropy
                np.mean(td_entropies),      # Mean time diff entropy
                np.std(td_entropies),       # Std time diff entropy
                np.min(ts_entropies),       # Min timestamp entropy
                np.max(ts_entropies),       # Max timestamp entropy
                np.min(td_entropies),       # Min time diff entropy
                np.max(td_entropies)        # Max time diff entropy
            ]
            data_for_heatmap.append(row_data)
        
        if not data_for_heatmap:
            print("No valid datasets found for heatmap.")
            return
        
        # Create DataFrame
        columns = ['TS Mean', 'TS Std', 'TD Mean', 'TD Std', 
                  'TS Min', 'TS Max', 'TD Min', 'TD Max']
        df = pd.DataFrame(data_for_heatmap, index=dataset_labels, columns=columns)
        
        # Create heatmap with improved styling
        plt.figure(figsize=figsize)
        plt.gca().set_facecolor('white')
        
        sns.heatmap(df, annot=True, fmt='.2f', cmap='viridis', 
                   cbar_kws={'label': 'Spectral Entropy'},
                   linewidths=0.5, linecolor='white')
        
        plt.title('Spectral Entropy Statistics Heatmap Across Datasets', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Entropy Statistics', fontsize=12, fontweight='bold')
        plt.ylabel('Datasets', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the plot
        output_file = self.output_dir / 'spectral_entropy_heatmap.eps'
        plt.savefig(output_file, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Comparison heatmap saved to: {output_file}")
        
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