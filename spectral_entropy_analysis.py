#!/usr/bin/env python3
"""
Spectral Entropy Analysis for Dynamic Graph D        # Try different possible file formats and locations
        # Prioritize ml_{dataset_name}.csv format as specified
        possible_files = [
            self.data_root / f"ml_{dataset_name}.csv",
            self.data_root / f"{dataset_name}" / f"ml_{dataset_name}.csv",
            self.data_root / f"{dataset_name}.csv",
            self.data_root / f"{dataset_name}" / f"{dataset_name}.csv",
            self.data_root / f"{dataset_name}" / "edges.csv"
        ]
This script analyzes the temporal patterns in dynamic graphs by computing spectral entropy
of node interaction patterns. It replicates and extends the analysis shown in Figure 8
of the paper, covering all 13 datasets.

The analysis computes spectral entropy using FFT on:
1. Normalized timestamps of interactions
2. Normalized time differences between consecutive interactions

Author: KAN-MAMMOTE Research Team
Date: October 7, 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
from scipy.stats import gaussian_kde
import os
import pickle
from collections import defaultdict
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SpectralEntropyAnalyzer:
    """
    Analyzes spectral entropy of temporal interaction patterns in dynamic graphs.
    """
    
    def __init__(self, data_root="./data", output_dir="./spectral_entropy_results"):
        """
        Initialize the analyzer.
        
        Args:
            data_root (str): Root directory containing dataset files
            output_dir (str): Directory to save results and plots
        """
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # All 13 datasets as specified
        self.datasets = [
            'wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
            'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis'
        ]
        
        # Store results for each dataset
        self.entropy_results = {}
        
    def load_dataset(self, dataset_name):
        """
        Load temporal graph dataset.
        
        Args:
            dataset_name (str): Name of the dataset to load
            
        Returns:
            pd.DataFrame: DataFrame with columns ['u', 'i', 'ts', 'label', 'idx']
        """
        print(f"Loading dataset: {dataset_name}")
        
        # Try different possible file formats and locations
        possible_files = [
            self.data_root / f"{dataset_name}.csv",
            self.data_root / f"{dataset_name}" / f"{dataset_name}.csv",
            self.data_root / f"{dataset_name}" / "edges.csv",
            self.data_root / f"{dataset_name}" / "ml_{dataset_name}.csv",
            self.data_root / f"ml_{dataset_name}.csv"
        ]
        
        df = None
        for file_path in possible_files:
            if file_path.exists():
                try:
                    print(f"  Trying to load: {file_path}")
                    df = pd.read_csv(file_path)
                    break
                except Exception as e:
                    print(f"  Failed to load {file_path}: {e}")
                    continue
        
        if df is None:
            raise FileNotFoundError(f"Could not find dataset file for {dataset_name}")
        
        # Standardize column names
        if 'u' not in df.columns:
            # Try to map common column names
            col_mapping = {
                'source': 'u', 'src': 'u', 'from': 'u', 'node1': 'u',
                'target': 'i', 'dst': 'i', 'to': 'i', 'node2': 'i',
                'timestamp': 'ts', 'time': 'ts', 't': 'ts'
            }
            
            for old_col, new_col in col_mapping.items():
                if old_col in df.columns:
                    df = df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        required_cols = ['u', 'i', 'ts']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns {missing_cols} in {dataset_name}")
        
        # Add missing columns with defaults
        if 'label' not in df.columns:
            df['label'] = 0
        if 'idx' not in df.columns:
            df['idx'] = range(len(df))
        
        # Sort by timestamp
        df = df.sort_values('ts').reset_index(drop=True)
        
        print(f"  Loaded {len(df)} interactions, {df['u'].nunique()} nodes")
        return df
    
    def normalize_timestamps(self, timestamps):
        """
        Normalize timestamps to [0, 1] range.
        
        Args:
            timestamps (array-like): Array of timestamps
            
        Returns:
            np.ndarray: Normalized timestamps
        """
        timestamps = np.array(timestamps)
        if len(timestamps) <= 1:
            return timestamps
        
        min_ts, max_ts = timestamps.min(), timestamps.max()
        if min_ts == max_ts:
            return np.zeros_like(timestamps)
        
        return (timestamps - min_ts) / (max_ts - min_ts)
    
    def compute_spectral_entropy(self, signal):
        """
        Compute spectral entropy of a signal using FFT.
        
        Args:
            signal (array-like): Input signal
            
        Returns:
            float: Spectral entropy value
        """
        signal = np.array(signal)
        
        # Need at least 2 points for FFT
        if len(signal) < 2:
            return 0.0
        
        # Apply FFT
        fft_result = fft(signal)
        
        # Compute magnitude spectrum
        magnitude = np.abs(fft_result)
        
        # Normalize to create probability distribution
        magnitude_sum = np.sum(magnitude)
        if magnitude_sum == 0:
            return 0.0
        
        prob_dist = magnitude / magnitude_sum
        
        # Remove zero probabilities to avoid log(0)
        prob_dist = prob_dist[prob_dist > 0]
        
        if len(prob_dist) == 0:
            return 0.0
        
        # Compute spectral entropy
        entropy = -np.sum(prob_dist * np.log(prob_dist))
        
        return entropy
    
    def analyze_node_interactions(self, df, min_interactions=5):
        """
        Analyze interaction patterns for each node with sufficient interactions.
        
        Args:
            df (pd.DataFrame): Dataset with interactions
            min_interactions (int): Minimum number of interactions required per node
            
        Returns:
            dict: Dictionary with entropy results
        """
        print(f"  Analyzing nodes with >= {min_interactions} interactions...")
        
        # Group by source node (u)
        node_groups = df.groupby('u')
        
        timestamp_entropies = []
        time_diff_entropies = []
        node_interaction_counts = []
        
        for node, group in node_groups:
            if len(group) < min_interactions:
                continue
            
            # Get timestamps for this node
            timestamps = group['ts'].values
            timestamps = np.sort(timestamps)  # Ensure chronological order
            
            # Normalize timestamps
            norm_timestamps = self.normalize_timestamps(timestamps)
            
            # Compute time differences
            time_diffs = np.diff(timestamps)
            if len(time_diffs) > 0:
                norm_time_diffs = self.normalize_timestamps(time_diffs)
            else:
                norm_time_diffs = np.array([])
            
            # Compute spectral entropy for timestamps
            ts_entropy = self.compute_spectral_entropy(norm_timestamps)
            timestamp_entropies.append(ts_entropy)
            
            # Compute spectral entropy for time differences
            if len(norm_time_diffs) > 0:
                td_entropy = self.compute_spectral_entropy(norm_time_diffs)
                time_diff_entropies.append(td_entropy)
            else:
                time_diff_entropies.append(0.0)
            
            node_interaction_counts.append(len(group))
        
        print(f"  Analyzed {len(timestamp_entropies)} nodes")
        
        return {
            'timestamp_entropies': np.array(timestamp_entropies),
            'time_diff_entropies': np.array(time_diff_entropies),
            'node_counts': np.array(node_interaction_counts),
            'total_nodes': len(timestamp_entropies)
        }
    
    def analyze_dataset(self, dataset_name, min_interactions=5):
        """
        Perform complete spectral entropy analysis for a dataset.
        
        Args:
            dataset_name (str): Name of the dataset
            min_interactions (int): Minimum interactions per node
            
        Returns:
            dict: Analysis results
        """
        try:
            # Load dataset
            df = self.load_dataset(dataset_name)
            
            # Analyze interactions
            results = self.analyze_node_interactions(df, min_interactions)
            
            # Add dataset metadata
            results['dataset_name'] = dataset_name
            results['total_interactions'] = len(df)
            results['total_unique_nodes'] = df['u'].nunique()
            
            # Store results
            self.entropy_results[dataset_name] = results
            
            print(f"✓ Completed analysis for {dataset_name}")
            print(f"  Total interactions: {results['total_interactions']}")
            print(f"  Nodes analyzed: {results['total_nodes']}")
            print(f"  Avg timestamp entropy: {np.mean(results['timestamp_entropies']):.3f}")
            print(f"  Avg time diff entropy: {np.mean(results['time_diff_entropies']):.3f}")
            print()
            
            return results
            
        except Exception as e:
            print(f"✗ Failed to analyze {dataset_name}: {e}")
            return None
    
    def analyze_all_datasets(self, min_interactions=5):
        """
        Analyze all datasets in the list.
        
        Args:
            min_interactions (int): Minimum interactions per node
        """
        print("Starting spectral entropy analysis for all datasets...")
        print("=" * 60)
        
        for dataset_name in self.datasets:
            self.analyze_dataset(dataset_name, min_interactions)
        
        print("Analysis complete!")
        
        # Save results
        results_file = self.output_dir / "spectral_entropy_results.pkl"
        with open(results_file, 'wb') as f:
            pickle.dump(self.entropy_results, f)
        print(f"Results saved to: {results_file}")


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='Spectral Entropy Analysis for Dynamic Graphs')
    parser.add_argument('--data_root', type=str, default='./data', 
                        help='Root directory containing dataset files')
    parser.add_argument('--output_dir', type=str, default='./spectral_entropy_results',
                        help='Directory to save results and plots')
    parser.add_argument('--min_interactions', type=int, default=5,
                        help='Minimum number of interactions per node')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to analyze (default: all)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = SpectralEntropyAnalyzer(args.data_root, args.output_dir)
    
    # Override dataset list if specified
    if args.datasets:
        analyzer.datasets = args.datasets
    
    # Run analysis
    analyzer.analyze_all_datasets(args.min_interactions)


if __name__ == "__main__":
    main()