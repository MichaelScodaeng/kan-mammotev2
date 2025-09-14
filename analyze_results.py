#!/usr/bin/env python3
"""
Results Analysis for KAN-MAMMOTE Time Encoder Comparison

Analyzes and compares the performance of different time encoders.
"""

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def parse_log_file(log_path):
    """Parse a log file to extract performance metrics."""
    if not os.path.exists(log_path):
        return None
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract metrics using regex
    metrics = {}
    
    # Look for validation and test results
    val_auc_pattern = r'validation auc: ([\d.]+)'
    test_auc_pattern = r'test auc: ([\d.]+)'
    val_ap_pattern = r'validation ap: ([\d.]+)'
    test_ap_pattern = r'test ap: ([\d.]+)'
    
    val_auc_matches = re.findall(val_auc_pattern, content)
    test_auc_matches = re.findall(test_auc_pattern, content)
    val_ap_matches = re.findall(val_ap_pattern, content)
    test_ap_matches = re.findall(test_ap_pattern, content)
    
    if val_auc_matches:
        metrics['val_auc'] = float(val_auc_matches[-1])  # Last (best) value
    if test_auc_matches:
        metrics['test_auc'] = float(test_auc_matches[-1])
    if val_ap_matches:
        metrics['val_ap'] = float(val_ap_matches[-1])
    if test_ap_matches:
        metrics['test_ap'] = float(test_ap_matches[-1])
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Analyze time encoder comparison results')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='Directory containing the experiment results')
    args = parser.parse_args()
    
    result_dir = Path(args.result_dir)
    
    if not result_dir.exists():
        print(f"Result directory {result_dir} does not exist!")
        return
    
    print("=== KAN-MAMMOTE Time Encoder Comparison Results ===\n")
    
    # Parse results for each encoder
    encoders = ['original', 'lete', 'kan_mammote']
    results = {}
    
    for encoder in encoders:
        log_file = result_dir / f"{encoder}.log"
        metrics = parse_log_file(log_file)
        
        if metrics:
            results[encoder] = metrics
            print(f"{encoder.upper()} Encoder:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
            print()
        else:
            print(f"{encoder.upper()} Encoder: No results found")
            print()
    
    # Create comparison table
    if len(results) > 1:
        print("=== Performance Comparison ===")
        df = pd.DataFrame(results).T
        print(df.round(4))
        print()
        
        # Find best performing encoder for each metric
        print("=== Best Performance ===")
        for metric in df.columns:
            best_encoder = df[metric].idxmax()
            best_score = df.loc[best_encoder, metric]
            print(f"{metric}: {best_encoder.upper()} ({best_score:.4f})")
        print()
        
        # Save results to CSV
        csv_path = result_dir / "comparison_results.csv"
        df.to_csv(csv_path)
        print(f"Results saved to: {csv_path}")
        
        # Create visualization if matplotlib is available
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Time Encoder Comparison', fontsize=16)
            
            metrics_to_plot = ['val_auc', 'test_auc', 'val_ap', 'test_ap']
            for i, metric in enumerate(metrics_to_plot):
                ax = axes[i // 2, i % 2]
                if metric in df.columns:
                    df[metric].plot(kind='bar', ax=ax, title=metric.upper())
                    ax.set_ylabel('Score')
                    ax.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plot_path = result_dir / "comparison_plot.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {plot_path}")
            
        except ImportError:
            print("matplotlib not available for plotting")


if __name__ == "__main__":
    main()
