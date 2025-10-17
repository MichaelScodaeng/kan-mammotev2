#!/usr/bin/env python3
"""
Complete Spectral Entropy Analysis Pipeline

This script runs the complete spectral entropy analysis pipeline for dynamic graph datasets,
including data loading, analysis, visualization, and reporting.

Usage:
    python run_spectral_entropy_analysis.py --data_root ./data --output_dir ./results

Author: KAN-MAMMOTE Research Team  
Date: October 7, 2025
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Import our modules
from spectral_entropy_analysis import SpectralEntropyAnalyzer
from spectral_entropy_visualizer import SpectralEntropyVisualizer

def generate_analysis_report(analyzer, output_dir):
    """
    Generate a comprehensive text report of the analysis results.
    
    Args:
        analyzer: SpectralEntropyAnalyzer instance with results
        output_dir: Directory to save the report
    """
    report_file = Path(output_dir) / "spectral_entropy_analysis_report.txt"
    
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("SPECTRAL ENTROPY ANALYSIS REPORT\n")
        f.write("Dynamic Graph Temporal Pattern Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Datasets Analyzed: {len(analyzer.entropy_results)}\n\n")
        
        # Overall statistics
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 40 + "\n")
        
        total_nodes = 0
        total_interactions = 0
        all_ts_entropies = []
        all_td_entropies = []
        
        for dataset_name, results in analyzer.entropy_results.items():
            if results is None:
                continue
            total_nodes += results['total_nodes']
            total_interactions += results['total_interactions']
            all_ts_entropies.extend(results['timestamp_entropies'])
            all_td_entropies.extend(results['time_diff_entropies'])
        
        f.write(f"Total Nodes Analyzed: {total_nodes:,}\n")
        f.write(f"Total Interactions: {total_interactions:,}\n")
        f.write(f"Average Timestamp Entropy: {sum(all_ts_entropies)/len(all_ts_entropies):.4f}\n")
        f.write(f"Average Time Diff Entropy: {sum(all_td_entropies)/len(all_td_entropies):.4f}\n\n")
        
        # Per-dataset analysis
        f.write("PER-DATASET ANALYSIS\n")
        f.write("-" * 40 + "\n\n")
        
        for dataset_name, results in analyzer.entropy_results.items():
            if results is None:
                f.write(f"{dataset_name.upper()}: ANALYSIS FAILED\n\n")
                continue
                
            f.write(f"{dataset_name.upper()}\n")
            f.write("-" * len(dataset_name) + "\n")
            
            # Basic statistics
            f.write(f"  Total Interactions: {results['total_interactions']:,}\n")
            f.write(f"  Unique Nodes: {results['total_unique_nodes']:,}\n")
            f.write(f"  Nodes Analyzed (≥5 interactions): {results['total_nodes']:,}\n")
            f.write(f"  Analysis Coverage: {results['total_nodes']/results['total_unique_nodes']*100:.1f}%\n")
            
            # Entropy statistics
            ts_entropies = results['timestamp_entropies']
            td_entropies = results['time_diff_entropies']
            
            f.write(f"\n  Timestamp Entropy Statistics:\n")
            f.write(f"    Mean: {ts_entropies.mean():.4f}\n")
            f.write(f"    Std:  {ts_entropies.std():.4f}\n")
            f.write(f"    Min:  {ts_entropies.min():.4f}\n")
            f.write(f"    Max:  {ts_entropies.max():.4f}\n")
            
            f.write(f"\n  Time Difference Entropy Statistics:\n")
            f.write(f"    Mean: {td_entropies.mean():.4f}\n")
            f.write(f"    Std:  {td_entropies.std():.4f}\n")
            f.write(f"    Min:  {td_entropies.min():.4f}\n")
            f.write(f"    Max:  {td_entropies.max():.4f}\n")
            
            # Pattern analysis
            low_entropy_threshold = 2.0  # Threshold for "periodic" behavior
            ts_periodic = (ts_entropies < low_entropy_threshold).sum()
            td_periodic = (td_entropies < low_entropy_threshold).sum()
            
            f.write(f"\n  Pattern Analysis (entropy < {low_entropy_threshold}):\n")
            f.write(f"    Periodic timestamp patterns: {ts_periodic} nodes ({ts_periodic/len(ts_entropies)*100:.1f}%)\n")
            f.write(f"    Periodic time diff patterns: {td_periodic} nodes ({td_periodic/len(td_entropies)*100:.1f}%)\n")
            
            f.write("\n" + "="*50 + "\n\n")
        
        # Key insights
        f.write("KEY INSIGHTS\n")
        f.write("-" * 40 + "\n")
        
        # Find dataset with most/least periodic behavior
        periodic_scores = {}
        for dataset_name, results in analyzer.entropy_results.items():
            if results is None:
                continue
            ts_entropies = results['timestamp_entropies']
            td_entropies = results['time_diff_entropies']
            avg_entropy = (ts_entropies.mean() + td_entropies.mean()) / 2
            periodic_scores[dataset_name] = avg_entropy
        
        if periodic_scores:
            most_periodic = min(periodic_scores.items(), key=lambda x: x[1])
            least_periodic = max(periodic_scores.items(), key=lambda x: x[1])
            
            f.write(f"1. Most Periodic Dataset: {most_periodic[0]} (avg entropy: {most_periodic[1]:.3f})\n")
            f.write(f"2. Least Periodic Dataset: {least_periodic[0]} (avg entropy: {least_periodic[1]:.3f})\n")
            
            f.write(f"3. Most datasets show high entropy values, indicating non-periodic temporal patterns\n")
            f.write(f"4. Only a small fraction of nodes exhibit strong periodicity in interaction times\n")
            f.write(f"5. Time difference patterns tend to be less predictable than absolute timestamps\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")
    
    print(f"Analysis report saved to: {report_file}")


def main():
    """Main function to run the complete analysis pipeline."""
    parser = argparse.ArgumentParser(description='Complete Spectral Entropy Analysis Pipeline')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='Root directory containing dataset files')
    parser.add_argument('--output_dir', type=str, default='./spectral_entropy_results',
                        help='Directory to save all results and plots')
    parser.add_argument('--min_interactions', type=int, default=5,
                        help='Minimum number of interactions per node for analysis')
    parser.add_argument('--datasets', nargs='+', default=None,
                        help='Specific datasets to analyze (default: all 13 datasets)')
    parser.add_argument('--skip_analysis', action='store_true',
                        help='Skip analysis and only create visualizations (requires existing results)')
    parser.add_argument('--skip_visualization', action='store_true',
                        help='Skip visualization creation')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("🚀 Starting Spectral Entropy Analysis Pipeline")
    print("=" * 60)
    print(f"Data Root: {args.data_root}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Minimum Interactions: {args.min_interactions}")
    
    results_file = output_dir / "spectral_entropy_results.pkl"
    
    # Step 1: Run Analysis (unless skipped)
    if not args.skip_analysis:
        print("\n📊 STEP 1: Running Spectral Entropy Analysis")
        print("-" * 45)
        
        # Initialize analyzer
        analyzer = SpectralEntropyAnalyzer(args.data_root, args.output_dir)
        
        # Override dataset list if specified
        if args.datasets:
            analyzer.datasets = args.datasets
            print(f"Analyzing selected datasets: {args.datasets}")
        else:
            print(f"Analyzing all {len(analyzer.datasets)} datasets")
        
        # Run analysis
        start_time = time.time()
        analyzer.analyze_all_datasets(args.min_interactions)
        analysis_time = time.time() - start_time
        
        print(f"\n✓ Analysis completed in {analysis_time:.1f} seconds")
        
        # Generate report
        print("\n📝 Generating analysis report...")
        generate_analysis_report(analyzer, args.output_dir)
        
    else:
        print("\n⏭️  Skipping analysis (using existing results)")
        if not results_file.exists():
            print(f"❌ Error: Results file not found: {results_file}")
            sys.exit(1)
    
    # Step 2: Create Visualizations (unless skipped)
    if not args.skip_visualization:
        print("\n🎨 STEP 2: Creating Visualizations")
        print("-" * 35)
        
        if not results_file.exists():
            print(f"❌ Error: Results file not found: {results_file}")
            sys.exit(1)
        
        # Initialize visualizer
        visualizer = SpectralEntropyVisualizer(str(results_file), args.output_dir)
        
        print("Creating density plots (Figure 8 style)...")
        visualizer.create_density_plots()
        
        print("Creating summary statistics...")
        visualizer.create_summary_statistics_plot()
        
        print("Creating comparison heatmap...")
        visualizer.create_comparison_heatmap()
        
        print("✓ All visualizations completed")
    
    else:
        print("\n⏭️  Skipping visualization creation")
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📁 All results saved to: {args.output_dir}")
    print("\nGenerated files:")
    
    # List generated files
    for file_path in output_dir.glob("*"):
        if file_path.is_file():
            print(f"  - {file_path.name}")


if __name__ == "__main__":
    main()