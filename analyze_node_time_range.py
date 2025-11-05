#!/usr/bin/env python3
"""
Helper script to analyze K-MOTE gating patterns for specific nodes and time ranges.

This script provides easy analysis of temporal patterns with focus on specific time periods.

Usage Examples:
1. Analyze node 12 in the first quarter of data:
   python analyze_node_time_range.py --node_id 12 --time_percentage_start 0 --time_percentage_end 25

2. Analyze node 12 in the middle period:
   python analyze_node_time_range.py --node_id 12 --time_percentage_start 40 --time_percentage_end 60

3. Analyze node 12 with absolute timestamps:
   python analyze_node_time_range.py --node_id 12 --time_range_start 1000000 --time_range_end 5000000

Author: KAN-MAMMOTE Research Team
Date: November 4, 2025
"""

import argparse
import numpy as np
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.DataLoader import get_link_prediction_data
from node_level_kmote_analysis import NodeLevelKMOTEAnalyzer


def get_dataset_time_range(dataset_name, seed=0):
    """Get the full time range of a dataset."""
    print(f"Loading {dataset_name} dataset to determine time range...")
    
    _, _, full_data, _, _, _, _, _ = get_link_prediction_data(
        dataset_name=dataset_name,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=seed,
        data_ratio=1.0
    )
    
    timestamps = full_data.node_interact_times
    min_time = timestamps.min()
    max_time = timestamps.max()
    
    print(f"Dataset time range: [{min_time:.0f}, {max_time:.0f}]")
    print(f"Total time span: {max_time - min_time:.0f} time units")
    
    return min_time, max_time


def percentage_to_absolute_time(min_time, max_time, start_pct, end_pct):
    """Convert percentage ranges to absolute timestamps."""
    time_span = max_time - min_time
    
    start_time = min_time + (start_pct / 100.0) * time_span
    end_time = min_time + (end_pct / 100.0) * time_span
    
    print(f"Converting {start_pct}%-{end_pct}% to absolute time range:")
    print(f"  [{start_time:.0f}, {end_time:.0f}]")
    
    return start_time, end_time


def analyze_node_with_interpretation(node_id, model_name='DyGMamba', dataset_name='uci', 
                                   time_encoder_type='kan_mammote_dual_kmote',
                                   time_range=None, output_dir='./node_analysis_results'):
    """Analyze a node with detailed interpretation of results."""
    
    print(f"\n🔍 K-MOTE Gating Analysis for Node {node_id}")
    print("=" * 60)
    
    analyzer = NodeLevelKMOTEAnalyzer(output_dir)
    
    result = analyzer.analyze_node(
        model_name=model_name,
        dataset_name=dataset_name,
        time_encoder_type=time_encoder_type,
        node_id=node_id,
        seed=0,
        time_range=time_range
    )
    
    if result:
        print("\n🎯 INTERPRETATION GUIDE:")
        print("=" * 40)
        print("📈 GATING WEIGHTS MEANING:")
        print("   • Values range from 0 to 1, sum to 1.0 at each timestamp")
        print("   • Higher values = expert is more active/important")
        print("   • Spline Expert: Good for smooth, polynomial-like patterns")
        print("   • Fourier Expert: Good for periodic, oscillatory patterns") 
        print("   • Wavelet Expert: Good for localized, burst-like patterns")
        print()
        print("📊 ABSOLUTE vs RELATIVE K-MOTE:")
        print("   • Absolute K-MOTE: Processes actual timestamps")
        print("   • Relative K-MOTE: Processes time differences between interactions")
        print()
        print("🎨 VISUALIZATION GUIDE:")
        print("   • Top panel: Node's interaction partners over time")
        print("   • Middle panels: Expert utilization (higher lines = more active)")
        print("   • Bottom panel: Which expert dominates at each time point")
        print()
        print("✅ Analysis complete! Check the generated plots for visual insights.")
        
        return result
    else:
        print("❌ Analysis failed!")
        return None


def main():
    """Main function with enhanced time range support."""
    parser = argparse.ArgumentParser(description='Node K-MOTE Analysis with Time Range Focus')
    
    # Basic parameters
    parser.add_argument('--model_name', type=str, default='DyGMamba',
                        choices=['TGAT', 'TCL', 'CAWN', 'GraphMixer', 'DyGFormer', 'DyGMamba'],
                        help='Model name to analyze')
    parser.add_argument('--dataset_name', type=str, default='uci',
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'uci',
                                'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis'],
                        help='Dataset name to analyze')
    parser.add_argument('--time_encoder_type', type=str, default='kan_mammote_dual_kmote',
                        choices=['kan_mammote_dual_kmote', 'kan_mammote_dual_kmote_tgat'],
                        help='Time encoder type')
    parser.add_argument('--node_id', type=int, required=True,
                        help='Node ID to analyze')
    parser.add_argument('--output_dir', type=str, default='./node_analysis_results',
                        help='Output directory')
    
    # Time range options
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument('--time_percentage_start', type=float, default=None,
                           help='Start time as percentage of dataset (0-100)')
    time_group.add_argument('--time_range_start', type=float, default=None,
                           help='Absolute start timestamp')
    
    parser.add_argument('--time_percentage_end', type=float, default=None,
                        help='End time as percentage of dataset (0-100)')
    parser.add_argument('--time_range_end', type=float, default=None,
                        help='Absolute end timestamp')
    
    # Analysis options
    parser.add_argument('--show_dataset_info', action='store_true',
                        help='Show dataset time range information')
    
    args = parser.parse_args()
    
    # Show dataset info if requested
    if args.show_dataset_info:
        get_dataset_time_range(args.dataset_name)
        return
    
    # Determine time range
    time_range = None
    
    if args.time_percentage_start is not None:
        if args.time_percentage_end is None:
            print("❌ Both --time_percentage_start and --time_percentage_end must be provided")
            return
        
        # Convert percentage to absolute time
        min_time, max_time = get_dataset_time_range(args.dataset_name)
        start_time, end_time = percentage_to_absolute_time(
            min_time, max_time, args.time_percentage_start, args.time_percentage_end
        )
        time_range = (start_time, end_time)
        
    elif args.time_range_start is not None:
        if args.time_range_end is None:
            print("❌ Both --time_range_start and --time_range_end must be provided")
            return
        time_range = (args.time_range_start, args.time_range_end)
    
    # Run analysis
    result = analyze_node_with_interpretation(
        node_id=args.node_id,
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        time_encoder_type=args.time_encoder_type,
        time_range=time_range,
        output_dir=args.output_dir
    )
    
    # Suggest interesting time ranges for further analysis
    if result and time_range is None:
        print("\n💡 SUGGESTIONS FOR FOCUSED ANALYSIS:")
        print("=" * 45)
        print("Try analyzing specific time periods:")
        print(f"  • Early period: --time_percentage_start 0 --time_percentage_end 25")
        print(f"  • Middle period: --time_percentage_start 40 --time_percentage_end 60") 
        print(f"  • Late period: --time_percentage_start 75 --time_percentage_end 100")
        print()
        print("To see dataset time info:")
        print(f"  python {__file__} --show_dataset_info --dataset_name {args.dataset_name}")


if __name__ == "__main__":
    main()