#!/usr/bin/env python3
"""
Quick Runtime Estimation Script
==============================

This script provides quick runtime estimates by testing a few representative combinations
and extrapolating to estimate full experiment times.

Usage:
    python quick_runtime_estimate.py
"""

import os
import sys
import time
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from benchmark_epoch_runtime import benchmark_single_combination, get_sampling_strategy
from utils.load_configs import get_link_prediction_args
from utils import set_random_seed

# Representative combinations for quick estimation
REPRESENTATIVE_COMBINATIONS = [
    # Small dataset combinations
    ('TGAT', 'CanParl', 'mercer'),
    ('TGN', 'Contacts', 'lete'),
    ('DyGMamba', 'UNtrade', 'kan_mammote_dual_kmote'),
    
    # Medium dataset combinations  
    ('TGAT', 'enron', 'mercer'),
    ('TGN', 'uci', 'time2vec'),
    ('DyGFormer', 'SocialEvo', 'kan_mammote_dual_kmote'),
    
    # Large dataset combinations
    ('TGAT', 'mooc', 'mercer'),
    ('TGN', 'lastfm', 'lete'),
    ('DyGMamba', 'mooc', 'kan_mammote_dual_kmote'),
    
    # Huge dataset combinations
    ('TGAT', 'reddit', 'mercer'),
    ('TGN', 'wikipedia', 'lete'),
]

# Dataset size estimates for extrapolation
DATASET_SCALE_FACTORS = {
    'small': 1.0,     # Baseline
    'medium': 5.0,    # 5x slower than small
    'large': 15.0,    # 15x slower than small  
    'huge': 50.0      # 50x slower than small
}

# Model complexity factors (relative to TGAT)
MODEL_COMPLEXITY_FACTORS = {
    'TGAT': 1.0,       # Baseline
    'JODIE': 1.2,      # Slightly more complex
    'TGN': 1.3,        # Memory updates
    'DyGFormer': 2.0,  # Transformer-based
    'DyGMamba': 3.0,   # Mamba + KAN-MAMMOTE
    'TCL': 1.5,        # Contrastive learning
}

# Encoder complexity factors (relative to mercer)
ENCODER_COMPLEXITY_FACTORS = {
    'mercer': 1.0,                    # Baseline
    'lete': 1.1,                      # Slightly more complex
    'time2vec': 1.2,                  # Learned embeddings
    'kan_mammote_dual_kmote': 5.0,    # Much more complex
}


def run_quick_estimation():
    """Run quick estimation on representative combinations"""
    
    print("🚀 Quick Runtime Estimation")
    print("="*50)
    
    results = []
    args = get_link_prediction_args(is_evaluation=False)
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for i, (model, dataset, encoder) in enumerate(REPRESENTATIVE_COMBINATIONS):
        print(f"\n🔧 Testing {i+1}/{len(REPRESENTATIVE_COMBINATIONS)}: {model} + {dataset} + {encoder}")
        
        set_random_seed(42)
        
        result = benchmark_single_combination(
            model, dataset, encoder, args, timeout_seconds=180  # 3 minutes timeout
        )
        
        results.append(result)
        
        if result['status'] == 'success':
            print(f"   ✅ {result['epoch_time_seconds']:.2f}s per epoch")
        else:
            print(f"   ❌ Failed: {result['error_message']}")
    
    return results


def extrapolate_estimates(quick_results):
    """Extrapolate estimates to all combinations"""
    
    print(f"\n📊 Extrapolating estimates...")
    
    # Get successful results grouped by dataset size
    successful_results = [r for r in quick_results if r['status'] == 'success']
    
    if not successful_results:
        print("❌ No successful results to extrapolate from")
        return
    
    # Calculate baseline times per dataset size category
    baseline_times = {}
    for result in successful_results:
        dataset_size = get_sampling_strategy(result['dataset'])
        size_category = None
        
        # Determine size category from sampling strategy
        if dataset_size['data_ratio'] >= 1.0:
            size_category = 'small'
        elif dataset_size['data_ratio'] >= 0.3:
            size_category = 'medium'
        elif dataset_size['data_ratio'] >= 0.1:
            size_category = 'large'
        else:
            size_category = 'huge'
        
        if size_category not in baseline_times:
            baseline_times[size_category] = []
        
        baseline_times[size_category].append(result['epoch_time_seconds'])
    
    # Calculate average baseline time per category
    avg_baseline_times = {}
    for category, times in baseline_times.items():
        avg_baseline_times[category] = sum(times) / len(times)
        print(f"   {category.capitalize()} datasets: ~{avg_baseline_times[category]:.1f}s per epoch (avg)")
    
    # Generate estimates for all combinations
    from benchmark_epoch_runtime import ALL_MODELS, ALL_DATASETS, ALL_ENCODERS, DATASET_SIZES
    
    estimates = []
    
    for model in ALL_MODELS:
        for dataset in ALL_DATASETS:
            for encoder in ALL_ENCODERS:
                
                # Get base time from dataset size
                dataset_size_category = DATASET_SIZES.get(dataset, 'medium')
                
                if dataset_size_category in avg_baseline_times:
                    base_time = avg_baseline_times[dataset_size_category]
                else:
                    # Use closest available category
                    base_time = list(avg_baseline_times.values())[0]
                
                # Apply complexity factors
                model_factor = MODEL_COMPLEXITY_FACTORS.get(model, 1.0)
                encoder_factor = ENCODER_COMPLEXITY_FACTORS.get(encoder, 1.0)
                
                estimated_time = base_time * model_factor * encoder_factor
                
                estimates.append({
                    'model': model,
                    'dataset': dataset,
                    'encoder': encoder,
                    'dataset_size_category': dataset_size_category,
                    'estimated_epoch_time_seconds': estimated_time,
                    'estimated_100_epoch_time_hours': estimated_time * 100 / 3600,
                    'model_complexity_factor': model_factor,
                    'encoder_complexity_factor': encoder_factor
                })
    
    return estimates


def analyze_estimates(estimates):
    """Analyze and present the estimates"""
    
    df = pd.DataFrame(estimates)
    
    print(f"\n📈 ANALYSIS RESULTS")
    print("="*50)
    
    # Overall statistics
    print(f"Total combinations: {len(df)}")
    print(f"Average epoch time: {df['estimated_epoch_time_seconds'].mean():.1f}s")
    print(f"Fastest combination: {df['estimated_epoch_time_seconds'].min():.1f}s")
    print(f"Slowest combination: {df['estimated_epoch_time_seconds'].max():.1f}s")
    
    # Time for 100 epochs
    print(f"\nEstimated time for 100 epochs:")
    print(f"  Average: {df['estimated_100_epoch_time_hours'].mean():.1f} hours")
    print(f"  Fastest: {df['estimated_100_epoch_time_hours'].min():.1f} hours")
    print(f"  Slowest: {df['estimated_100_epoch_time_hours'].max():.1f} hours")
    print(f"  Total for all combinations: {df['estimated_100_epoch_time_hours'].sum():.0f} hours")
    
    # By model
    print(f"\n📊 BY MODEL (avg epoch time):")
    model_avg = df.groupby('model')['estimated_epoch_time_seconds'].mean().sort_values()
    for model, time in model_avg.items():
        print(f"  {model:12s}: {time:6.1f}s")
    
    # By encoder
    print(f"\n📊 BY ENCODER (avg epoch time):")
    encoder_avg = df.groupby('encoder')['estimated_epoch_time_seconds'].mean().sort_values()
    for encoder, time in encoder_avg.items():
        print(f"  {encoder:25s}: {time:6.1f}s")
    
    # By dataset size
    print(f"\n📊 BY DATASET SIZE (avg epoch time):")
    size_avg = df.groupby('dataset_size_category')['estimated_epoch_time_seconds'].mean().sort_values()
    for size, time in size_avg.items():
        print(f"  {size.capitalize():8s}: {time:6.1f}s")
    
    # Top 10 slowest combinations
    print(f"\n🐌 TOP 10 SLOWEST COMBINATIONS:")
    slowest = df.nlargest(10, 'estimated_epoch_time_seconds')
    for _, row in slowest.iterrows():
        print(f"  {row['model']} + {row['dataset']} + {row['encoder']}: {row['estimated_epoch_time_seconds']:.1f}s")
    
    # Top 10 fastest combinations
    print(f"\n🚀 TOP 10 FASTEST COMBINATIONS:")
    fastest = df.nsmallest(10, 'estimated_epoch_time_seconds')
    for _, row in fastest.iterrows():
        print(f"  {row['model']} + {row['dataset']} + {row['encoder']}: {row['estimated_epoch_time_seconds']:.1f}s")
    
    # Save estimates
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'runtime_estimates_{timestamp}.csv'
    df.to_csv(output_file, index=False)
    print(f"\n💾 Estimates saved to: {output_file}")
    
    return df


def main():
    """Main execution"""
    
    # Run quick estimation
    quick_results = run_quick_estimation()
    
    # Extrapolate to all combinations
    estimates = extrapolate_estimates(quick_results)
    
    if estimates:
        # Analyze and present results
        analyze_estimates(estimates)
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"  1. Start with small datasets (CanParl, Contacts, UNtrade) for testing")
        print(f"  2. Use simpler encoders (mercer, lete) for initial experiments")
        print(f"  3. Test KAN-MAMMOTE on small datasets first")
        print(f"  4. Consider data_ratio sampling for large datasets (reddit, wikipedia)")
        print(f"  5. Plan for ~{sum(est['estimated_100_epoch_time_hours'] for est in estimates):.0f} total hours for all combinations")


if __name__ == '__main__':
    main()