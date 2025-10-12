#!/usr/bin/env python3
"""
Model Availability Checker

This script checks what trained models are available for evaluation
and creates a comprehensive report of training completion status.

Usage:
    python check_model_availability.py                    # Check all combinations
    python check_model_availability.py --csv              # Output as CSV
    python check_model_availability.py --summary          # Show summary only
"""

import glob
import os
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple

# Same configurations as other scripts
ALL_TIME_ENCODERS = ['mercer', 'bochner', 'time2vec', 'lete', 'kan_mammote', 'kan_mammote_lite', 'kan_mammote_dual_kmote', 'original']
ALL_MODELS = ['TGAT', 'JODIE', 'TGN', 'GraphMixer', 'DyGFormer', 'DyGMamba', 'TCL']
ALL_DATASETS = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
                'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']

def check_model_exists(model: str, dataset: str, encoder: str) -> Tuple[bool, List[str], int]:
    """
    Check if trained model exists for given combination
    Returns: (exists, model_files, num_seeds)
    """
    possible_patterns = [
        f"./saved_models/{model}/{dataset}/*{encoder}*seed*/*.pth",
        f"./saved_models/{model}/{dataset}/*{encoder}*seed*/*.pkl",
        f"./saved_models/{model}/{dataset}/{model}_{encoder}_seed*/*.pth",
        f"./saved_models/{model}/{dataset}/{model}_{encoder}_seed*/*.pkl",
        f"./saved_models/{model}/{dataset}/*{encoder}*/*.pth",
        f"./saved_models/{model}/{dataset}/*{encoder}*/*.pkl"
    ]
    
    all_found_models = []
    seed_counts = set()
    
    for pattern in possible_patterns:
        model_files = glob.glob(pattern)
        all_found_models.extend(model_files)
        
        # Count seeds
        for model_file in model_files:
            if 'seed' in model_file:
                # Extract seed number
                parts = model_file.split('seed')
                if len(parts) > 1:
                    seed_part = parts[1].split('/')[0].split('_')[0]
                    try:
                        seed_num = int(seed_part)
                        seed_counts.add(seed_num)
                    except ValueError:
                        pass
    
    # Remove duplicates
    unique_models = list(set(all_found_models))
    
    return len(unique_models) > 0, unique_models, len(seed_counts)

def check_all_combinations():
    """Check all model combinations and return status"""
    results = []
    
    print("🔍 Checking trained model availability...")
    print(f"Checking {len(ALL_MODELS)} models × {len(ALL_DATASETS)} datasets × {len(ALL_TIME_ENCODERS)} encoders = {len(ALL_MODELS) * len(ALL_DATASETS) * len(ALL_TIME_ENCODERS)} combinations")
    
    total_combinations = len(ALL_MODELS) * len(ALL_DATASETS) * len(ALL_TIME_ENCODERS)
    count = 0
    
    for model in ALL_MODELS:
        for dataset in ALL_DATASETS:
            for encoder in ALL_TIME_ENCODERS:
                count += 1
                if count % 50 == 0 or count == total_combinations:
                    print(f"  Progress: {count}/{total_combinations} ({count/total_combinations*100:.1f}%)")
                
                exists, model_files, num_seeds = check_model_exists(model, dataset, encoder)
                
                results.append({
                    'model': model,
                    'dataset': dataset,
                    'encoder': encoder,
                    'trained': 'Yes' if exists else 'No',
                    'num_model_files': len(model_files),
                    'num_seeds': num_seeds,
                    'model_files': '; '.join([os.path.basename(f) for f in model_files[:3]]) + ('; ...' if len(model_files) > 3 else '')
                })
    
    return results

def print_summary(results: List[Dict]):
    """Print a summary of results"""
    df = pd.DataFrame(results)
    
    total = len(results)
    trained = len(df[df['trained'] == 'Yes'])
    not_trained = total - trained
    
    print(f"\n{'='*80}")
    print("TRAINING COMPLETION SUMMARY")
    print(f"{'='*80}")
    print(f"Total Combinations: {total}")
    print(f"Trained: {trained} ({trained/total*100:.1f}%)")
    print(f"Not Trained: {not_trained} ({not_trained/total*100:.1f}%)")
    
    # Summary by model
    print(f"\nBy Model:")
    model_summary = df.groupby('model')['trained'].apply(lambda x: (x == 'Yes').sum()).sort_values(ascending=False)
    total_per_model = len(ALL_DATASETS) * len(ALL_TIME_ENCODERS)
    for model, count in model_summary.items():
        percentage = count / total_per_model * 100
        print(f"  {model:12s}: {count:3d}/{total_per_model} ({percentage:5.1f}%)")
    
    # Summary by encoder
    print(f"\nBy Encoder:")
    encoder_summary = df.groupby('encoder')['trained'].apply(lambda x: (x == 'Yes').sum()).sort_values(ascending=False)
    total_per_encoder = len(ALL_MODELS) * len(ALL_DATASETS)
    for encoder, count in encoder_summary.items():
        percentage = count / total_per_encoder * 100
        print(f"  {encoder:20s}: {count:3d}/{total_per_encoder} ({percentage:5.1f}%)")
    
    # Summary by dataset
    print(f"\nBy Dataset:")
    dataset_summary = df.groupby('dataset')['trained'].apply(lambda x: (x == 'Yes').sum()).sort_values(ascending=False)
    total_per_dataset = len(ALL_MODELS) * len(ALL_TIME_ENCODERS)
    for dataset, count in dataset_summary.items():
        percentage = count / total_per_dataset * 100
        print(f"  {dataset:12s}: {count:3d}/{total_per_dataset} ({percentage:5.1f}%)")
    
    # Most complete combinations
    print(f"\nMost Complete Model-Dataset Pairs:")
    model_dataset_summary = df.groupby(['model', 'dataset'])['trained'].apply(lambda x: (x == 'Yes').sum()).sort_values(ascending=False)
    total_encoders = len(ALL_TIME_ENCODERS)
    for (model, dataset), count in model_dataset_summary.head(10).items():
        percentage = count / total_encoders * 100
        print(f"  {model:12s} + {dataset:12s}: {count:2d}/{total_encoders} encoders ({percentage:5.1f}%)")
    
    # Least complete combinations
    if len(model_dataset_summary) > 10:
        print(f"\nLeast Complete Model-Dataset Pairs:")
        for (model, dataset), count in model_dataset_summary.tail(10).items():
            percentage = count / total_encoders * 100
            print(f"  {model:12s} + {dataset:12s}: {count:2d}/{total_encoders} encoders ({percentage:5.1f}%)")

def save_csv(results: List[Dict], output_file: str = None):
    """Save results to CSV"""
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"model_availability_{timestamp}.csv"
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"📊 Results saved to: {output_file}")
    
    # Also save a summary CSV
    summary_file = output_file.replace('.csv', '_summary.csv')
    
    # Create summary matrices
    
    # Model x Dataset matrix
    model_dataset_matrix = df.pivot_table(
        index='model', 
        columns='dataset', 
        values='trained', 
        aggfunc=lambda x: (x == 'Yes').sum()
    ).fillna(0).astype(int)
    
    # Model x Encoder matrix  
    model_encoder_matrix = df.pivot_table(
        index='model',
        columns='encoder', 
        values='trained',
        aggfunc=lambda x: (x == 'Yes').sum()
    ).fillna(0).astype(int)
    
    # Save matrices to separate sheets if possible, or separate files
    try:
        with pd.ExcelWriter(summary_file.replace('.csv', '.xlsx')) as writer:
            model_dataset_matrix.to_excel(writer, sheet_name='Model_Dataset_Matrix')
            model_encoder_matrix.to_excel(writer, sheet_name='Model_Encoder_Matrix')
            df.to_excel(writer, sheet_name='Complete_Results', index=False)
        print(f"📊 Summary matrices saved to: {summary_file.replace('.csv', '.xlsx')}")
    except ImportError:
        # Fallback to CSV if openpyxl not available
        model_dataset_matrix.to_csv(summary_file.replace('.csv', '_model_dataset_matrix.csv'))
        model_encoder_matrix.to_csv(summary_file.replace('.csv', '_model_encoder_matrix.csv'))
        print(f"📊 Summary matrices saved to CSV files")
    
    return output_file

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Check trained model availability')
    parser.add_argument('--csv', action='store_true',
                        help='Save results as CSV file')
    parser.add_argument('--output', type=str,
                        help='Output CSV filename')
    parser.add_argument('--summary', action='store_true',
                        help='Show summary only (no detailed list)')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS,
                        help='Check specific models only')
    parser.add_argument('--datasets', nargs='+', choices=ALL_DATASETS,
                        help='Check specific datasets only')
    parser.add_argument('--encoders', nargs='+', choices=ALL_TIME_ENCODERS,
                        help='Check specific encoders only')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    print("🔍 Model Availability Checker")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Override global lists if specific items requested
    global ALL_MODELS, ALL_DATASETS, ALL_TIME_ENCODERS
    if args.models:
        ALL_MODELS = args.models
    if args.datasets:
        ALL_DATASETS = args.datasets  
    if args.encoders:
        ALL_TIME_ENCODERS = args.encoders
    
    # Check all combinations
    results = check_all_combinations()
    
    # Print summary
    print_summary(results)
    
    # Show detailed results if not summary-only
    if not args.summary:
        print(f"\n{'='*80}")
        print("DETAILED RESULTS")
        print(f"{'='*80}")
        
        df = pd.DataFrame(results)
        
        # Show trained combinations
        trained_df = df[df['trained'] == 'Yes']
        if not trained_df.empty:
            print(f"\nTRAINED COMBINATIONS ({len(trained_df)}):")
            for _, row in trained_df.iterrows():
                print(f"✅ {row['model']:12s} + {row['dataset']:12s} + {row['encoder']:20s} ({row['num_seeds']} seeds, {row['num_model_files']} files)")
        
        # Show some untrained combinations  
        untrained_df = df[df['trained'] == 'No']
        if not untrained_df.empty:
            print(f"\nNOT TRAINED COMBINATIONS ({len(untrained_df)}) - showing first 20:")
            for _, row in untrained_df.head(20).iterrows():
                print(f"❌ {row['model']:12s} + {row['dataset']:12s} + {row['encoder']:20s}")
            if len(untrained_df) > 20:
                print(f"   ... and {len(untrained_df) - 20} more")
    
    # Save CSV if requested
    if args.csv or args.output:
        save_csv(results, args.output)
    
    print(f"\n✅ Check completed!")

if __name__ == "__main__":
    main()