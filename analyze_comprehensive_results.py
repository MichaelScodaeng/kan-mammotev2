#!/usr/bin/env python3
"""
Analyze comprehensive evaluation results across all negative sampling strategies.

This script reads the comprehensive evaluation JSON files and creates a summary
showing performance across different negative sampling strategies.

Usage:
    python analyze_comprehensive_results.py --results_dir ./saved_results
    python analyze_comprehensive_results.py --model TGAT --dataset wikipedia
"""

import json
import argparse
import os
from pathlib import Path
import pandas as pd
from collections import defaultdict


def find_comprehensive_results(results_dir, model=None, dataset=None):
    """Find all comprehensive evaluation result files"""
    
    results_dir = Path(results_dir)
    comprehensive_files = []
    
    # Search pattern
    if model and dataset:
        pattern = f"{results_dir}/{model}/{dataset}/*_comprehensive_*.json"
    elif model:
        pattern = f"{results_dir}/{model}/**/*_comprehensive_*.json"
    elif dataset:
        pattern = f"{results_dir}/**/{dataset}/*_comprehensive_*.json"
    else:
        pattern = f"{results_dir}/**/*_comprehensive_*.json"
    
    # Find all comprehensive result files
    for file_path in results_dir.rglob("*_comprehensive_*.json"):
        if model and model not in str(file_path):
            continue
        if dataset and dataset not in str(file_path):
            continue
        comprehensive_files.append(file_path)
    
    return comprehensive_files


def load_comprehensive_result(file_path):
    """Load and parse a comprehensive result file"""
    
    try:
        with open(file_path) as f:
            data = json.load(f)
        
        # Extract metadata from path
        parts = file_path.parts
        dataset = parts[-2]
        model = parts[-3]
        
        result = {
            'file': str(file_path),
            'model': model,
            'dataset': dataset,
            'time_encoder': data.get('time_encoder_type', 'unknown'),
            'seed': data.get('seed', 0),
            'primary_strategy': data.get('primary_strategy', 'unknown'),
            'strategies': data.get('strategies', {})
        }
        
        return result
        
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def create_comparison_table(results):
    """Create a comparison table across all strategies"""
    
    rows = []
    
    for result in results:
        base_row = {
            'Model': result['model'],
            'Dataset': result['dataset'],
            'Encoder': result['time_encoder'],
            'Seed': result['seed'],
            'Primary': result['primary_strategy']
        }
        
        for strategy, metrics in result['strategies'].items():
            row = base_row.copy()
            row['Strategy'] = strategy
            
            # Transductive metrics
            row['Trans_Loss'] = metrics['transductive_test']['loss']
            row['Trans_AP'] = metrics['transductive_test']['metrics'].get('average_precision', 0)
            row['Trans_AUC'] = metrics['transductive_test']['metrics'].get('roc_auc', 0)
            
            # Inductive metrics
            row['Ind_Loss'] = metrics['inductive_test']['loss']
            row['Ind_AP'] = metrics['inductive_test']['metrics'].get('average_precision', 0)
            row['Ind_AUC'] = metrics['inductive_test']['metrics'].get('roc_auc', 0)
            
            rows.append(row)
    
    return pd.DataFrame(rows)


def print_summary(df):
    """Print summary statistics"""
    
    print("\n" + "="*100)
    print("📊 COMPREHENSIVE EVALUATION SUMMARY")
    print("="*100)
    
    # Group by model and dataset
    for (model, dataset), group in df.groupby(['Model', 'Dataset']):
        print(f"\n{'─'*100}")
        print(f"Model: {model} | Dataset: {dataset}")
        print(f"{'─'*100}")
        
        # Average across seeds for each strategy
        strategy_avg = group.groupby('Strategy').agg({
            'Trans_AP': ['mean', 'std'],
            'Trans_AUC': ['mean', 'std'],
            'Ind_AP': ['mean', 'std'],
            'Ind_AUC': ['mean', 'std']
        })
        
        print(f"\n{'Strategy':<15} {'Trans AP':<20} {'Trans AUC':<20} {'Ind AP':<20} {'Ind AUC':<20}")
        print("─"*100)
        
        for strategy in ['random', 'historical', 'inductive']:
            if strategy not in strategy_avg.index:
                continue
            
            trans_ap = f"{strategy_avg.loc[strategy, ('Trans_AP', 'mean')]:.4f} ± {strategy_avg.loc[strategy, ('Trans_AP', 'std')]:.4f}"
            trans_auc = f"{strategy_avg.loc[strategy, ('Trans_AUC', 'mean')]:.4f} ± {strategy_avg.loc[strategy, ('Trans_AUC', 'std')]:.4f}"
            ind_ap = f"{strategy_avg.loc[strategy, ('Ind_AP', 'mean')]:.4f} ± {strategy_avg.loc[strategy, ('Ind_AP', 'std')]:.4f}"
            ind_auc = f"{strategy_avg.loc[strategy, ('Ind_AUC', 'mean')]:.4f} ± {strategy_avg.loc[strategy, ('Ind_AUC', 'std')]:.4f}"
            
            print(f"{strategy:<15} {trans_ap:<20} {trans_auc:<20} {ind_ap:<20} {ind_auc:<20}")
        
        # Find best strategy
        best_strategy = strategy_avg[('Trans_AP', 'mean')].idxmax()
        print(f"\n✅ Best strategy for transductive test: {best_strategy}")
        
        best_strategy_ind = strategy_avg[('Ind_AP', 'mean')].idxmax()
        print(f"✅ Best strategy for inductive test: {best_strategy_ind}")
    
    print("\n" + "="*100 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze comprehensive evaluation results'
    )
    parser.add_argument('--results_dir', type=str, default='./saved_results',
                        help='Directory containing result files')
    parser.add_argument('--model', type=str, default=None,
                        help='Filter by model name')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Filter by dataset name')
    parser.add_argument('--output_csv', type=str, default=None,
                        help='Save results to CSV file')
    
    args = parser.parse_args()
    
    # Find comprehensive result files
    print(f"Searching for comprehensive results in: {args.results_dir}")
    files = find_comprehensive_results(args.results_dir, args.model, args.dataset)
    
    if not files:
        print("❌ No comprehensive result files found!")
        print(f"   Expected pattern: *_comprehensive_*.json")
        return
    
    print(f"✅ Found {len(files)} comprehensive result files")
    
    # Load all results
    results = []
    for file_path in files:
        result = load_comprehensive_result(file_path)
        if result:
            results.append(result)
    
    if not results:
        print("❌ No valid results could be loaded!")
        return
    
    print(f"✅ Loaded {len(results)} valid results")
    
    # Create comparison table
    df = create_comparison_table(results)
    
    # Print summary
    print_summary(df)
    
    # Save to CSV if requested
    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"✅ Results saved to: {args.output_csv}")
    
    # Print raw data for detailed inspection
    print("\n" + "="*100)
    print("📋 DETAILED RESULTS TABLE")
    print("="*100)
    print(df.to_string(index=False))
    print("\n")


if __name__ == '__main__':
    main()
