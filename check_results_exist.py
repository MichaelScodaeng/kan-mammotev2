import subprocess
import itertools
import os
import time
import argparse
import sys
import logging
import glob
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path

# All configurations - same as test_integration.py
ALL_TIME_ENCODERS = ['mercer', 'time2vec', 'lete', 'kan_mammote_dual_kmote', 'original']
ALL_MODELS = ['TGAT', 'JODIE', 'TGN', 'DyGFormer', 'DyGMamba', 'TCL']  # Exclude CAWN
ALL_DATASETS = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
                'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']
ALL_NEG_STRATEGIES = ['random', 'historical', 'inductive']

# Quick test configurations
QUICK_DATASETS = ['wikipedia', 'reddit', 'mooc']
QUICK_MODELS = ['TGAT', 'TGN', 'DyGMamba']
QUICK_ENCODERS = ['mercer', 'kan_mammote_dual_kmote']


def check_evaluation_results_exist(model: str, dataset: str, encoder: str, neg_strategy: str = None) -> Tuple[bool, List[str], Dict]:
    """
    Check if evaluation results already exist for the given combination.
    
    File naming patterns:
    - Random (default): {model}_{encoder}_seed{N}_{timestamp}.json
    - Historical: historical_negative_sampling_{model}_{encoder}_seed{N}.json
    - Inductive: inductive_negative_sampling_{model}_{encoder}_seed{N}.json
    
    Args:
        model: Model name (e.g., 'JODIE', 'TGN')
        dataset: Dataset name (e.g., 'wikipedia', 'reddit')
        encoder: Time encoder type (e.g., 'time2vec', 'kan_mammote_dual_kmote')
        neg_strategy: Negative sampling strategy ('random', 'historical', 'inductive', or None to check all)
    
    Returns:
        Tuple of (exists: bool, files: List[str], info: Dict)
    """
    
    base_dir = f"./saved_results/{model}/{dataset}"
    
    if neg_strategy is None or neg_strategy == 'random':
        # Random strategy: {model}_{encoder}_seed0_{timestamp}.json
        # Also matches patterns with validation suffixes like *_val_*
        result_pattern = f"{base_dir}/{model}_{encoder}_seed0*.json"
    elif neg_strategy == 'historical':
        # Historical strategy: historical_negative_sampling_{model}_{encoder}_seed0.json
        result_pattern = f"{base_dir}/historical_negative_sampling_{model}_{encoder}_seed0*.json"
    elif neg_strategy == 'inductive':
        # Inductive strategy: inductive_negative_sampling_{model}_{encoder}_seed0.json
        result_pattern = f"{base_dir}/inductive_negative_sampling_{model}_{encoder}_seed0*.json"
    else:
        raise ValueError(f"Unknown neg_strategy: {neg_strategy}. Must be 'random', 'historical', 'inductive', or None.")
    
    existing_results = glob.glob(result_pattern)
    
    # Filter out validation files if looking for baseline (no _val_ in filename)
    if neg_strategy is not None:
        # Keep only files that match the expected pattern exactly
        if neg_strategy == 'random':
            # Exclude historical/inductive prefixed files
            existing_results = [f for f in existing_results 
                              if not os.path.basename(f).startswith('historical_negative_sampling_')
                              and not os.path.basename(f).startswith('inductive_negative_sampling_')]
    
    result_info = {
        'pattern': result_pattern,
        'count': len(existing_results),
        'files': existing_results,
        'neg_strategy': neg_strategy
    }
    
    if existing_results:
        # Check if results contain actual evaluation metrics
        return True, existing_results, result_info
    
    return False, [], result_info


def get_metrics_from_result_file(model: str, dataset: str, encoder: str, neg_strategy: str = 'random') -> Optional[Dict]:
    """
    Extract test metrics from result JSON file for the given combination.
    
    Args:
        model: Model name
        dataset: Dataset name
        encoder: Time encoder type
        neg_strategy: Negative sampling strategy ('random', 'historical', 'inductive')
    
    Returns:
        Dictionary of metrics or None if file not found
    """
    
    base_dir = f"./saved_results/{model}/{dataset}"
    
    if neg_strategy == 'random':
        result_pattern = f"{base_dir}/{model}_{encoder}_seed0_*.json"
    elif neg_strategy == 'historical':
        result_pattern = f"{base_dir}/historical_negative_sampling_{model}_{encoder}_seed0*.json"
    elif neg_strategy == 'inductive':
        result_pattern = f"{base_dir}/inductive_negative_sampling_{model}_{encoder}_seed0*.json"
    else:
        raise ValueError(f"Unknown neg_strategy: {neg_strategy}")
    
    existing_results = glob.glob(result_pattern)
    
    # Filter out unwanted files for random strategy
    if neg_strategy == 'random':
        existing_results = [f for f in existing_results 
                          if not os.path.basename(f).startswith('historical_negative_sampling_')
                          and not os.path.basename(f).startswith('inductive_negative_sampling_')]
    
    if not existing_results:
        return None
    
    # Use the first (most recent) result file if multiple exist
    result_file = existing_results[0]
    
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        # Extract the test metrics
        metrics = {}
        
        if "test metrics" in data:
            metrics["test_metrics"] = data["test metrics"]
        
        if "new node test metrics" in data:
            metrics["new_node_test_metrics"] = data["new node test metrics"]
        
        # Also include validation metrics for completeness
        if "validate metrics" in data:
            metrics["validate_metrics"] = data["validate metrics"]
            
        if "new node validate metrics" in data:
            metrics["new_node_validate_metrics"] = data["new node validate metrics"]
        
        return metrics if metrics else None
        
    except (json.JSONDecodeError, FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not read metrics from {result_file}: {e}")
        return None


def create_completion_matrix(combinations: List, check_func, matrix_name: str, output_dir: str) -> pd.DataFrame:
    """Create a completion matrix and save as CSV"""
    
    print(f"\n📊 Creating {matrix_name} completion matrix...")
    
    # Extract unique values for each dimension
    if matrix_name == "Model-Dataset":
        row_items = sorted(list(set([combo[0] for combo in combinations])))  # models
        col_items = sorted(list(set([combo[1] for combo in combinations])))  # datasets
        get_row = lambda combo: combo[0]
        get_col = lambda combo: combo[1]
    elif matrix_name == "Model-Encoder":
        row_items = sorted(list(set([combo[0] for combo in combinations])))  # models  
        col_items = sorted(list(set([combo[2] for combo in combinations])))  # encoders
        get_row = lambda combo: combo[0]
        get_col = lambda combo: combo[2]
    elif matrix_name == "Dataset-Encoder":
        row_items = sorted(list(set([combo[1] for combo in combinations])))  # datasets
        col_items = sorted(list(set([combo[2] for combo in combinations])))  # encoders
        get_row = lambda combo: combo[1]
        get_col = lambda combo: combo[2]
    else:
        raise ValueError(f"Unknown matrix_name: {matrix_name}")
    
    # Initialize matrix with 0s
    matrix = pd.DataFrame(0, index=row_items, columns=col_items)
    
    # Fill matrix based on completion status
    for combo in combinations:
        model, dataset, encoder, neg_strategy = combo
        row_key = get_row(combo)
        col_key = get_col(combo)
        
        completed, files, info = check_func(model, dataset, encoder, neg_strategy)
        
        if completed:
            matrix.loc[row_key, col_key] = 1
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, f"{matrix_name.lower().replace('-', '_')}_completion.csv")
    matrix.to_csv(csv_path)
    
    print(f"   💾 Saved to: {csv_path}")
    print(f"   ✅ Completed: {matrix.sum().sum()}/{matrix.size} ({matrix.sum().sum()/matrix.size*100:.1f}%)")
    
    return matrix


def create_comprehensive_results_table(models: List[str], datasets: List[str], 
                                      encoders: List[str], neg_strategies: List[str],
                                      output_dir: str) -> pd.DataFrame:
    """
    Create a comprehensive long-format table with all results across all strategies.
    
    Table columns: Model | Dataset | Encoder | Strategy | Test_AP | Test_AUC | New_Node_AP | New_Node_AUC | ...
    
    Args:
        models: List of models
        datasets: List of datasets
        encoders: List of time encoders
        neg_strategies: List of negative sampling strategies
        output_dir: Directory to save the table
    
    Returns:
        DataFrame in long format with all results
    """
    
    print(f"\n📊 Creating comprehensive results table...")
    print(f"   Models: {len(models)}")
    print(f"   Datasets: {len(datasets)}")
    print(f"   Encoders: {len(encoders)}")
    print(f"   Strategies: {neg_strategies}")
    
    results_list = []
    
    total_combinations = len(models) * len(datasets) * len(encoders) * len(neg_strategies)
    processed = 0
    found = 0
    
    for model in sorted(models):
        for dataset in sorted(datasets):
            for encoder in sorted(encoders):
                for neg_strategy in neg_strategies:
                    processed += 1
                    
                    # Get metrics for this combination
                    metrics = get_metrics_from_result_file(model, dataset, encoder, neg_strategy)
                    
                    if metrics:
                        found += 1
                        # Create a row for this combination
                        row = {
                            'model': model,
                            'dataset': dataset,
                            'encoder': encoder,
                            'neg_strategy': neg_strategy
                        }
                        
                        # Extract test metrics
                        if 'test_metrics' in metrics and metrics['test_metrics']:
                            for metric_name, metric_value in metrics['test_metrics'].items():
                                row[f'test_{metric_name}'] = metric_value
                        
                        # Extract new node test metrics
                        if 'new_node_test_metrics' in metrics and metrics['new_node_test_metrics']:
                            for metric_name, metric_value in metrics['new_node_test_metrics'].items():
                                row[f'new_node_test_{metric_name}'] = metric_value
                        
                        # Extract validation metrics (optional)
                        if 'validate_metrics' in metrics and metrics['validate_metrics']:
                            for metric_name, metric_value in metrics['validate_metrics'].items():
                                row[f'val_{metric_name}'] = metric_value
                        
                        # Extract new node validation metrics (optional)
                        if 'new_node_validate_metrics' in metrics and metrics['new_node_validate_metrics']:
                            for metric_name, metric_value in metrics['new_node_validate_metrics'].items():
                                row[f'new_node_val_{metric_name}'] = metric_value
                        
                        results_list.append(row)
                    
                    # Progress indicator
                    if processed % 100 == 0:
                        print(f"   Progress: {processed}/{total_combinations} ({processed/total_combinations*100:.1f}%) - Found: {found}")
    
    # Create DataFrame
    if results_list:
        df = pd.DataFrame(results_list)
        
        # Reorder columns for readability
        base_cols = ['model', 'dataset', 'encoder', 'neg_strategy']
        other_cols = [col for col in df.columns if col not in base_cols]
        df = df[base_cols + sorted(other_cols)]
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "comprehensive_results_table.csv")
        df.to_csv(csv_path, index=False)
        
        print(f"\n   💾 Saved comprehensive table to: {csv_path}")
        print(f"   ✅ Total rows: {len(df)}")
        print(f"   📊 Coverage: {len(df)}/{total_combinations} ({len(df)/total_combinations*100:.1f}%)")
        
        # Also create a pivot table for quick reference (Test AP only)
        if 'test_average_precision' in df.columns:
            pivot_df = df.pivot_table(
                index=['model', 'dataset'],
                columns=['encoder', 'neg_strategy'],
                values='test_average_precision'
            )
            pivot_path = os.path.join(output_dir, "test_ap_pivot_table.csv")
            pivot_df.to_csv(pivot_path)
            print(f"   💾 Saved pivot table (Test AP) to: {pivot_path}")
        
        return df
    else:
        print(f"   ⚠️  No results found!")
        return pd.DataFrame()


def create_time_encoder_metrics_matrix(time_encoder: str, models: List[str], datasets: List[str], 
                                     output_dir: str) -> pd.DataFrame:
    """
    Create a metrics matrix for a specific time encoder showing models vs datasets with actual metrics values.
    
    Args:
        time_encoder: Specific time encoder to analyze
        models: List of models to check
        datasets: List of datasets to check  
        output_dir: Directory to save the matrix
    
    Returns:
        DataFrame with models as rows, datasets as columns, metrics as values
    """
    
    print(f"\n🎯 Creating metrics matrix for time encoder: {time_encoder}")
    print(f"   Models: {len(models)}")
    print(f"   Datasets: {len(datasets)}")
    
    # Create a more complex data structure to store metrics
    metrics_data = {}
    
    for model in sorted(models):
        metrics_data[model] = {}
        for dataset in sorted(datasets):
            metrics = get_metrics_from_result_file(model, dataset, time_encoder)
            if metrics:
                # Convert metrics to a more flat structure for CSV
                flattened_metrics = {}
                for metric_type, metric_values in metrics.items():
                    if isinstance(metric_values, dict):
                        for metric_name, metric_value in metric_values.items():
                            flattened_metrics[f"{metric_type}_{metric_name}"] = metric_value
                    else:
                        flattened_metrics[metric_type] = metric_values
                metrics_data[model][dataset] = flattened_metrics
            else:
                metrics_data[model][dataset] = None
    
    # Create separate DataFrames for different metrics
    metric_types = set()
    for model_data in metrics_data.values():
        for dataset_data in model_data.values():
            if dataset_data:
                metric_types.update(dataset_data.keys())
    
    # Create a DataFrame for each metric type
    os.makedirs(output_dir, exist_ok=True)
    
    for metric_type in sorted(metric_types):
        df = pd.DataFrame(index=sorted(models), columns=sorted(datasets))
        
        for model in sorted(models):
            for dataset in sorted(datasets):
                if metrics_data[model][dataset] and metric_type in metrics_data[model][dataset]:
                    df.loc[model, dataset] = metrics_data[model][dataset][metric_type]
                else:
                    df.loc[model, dataset] = None
        
        # Save individual metric DataFrame
        csv_filename = f"time_encoder_{time_encoder}_{metric_type}_metrics.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path)
        print(f"   💾 Saved {metric_type} metrics to: {csv_path}")
    
    # Also create a comprehensive JSON file with all metrics
    json_filename = f"time_encoder_{time_encoder}_all_metrics.json"
    json_path = os.path.join(output_dir, json_filename)
    with open(json_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    print(f"   💾 Saved all metrics to: {json_path}")
    
    # Create a summary completion matrix (same as before but for reference)
    completion_matrix = pd.DataFrame(0, index=sorted(models), columns=sorted(datasets))
    completed_combinations = 0
    total_combinations = len(models) * len(datasets)
    
    for model in sorted(models):
        for dataset in sorted(datasets):
            if metrics_data[model][dataset] is not None:
                completion_matrix.loc[model, dataset] = 1
                completed_combinations += 1
    
    completion_csv_path = os.path.join(output_dir, f"time_encoder_{time_encoder}_completion_summary.csv")
    completion_matrix.to_csv(completion_csv_path)
    
    completion_rate = completed_combinations / total_combinations * 100 if total_combinations > 0 else 0
    print(f"   ✅ Completed: {completed_combinations}/{total_combinations} ({completion_rate:.1f}%)")
    
    return completion_matrix


def create_time_encoder_completion_matrix(time_encoder: str, models: List[str], datasets: List[str], 
                                        neg_strategies: List[str], check_func, output_dir: str) -> pd.DataFrame:
    """
    Create a completion matrix for a specific time encoder showing models vs datasets.
    
    Args:
        time_encoder: Specific time encoder to analyze
        models: List of models to check
        datasets: List of datasets to check  
        neg_strategies: List of negative strategies (used for checking but not in matrix)
        check_func: Function to check completion status
        output_dir: Directory to save the matrix
    
    Returns:
        DataFrame with models as rows, datasets as columns, completion status as values
    """
    
    print(f"\n🎯 Creating completion matrix for time encoder: {time_encoder}")
    print(f"   Models: {len(models)}")
    print(f"   Datasets: {len(datasets)}")
    
    # Initialize matrix with 0s (not completed)
    matrix = pd.DataFrame(0, index=sorted(models), columns=sorted(datasets))
    
    # Check each model-dataset combination for the specific time encoder
    total_combinations = len(models) * len(datasets)
    completed_combinations = 0
    
    for model in models:
        for dataset in datasets:
            # Check if any neg_strategy has completed results for this combination
            completed = False
            for neg_strategy in neg_strategies:
                is_completed, files, info = check_func(model, dataset, time_encoder, neg_strategy)
                if is_completed:
                    completed = True
                    break
            
            if completed:
                matrix.loc[model, dataset] = 1
                completed_combinations += 1
    
    # Calculate completion statistics
    completion_rate = completed_combinations / total_combinations * 100 if total_combinations > 0 else 0
    
    # Save to CSV
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = f"time_encoder_{time_encoder}_completion.csv"
    csv_path = os.path.join(output_dir, csv_filename)
    matrix.to_csv(csv_path)
    
    print(f"   💾 Saved to: {csv_path}")
    print(f"   ✅ Completed: {completed_combinations}/{total_combinations} ({completion_rate:.1f}%)")
    
    # Print summary by model
    print(f"\n📊 Completion by Model for {time_encoder}:")
    for model in sorted(models):
        model_completed = matrix.loc[model].sum()
        model_total = len(datasets)
        model_rate = model_completed / model_total * 100 if model_total > 0 else 0
        print(f"   {model:12s}: {model_completed:3d}/{model_total:3d} ({model_rate:5.1f}%)")
    
    # Print summary by dataset
    print(f"\n📊 Completion by Dataset for {time_encoder}:")
    for dataset in sorted(datasets):
        dataset_completed = matrix[dataset].sum()
        dataset_total = len(models)
        dataset_rate = dataset_completed / dataset_total * 100 if dataset_total > 0 else 0
        print(f"   {dataset:12s}: {dataset_completed:3d}/{dataset_total:3d} ({dataset_rate:5.1f}%)")
    
    return matrix


def generate_summary_stats(combinations: List, check_func, output_dir: str) -> Dict:
    """Generate detailed summary statistics"""
    
    print(f"\n📈 Generating summary statistics...")
    
    total_combinations = len(combinations)
    completed_combinations = 0
    failed_combinations = []
    completed_combinations_list = []
    
    model_stats = {}
    dataset_stats = {}
    encoder_stats = {}
    
    for combo in combinations:
        model, dataset, encoder, neg_strategy = combo
        
        completed, files, info = check_func(model, dataset, encoder, neg_strategy)
        
        if completed:
            completed_combinations += 1
            completed_combinations_list.append(combo)
        else:
            failed_combinations.append(combo)
        
        # Update stats by category
        if model not in model_stats:
            model_stats[model] = {'total': 0, 'completed': 0}
        model_stats[model]['total'] += 1
        if completed:
            model_stats[model]['completed'] += 1
            
        if dataset not in dataset_stats:
            dataset_stats[dataset] = {'total': 0, 'completed': 0}
        dataset_stats[dataset]['total'] += 1
        if completed:
            dataset_stats[dataset]['completed'] += 1
            
        if encoder not in encoder_stats:
            encoder_stats[encoder] = {'total': 0, 'completed': 0}
        encoder_stats[encoder]['total'] += 1
        if completed:
            encoder_stats[encoder]['completed'] += 1
    
    # Calculate completion rates
    for stats in [model_stats, dataset_stats, encoder_stats]:
        for key in stats:
            total = stats[key]['total']
            completed = stats[key]['completed']
            stats[key]['completion_rate'] = completed / total if total > 0 else 0
    
    summary = {
        'total_combinations': total_combinations,
        'completed_combinations_count': completed_combinations,
        'completion_rate': completed_combinations / total_combinations if total_combinations > 0 else 0,
        'failed_combinations_count': len(failed_combinations),
        'model_stats': model_stats,
        'dataset_stats': dataset_stats,
        'encoder_stats': encoder_stats,
        'failed_combinations': failed_combinations,
        'completed_combinations': completed_combinations_list
    }
    
    # Save detailed stats to JSON
    os.makedirs(output_dir, exist_ok=True)
    stats_path = os.path.join(output_dir, "completion_summary.json")
    with open(stats_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Save stats tables to CSV
    pd.DataFrame(model_stats).T.to_csv(os.path.join(output_dir, "model_completion_stats.csv"))
    pd.DataFrame(dataset_stats).T.to_csv(os.path.join(output_dir, "dataset_completion_stats.csv"))
    pd.DataFrame(encoder_stats).T.to_csv(os.path.join(output_dir, "encoder_completion_stats.csv"))
    
    return summary


def print_summary_report(summary: Dict):
    """Print a nice summary report"""
    
    print(f"\n{'='*80}")
    print(f"📋 EXPERIMENT COMPLETION SUMMARY REPORT")
    print(f"{'='*80}")
    
    print(f"\n🎯 OVERALL COMPLETION:")
    print(f"   Total Combinations: {summary['total_combinations']:,}")
    print(f"   Completed: {summary['completed_combinations_count']:,}")
    print(f"   Failed/Missing: {summary['failed_combinations_count']:,}")
    print(f"   Completion Rate: {summary['completion_rate']*100:.1f}%")
    
    print(f"\n🏷️  BY MODEL:")
    for model, stats in sorted(summary['model_stats'].items()):
        rate = stats['completion_rate'] * 100
        print(f"   {model:12s}: {stats['completed']:3d}/{stats['total']:3d} ({rate:5.1f}%)")
    
    print(f"\n📊 BY DATASET:")
    for dataset, stats in sorted(summary['dataset_stats'].items()):
        rate = stats['completion_rate'] * 100
        print(f"   {dataset:12s}: {stats['completed']:3d}/{stats['total']:3d} ({rate:5.1f}%)")
    
    print(f"\n🔧 BY ENCODER:")
    for encoder, stats in sorted(summary['encoder_stats'].items()):
        rate = stats['completion_rate'] * 100
        print(f"   {encoder:20s}: {stats['completed']:3d}/{stats['total']:3d} ({rate:5.1f}%)")
    
    # Show some failed combinations
    if summary['failed_combinations']:
        print(f"\n❌ SAMPLE MISSING COMBINATIONS (first 10):")
        for i, combo in enumerate(summary['failed_combinations'][:10]):
            model, dataset, encoder, neg_strategy = combo
            print(f"   {i+1:2d}. {model} + {dataset} + {encoder} + {neg_strategy}")
        
        if len(summary['failed_combinations']) > 10:
            print(f"   ... and {len(summary['failed_combinations'])-10} more")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Check completion status of all experiment combinations')
    parser.add_argument('--quick', action='store_true',
                        help='Check quick test combinations only')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS,
                        help='Check specific models only')
    parser.add_argument('--datasets', nargs='+', choices=ALL_DATASETS,
                        help='Check specific datasets only')
    parser.add_argument('--encoders', nargs='+', choices=ALL_TIME_ENCODERS,
                        help='Check specific encoders only')
    parser.add_argument('--time_encoder', type=str, choices=ALL_TIME_ENCODERS,
                        help='Generate completion matrix for specific time encoder (models vs datasets)')
    parser.add_argument('--time_encoder_metrics', type=str, choices=ALL_TIME_ENCODERS,
                        help='Generate metrics matrix for specific time encoder with actual metric values')
    parser.add_argument('--comprehensive_table', action='store_true',
                        help='Generate comprehensive long-format table with all strategies')
    parser.add_argument('--neg_strategies', nargs='+', choices=ALL_NEG_STRATEGIES,
                        help='Check specific negative sampling strategies only')
    parser.add_argument('--output_dir', type=str, default='completion_analysis',
                        help='Directory to save CSV reports (default: completion_analysis)')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output for each combination')
    
    return parser.parse_args()


def get_combinations(args):
    """Get combinations to check based on arguments"""
    if args.quick:
        models = args.models or QUICK_MODELS
        datasets = args.datasets or QUICK_DATASETS
        encoders = args.encoders or QUICK_ENCODERS
        neg_strategies = args.neg_strategies or ALL_NEG_STRATEGIES
    else:
        models = args.models or ALL_MODELS
        datasets = args.datasets or ALL_DATASETS
        encoders = args.encoders or ALL_TIME_ENCODERS
        neg_strategies = args.neg_strategies or ALL_NEG_STRATEGIES
    
    return models, datasets, encoders, neg_strategies
if __name__ == "__main__":
    args = parse_arguments()
    
    print("🔍 Experiment Completion Analysis")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f" Output directory: {args.output_dir}")
    
    models, datasets, encoders, neg_strategies = get_combinations(args)
    
    print(f"\nAnalysis Configuration:")
    print(f"  Models: {models}")
    print(f"  Datasets: {datasets}")
    print(f"  Encoders: {encoders}")
    print(f"  Negative Strategies: {neg_strategies}")
    print(f"  Quick Mode: {args.quick}")
    
    # Generate all combinations
    all_combinations = list(itertools.product(models, datasets, encoders, neg_strategies))
    total_combinations = len(all_combinations)
    
    print(f"\nTotal Combinations to Check: {total_combinations:,}")
    
    # Check if we're doing time_encoder specific analysis
    if args.time_encoder:
        print(f"\n🎯 Time Encoder Specific Analysis: {args.time_encoder}")
        
        # Generate time-encoder specific completion matrix
        time_encoder_matrix = create_time_encoder_completion_matrix(
            time_encoder=args.time_encoder,
            models=models,
            datasets=datasets,
            neg_strategies=neg_strategies,
            check_func=check_evaluation_results_exist,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Time encoder analysis completed!")
        print(f"📁 Generated file: {args.output_dir}/time_encoder_{args.time_encoder}_completion.csv")
        sys.exit(0)
    
    # Check if we're doing time_encoder metrics analysis
    if args.time_encoder_metrics:
        print(f"\n🎯 Time Encoder Metrics Analysis: {args.time_encoder_metrics}")
        
        # Generate time-encoder specific metrics matrix
        metrics_matrix = create_time_encoder_metrics_matrix(
            time_encoder=args.time_encoder_metrics,
            models=models,
            datasets=datasets,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Time encoder metrics analysis completed!")
        print(f"📁 Generated files in: {args.output_dir}/")
        print(f"   - Individual metric CSV files")
        print(f"   - time_encoder_{args.time_encoder_metrics}_all_metrics.json")
        print(f"   - time_encoder_{args.time_encoder_metrics}_completion_summary.csv")
        sys.exit(0)
    
    # Check if we're doing comprehensive table generation
    if args.comprehensive_table:
        print(f"\n🎯 Comprehensive Results Table Generation")
        print(f"   This will include ALL negative sampling strategies in long format")
        
        # Generate comprehensive table
        comprehensive_df = create_comprehensive_results_table(
            models=models,
            datasets=datasets,
            encoders=encoders,
            neg_strategies=neg_strategies,
            output_dir=args.output_dir
        )
        
        print(f"\n✅ Comprehensive table generation completed!")
        print(f"📁 Generated files:")
        print(f"   - {args.output_dir}/comprehensive_results_table.csv (long format)")
        print(f"   - {args.output_dir}/test_ap_pivot_table.csv (pivot view)")
        print(f"\n💡 Usage tips:")
        print(f"   - Use Excel/Pandas to filter by model, dataset, encoder, or strategy")
        print(f"   - Compare strategies side-by-side for same model+dataset+encoder")
        print(f"   - Pivot table shows Test AP across all combinations")
        sys.exit(0)
    
    if args.verbose:
        print(f"\n🔍 Checking each combination...")
    
    # Check completion status
    print(f"\n📊 Analyzing completion status...")
    
    # Generate completion matrices
    matrices = {}
    
    # For matrix generation, we'll check unique model-dataset-encoder combinations
    # (ignoring neg_strategy since one result file covers all neg strategies)
    unique_combinations = list(itertools.product(models, datasets, encoders, ['combined']))
    
    matrices['model_dataset'] = create_completion_matrix(
        unique_combinations, 
        check_evaluation_results_exist, 
        "Model-Dataset", 
        args.output_dir
    )
    
    matrices['model_encoder'] = create_completion_matrix(
        unique_combinations,
        check_evaluation_results_exist,
        "Model-Encoder", 
        args.output_dir
    )
    
    matrices['dataset_encoder'] = create_completion_matrix(
        unique_combinations,
        check_evaluation_results_exist,
        "Dataset-Encoder",
        args.output_dir
    )
    
    # Generate summary statistics
    summary = generate_summary_stats(unique_combinations, check_evaluation_results_exist, args.output_dir)
    
    # Print summary report
    print_summary_report(summary)
    
    print(f"\n📁 Generated Files:")
    print(f"   📊 CSV Matrices:")
    print(f"      - {args.output_dir}/model_dataset_completion.csv")
    print(f"      - {args.output_dir}/model_encoder_completion.csv") 
    print(f"      - {args.output_dir}/dataset_encoder_completion.csv")
    print(f"   📈 Statistics:")
    print(f"      - {args.output_dir}/completion_summary.json")
    print(f"      - {args.output_dir}/model_completion_stats.csv")
    print(f"      - {args.output_dir}/dataset_completion_stats.csv")
    print(f"      - {args.output_dir}/encoder_completion_stats.csv")
    
    print(f"\n✅ Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Optional: Show some specific missing combinations if verbose
    if args.verbose and summary['failed_combinations']:
        print(f"\n🔍 Detailed Missing Combinations:")
        for i, combo in enumerate(summary['failed_combinations']):
            model, dataset, encoder, _ = combo
            completed, files, info = check_evaluation_results_exist(model, dataset, encoder)
            print(f"   {i+1:3d}. {model:12s} + {dataset:12s} + {encoder:20s}")
            if args.verbose:
                print(f"        Pattern: {info['pattern']}")
                print(f"        Found files: {info['count']}")
                if info['count'] > 0:
                    print(f"        Valid results: {info['valid_count']}")
                print()