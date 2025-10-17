#!/usr/bin/env python3
"""
Combine HPC Results Script
==========================

This script combines experimental results from multiple HPC clusters/runs into unified files.
It processes CSV metrics files and JSON all_metrics files, merging data from different sources
while preserving the original structure and converting CSV data to long format.

The output includes:
- Long-format CSV files with columns: model, dataset, time_encoder, training_method, 
  neg_sampling, evaluation_metric, value
- Combined JSON files for all_metrics
- Global long-format CSV combining all time encoders
- Summary JSON with processing statistics

Usage:
    python combine_hpc_results.py [--input_dir PATH] [--output_dir PATH]
    
Arguments:
    --input_dir: Directory containing results_* folders (default: ./results_all_cal/results_extracted)
    --output_dir: Output directory for combined results (default: ./combined_results)
"""

import os
import sys
import json
import argparse
import pandas as pd
from pathlib import Path
from collections import defaultdict
import numpy as np

def find_hpc_folders(input_dir):
    """Find all HPC result folders (results_*, results_csvhpc*, etc.)"""
    hpc_folders = []
    input_path = Path(input_dir)
    
    for item in input_path.iterdir():
        if item.is_dir() and (item.name.startswith('results_csv') or item.name.startswith('results_')):
            # Check if it has the expected structure
            csv_path = item / 'results_csv'
            if csv_path.exists():
                hpc_folders.append(item)
                print(f"Found HPC folder: {item.name}")
    
    return sorted(hpc_folders)

def find_time_encoder_folders(hpc_folder):
    """Find all time encoder result folders within an HPC folder"""
    csv_path = hpc_folder / 'results_csv'
    encoder_folders = []
    encoder_mapping = get_encoder_name_mapping()
    
    if csv_path.exists():
        for item in csv_path.iterdir():
            if item.is_dir() and item.name.startswith('results_'):
                # Use mapping to get the actual encoder name
                actual_encoder_name = encoder_mapping.get(item.name, item.name.replace('results_', ''))
                encoder_folders.append((actual_encoder_name, item))
    
    return encoder_folders

def get_encoder_name_mapping():
    """Map folder names to actual encoder names found in files"""
    return {
        'results_t2v': 'time2vec',
        'results_kmammote': 'kan_mammote_dual_kmote',
        'results_lete': 'lete',
        'results_mercer': 'mercer'
    }

def get_file_patterns():
    """Define the file patterns we want to combine"""
    return [
        'completion_summary.csv',
        'test_metrics_average_precision_metrics.csv',
        'test_metrics_roc_auc_metrics.csv', 
        'new_node_test_metrics_average_precision_metrics.csv',
        'new_node_test_metrics_roc_auc_metrics.csv',
        'validate_metrics_average_precision_metrics.csv',
        'validate_metrics_roc_auc_metrics.csv',
        'new_node_validate_metrics_average_precision_metrics.csv',
        'new_node_validate_metrics_roc_auc_metrics.csv',
        'all_metrics.json'
    ]

def combine_csv_files(csv_files, metric_name, time_encoder):
    """Combine multiple CSV files and convert to long format"""
    if not csv_files:
        return None
    
    print(f"  Combining {len(csv_files)} CSV files for {metric_name}")
    
    # Read all CSV files and combine data
    all_data = {}  # {(model, dataset): value}
    source_info = []
    
    for file_path, source in csv_files:
        try:
            df = pd.read_csv(file_path, index_col=0)
            source_info.append(source)
            
            # Extract data from this source
            for model in df.index:
                for dataset in df.columns:
                    key = (model, dataset)
                    value = df.loc[model, dataset]
                    
                    # Only add if we don't have this combination yet and value is not NaN/empty
                    if key not in all_data and not pd.isna(value) and value != '':
                        all_data[key] = value
                    
        except Exception as e:
            print(f"    Warning: Could not read {file_path}: {e}")
            continue
    
    if not all_data:
        return None
    
    # Convert to long format
    long_data = []
    
    # Determine training method and evaluation metric from filename
    if 'new_node' in metric_name:
        training_method = 'inductive'
    else:
        training_method = 'transductive'
    
    if 'average_precision' in metric_name:
        evaluation_metric = 'average_precision'
    elif 'roc_auc' in metric_name:
        evaluation_metric = 'roc_auc'
    else:
        evaluation_metric = 'unknown'
    
    # Convert each data point to long format
    for (model, dataset), value in all_data.items():
        long_data.append({
            'model': model,
            'dataset': dataset,
            'time_encoder': time_encoder,
            'training_method': training_method,
            'neg_sampling': 'random',  # As specified
            'evaluation_metric': evaluation_metric,
            'value': value
        })
    
    if long_data:
        return pd.DataFrame(long_data)
    else:
        return None

def combine_json_files(json_files):
    """Combine multiple JSON all_metrics files"""
    if not json_files:
        return None
    
    print(f"  Combining {len(json_files)} JSON files")
    
    combined_data = {}
    
    for file_path, source in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            for model, model_data in data.items():
                if model not in combined_data:
                    combined_data[model] = {}
                
                for dataset, dataset_metrics in model_data.items():
                    if dataset not in combined_data[model]:
                        combined_data[model][dataset] = dataset_metrics
                    elif dataset_metrics is not None and combined_data[model][dataset] is None:
                        # Fill missing data
                        combined_data[model][dataset] = dataset_metrics
                        print(f"    Filled {model}-{dataset} from {source}")
                        
        except Exception as e:
            print(f"    Warning: Could not read {file_path}: {e}")
            continue
    
    return combined_data

def combine_results_for_encoder(encoder_name, encoder_folders, output_dir):
    """Combine all results for a specific time encoder"""
    print(f"\n🔄 Processing time encoder: {encoder_name}")
    
    # Create output directory for this encoder
    encoder_output_dir = output_dir / f"results_{encoder_name}"
    encoder_output_dir.mkdir(parents=True, exist_ok=True)
    
    file_patterns = get_file_patterns()
    
    # Group files by metric type
    files_by_metric = defaultdict(list)
    
    for source_name, encoder_folder in encoder_folders:
        for pattern in file_patterns:
            # Extract metric name from pattern
            if pattern == 'all_metrics.json':
                metric_name = 'all_metrics'
                file_name = f"time_encoder_{encoder_name}_{pattern}"
            else:
                metric_name = pattern.replace('.csv', '')
                file_name = f"time_encoder_{encoder_name}_{pattern}"
            
            file_path = encoder_folder / file_name
            
            if file_path.exists():
                files_by_metric[metric_name].append((file_path, source_name))
    
    # Process each metric type and collect all long-format data
    all_long_data = []
    results_summary = {
        'encoder': encoder_name,
        'total_sources': len(encoder_folders),
        'files_combined': {},
        'files_created': []
    }
    
    for metric_name, files in files_by_metric.items():
        if not files:
            continue
            
        print(f"  📊 Processing {metric_name} ({len(files)} files)")
        
        if metric_name == 'all_metrics':
            # Handle JSON files (keep existing functionality)
            combined_data = combine_json_files(files)
            if combined_data:
                output_file = encoder_output_dir / f"time_encoder_{encoder_name}_all_metrics_combined.json"
                with open(output_file, 'w') as f:
                    json.dump(combined_data, f, indent=2)
                results_summary['files_created'].append(output_file.name)
                print(f"    ✅ Saved: {output_file.name}")
        elif metric_name != 'completion_summary':  # Skip completion summary for long format
            # Handle CSV files - convert to long format
            combined_df = combine_csv_files(files, metric_name, encoder_name)
            if combined_df is not None and not combined_df.empty:
                all_long_data.append(combined_df)
                print(f"    ✅ Processed {len(combined_df)} records from {metric_name}")
        
        results_summary['files_combined'][metric_name] = len(files)
    
    # Combine all long-format data into a single DataFrame
    if all_long_data:
        final_long_df = pd.concat(all_long_data, ignore_index=True)
        
        # Sort by model, dataset, training_method, evaluation_metric for consistency
        final_long_df = final_long_df.sort_values([
            'model', 'dataset', 'training_method', 'evaluation_metric'
        ]).reset_index(drop=True)
        
        # Save the combined long-format CSV
        output_file = encoder_output_dir / f"time_encoder_{encoder_name}_all_metrics_long_format.csv"
        final_long_df.to_csv(output_file, index=False)
        results_summary['files_created'].append(output_file.name)
        
        print(f"    ✅ Saved long-format data: {output_file.name}")
        print(f"    📊 Total records: {len(final_long_df)}")
        print(f"    📊 Unique combinations: {final_long_df[['model', 'dataset']].drop_duplicates().shape[0]}")
    
    return results_summary

def get_experiment_definitions():
    """Define all possible experiment combinations"""
    datasets = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci', 
                'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']
    models = ['JODIE', 'TGAT', 'TGN', 'TCL', 'DyGFormer', 'DyGMamba']
    time_encoders = ['lete', 'time2vec', 'kan_mammote_dual_kmote', 'mercer']
    training_methods = ['transductive', 'inductive']
    evaluation_metrics = ['average_precision', 'roc_auc']
    
    return datasets, models, time_encoders, training_methods, evaluation_metrics

def analyze_completion_status(output_dir):
    """Analyze which experiment combinations are completed and which are missing"""
    print(f"\n🔍 Analyzing completion status...")
    
    # Load the global combined data
    global_long_file = output_dir / 'all_encoders_combined_long_format.csv'
    
    if not global_long_file.exists():
        print(f"❌ Global long-format file not found: {global_long_file}")
        return None
    
    try:
        completed_df = pd.read_csv(global_long_file)
        print(f"📊 Loaded {len(completed_df)} completed experiments")
    except Exception as e:
        print(f"❌ Error loading global file: {e}")
        return None
    
    # Get all possible combinations
    datasets, models, time_encoders, training_methods, evaluation_metrics = get_experiment_definitions()
    
    # Create all possible combinations
    all_combinations = []
    for dataset in datasets:
        for model in models:
            for time_encoder in time_encoders:
                for training_method in training_methods:
                    for evaluation_metric in evaluation_metrics:
                        all_combinations.append({
                            'dataset': dataset,
                            'model': model,
                            'time_encoder': time_encoder,
                            'training_method': training_method,
                            'evaluation_metric': evaluation_metric,
                            'neg_sampling': 'random'  # Default value
                        })
    
    print(f"📊 Total possible experiment combinations: {len(all_combinations)}")
    
    # Create DataFrame with all combinations
    all_exp_df = pd.DataFrame(all_combinations)
    
    # Create a set of completed combinations for fast lookup
    completed_combinations = set()
    for _, row in completed_df.iterrows():
        combo = (row['dataset'], row['model'], row['time_encoder'], 
                row['training_method'], row['evaluation_metric'])
        completed_combinations.add(combo)
    
    # Mark completed status
    all_exp_df['completed'] = all_exp_df.apply(
        lambda row: (row['dataset'], row['model'], row['time_encoder'], 
                    row['training_method'], row['evaluation_metric']) in completed_combinations,
        axis=1
    )
    
    # Add value column (NaN for incomplete, actual value for completed)
    all_exp_df['value'] = np.nan
    for _, row in completed_df.iterrows():
        mask = (
            (all_exp_df['dataset'] == row['dataset']) &
            (all_exp_df['model'] == row['model']) &
            (all_exp_df['time_encoder'] == row['time_encoder']) &
            (all_exp_df['training_method'] == row['training_method']) &
            (all_exp_df['evaluation_metric'] == row['evaluation_metric'])
        )
        all_exp_df.loc[mask, 'value'] = row['value']
    
    # Sort for better readability
    all_exp_df = all_exp_df.sort_values([
        'time_encoder', 'model', 'dataset', 'training_method', 'evaluation_metric'
    ]).reset_index(drop=True)
    
    # Save completion status file
    completion_file = output_dir / 'experiment_completion_status.csv'
    all_exp_df.to_csv(completion_file, index=False)
    
    print(f"💾 Completion status saved to: {completion_file}")
    
    return all_exp_df, completed_df

def generate_completion_report(all_exp_df, completed_df, output_dir):
    """Generate detailed completion statistics and reports"""
    print(f"\n📊 Generating completion report...")
    
    datasets, models, time_encoders, training_methods, evaluation_metrics = get_experiment_definitions()
    
    # Overall completion statistics
    total_experiments = len(all_exp_df)
    completed_experiments = all_exp_df['completed'].sum()
    completion_rate = (completed_experiments / total_experiments) * 100
    
    print(f"🎯 Overall Completion: {completed_experiments}/{total_experiments} ({completion_rate:.1f}%)")
    
    # Create detailed completion report
    completion_report = {
        'overall_statistics': {
            'total_possible_experiments': total_experiments,
            'completed_experiments': int(completed_experiments),
            'missing_experiments': int(total_experiments - completed_experiments),
            'completion_percentage': round(completion_rate, 2)
        },
        'completion_by_category': {}
    }
    
    # Completion by time encoder
    encoder_stats = []
    for encoder in time_encoders:
        encoder_mask = all_exp_df['time_encoder'] == encoder
        encoder_total = encoder_mask.sum()
        encoder_completed = all_exp_df[encoder_mask]['completed'].sum()
        encoder_rate = (encoder_completed / encoder_total) * 100 if encoder_total > 0 else 0
        
        encoder_stats.append({
            'time_encoder': encoder,
            'total': int(encoder_total),
            'completed': int(encoder_completed),
            'missing': int(encoder_total - encoder_completed),
            'completion_rate': round(encoder_rate, 2)
        })
        
        print(f"  📡 {encoder}: {encoder_completed}/{encoder_total} ({encoder_rate:.1f}%)")
    
    completion_report['completion_by_category']['time_encoders'] = encoder_stats
    
    # Completion by model
    model_stats = []
    for model in models:
        model_mask = all_exp_df['model'] == model
        model_total = model_mask.sum()
        model_completed = all_exp_df[model_mask]['completed'].sum()
        model_rate = (model_completed / model_total) * 100 if model_total > 0 else 0
        
        model_stats.append({
            'model': model,
            'total': int(model_total),
            'completed': int(model_completed),
            'missing': int(model_total - model_completed),
            'completion_rate': round(model_rate, 2)
        })
    
    completion_report['completion_by_category']['models'] = model_stats
    
    # Completion by dataset
    dataset_stats = []
    for dataset in datasets:
        dataset_mask = all_exp_df['dataset'] == dataset
        dataset_total = dataset_mask.sum()
        dataset_completed = all_exp_df[dataset_mask]['completed'].sum()
        dataset_rate = (dataset_completed / dataset_total) * 100 if dataset_total > 0 else 0
        
        dataset_stats.append({
            'dataset': dataset,
            'total': int(dataset_total),
            'completed': int(dataset_completed),
            'missing': int(dataset_total - dataset_completed),
            'completion_rate': round(dataset_rate, 2)
        })
    
    completion_report['completion_by_category']['datasets'] = dataset_stats
    
    # Find missing combinations
    missing_combinations = all_exp_df[~all_exp_df['completed']].copy()
    
    # Save missing combinations separately (full detail)
    missing_file = output_dir / 'missing_experiments_detailed.csv'
    missing_combinations[['dataset', 'model', 'time_encoder', 'training_method', 'evaluation_metric']].to_csv(
        missing_file, index=False
    )
    
    print(f"📋 Detailed missing experiments saved to: {missing_file}")
    print(f"❌ Total missing experiments: {len(missing_combinations)}")
    
    # Create simplified missing experiments (unique model-dataset-encoder combinations only)
    simple_missing = missing_combinations[['dataset', 'model', 'time_encoder']].drop_duplicates()
    simple_missing_file = output_dir / 'missing_experiments.csv'
    simple_missing.to_csv(simple_missing_file, index=False)
    
    print(f"🎯 Simple missing combinations (model-dataset-encoder): {len(simple_missing)}")
    print(f"💾 Simple missing saved to: {simple_missing_file}")
    
    # Create priority missing experiments (same as simple for now)
    priority_file = output_dir / 'priority_missing_experiments.csv'
    simple_missing.to_csv(priority_file, index=False)
    
    # Save completion report
    report_file = output_dir / 'completion_report.json'
    with open(report_file, 'w') as f:
        json.dump(completion_report, f, indent=2)
    
    print(f"📊 Detailed completion report saved to: {report_file}")
    
    # Print summary table
    print(f"\n📋 Completion Summary by Time Encoder:")
    print(f"{'Encoder':<25} {'Completed':<10} {'Total':<8} {'Rate':<8}")
    print("-" * 55)
    for stat in encoder_stats:
        print(f"{stat['time_encoder']:<25} {stat['completed']:<10} {stat['total']:<8} {stat['completion_rate']:.1f}%")
    
    print(f"\n📋 Top 5 Models by Completion:")
    sorted_models = sorted(model_stats, key=lambda x: x['completion_rate'], reverse=True)[:5]
    for stat in sorted_models:
        print(f"  {stat['model']}: {stat['completed']}/{stat['total']} ({stat['completion_rate']:.1f}%)")
    
    print(f"\n📋 Top 5 Datasets by Completion:")
    sorted_datasets = sorted(dataset_stats, key=lambda x: x['completion_rate'], reverse=True)[:5]
    for stat in sorted_datasets:
        print(f"  {stat['dataset']}: {stat['completed']}/{stat['total']} ({stat['completion_rate']:.1f}%)")
    
    return completion_report

def create_global_summary(all_summaries, output_dir):
    """Create a global summary of the combination process and combine all long-format data"""
    summary_data = {
        'combination_summary': {
            'total_encoders_processed': len(all_summaries),
            'encoders': {}
        }
    }
    
    total_files = 0
    all_global_data = []
    
    for summary in all_summaries:
        encoder = summary['encoder']
        summary_data['combination_summary']['encoders'][encoder] = {
            'sources_found': summary['total_sources'],
            'metrics_combined': len(summary['files_combined']),
            'files_created': len(summary['files_created']),
            'files_by_metric': summary['files_combined']
        }
        total_files += len(summary['files_created'])
        
        # Load the long-format data for this encoder
        encoder_long_file = output_dir / f"results_{encoder}" / f"time_encoder_{encoder}_all_metrics_long_format.csv"
        if encoder_long_file.exists():
            try:
                encoder_df = pd.read_csv(encoder_long_file)
                all_global_data.append(encoder_df)
                print(f"  📊 Loaded {len(encoder_df)} records from {encoder}")
            except Exception as e:
                print(f"    Warning: Could not load {encoder_long_file}: {e}")
    
    summary_data['combination_summary']['total_files_created'] = total_files
    
    # Create global long-format CSV combining all encoders
    if all_global_data:
        global_long_df = pd.concat(all_global_data, ignore_index=True)
        
        # Sort for consistency
        global_long_df = global_long_df.sort_values([
            'time_encoder', 'model', 'dataset', 'training_method', 'evaluation_metric'
        ]).reset_index(drop=True)
        
        # Save global long-format file
        global_long_file = output_dir / 'all_encoders_combined_long_format.csv'
        global_long_df.to_csv(global_long_file, index=False)
        
        print(f"\n📊 Global long-format data saved to: {global_long_file}")
        print(f"📊 Total records: {len(global_long_df)}")
        print(f"📊 Unique time encoders: {global_long_df['time_encoder'].nunique()}")
        print(f"📊 Unique models: {global_long_df['model'].nunique()}")
        print(f"📊 Unique datasets: {global_long_df['dataset'].nunique()}")
        
        # Add global stats to summary
        summary_data['combination_summary']['global_long_format'] = {
            'total_records': len(global_long_df),
            'unique_encoders': global_long_df['time_encoder'].nunique(),
            'unique_models': global_long_df['model'].nunique(),
            'unique_datasets': global_long_df['dataset'].nunique(),
            'file_path': str(global_long_file)
        }
    
    # Save summary
    summary_file = output_dir / 'combination_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary_data, f, indent=2)
    
    print(f"\n📋 Global summary saved to: {summary_file}")
    print(f"📊 Total files created: {total_files}")
    
    return summary_data

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Combine HPC experimental results')
    parser.add_argument('--input_dir', type=str, 
                       default='./results_all_cal',
                       help='Directory containing results_* folders')
    parser.add_argument('--output_dir', type=str, 
                       default='./combined_results',
                       help='Output directory for combined results')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    print(f"🚀 HPC Results Combination Script")
    print(f"📁 Input directory: {input_dir}")
    print(f"📁 Output directory: {output_dir}")
    print("="*60)
    
    # Check input directory exists
    if not input_dir.exists():
        print(f"❌ Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all HPC folders
    hpc_folders = find_hpc_folders(input_dir)
    if not hpc_folders:
        print(f"❌ No HPC result folders found in {input_dir}")
        sys.exit(1)
    
    print(f"🔍 Found {len(hpc_folders)} HPC result folders")
    
    # Collect all encoder folders across HPC runs
    encoder_data = defaultdict(list)  # encoder_name -> [(source_name, folder_path), ...]
    
    for hpc_folder in hpc_folders:
        source_name = hpc_folder.name
        encoder_folders = find_time_encoder_folders(hpc_folder)
        
        for encoder_name, encoder_folder in encoder_folders:
            encoder_data[encoder_name].append((source_name, encoder_folder))
    
    if not encoder_data:
        print("❌ No time encoder folders found")
        sys.exit(1)
    
    print(f"🔍 Found {len(encoder_data)} unique time encoders:")
    for encoder_name, sources in encoder_data.items():
        print(f"  - {encoder_name}: {len(sources)} sources")
    
    # Process each encoder
    all_summaries = []
    for encoder_name, encoder_folders in encoder_data.items():
        summary = combine_results_for_encoder(encoder_name, encoder_folders, output_dir)
        all_summaries.append(summary)
    
    # Create global summary
    global_summary = create_global_summary(all_summaries, output_dir)
    
    # Analyze completion status
    completion_results = analyze_completion_status(output_dir)
    if completion_results:
        all_exp_df, completed_df = completion_results
        completion_report = generate_completion_report(all_exp_df, completed_df, output_dir)
    
    print(f"\n✅ Combination complete!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"📊 Processed {len(encoder_data)} encoders from {len(hpc_folders)} HPC sources")

if __name__ == '__main__':
    main()