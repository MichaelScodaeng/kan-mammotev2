#!/usr/bin/env python3
"""
Comprehensive training time benchmarking script
Tests all combinations of datasets, models, and encoders and saves results to CSV
"""

import subprocess
import sys
import time
import pandas as pd
import json
import os
from datetime import datetime
from itertools import product
import argparse

# Configuration
DATASETS = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
           'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']

MODELS = ['JODIE', 'TGAT', 'TGN', 'TCL', 'DyGFormer', 'DyGMamba']

ENCODERS = ['lete', 'kan_mammote_dual_kmote', 'mercer', 'time2vec']

# Default parameters
DEFAULT_PARAMS = {
    'num_epochs': 200,
    'batch_size': 200,
    'data_ratio': 1.0,
    'num_runs': 1,
    'disable_progress_bar': True
}

def run_estimation(dataset, model, encoder, timeout=600):
    """
    Run training time estimation for a specific combination
    
    Args:
        dataset: Dataset name
        model: Model name  
        encoder: Encoder type
        timeout: Timeout in seconds (default 10 minutes)
    
    Returns:
        dict: Results dictionary or None if failed
    """
    
    cmd = [
        "python", "experiments/train_link_prediction_estimate.py",
        "--dataset_name", dataset,
        "--model_name", model,
        "--time_encoder_type", encoder,
        "--num_epochs", str(DEFAULT_PARAMS['num_epochs']),
        "--batch_size", str(DEFAULT_PARAMS['batch_size']),
        "--data_ratio", str(DEFAULT_PARAMS['data_ratio']),
        "--num_runs", str(DEFAULT_PARAMS['num_runs']),
        "--disable_progress_bar"
    ]
    
    print(f"🔄 Testing: {dataset} + {model} + {encoder}")
    print(f"   Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            print(f"   ✅ Success in {elapsed_time:.1f}s")
            
            # Try to parse the saved JSON file
            estimate_file = f"./time_estimates/{model}_{encoder}_{dataset}_dr{DEFAULT_PARAMS['data_ratio']}_estimate.json"
            
            if os.path.exists(estimate_file):
                try:
                    with open(estimate_file, 'r') as f:
                        estimate_data = json.load(f)
                    
                    # Extract key metrics
                    return {
                        'dataset': dataset,
                        'model': model,
                        'encoder': encoder,
                        'status': 'success',
                        'avg_batch_time_seconds': estimate_data.get('avg_batch_time_seconds', None),
                        'estimated_epoch_time_minutes': estimate_data.get('estimated_epoch_time_minutes', None),
                        'estimated_total_time_hours': estimate_data.get('estimated_total_time_hours', None),
                        'estimated_total_time_days': estimate_data.get('estimated_total_time_days', None),
                        'total_batches': estimate_data.get('total_batches', None),
                        'training_data_size': estimate_data.get('training_data_size', None),
                        'full_data_size': estimate_data.get('full_data_size', None),
                        'batch_size': estimate_data.get('batch_size', None),
                        'num_epochs': estimate_data.get('num_epochs', None),
                        'data_ratio': estimate_data.get('data_ratio', None),
                        'sample_batch_times': estimate_data.get('sample_batch_times_seconds', []),
                        'benchmark_time_seconds': elapsed_time,
                        'timestamp': datetime.now().isoformat(),
                        'error_message': None
                    }
                except Exception as e:
                    print(f"   ⚠️  Success but failed to parse JSON: {e}")
                    return {
                        'dataset': dataset,
                        'model': model,
                        'encoder': encoder,
                        'status': 'success_no_json',
                        'benchmark_time_seconds': elapsed_time,
                        'timestamp': datetime.now().isoformat(),
                        'error_message': f"JSON parse error: {e}"
                    }
            else:
                print(f"   ⚠️  Success but no output file found: {estimate_file}")
                return {
                    'dataset': dataset,
                    'model': model,
                    'encoder': encoder,
                    'status': 'success_no_file',
                    'benchmark_time_seconds': elapsed_time,
                    'timestamp': datetime.now().isoformat(),
                    'error_message': "No output file generated"
                }
        else:
            print(f"   ❌ Failed in {elapsed_time:.1f}s")
            print(f"      Error: {result.stderr[:200]}...")
            return {
                'dataset': dataset,
                'model': model,
                'encoder': encoder,
                'status': 'failed',
                'benchmark_time_seconds': elapsed_time,
                'timestamp': datetime.now().isoformat(),
                'error_message': result.stderr[:500]  # Limit error message length
            }
            
    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"   ⏰ Timeout after {elapsed_time:.1f}s")
        return {
            'dataset': dataset,
            'model': model,
            'encoder': encoder,
            'status': 'timeout',
            'benchmark_time_seconds': elapsed_time,
            'timestamp': datetime.now().isoformat(),
            'error_message': f"Timeout after {timeout}s"
        }
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"   💥 Exception in {elapsed_time:.1f}s: {e}")
        return {
            'dataset': dataset,
            'model': model,
            'encoder': encoder,
            'status': 'exception',
            'benchmark_time_seconds': elapsed_time,
            'timestamp': datetime.now().isoformat(),
            'error_message': str(e)
        }

def save_results(results, output_file):
    """Save results to CSV file"""
    df = pd.DataFrame(results)
    
    # Reorder columns for better readability
    column_order = [
        'dataset', 'model', 'encoder', 'status',
        'avg_batch_time_seconds', 'estimated_epoch_time_minutes', 
        'estimated_total_time_hours', 'estimated_total_time_days',
        'total_batches', 'training_data_size', 'full_data_size',
        'batch_size', 'num_epochs', 'data_ratio',
        'benchmark_time_seconds', 'timestamp', 'error_message',
        'sample_batch_times'
    ]
    
    # Only include columns that exist
    existing_columns = [col for col in column_order if col in df.columns]
    df = df[existing_columns]
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")
    
    # Print summary statistics
    print(f"\n📊 BENCHMARK SUMMARY:")
    print(f"   Total combinations tested: {len(df)}")
    print(f"   Successful: {len(df[df['status'] == 'success'])}")
    print(f"   Failed: {len(df[df['status'] == 'failed'])}")
    print(f"   Timeouts: {len(df[df['status'] == 'timeout'])}")
    print(f"   Exceptions: {len(df[df['status'] == 'exception'])}")
    
    if len(df[df['status'] == 'success']) > 0:
        success_df = df[df['status'] == 'success']
        print(f"\n🎯 SUCCESS STATISTICS:")
        print(f"   Average batch time: {success_df['avg_batch_time_seconds'].mean():.3f}s")
        print(f"   Average epoch time: {success_df['estimated_epoch_time_minutes'].mean():.1f} minutes")
        print(f"   Average total time: {success_df['estimated_total_time_hours'].mean():.1f} hours")
        
        # Show fastest and slowest combinations
        if not success_df['estimated_total_time_hours'].isna().all():
            fastest = success_df.loc[success_df['estimated_total_time_hours'].idxmin()]
            slowest = success_df.loc[success_df['estimated_total_time_hours'].idxmax()]
            
            print(f"\n🏃 FASTEST: {fastest['dataset']} + {fastest['model']} + {fastest['encoder']}")
            print(f"   Estimated time: {fastest['estimated_total_time_hours']:.1f} hours")
            
            print(f"\n🐌 SLOWEST: {slowest['dataset']} + {slowest['model']} + {slowest['encoder']}")
            print(f"   Estimated time: {slowest['estimated_total_time_hours']:.1f} hours")

def main():
    """Main benchmarking function"""
    parser = argparse.ArgumentParser(description='Benchmark training times for all model/dataset/encoder combinations')
    parser.add_argument('--output', type=str, default='training_time_benchmark.csv',
                       help='Output CSV file name')
    parser.add_argument('--timeout', type=int, default=600,
                       help='Timeout per test in seconds (default: 600)')
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                       help='Datasets to test (default: all)')
    parser.add_argument('--models', nargs='+', default=MODELS,
                       help='Models to test (default: all)')
    parser.add_argument('--encoders', nargs='+', default=ENCODERS,
                       help='Encoders to test (default: all)')
    parser.add_argument('--skip_existing', action='store_true',
                       help='Skip combinations that already have results')
    
    args = parser.parse_args()
    
    print(f"🚀 Starting comprehensive training time benchmark")
    print(f"   Datasets: {args.datasets}")
    print(f"   Models: {args.models}")
    print(f"   Encoders: {args.encoders}")
    print(f"   Total combinations: {len(args.datasets) * len(args.models) * len(args.encoders)}")
    print(f"   Timeout per test: {args.timeout}s")
    print(f"   Output file: {args.output}")
    
    # Create time_estimates directory
    os.makedirs("./time_estimates", exist_ok=True)
    
    # Load existing results if skip_existing is enabled
    existing_results = set()
    if args.skip_existing and os.path.exists(args.output):
        try:
            existing_df = pd.read_csv(args.output)
            for _, row in existing_df.iterrows():
                existing_results.add((row['dataset'], row['model'], row['encoder']))
            print(f"   Found {len(existing_results)} existing results to skip")
        except Exception as e:
            print(f"   Warning: Could not load existing results: {e}")
    
    results = []
    total_combinations = len(list(product(args.datasets, args.models, args.encoders)))
    
    start_time = time.time()
    
    for i, (dataset, model, encoder) in enumerate(product(args.datasets, args.models, args.encoders), 1):
        print(f"\n{'='*60}")
        print(f"Progress: {i}/{total_combinations} ({i/total_combinations*100:.1f}%)")
        
        # Skip if already exists
        if (dataset, model, encoder) in existing_results:
            print(f"⏭️  Skipping: {dataset} + {model} + {encoder} (already exists)")
            continue
        
        # Run the estimation
        result = run_estimation(dataset, model, encoder, timeout=args.timeout)
        if result:
            results.append(result)
            
            # Save intermediate results every 10 tests
            if len(results) % 10 == 0:
                temp_file = f"{args.output}.tmp"
                save_results(results, temp_file)
                print(f"💾 Intermediate save: {len(results)} results")
    
    # Final save
    if results:
        save_results(results, args.output)
        
        total_time = time.time() - start_time
        print(f"\n🏁 Benchmark completed in {total_time/60:.1f} minutes")
        print(f"   Results saved to: {args.output}")
    else:
        print(f"\n⚠️  No results collected!")

if __name__ == "__main__":
    main()