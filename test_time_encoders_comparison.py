#!/usr/bin/env python3
"""
Comprehensive Test Script for Time Encoder Comparison

This script compares different time encoding approaches:
1. Original TimeEncoder
2. LeTE (if available)
3. K-MOTE standalone (ablation study)
4. KAN-MAMMOTE with SM-kernel (default)
5. KAN-MAMMOTE with dual K-MOTE (new variant)

Author: GitHub Copilot
Date: October 2025
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

def create_test_config(encoder_type, base_config=None):
    """Create a test configuration for a specific encoder type."""
    if base_config is None:
        base_config = {
            'model_name': 'TGAT',
            'dataset_name': 'wikipedia',
            'data_ratio': 0.1,  # Use 10% of data for faster testing
            'num_epochs': 50,   # Fewer epochs for quick comparison
            'batch_size': 200,
            'num_runs': 1,      # 3 runs for statistical significance
            'patience': 10,
            'learning_rate': 0.0001,
            'time_feat_dim': 100,
            'expert_dim': 128,
            'num_mixtures': 16,
            'num_neighbors': 20,
            'num_layers': 2,
            'num_heads': 2,
            'dropout': 0.1,
            'save_results': True
        }
    
    config = base_config.copy()
    config['time_encoder_type'] = encoder_type
    
    # Encoder-specific adjustments
    if encoder_type == 'original':
        config['description'] = 'Original TimeEncoder baseline'
    elif encoder_type == 'lete':
        config['description'] = 'LeTE: Learnable Time Encoder'
    elif encoder_type == 'kmote_abs_only':
        config['description'] = 'K-MOTE standalone (absolute time only)'
    elif encoder_type == 'kan_mammote':
        config['description'] = 'KAN-MAMMOTE with SM-kernel for relative time'
    elif encoder_type == 'kan_mammote_dual_kmote':
        config['description'] = 'KAN-MAMMOTE with dual K-MOTE (no SM-kernel)'
    else:
        config['description'] = f'Time encoder: {encoder_type}'
    
    return config

def run_experiment(config, output_dir):
    """Run a single experiment with the given configuration."""
    encoder_type = config['time_encoder_type']
    print(f"\n{'='*60}")
    print(f"🔬 Running experiment: {encoder_type}")
    print(f"📝 Description: {config['description']}")
    print(f"{'='*60}")
    
    # Create output directory for this encoder
    encoder_output_dir = output_dir / encoder_type
    encoder_output_dir.mkdir(exist_ok=True)
    
    # Save config
    config_path = encoder_output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Prepare command line arguments
    cmd = [
        'python', 'experiments/train_link_prediction.py',
        '--model_name', config['model_name'],
        '--dataset_name', config['dataset_name'],
        '--time_encoder_type', config['time_encoder_type'],
        '--data_ratio', str(config['data_ratio']),
        '--num_epochs', str(config['num_epochs']),
        '--batch_size', str(config['batch_size']),
        '--num_runs', str(config['num_runs']),
        '--patience', str(config['patience']),
        '--learning_rate', str(config['learning_rate']),
        '--time_feat_dim', str(config['time_feat_dim']),
        '--expert_dim', str(config['expert_dim']),
        '--num_mixtures', str(config['num_mixtures']),
        '--num_neighbors', str(config['num_neighbors']),
        '--num_layers', str(config['num_layers']),
        '--num_heads', str(config['num_heads']),
        '--dropout', str(config['dropout']),
    ]
    
    # Run the experiment
    start_time = time.time()
    log_file = encoder_output_dir / 'training.log'
    
    try:
        with open(log_file, 'w') as f:
            process = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                check=True
            )
        
        duration = time.time() - start_time
        
        # Log success
        result = {
            'encoder_type': encoder_type,
            'status': 'success',
            'duration_seconds': duration,
            'duration_formatted': f"{duration/60:.1f} minutes",
            'config': config
        }
        
        print(f"✅ Experiment {encoder_type} completed successfully in {duration/60:.1f} minutes")
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        
        # Log failure
        result = {
            'encoder_type': encoder_type,
            'status': 'failed',
            'error_code': e.returncode,
            'duration_seconds': duration,
            'config': config
        }
        
        print(f"❌ Experiment {encoder_type} failed after {duration/60:.1f} minutes")
        print(f"   Error code: {e.returncode}")
        print(f"   Check log file: {log_file}")
    
    # Save result
    result_file = encoder_output_dir / 'result.json'
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

def collect_results(output_dir):
    """Collect and summarize results from all experiments."""
    print(f"\n{'='*60}")
    print("📊 COLLECTING RESULTS")
    print(f"{'='*60}")
    
    results = []
    
    for encoder_dir in output_dir.iterdir():
        if encoder_dir.is_dir():
            result_file = encoder_dir / 'result.json'
            if result_file.exists():
                with open(result_file) as f:
                    result = json.load(f)
                    results.append(result)
    
    # Sort by status (success first) then by duration
    results.sort(key=lambda x: (x['status'] != 'success', x.get('duration_seconds', float('inf'))))
    
    # Print summary
    print("\n📈 EXPERIMENT SUMMARY:")
    print("-" * 80)
    print(f"{'Encoder':<25} {'Status':<10} {'Duration':<12} {'Description'}")
    print("-" * 80)
    
    for result in results:
        encoder = result['encoder_type']
        status = result['status']
        duration = result.get('duration_formatted', 'N/A')
        description = result['config'].get('description', '')
        
        status_icon = "✅" if status == 'success' else "❌"
        print(f"{encoder:<25} {status_icon} {status:<8} {duration:<12} {description}")
    
    # Save summary
    summary_file = output_dir / 'experiment_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'total_experiments': len(results),
            'successful': len([r for r in results if r['status'] == 'success']),
            'failed': len([r for r in results if r['status'] == 'failed']),
            'results': results
        }, f, indent=2)
    
    print(f"\n💾 Summary saved to: {summary_file}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Compare different time encoders')
    parser.add_argument('--encoders', nargs='+', 
                       choices=['original', 'lete', 'kmote_abs_only', 'kan_mammote', 'kan_mammote_dual_kmote', 'all'],
                       default=['all'],
                       help='Time encoders to test (default: all)')
    parser.add_argument('--data_ratio', type=float, default=0.1,
                       help='Fraction of data to use for testing (default: 0.1)')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of epochs per experiment (default: 50)')
    parser.add_argument('--num_runs', type=int, default=1,
                       help='Number of runs per encoder (default: 1)')
    parser.add_argument('--dataset', default='wikipedia',
                       help='Dataset to use (default: wikipedia)')
    parser.add_argument('--output_dir', default='./time_encoder_comparison_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Define all available encoders
    all_encoders = ['original', 'lete', 'kmote_abs_only', 'kan_mammote', 'kan_mammote_dual_kmote']
    
    # Select encoders to test
    if 'all' in args.encoders:
        encoders_to_test = all_encoders
    else:
        encoders_to_test = args.encoders
    
    print(f"🧪 Time Encoder Comparison Test")
    print(f"Dataset: {args.dataset}")
    print(f"Data ratio: {args.data_ratio}")
    print(f"Epochs per experiment: {args.num_epochs}")
    print(f"Runs per encoder: {args.num_runs}")
    print(f"Encoders to test: {', '.join(encoders_to_test)}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Base configuration
    base_config = {
        'model_name': 'TGAT',
        'dataset_name': args.dataset,
        'data_ratio': args.data_ratio,
        'num_epochs': args.num_epochs,
        'num_runs': args.num_runs,
        # Fixed parameters for fair comparison
        'batch_size': 200,
        'patience': 10,
        'learning_rate': 0.0001,
        'time_feat_dim': 100,
        'expert_dim': 128,
        'num_mixtures': 16,
        'num_neighbors': 20,
        'num_layers': 2,
        'num_heads': 2,
        'dropout': 0.1
    }
    
    # Run experiments
    results = []
    total_start_time = time.time()
    
    for i, encoder_type in enumerate(encoders_to_test, 1):
        print(f"\n🔄 Progress: {i}/{len(encoders_to_test)} experiments")
        
        # Skip encoders that might not be available
        if encoder_type == 'lete':
            print("ℹ️  Note: LeTE encoder may not be available. If it fails, it will be skipped.")
        
        config = create_test_config(encoder_type, base_config)
        result = run_experiment(config, output_dir)
        results.append(result)
    
    total_duration = time.time() - total_start_time
    
    # Collect and summarize results
    final_results = collect_results(output_dir)
    
    print(f"\n🎉 ALL EXPERIMENTS COMPLETED!")
    print(f"⏱️  Total time: {total_duration/60:.1f} minutes")
    print(f"📁 Results saved in: {output_dir}")
    print(f"✅ Successful: {len([r for r in final_results if r['status'] == 'success'])}")
    print(f"❌ Failed: {len([r for r in final_results if r['status'] == 'failed'])}")
    
    # Recommendations
    successful_results = [r for r in final_results if r['status'] == 'success']
    if successful_results:
        fastest = min(successful_results, key=lambda x: x['duration_seconds'])
        print(f"\n🏆 Fastest successful encoder: {fastest['encoder_type']} ({fastest['duration_formatted']})")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Check detailed logs in {output_dir}")
        print(f"   2. Compare final test metrics between successful encoders")
        print(f"   3. Run full experiments with best-performing encoders")

if __name__ == "__main__":
    main()