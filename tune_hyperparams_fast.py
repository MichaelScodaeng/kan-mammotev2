#!/usr/bin/env python3
"""
Fast hyperparameter tuning script for learning rate and weight decay.
Uses 10% temporal prefix training data and early stopping with 10 epochs max, patience 3.
This won't interfere with actual training due to special save_model_name_suffix.
"""

import os
import sys
import subprocess
import itertools
from pathlib import Path
import json
from datetime import datetime
import argparse

# Configuration
DATASETS = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
            'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']

MODELS = ['JODIE', 'TGAT', 'TGN', 'TCL', 'DyGFormer', 'DyGMamba']

TIME_ENCODERS = ['lete', 'kan_mammote_dual_kmote', 'mercer', 'time2vec']

# Hyperparameter search space - keep it small for speed
LEARNING_RATES = [1e-4, 5e-4, 1e-3, 5e-5, 1e-5]
WEIGHT_DECAYS = [0.0, 1e-5, 1e-4, 1e-3]

# Fixed parameters for fast tuning
DATA_RATIO = 0.1  # 10% temporal prefix
NUM_EPOCHS = 10
PATIENCE = 3
NUM_RUNS = 1  # Single run for speed
GPU = 0
BATCH_SIZE = 200

# Output directory
TUNING_OUTPUT_DIR = Path('./hyperparameter_tuning_results')
TUNING_OUTPUT_DIR.mkdir(exist_ok=True)


def get_training_script():
    """Find the main training script."""
    candidates = ['train_link_prediction.py', 'experiment_unified.py', 'train.py']
    for script in candidates:
        if Path(script).exists():
            return script
    raise FileNotFoundError("Could not find training script. Please specify manually.")


def run_experiment(model, dataset, time_encoder, lr, wd, gpu_id, dry_run=False):
    """
    Run a single hyperparameter configuration.
    
    Args:
        model: Model name
        dataset: Dataset name
        time_encoder: Time encoder type
        lr: Learning rate
        wd: Weight decay
        gpu_id: GPU device ID
        dry_run: If True, only print commands without running
    
    Returns:
        dict: Results dictionary with metrics
    """
    # Create unique identifier for this configuration
    config_id = f"{model}_{dataset}_{time_encoder}_lr{lr}_wd{wd}"
    
    # Special suffix to avoid conflicts with actual training
    save_suffix = f"_HPTUNE_{datetime.now().strftime('%Y%m%d')}"
    ablation_dir = TUNING_OUTPUT_DIR / config_id
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    # Build command
    training_script = get_training_script()
    cmd = [
        'python', training_script,
        '--model_name', model,
        '--dataset_name', dataset,
        '--time_encoder_type', time_encoder,
        '--learning_rate', str(lr),
        '--weight_decay', str(wd),
        '--data_ratio', str(DATA_RATIO),
        '--num_epochs', str(NUM_EPOCHS),
        '--patience', str(PATIENCE),
        '--num_runs', str(NUM_RUNS),
        '--gpu', str(gpu_id),
        '--batch_size', str(BATCH_SIZE),
        '--save_model_name_suffix', save_suffix,
        '--ablation_dir', str(ablation_dir),
        '--load_best_configs',  # Use model-specific best configs for other hyperparams
        '--disable_progress_bar',  # Cleaner output for batch jobs
    ]
    
    print(f"\n{'='*80}")
    print(f"Running: {config_id}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")
    
    if dry_run:
        return {'config': config_id, 'status': 'dry_run'}
    
    # Run the experiment
    log_file = ablation_dir / 'training.log'
    try:
        with open(log_file, 'w') as f:
            result = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=7200  # 2 hour timeout per experiment
            )
        
        # Parse results
        metrics = parse_results(ablation_dir)
        metrics['config'] = config_id
        metrics['lr'] = lr
        metrics['wd'] = wd
        metrics['status'] = 'success' if result.returncode == 0 else 'failed'
        metrics['exit_code'] = result.returncode
        
        return metrics
        
    except subprocess.TimeoutExpired:
        print(f"WARNING: {config_id} timed out!")
        return {'config': config_id, 'status': 'timeout', 'lr': lr, 'wd': wd}
    except Exception as e:
        print(f"ERROR running {config_id}: {e}")
        return {'config': config_id, 'status': 'error', 'lr': lr, 'wd': wd, 'error': str(e)}


def parse_results(result_dir):
    """
    Parse results from experiment output directory.
    
    Args:
        result_dir: Path to experiment results
    
    Returns:
        dict: Parsed metrics
    """
    metrics = {}
    
    # Try to find and parse result files
    result_patterns = [
        result_dir / 'saved_results' / '*.json',
        result_dir / 'results' / '*.json',
        result_dir / '*.json'
    ]
    
    import glob
    for pattern in result_patterns:
        json_files = list(glob.glob(str(pattern)))
        if json_files:
            try:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                    # Extract key metrics
                    if 'validate_ap' in data:
                        metrics['val_ap'] = data['validate_ap']
                    if 'validate_auc' in data:
                        metrics['val_auc'] = data['validate_auc']
                    if 'test_ap' in data:
                        metrics['test_ap'] = data['test_ap']
                    if 'test_auc' in data:
                        metrics['test_auc'] = data['test_auc']
                    break
            except Exception as e:
                print(f"Warning: Could not parse {json_files[0]}: {e}")
    
    # Try to parse from log file
    log_file = result_dir / 'training.log'
    if log_file.exists() and not metrics:
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                for line in reversed(lines[-100:]):  # Check last 100 lines
                    if 'validate_ap' in line.lower() or 'val_ap' in line.lower():
                        # Try to extract numeric values
                        import re
                        numbers = re.findall(r'[-+]?\d*\.\d+|\d+', line)
                        if numbers:
                            metrics['val_metric'] = float(numbers[0])
                            break
        except Exception as e:
            print(f"Warning: Could not parse log file: {e}")
    
    return metrics


def save_results(all_results, output_file):
    """Save all results to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def create_summary_report(all_results, output_file):
    """Create a human-readable summary report."""
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("HYPERPARAMETER TUNING SUMMARY\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*80 + "\n\n")
        
        # Group by model, dataset, time_encoder
        grouped = {}
        for result in all_results:
            if result.get('status') != 'success':
                continue
            
            key = (result.get('model', 'unknown'), 
                   result.get('dataset', 'unknown'), 
                   result.get('time_encoder', 'unknown'))
            
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        
        # Find best configurations
        for (model, dataset, time_encoder), results in sorted(grouped.items()):
            f.write(f"\n{model} + {dataset} + {time_encoder}\n")
            f.write("-" * 60 + "\n")
            
            if not results:
                f.write("  No successful runs\n")
                continue
            
            # Sort by validation metric (use val_ap if available)
            metric_key = 'val_ap' if 'val_ap' in results[0] else 'val_metric'
            if metric_key in results[0]:
                results_sorted = sorted(results, key=lambda x: x.get(metric_key, 0), reverse=True)
                
                f.write(f"  Best configuration:\n")
                best = results_sorted[0]
                f.write(f"    LR: {best.get('lr')}, WD: {best.get('wd')}\n")
                f.write(f"    Val Metric: {best.get(metric_key, 'N/A')}\n")
                
                if len(results_sorted) > 1:
                    f.write(f"\n  Top 3 configurations:\n")
                    for i, r in enumerate(results_sorted[:3], 1):
                        f.write(f"    {i}. LR={r.get('lr')}, WD={r.get('wd')}, "
                               f"Val={r.get(metric_key, 'N/A'):.4f}\n")
            else:
                f.write("  Completed but no metrics found\n")
    
    print(f"Summary report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Fast hyperparameter tuning for temporal graph models')
    parser.add_argument('--models', nargs='+', default=MODELS,
                        choices=MODELS, help='Models to tune')
    parser.add_argument('--datasets', nargs='+', default=DATASETS,
                        choices=DATASETS, help='Datasets to use')
    parser.add_argument('--time_encoders', nargs='+', default=TIME_ENCODERS,
                        choices=TIME_ENCODERS, help='Time encoders to test')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--dry_run', action='store_true', 
                        help='Print commands without running')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel jobs (experimental)')
    parser.add_argument('--subset', type=int, default=None,
                        help='Only run first N experiments (for testing)')
    
    args = parser.parse_args()
    
    # Generate all combinations
    experiments = list(itertools.product(
        args.models,
        args.datasets,
        args.time_encoders,
        LEARNING_RATES,
        WEIGHT_DECAYS
    ))
    
    if args.subset:
        experiments = experiments[:args.subset]
    
    print(f"\n{'='*80}")
    print(f"FAST HYPERPARAMETER TUNING")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiments)}")
    print(f"Models: {args.models}")
    print(f"Datasets: {args.datasets}")
    print(f"Time Encoders: {args.time_encoders}")
    print(f"Learning Rates: {LEARNING_RATES}")
    print(f"Weight Decays: {WEIGHT_DECAYS}")
    print(f"Data ratio: {DATA_RATIO} (10% temporal prefix)")
    print(f"Epochs: {NUM_EPOCHS}, Patience: {PATIENCE}")
    print(f"GPU: {args.gpu}")
    print(f"Dry run: {args.dry_run}")
    print(f"{'='*80}\n")
    
    if not args.dry_run:
        response = input("Continue? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Run experiments
    all_results = []
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for i, (model, dataset, time_encoder, lr, wd) in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Running experiment...")
        
        result = run_experiment(
            model=model,
            dataset=dataset,
            time_encoder=time_encoder,
            lr=lr,
            wd=wd,
            gpu_id=args.gpu,
            dry_run=args.dry_run
        )
        
        result['model'] = model
        result['dataset'] = dataset
        result['time_encoder'] = time_encoder
        result['experiment_id'] = i
        
        all_results.append(result)
        
        # Save intermediate results
        if not args.dry_run and i % 10 == 0:
            intermediate_file = TUNING_OUTPUT_DIR / f'results_intermediate_{timestamp}.json'
            save_results(all_results, intermediate_file)
    
    # Save final results
    if not args.dry_run:
        results_file = TUNING_OUTPUT_DIR / f'results_final_{timestamp}.json'
        save_results(all_results, results_file)
        
        summary_file = TUNING_OUTPUT_DIR / f'summary_{timestamp}.txt'
        create_summary_report(all_results, summary_file)
        
        print(f"\n{'='*80}")
        print(f"TUNING COMPLETE!")
        print(f"Total experiments: {len(all_results)}")
        print(f"Successful: {sum(1 for r in all_results if r.get('status') == 'success')}")
        print(f"Failed: {sum(1 for r in all_results if r.get('status') == 'failed')}")
        print(f"Timeout: {sum(1 for r in all_results if r.get('status') == 'timeout')}")
        print(f"Results: {results_file}")
        print(f"Summary: {summary_file}")
        print(f"{'='*80}\n")
    else:
        print("\nDry run complete. No experiments were executed.")


if __name__ == '__main__':
    main()
