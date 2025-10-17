#!/usr/bin/env python3
"""
KAN-MAMMOTE Ablation Study Experiment
====================================

This script runs ablation experiments to understand which components of KAN-MAMMOTE
contribute most to performance. Tests the following configurations:

1. SM-Kernel Only (relative time encoding only)
2. K-MOTE Absolute Only (absolute time encoding only) 
3. K-MOTE Relative Only (relative time encoding only)
4. Dual Stream Baseline (K-MOTE abs + SM-Kernel, no Mamba)
5. KAN-MAMMOTE Lite (production-ready without Mamba)
6. LeTE (for comparison)
7. Bochner (Gaussian Fourier features)
8. Mercer (harmonic eigenfunction expansion)
9. Original (cosine-based from DyGMamba)
10. Full KAN-MAMMOTE (for reference)

Supports multiple GNN models: TGAT, TGN, CAWN, TCL, GraphMixer, DyGFormer, DyGMamba
Supports 13 datasets: wikipedia, reddit, mooc, lastfm, enron, SocialEvo, uci,
                      CanParl, Contacts, Flights, UNtrade, UNvote, USLegis

Usage:
    # Single model + single dataset with all encoders
    python experiment_ablation.py --model TGAT --dataset wikipedia --data_ratio 0.1
    
    # Run on ALL models with ALL datasets and all encoders
    python experiment_ablation.py --all_models --all_datasets --data_ratio 0.1
    
    # Run multiple models on one dataset
    python experiment_ablation.py --model TGAT --all_datasets --data_ratio 0.1
    
    # Run one model on all datasets
    python experiment_ablation.py --all_models --dataset wikipedia --data_ratio 0.1
    
    # Specific encoders only on one model/dataset
    python experiment_ablation.py --model TGAT --dataset reddit --encoders kan_mammote_full lete
    
    # Quick dry-run test
    python experiment_ablation.py --model TGAT --dataset wikipedia --num_epochs 2 --data_ratio 0.02 --dry_run
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime
import subprocess

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def create_experiment_config():
    """Create the base configuration for ablation experiments."""
    
    # Ablation encoder configurations
    ablation_configs = {
        'lete': {
            'description': 'LeTE encoder for comparison',
            'time_encoder': 'lete',
            'save_suffix': 'lete'
        },
        
        'mercer': {
            'description': 'Mercer encoder (harmonic eigenfunction expansion)',
            'time_encoder': 'mercer',
            'save_suffix': 'mercer'
        },
        'original': {
            'description': 'Original encoder (cosine-based from DyGMamba)',
            'time_encoder': 'original',
            'save_suffix': 'original'
        },
        
        'kan_mammote_full': {
            'description': 'Full KAN-MAMMOTE for reference',
            'time_encoder': 'kan_mammote',
            'save_suffix': 'kan_mammote_full'
        },
        'kan_mammote_dual_kmote': {
            'description': 'KAN-MAMMOTE with dual K-MOTE streams (abs + rel) + Mamba',
            'time_encoder': 'kan_mammote_dual_kmote',
            'save_suffix': 'kan_mammote_dual_kmote'
        },
        
    }

    """ 
    'sm_kernel_only': {
            'description': 'SM-Kernel only for relative time encoding',
            'time_encoder': 'sm_kernel_only',
            'save_suffix': 'sm_kernel_only'
        },
        'kmote_abs_only': {
            'description': 'K-MOTE for absolute time encoding only',
            'time_encoder': 'kmote_abs_only', 
            'save_suffix': 'kmote_abs_only'
        },
        'kmote_rel_only': {
            'description': 'K-MOTE for relative time encoding only',
            'time_encoder': 'kmote_rel_only',
            'save_suffix': 'kmote_rel_only'
        },
    'dual_stream_baseline': {
            'description': 'Dual stream (K-MOTE abs + SM-Kernel) without Mamba',
            'time_encoder': 'dual_stream_baseline',
            'save_suffix': 'dual_stream_baseline'
        },
    'kan_mammote_lite': {
            'description': 'KAN-MAMMOTE Lite (production-ready without Mamba)',
            'time_encoder': 'kan_mammote_lite',
            'save_suffix': 'kan_mammote_lite'
        },
        'bochner': {
            'description': 'Bochner encoder (Gaussian Fourier features)',
            'time_encoder': 'bochner',
            'save_suffix': 'bochner'
        },
    """
    return ablation_configs

def run_single_experiment(encoder_name, config, args, ablation_dir):
    """Run a single ablation experiment."""
    
    print(f"\n{'='*60}")
    print(f"Running Ablation Experiment: {encoder_name}")
    print(f"Description: {config['description']}")
    print(f"{'='*60}")
    
    # Create command arguments
    cmd_args = [
        'python', '-m', 'experiments.train_link_prediction',
        '--model_name', args.model,  # Use configurable model
        '--dataset_name', args.dataset,
        '--time_encoder_type', config['time_encoder'],  # Note: training script uses time_encoder_type
        '--num_epochs', str(args.num_epochs),
        '--batch_size', str(args.batch_size),
        '--num_neighbors', str(args.num_neighbors),
        '--learning_rate', str(args.learning_rate),
        '--dropout', str(args.dropout),  # Note: training script uses dropout not drop_out
        '--patience', str(args.tolerance),  # Note: training script uses patience not tolerance
        '--save_model_name_suffix', config['save_suffix'],
        '--sort_neighbors_by_time',  # Important for Mamba compatibility
        '--ablation_dir', ablation_dir,  # Add ablation directory for organized output
        '--seed', '42'  # ✅ Fixed seed for reproducibility across all experiments
    ]
    
    # Add data ratio if specified (now applies BEFORE splitting in data loader)
    if args.data_ratio < 1.0:
        cmd_args.extend(['--data_ratio', str(args.data_ratio)])
        print(f"   Using data_ratio={args.data_ratio} (applied BEFORE train/val/test split)")
    
    # Add encoder-specific parameters for K-MOTE variants and KAN-MAMMOTE
    if config['time_encoder'] in ['kmote_abs_only', 'kmote_rel_only', 'dual_stream_baseline', 'kan_mammote', 'kan_mammote_lite']:
        cmd_args.extend([
            '--expert_dim', str(args.expert_dim),
            '--num_mixtures', str(args.num_mixtures)
        ])
        
        # Add Mamba parameters for full KAN-MAMMOTE only
        if config['time_encoder'] == 'kan_mammote':
            cmd_args.extend([
                '--mamba_d_state', str(args.mamba_d_state),
                '--mamba_d_conv', str(args.mamba_d_conv),
                '--mamba_expand', str(args.mamba_expand),
                '--mamba_headdim', str(args.mamba_headdim)
            ])
    
    # Add verbosity
    if args.verbose:
        cmd_args.append('--verbose')
    
    print(f"Command: {' '.join(cmd_args)}")
    print(f"Starting experiment at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        # Run the experiment
        result = subprocess.run(cmd_args, text=True, timeout=args.timeout)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(f"✅ Experiment {encoder_name} completed successfully in {duration:.2f}s")
            if args.verbose:
                print("STDOUT:")
                print(result.stdout[-1000:])  # Last 1000 chars
        else:
            print(f"❌ Experiment {encoder_name} failed with return code {result.returncode}")
            print("STDERR:")
            print(result.stderr[-1000:])  # Last 1000 chars
            
        return {
            'encoder_name': encoder_name,
            'config': config,
            'success': result.returncode == 0,
            'duration': duration,
            'return_code': result.returncode,
            'stdout_tail': result.stdout[-500:] if result.stdout else "",
            'stderr_tail': result.stderr[-500:] if result.stderr else ""
        }
        
    except subprocess.TimeoutExpired:
        print(f"⏰ Experiment {encoder_name} timed out after {args.timeout}s")
        return {
            'encoder_name': encoder_name,
            'config': config,
            'success': False,
            'duration': args.timeout,
            'return_code': -1,
            'error': 'Timeout',
            'stdout_tail': "",
            'stderr_tail': ""
        }
    except Exception as e:
        print(f"💥 Experiment {encoder_name} crashed: {str(e)}")
        return {
            'encoder_name': encoder_name,
            'config': config,
            'success': False,
            'duration': time.time() - start_time,
            'return_code': -2,
            'error': str(e),
            'stdout_tail': "",
            'stderr_tail': ""
        }

def save_experiment_summary(results, args, ablation_dir):
    """Save a summary of all experiment results."""
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Handle multiple models and datasets
    models_str = '_'.join(args.models) if hasattr(args, 'models') and len(args.models) > 1 else (args.model if hasattr(args, 'model') and args.model else 'multiple')
    if len(models_str) > 30:  # Truncate if too long
        models_str = f"{len(args.models)}models"
    
    datasets_str = '_'.join(args.datasets) if hasattr(args, 'datasets') and len(args.datasets) > 1 else (args.dataset if args.dataset else 'multiple')
    if len(datasets_str) > 30:  # Truncate if too long
        datasets_str = f"{len(args.datasets)}datasets"
    
    summary_file = os.path.join(ablation_dir, f"ablation_study_summary_{models_str}_{datasets_str}_{timestamp}.json")
    
    summary = {
        'experiment_info': {
            'models': args.models if hasattr(args, 'models') else [args.model],
            'datasets': args.datasets if hasattr(args, 'datasets') else [args.dataset],
            'data_ratio': args.data_ratio,
            'num_epochs': args.num_epochs,
            'batch_size': args.batch_size,
            'num_neighbors': args.num_neighbors,
            'learning_rate': args.learning_rate,
            'timestamp': timestamp,
            'ablation_dir': ablation_dir,
            'total_experiments': len(results)
        },
        'results': results
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n📊 Experiment summary saved to: {summary_file}")
    
    # Print summary table
    print(f"\n{'='*110}")
    print("ABLATION STUDY RESULTS SUMMARY")
    print(f"{'='*110}")
    print(f"{'Model':<12} {'Dataset':<15} {'Encoder':<25} {'Status':<10} {'Duration':<12} {'Description':<25}")
    print(f"{'-'*110}")
    
    for result in results:
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        duration = f"{result['duration']:.1f}s"
        description = result['config']['description'][:24]
        dataset = result.get('dataset', 'N/A')[:14]
        model = result.get('model', 'N/A')[:11]
        print(f"{model:<12} {dataset:<15} {result['encoder_name']:<25} {status:<10} {duration:<12} {description:<25}")
    
    success_count = sum(1 for r in results if r['success'])
    print(f"{'-'*110}")
    print(f"Total: {len(results)} experiments, {success_count} successful, {len(results)-success_count} failed")
    print(f"{'='*110}")

def main():
    parser = argparse.ArgumentParser(description='Run KAN-MAMMOTE ablation study experiments')
    
    # Model and dataset parameters
    parser.add_argument('--model', type=str, default=None,
                        choices=['TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DyGMamba'],
                        help='Model to use for experiments (omit to run all models)')
    parser.add_argument('--all_models', action='store_true',
                        help='Run experiments on all available GNN models')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
                                'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis'],
                        help='Dataset to use for the ablation study (omit to run all datasets)')
    parser.add_argument('--all_datasets', action='store_true',
                        help='Run experiments on all available datasets')
    parser.add_argument('--data_ratio', type=float, default=0.1,
                        help='Ratio of data to use (0.1 = 10%% for faster experiments)')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='Batch size for training')
    parser.add_argument('--num_neighbors', type=int, default=20,
                        help='Number of temporal neighbors')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--tolerance', type=int, default=5,
                        help='Early stopping tolerance')
    
    # K-MOTE parameters
    parser.add_argument('--expert_dim', type=int, default=128,
                        help='Expert dimension for K-MOTE encoders')
    parser.add_argument('--num_mixtures', type=int, default=16,
                        help='Number of mixtures for K-MOTE encoders')
    
    # Mamba parameters (for full KAN-MAMMOTE)
    parser.add_argument('--mamba_d_state', type=int, default=256,
                        help='Mamba state dimension')
    parser.add_argument('--mamba_d_conv', type=int, default=4,
                        help='Mamba convolution dimension')
    parser.add_argument('--mamba_expand', type=int, default=2,
                        help='Mamba expansion factor')
    parser.add_argument('--mamba_headdim', type=int, default=64,
                        help='Mamba head dimension')
    
    # Experiment control
    parser.add_argument('--encoders', nargs='+', 
                        choices=['sm_kernel_only', 'kmote_abs_only', 'kmote_rel_only', 
                                'dual_stream_baseline', 'kan_mammote_lite', 'lete', 'kan_mammote_full',
                                'bochner', 'mercer', 'original',"kan_mammote_dual_kmote"],
                        help='Specific encoders to test (default: all)')
    parser.add_argument('--timeout', type=int, default=3600,
                        help='Timeout per experiment in seconds (default: 1 hour)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without running them')
    
    args = parser.parse_args()
    
    # Determine which models to run
    all_available_models = ['TGAT', 'TGN', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'DyGMamba']
    
    if args.all_models:
        models_to_run = all_available_models
    elif args.model:
        models_to_run = [args.model]
    else:
        # Default to TGAT if no model specified
        models_to_run = ['TGAT']
        print("ℹ️  No model specified, defaulting to TGAT. Use --model <name> or --all_models to change.")
    
    # Determine which datasets to run
    all_available_datasets = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
                             'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']
    
    if args.all_datasets:
        datasets_to_run = all_available_datasets
    elif args.dataset:
        datasets_to_run = [args.dataset]
    else:
        # Default to wikipedia if no dataset specified
        datasets_to_run = ['wikipedia']
        print("ℹ️  No dataset specified, defaulting to wikipedia. Use --dataset <name> or --all_datasets to change.")
    
    # Create timestamped ablation directory
    ablation_dir = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(ablation_dir, exist_ok=True)
    
    print(f"""
🧪 KAN-MAMMOTE Ablation Study
============================
Model(s): {', '.join(models_to_run)}
Dataset(s): {', '.join(datasets_to_run)}
Data Ratio: {args.data_ratio} ({args.data_ratio*100:.1f}%)
Epochs: {args.num_epochs}
Batch Size: {args.batch_size}
Neighbors: {args.num_neighbors}
Learning Rate: {args.learning_rate}
Output Directory: {ablation_dir}
    """)
    
    # Get experiment configurations
    ablation_configs = create_experiment_config()
    
    # Filter encoders if specified
    if args.encoders:
        ablation_configs = {k: v for k, v in ablation_configs.items() if k in args.encoders}
    
    total_experiments = len(ablation_configs) * len(datasets_to_run) * len(models_to_run)
    print(f"Running {total_experiments} total experiments ({len(ablation_configs)} encoders × {len(datasets_to_run)} dataset(s) × {len(models_to_run)} model(s)):")
    for name, config in ablation_configs.items():
        print(f"  - {name}: {config['description']}")
    print(f"  Models: {', '.join(models_to_run)}")
    print(f"  Datasets: {', '.join(datasets_to_run)}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - Commands will be printed but not executed")
        for model in models_to_run:
            for dataset in datasets_to_run:
                print(f"\n🤖 Model: {model}, 📊 Dataset: {dataset}")
                for encoder_name, config in ablation_configs.items():
                    cmd_args = [
                        'python', '-m', 'experiments.train_link_prediction',
                        '--model_name', model,
                        '--dataset_name', dataset,
                        '--time_encoder_type', config['time_encoder'],  # Note: training script uses time_encoder_type
                        '--save_model_name_suffix', config['save_suffix'],
                        '--ablation_dir', ablation_dir,
                    '--sort_neighbors_by_time'
                ]
                print(f"  {encoder_name}: {' '.join(cmd_args)}")
        return
    
    # Confirm before running
    if not args.verbose:
        response = input(f"\nProceed with {total_experiments} experiments? [y/N]: ")
        if response.lower() not in ['y', 'yes']:
            print("Experiment cancelled.")
            return
    
    # Run experiments across all models, datasets and encoders
    results = []
    start_time = time.time()
    
    experiment_count = 0
    for model_idx, model in enumerate(models_to_run, 1):
        print(f"\n{'='*90}")
        print(f"🤖 MODEL [{model_idx}/{len(models_to_run)}]: {model}")
        print(f"{'='*90}")
        
        # Temporarily set args.model for compatibility with run_single_experiment
        args.model = model
        
        for dataset_idx, dataset in enumerate(datasets_to_run, 1):
            print(f"\n{'='*80}")
            print(f"📊 DATASET [{dataset_idx}/{len(datasets_to_run)}]: {dataset} (Model: {model})")
            print(f"{'='*80}")
            
            # Temporarily set args.dataset for compatibility with run_single_experiment
            args.dataset = dataset
            
            for encoder_idx, (encoder_name, config) in enumerate(ablation_configs.items(), 1):
                experiment_count += 1
                print(f"\n[{experiment_count}/{total_experiments}] Running {encoder_name} on {model}/{dataset}...")
                result = run_single_experiment(encoder_name, config, args, ablation_dir)
                result['model'] = model  # Add model info to result
                result['dataset'] = dataset  # Add dataset info to result
                results.append(result)
                
                # Brief pause between experiments
                if experiment_count < total_experiments:
                    time.sleep(2)
    
    total_time = time.time() - start_time
    print(f"\n🏁 All experiments completed in {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    # Save summary (restore original model and dataset lists for summary)
    args.models = models_to_run
    args.datasets = datasets_to_run
    save_experiment_summary(results, args, ablation_dir)
    
    # Final recommendations
    successful_results = [r for r in results if r['success']]
    if successful_results:
        print(f"\n💡 Next Steps:")
        print(f"1. Check the experiment results in {ablation_dir}/ directory")
        print(f"2. All models, metrics, and results are saved in {ablation_dir}/")
        print(f"3. Compare train/val metrics CSV files for each encoder")
        print(f"4. Run analysis script on the ablation directory results")
        print(f"5. Model suffixes used: {[r['config']['save_suffix'] for r in successful_results]}")
    else:
        print(f"\n⚠️  No experiments succeeded. Check error messages above.")

if __name__ == '__main__':
    main()