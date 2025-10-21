#!/usr/bin/env python3
"""
Direct KAN-MAMMOTE Hyperparameter Tuning (No Job Script Generation)
===================================================================

Runs hyperparameter tuning experiments directly without generating PBS scripts.

Usage:
    python tune_kan_mammote_direct.py --models TCL --datasets wikipedia
    python tune_kan_mammote_direct.py --models TGAT TGN --datasets wikipedia reddit mooc
    python tune_kan_mammote_direct.py --max_configs 5  # Test with fewer configs
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from itertools import product

# Same config as tune_kan_mammote_fast.py
UNIFORM_CONFIG = {
    'expert_dim': [64, 128, 256],
    'mamba_d_state': [128, 256, 512],
    'mamba_expand': [2, 4],
    'dropout': [0, 0.1, 0.2, 0.3],
    'mamba_headdim': [64],
    'mamba_d_conv': [4]
}

DATASET_CONFIGS = {
    'Contacts': UNIFORM_CONFIG, 'USLegis': UNIFORM_CONFIG, 'Flights': UNIFORM_CONFIG,
    'UNvote': UNIFORM_CONFIG, 'wikipedia': UNIFORM_CONFIG, 'reddit': UNIFORM_CONFIG,
    'mooc': UNIFORM_CONFIG, 'lastfm': UNIFORM_CONFIG, 'enron': UNIFORM_CONFIG,
    'UNtrade': UNIFORM_CONFIG, 'uci': UNIFORM_CONFIG, 'CanParl': UNIFORM_CONFIG,
    'SocialEvo': UNIFORM_CONFIG,
}

GNN_MODELS = ['TGAT', 'TGN', 'TCL', 'JODIE', 'DyGFormer', 'DyGMamba']

FAST_TUNING_PARAMS = {
    'train_only_ratio': 0.1,
    'num_epochs': 10,
    'patience': 3,
    'num_runs': 1,
    'seed': 0,
    'test_interval_epochs': 1,
    'checkpoint_strategy': 'minimal',
    'disable_progress_bar': True
}

def is_valid_mamba_config(expert_dim, mamba_expand, mamba_headdim):
    """Validate Mamba2 configuration constraint"""
    inner_dim = expert_dim * mamba_expand
    if inner_dim % mamba_headdim != 0:
        return False
    ngroups = inner_dim // mamba_headdim
    return ngroups % 8 == 0

def generate_config_grid(dataset):
    """Generate valid hyperparameter configurations"""
    config_space = DATASET_CONFIGS.get(dataset, UNIFORM_CONFIG)
    
    all_configs = []
    for values in product(
        config_space['expert_dim'],
        config_space['mamba_d_state'],
        config_space['mamba_expand'],
        config_space['dropout'],
        config_space['mamba_headdim'],
        config_space['mamba_d_conv']
    ):
        config = {
            'expert_dim': values[0],
            'mamba_d_state': values[1],
            'mamba_expand': values[2],
            'dropout': values[3],
            'mamba_headdim': values[4],
            'mamba_d_conv': values[5]
        }
        
        if is_valid_mamba_config(config['expert_dim'], config['mamba_expand'], config['mamba_headdim']):
            all_configs.append(config)
    
    return all_configs

def run_experiment(dataset, model, config, config_idx, verbose=True):
    """Run a single tuning experiment"""
    
    cmd = [
        'python', 'experiments/train_link_prediction.py',
        '--dataset_name', dataset,
        '--model_name', model,
        '--time_encoder_type', 'kan_mammote_dual_kmote',
        '--expert_dim', str(config['expert_dim']),
        '--mamba_d_state', str(config['mamba_d_state']),
        '--mamba_expand', str(config['mamba_expand']),
        '--dropout', str(config['dropout']),
        '--mamba_headdim', str(config['mamba_headdim']),
        '--mamba_d_conv', str(config['mamba_d_conv']),
        '--data_ratio', str(0.25) if model in ['JODIE', 'TCL'] else str(FAST_TUNING_PARAMS['train_only_ratio']),
        '--num_epochs', str(FAST_TUNING_PARAMS['num_epochs']),
        '--patience', str(FAST_TUNING_PARAMS['patience']),
        '--num_runs', str(FAST_TUNING_PARAMS['num_runs']),
        '--seed', str(FAST_TUNING_PARAMS['seed']),
        '--test_interval_epochs', str(FAST_TUNING_PARAMS['test_interval_epochs']),
        '--checkpoint_strategy', FAST_TUNING_PARAMS['checkpoint_strategy'],
        '--save_model_name_suffix', f'hptune_c{config_idx:03d}_ed{config["expert_dim"]}_ds{config["mamba_d_state"]}_ex{config["mamba_expand"]}',
        '--ablation_dir', f'./hptune_results/{dataset}/{model}'
    ]
    
    # Add --train_only_ratio flag (no value) for JODIE and TCL
    if model in ['JODIE', 'TCL']:
        cmd.append('--train_only_ratio')
    
    if FAST_TUNING_PARAMS['disable_progress_bar']:
        cmd.append('--disable_progress_bar')
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Config {config_idx}: {dataset} / {model}")
        print(f"Params: {json.dumps(config, indent=2)}")
        print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=not verbose, text=True)
        if verbose:
            print(f"✅ Config {config_idx} completed successfully")
        return True, None
    except subprocess.CalledProcessError as e:
        error_msg = f"❌ Config {config_idx} failed: {e}"
        if verbose:
            print(error_msg)
        return False, error_msg

def main():
    parser = argparse.ArgumentParser(description='Run KAN-MAMMOTE hyperparameter tuning directly')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets to tune (default: all)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to tune (default: all)')
    parser.add_argument('--max_configs', type=int, default=None,
                        help='Maximum configs per dataset/model (for testing)')
    parser.add_argument('--start_from', type=int, default=0,
                        help='Start from config index (for resuming)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Show detailed output')
    
    args = parser.parse_args()
    
    datasets = args.datasets if args.datasets else list(DATASET_CONFIGS.keys())
    models = args.models if args.models else GNN_MODELS
    
    print("="*80)
    print("KAN-MAMMOTE Direct Hyperparameter Tuning")
    print("="*80)
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Tuning params: {json.dumps(FAST_TUNING_PARAMS, indent=2)}")
    print("="*80)
    
    results = []
    
    for dataset in datasets:
        for model in models:
            print(f"\n🎯 Tuning {dataset} / {model}...")
            
            configs = generate_config_grid(dataset)
            
            if args.max_configs:
                configs = configs[:args.max_configs]
            
            print(f"   Total configurations: {len(configs)}")
            
            for idx, config in enumerate(configs):
                if idx < args.start_from:
                    continue
                    
                success, error = run_experiment(dataset, model, config, idx, args.verbose)
                results.append({
                    'dataset': dataset,
                    'model': model,
                    'config_idx': idx,
                    'config': config,
                    'success': success,
                    'error': error
                })
    
    # Summary
    print("\n" + "="*80)
    print("Tuning Summary")
    print("="*80)
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total - successful
    print(f"Total experiments: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    if failed > 0:
        print("\nFailed experiments:")
        for r in results:
            if not r['success']:
                print(f"  - {r['dataset']}/{r['model']} config {r['config_idx']}: {r['error']}")

if __name__ == '__main__':
    main()