#!/usr/bin/env python3
"""
Quick test script to run a single hyperparameter tuning experiment.
Useful for debugging before running full sequential tuning.
"""

import subprocess
import sys

def main():
    # Single test configuration
    cmd = [
        'python', 'experiments/train_link_prediction.py',
        '--dataset_name', 'wikipedia',
        '--model_name', 'TGAT',
        '--time_encoder_type', 'kan_mammote_dual_kmote',
        '--gpu', '0',
        
        # Test configuration
        '--expert_dim', '128',
        '--mamba_d_state', '256',
        '--mamba_expand', '4',
        '--dropout', '0.1',
        '--mamba_headdim', '64',
        '--mamba_d_conv', '4',
        
        # Fast tuning
        '--data_ratio', '0.1',
        '--train_only_ratio',
        '--num_epochs', '10',
        '--patience', '3',
        '--num_runs', '1',
        '--seed', '0',
        '--test_interval_epochs', '1',
        
        # Output
        '--save_model_name_suffix', 'test_hptune',
        '--ablation_dir', './test_hptune_output'
    ]
    
    print("Testing single hyperparameter tuning experiment...")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✅ Test successful! You can now run full sequential tuning.")
    else:
        print("\n❌ Test failed! Check the error above.")
        sys.exit(1)

if __name__ == '__main__':
    main()