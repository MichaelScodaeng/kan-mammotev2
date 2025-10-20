#!/usr/bin/env python3
"""
Quick test script to verify hyperparameter tuning setup.
Runs a single fast experiment to ensure everything works.
"""

import subprocess
import sys
from pathlib import Path

def test_environment():
    """Check if environment is properly set up."""
    print("Checking environment...")
    
    # Check for training script
    scripts = ['experiment_unified.py', 'train_link_prediction.py', 'train.py']
    training_script = None
    for script in scripts:
        if Path(script).exists():
            training_script = script
            print(f"✓ Found training script: {script}")
            break
    
    if not training_script:
        print("✗ No training script found!")
        return False
    
    # Check for utils
    if not Path('utils/load_configs.py').exists():
        print("✗ utils/load_configs.py not found!")
        return False
    print("✓ Found utils/load_configs.py")
    
    # Check for data directory
    if not Path('data').exists():
        print("⚠ Warning: data/ directory not found (may need to download datasets)")
    else:
        print("✓ Found data/ directory")
    
    return True


def run_test_experiment():
    """Run a minimal test experiment."""
    print("\n" + "="*80)
    print("Running test experiment...")
    print("Model: TGAT | Dataset: Contacts | Encoder: lete")
    print("This should take ~2-5 minutes")
    print("="*80 + "\n")
    
    # Find training script
    training_script = None
    for script in ['experiment_unified.py', 'train_link_prediction.py']:
        if Path(script).exists():
            training_script = script
            break
    
    cmd = [
        'python', training_script,
        '--model_name', 'TGAT',
        '--dataset_name', 'Contacts',
        '--time_encoder_type', 'lete',
        '--learning_rate', '0.001',
        '--weight_decay', '0.0',
        '--data_ratio', '0.1',
        '--num_epochs', '3',
        '--patience', '2',
        '--num_runs', '1',
        '--batch_size', '200',
        '--gpu', '0',
        '--save_model_name_suffix', '_HPTUNE_TEST',
        '--ablation_dir', './test_hptune_output',
        '--load_best_configs',
        '--disable_progress_bar',
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            timeout=600,  # 10 minute timeout
            capture_output=False
        )
        
        if result.returncode == 0:
            print("\n" + "="*80)
            print("✓ TEST PASSED! Hyperparameter tuning setup is working.")
            print("="*80)
            print("\nYou can now run the full tuning:")
            print("  Sequential: python tune_hyperparams_fast.py --subset 10")
            print("  Parallel:   python generate_hptune_jobs.py && qsub run_hptune_array.sh")
            return True
        else:
            print("\n" + "="*80)
            print(f"✗ TEST FAILED with exit code {result.returncode}")
            print("="*80)
            print("\nCheck the error messages above.")
            return False
            
    except subprocess.TimeoutExpired:
        print("\n✗ TEST TIMED OUT (>10 minutes)")
        print("This might indicate a problem with data loading or GPU.")
        return False
    except Exception as e:
        print(f"\n✗ TEST ERROR: {e}")
        return False


def main():
    print("="*80)
    print("HYPERPARAMETER TUNING - QUICK TEST")
    print("="*80)
    print()
    
    # Check environment
    if not test_environment():
        print("\n✗ Environment check failed!")
        print("Please ensure you're in the correct directory and have all dependencies.")
        sys.exit(1)
    
    print("\n✓ Environment check passed!")
    
    # Ask to run test
    print("\nThis will run a quick test experiment (~2-5 min).")
    response = input("Continue? [y/N]: ")
    
    if response.lower() != 'y':
        print("Skipped test experiment.")
        print("\nTo run manually:")
        print("  python tune_hyperparams_fast.py --subset 1")
        sys.exit(0)
    
    # Run test
    success = run_test_experiment()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
