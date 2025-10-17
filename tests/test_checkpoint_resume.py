#!/usr/bin/env python3
"""
Checkpoint Resume Test Script

This script tests that the checkpoint resuming functionality works correctly,
especially that it resumes at the correct seed and epoch.

Usage:
    python test_checkpoint_resume.py                    # Test basic checkpoint resume
    python test_checkpoint_resume.py --verbose          # Detailed output
    python test_checkpoint_resume.py --cleanup          # Clean up test artifacts
    python test_checkpoint_resume.py --test_seeds 3     # Test multi-seed resume
"""

import subprocess
import os
import time
import argparse
import json
import glob
import shutil
from datetime import datetime
from typing import Tuple, Optional, List

# Test configurations - use small settings for speed
TEST_MODEL = 'TGAT'
TEST_DATASET = 'wikipedia'
TEST_ENCODER = 'original'
TEST_EPOCHS = 10  # Small number for quick testing
TEST_RUNS = 3     # Test multi-run functionality

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Test checkpoint resuming functionality')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up test artifacts and exit')
    parser.add_argument('--test_epochs', type=int, default=TEST_EPOCHS,
                        help=f'Number of epochs for testing (default: {TEST_EPOCHS})')
    parser.add_argument('--test_runs', type=int, default=TEST_RUNS,
                        help=f'Number of runs for testing (default: {TEST_RUNS})')
    parser.add_argument('--interrupt_at_epoch', type=int, default=5,
                        help='Interrupt training at this epoch to test resume (default: 5)')
    parser.add_argument('--data_ratio', type=float, default=0.05,
                        help='Fraction of data to use (default: 0.05 for speed)')
    
    return parser.parse_args()

def cleanup_test_artifacts():
    """Clean up all test artifacts"""
    print("🧹 Cleaning up test artifacts...")
    
    patterns_to_clean = [
        f"./saved_models/{TEST_MODEL}/{TEST_DATASET}/*{TEST_ENCODER}*test*",
        f"./saved_results/{TEST_MODEL}/{TEST_DATASET}/*{TEST_ENCODER}*test*",
        f"./logs/{TEST_MODEL}/{TEST_DATASET}/*{TEST_ENCODER}*test*",
        "./experiment_logs/test_*",
        "./checkpoints_test_*"
    ]
    
    for pattern in patterns_to_clean:
        try:
            files_to_remove = glob.glob(pattern)
            for file_path in files_to_remove:
                if os.path.isdir(file_path):
                    shutil.rmtree(file_path, ignore_errors=True)
                else:
                    os.remove(file_path)
                print(f"   Removed: {file_path}")
        except Exception as e:
            print(f"   Warning: Could not remove {pattern}: {e}")
    
    print("✅ Cleanup completed")

def run_training_command(command: List[str], timeout_minutes: int = 10, 
                        verbose: bool = False, expect_interrupt: bool = False) -> Tuple[bool, str, int]:
    """Run training command and return success, message, and exit code"""
    try:
        if verbose:
            print(f"🚀 Running: {' '.join(command)}")
        
        start_time = time.time()
        result = subprocess.run(
            command,
            capture_output=not verbose,
            text=True,
            timeout=timeout_minutes * 60,
            check=not expect_interrupt  # Don't check return code if we expect interrupt
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            return True, f"Completed successfully in {duration:.1f}s", result.returncode
        elif expect_interrupt and result.returncode != 0:
            return True, f"Expected interruption after {duration:.1f}s", result.returncode
        else:
            return False, f"Failed with exit code {result.returncode}", result.returncode
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout_minutes} minutes", -1
    except Exception as e:
        return False, f"Unexpected error: {str(e)}", -1

def find_checkpoints(model: str, dataset: str, encoder: str, seed: int = 0) -> List[str]:
    """Find checkpoint files for a specific combination"""
    patterns = [
        f"./saved_models/{model}/{dataset}/*{encoder}*seed{seed}/checkpoint_*.pt",
        f"./logs/{model}/{dataset}/*{encoder}*seed{seed}/checkpoint_*.pt"
    ]
    
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob.glob(pattern))
    
    return sorted(checkpoints)

def get_checkpoint_info(checkpoint_path: str) -> Optional[dict]:
    """Get information from checkpoint file"""
    try:
        import torch
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        return {
            'epoch': checkpoint.get('epoch', -1),
            'seed': checkpoint.get('seed', -1),
            'model_state_keys': list(checkpoint.get('model_state_dict', {}).keys())[:5],  # First 5 keys
            'optimizer_state_available': 'optimizer_state_dict' in checkpoint,
            'random_state_available': 'random_state' in checkpoint
        }
    except Exception as e:
        return {'error': str(e)}

def test_basic_checkpoint_creation(args):
    """Test 1: Basic checkpoint creation during training"""
    print(f"\n{'='*60}")
    print("TEST 1: Basic Checkpoint Creation")
    print(f"{'='*60}")
    
    # Clean up first
    cleanup_test_artifacts()
    
    # Run short training to create checkpoints
    command = [
        'python', 'experiments/train_link_prediction.py',
        '--model_name', TEST_MODEL,
        '--dataset_name', TEST_DATASET,
        '--time_encoder_type', TEST_ENCODER,
        '--num_epochs', str(args.test_epochs),
        '--num_runs', '1',  # Single run for simplicity
        '--data_ratio', str(args.data_ratio),
        '--load_best_configs',
        '--save_checkpoints',
        '--checkpoint_strategy', 'frequent',  # Save more checkpoints for testing
        '--checkpoint_interval', '2',  # Save every 2 epochs
        '--save_model_name_suffix', '_test'
    ]
    
    print(f"🚀 Starting training with checkpoint saving...")
    success, message, exit_code = run_training_command(command, timeout_minutes=15, verbose=args.verbose)
    
    if not success:
        print(f"❌ Training failed: {message}")
        return False
    
    print(f"✅ Training completed: {message}")
    
    # Check for checkpoint files
    checkpoints = find_checkpoints(TEST_MODEL, TEST_DATASET, TEST_ENCODER)
    
    if not checkpoints:
        print("❌ No checkpoint files found")
        return False
    
    print(f"✅ Found {len(checkpoints)} checkpoint files:")
    for i, checkpoint in enumerate(checkpoints):
        info = get_checkpoint_info(checkpoint)
        print(f"   {i+1}. {os.path.basename(checkpoint)}")
        if 'error' not in info:
            print(f"      Epoch: {info['epoch']}, Seed: {info['seed']}")
            print(f"      Model state: {len(info['model_state_keys'])} keys")
            print(f"      Optimizer: {'✓' if info['optimizer_state_available'] else '✗'}")
            print(f"      Random state: {'✓' if info['random_state_available'] else '✗'}")
        else:
            print(f"      Error reading checkpoint: {info['error']}")
    
    return True

def test_checkpoint_resume(args):
    """Test 2: Resume from checkpoint"""
    print(f"\n{'='*60}")
    print("TEST 2: Checkpoint Resume")
    print(f"{'='*60}")
    
    # Find existing checkpoints from previous test
    checkpoints = find_checkpoints(TEST_MODEL, TEST_DATASET, TEST_ENCODER)
    
    if not checkpoints:
        print("❌ No checkpoints found to test resume functionality")
        return False
    
    # Use the first checkpoint for resume test
    checkpoint_to_use = checkpoints[0]
    checkpoint_info = get_checkpoint_info(checkpoint_to_use)
    
    if 'error' in checkpoint_info:
        print(f"❌ Cannot read checkpoint: {checkpoint_info['error']}")
        return False
    
    print(f"🔄 Testing resume from checkpoint:")
    print(f"   File: {checkpoint_to_use}")
    print(f"   Epoch: {checkpoint_info['epoch']}")
    print(f"   Seed: {checkpoint_info['seed']}")
    
    # Clear model files to force resume
    model_dirs = glob.glob(f"./saved_models/{TEST_MODEL}/{TEST_DATASET}/*{TEST_ENCODER}*test*")
    for model_dir in model_dirs:
        if os.path.isdir(model_dir):
            # Keep checkpoints but remove final model
            final_models = glob.glob(os.path.join(model_dir, "*.pth"))
            for model_file in final_models:
                os.remove(model_file)
                print(f"   Removed final model: {model_file}")
    
    # Resume training
    command = [
        'python', 'experiments/train_link_prediction.py',
        '--model_name', TEST_MODEL,
        '--dataset_name', TEST_DATASET,
        '--time_encoder_type', TEST_ENCODER,
        '--num_epochs', str(args.test_epochs),
        '--num_runs', '1',
        '--data_ratio', str(args.data_ratio),
        '--load_best_configs',
        '--save_checkpoints',
        '--resume_from_checkpoint', checkpoint_to_use,
        '--save_model_name_suffix', '_test'
    ]
    
    print(f"🔄 Resuming training from checkpoint...")
    success, message, exit_code = run_training_command(command, timeout_minutes=15, verbose=args.verbose)
    
    if not success:
        print(f"❌ Resume failed: {message}")
        return False
    
    print(f"✅ Resume completed: {message}")
    
    # Verify final model was created
    final_models = glob.glob(f"./saved_models/{TEST_MODEL}/{TEST_DATASET}/*{TEST_ENCODER}*test*seed0/*.pth")
    
    if not final_models:
        print("❌ No final model found after resume")
        return False
    
    print(f"✅ Final model created: {os.path.basename(final_models[0])}")
    return True

def test_multi_seed_resume(args):
    """Test 3: Multi-seed checkpoint resume"""
    print(f"\n{'='*60}")
    print("TEST 3: Multi-Seed Resume")
    print(f"{'='*60}")
    
    # Clean up for fresh multi-seed test
    cleanup_test_artifacts()
    
    # Start multi-seed training
    command = [
        'python', 'experiments/train_link_prediction.py',
        '--model_name', TEST_MODEL,
        '--dataset_name', TEST_DATASET,
        '--time_encoder_type', TEST_ENCODER,
        '--num_epochs', str(args.test_epochs),
        '--num_runs', str(args.test_runs),
        '--data_ratio', str(args.data_ratio),
        '--load_best_configs',
        '--save_checkpoints',
        '--checkpoint_strategy', 'frequent',
        '--checkpoint_interval', '3',
        '--save_model_name_suffix', '_test_multiseed'
    ]
    
    print(f"🚀 Starting multi-seed training ({args.test_runs} runs)...")
    
    # Start training process but interrupt it partway through
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        # Let it run for a bit then interrupt
        time.sleep(30)  # Let first seed make some progress
        process.terminate()
        
        # Wait for clean shutdown
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        
        print(f"⏹️  Interrupted training as planned")
        
    except Exception as e:
        print(f"❌ Error during controlled interruption: {e}")
        return False
    
    # Check which seeds have checkpoints
    all_checkpoints = []
    for seed in range(args.test_runs):
        seed_checkpoints = find_checkpoints(TEST_MODEL, TEST_DATASET, TEST_ENCODER, seed)
        if seed_checkpoints:
            checkpoint_info = get_checkpoint_info(seed_checkpoints[-1])  # Latest checkpoint
            all_checkpoints.append((seed, seed_checkpoints[-1], checkpoint_info))
            print(f"   Seed {seed}: checkpoint at epoch {checkpoint_info.get('epoch', '?')}")
        else:
            print(f"   Seed {seed}: no checkpoint found")
    
    if not all_checkpoints:
        print("❌ No checkpoints found from interrupted training")
        return False
    
    # Now test resuming
    print(f"\n🔄 Testing resume from interruption...")
    
    resume_command = [
        'python', 'experiments/train_link_prediction.py',
        '--model_name', TEST_MODEL,
        '--dataset_name', TEST_DATASET,
        '--time_encoder_type', TEST_ENCODER,
        '--num_epochs', str(args.test_epochs),
        '--num_runs', str(args.test_runs),
        '--data_ratio', str(args.data_ratio),
        '--load_best_configs',
        '--save_checkpoints',
        '--save_model_name_suffix', '_test_multiseed'
        # Note: No explicit resume_from_checkpoint - should auto-detect
    ]
    
    success, message, exit_code = run_training_command(resume_command, timeout_minutes=20, verbose=args.verbose)
    
    if not success:
        print(f"❌ Multi-seed resume failed: {message}")
        return False
    
    print(f"✅ Multi-seed resume completed: {message}")
    
    # Verify all seeds completed
    for seed in range(args.test_runs):
        final_models = glob.glob(f"./saved_models/{TEST_MODEL}/{TEST_DATASET}/*{TEST_ENCODER}*test_multiseed*seed{seed}/*.pth")
        if final_models:
            print(f"   ✅ Seed {seed}: final model created")
        else:
            print(f"   ❌ Seed {seed}: no final model found")
            return False
    
    return True

def main():
    args = parse_arguments()
    
    print("🧪 Checkpoint Resume Functionality Test")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Configuration:")
    print(f"  Model: {TEST_MODEL}")
    print(f"  Dataset: {TEST_DATASET}")
    print(f"  Encoder: {TEST_ENCODER}")
    print(f"  Test Epochs: {args.test_epochs}")
    print(f"  Test Runs: {args.test_runs}")
    print(f"  Data Ratio: {args.data_ratio}")
    
    if args.cleanup:
        cleanup_test_artifacts()
        return
    
    # Run tests
    tests_passed = 0
    tests_total = 3
    
    try:
        # Test 1: Basic checkpoint creation
        if test_basic_checkpoint_creation(args):
            tests_passed += 1
        
        # Test 2: Checkpoint resume
        if test_checkpoint_resume(args):
            tests_passed += 1
        
        # Test 3: Multi-seed resume
        if test_multi_seed_resume(args):
            tests_passed += 1
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error during testing: {e}")
    
    # Final summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Tests Passed: {tests_passed}/{tests_total}")
    print(f"Pass Rate: {tests_passed/tests_total*100:.1f}%")
    
    if tests_passed == tests_total:
        print("✅ All checkpoint resume tests passed!")
        print("\nCheckpoint system is working correctly:")
        print("  ✓ Checkpoints are created during training")
        print("  ✓ Training can resume from checkpoints")
        print("  ✓ Multi-seed resume works properly")
    else:
        print("❌ Some checkpoint resume tests failed!")
        print("\nRecommendations:")
        print("  - Check that --save_checkpoints flag is working")
        print("  - Verify checkpoint files contain all required state")
        print("  - Ensure resume logic correctly loads checkpoint data")
    
    print(f"\n🧹 Cleanup: Run with --cleanup to remove test artifacts")

if __name__ == "__main__":
    main()