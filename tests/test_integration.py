"""
Integration Test Script for Time Encoder System

This script tests that all time encoders work correctly with all models (except CAWN)
and datasets, and can be evaluated with different negative sampling strategies.

Features:
- Tests all time encoder types
- Tests all models except CAWN
- Tests subset of datasets (representative sample)
- Tests all negative sampling strategies
- Tests multi-run functionality
- Uses minimal epochs for speed
- Provides detailed progress and error reporting

Usage:
    python test_integration.py                    # Run full test suite (10% data)
    python test_integration.py --quick            # Quick test (fewer combinations, 10% data)
    python test_integration.py --data_ratio 0.05  # Use only 5% of data (super fast)
    python test_integration.py --models TGAT      # Test specific model only
    python test_integration.py --encoders kan_mammote original  # Test specific encoders
    python test_integration.py --dry_run          # Show what would be tested
"""

import subprocess
import itertools
import os
import time
import argparse
import sys
import logging
from datetime import datetime
from typing import List, Dict, Tuple

# Test configurations
ALL_TIME_ENCODERS = ['mercer', 'bochner', 'time2vec',"lete" ] # 'original',
ALL_MODELS = ['TGAT', 'JODIE', 'TGN', 'GraphMixer', 'DyGFormer', 'DyGMamba', 'TCL']  # Exclude CAWN as requested
ALL_DATASETS = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
                                'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']
ALL_NEG_STRATEGIES = ['random', 'historical', 'inductive']

# Quick test configurations (reduced for speed)
QUICK_DATASETS = ALL_DATASETS# Representative subset
QUICK_MODELS = ALL_MODELS  # Cover different model types
QUICK_ENCODERS = ['kan_mammote_dual_kmote']    # Cover main encoder types

def setup_logging(log_dir: str = "test_logs"):
    """Setup logging to save test output to files with timestamps"""
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create timestamped log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f"integration_test_{timestamp}.log")
    
    # Setup logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Still show in terminal too
        ]
    )
    
    return log_file

def run_command_with_logging(command: List[str], test_name: str, timeout_minutes: int, 
                           verbose: bool = False, log_dir: str = "test_logs") -> subprocess.CompletedProcess:
    """Run command and save output to both log file and return result"""
    
    # Create command-specific log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_test_name = test_name.replace('/', '_').replace(' ', '_')
    cmd_log_file = os.path.join(log_dir, f"{safe_test_name}_{timestamp}.log")
    
    logging.info(f"Starting command for {test_name}")
    logging.info(f"Command: {' '.join(command)}")
    logging.info(f"Command output will be saved to: {cmd_log_file}")
    
    if verbose:
        print(f"   📝 Command log: {cmd_log_file}")
    
    try:
        # Run command and capture output
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60,
            check=True
        )
        
        # Save command output to file
        with open(cmd_log_file, 'w') as f:
            f.write(f"Command: {' '.join(command)}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Test: {test_name}\n")
            f.write("=" * 80 + "\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n" + "=" * 80 + "\n")
            f.write("STDERR:\n")
            f.write(result.stderr)
            f.write(f"\nReturn code: {result.returncode}\n")
        
        logging.info(f"Command completed successfully for {test_name}")
        
        if verbose:
            print(f"   ✅ Command output saved to: {cmd_log_file}")
        
        return result
        
    except subprocess.TimeoutExpired as e:
        # Save timeout info to log file
        with open(cmd_log_file, 'w') as f:
            f.write(f"Command: {' '.join(command)}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Test: {test_name}\n")
            f.write("=" * 80 + "\n")
            f.write(f"TIMEOUT after {timeout_minutes} minutes\n")
            f.write("STDOUT (partial):\n")
            f.write(e.stdout if e.stdout else "No stdout captured")
            f.write("\n" + "=" * 80 + "\n")
            f.write("STDERR (partial):\n")
            f.write(e.stderr if e.stderr else "No stderr captured")
        
        logging.error(f"Command timeout for {test_name} after {timeout_minutes} minutes")
        raise e
        
    except subprocess.CalledProcessError as e:
        # Save error info to log file
        with open(cmd_log_file, 'w') as f:
            f.write(f"Command: {' '.join(command)}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Test: {test_name}\n")
            f.write("=" * 80 + "\n")
            f.write(f"FAILED with return code: {e.returncode}\n")
            f.write("STDOUT:\n")
            f.write(e.stdout if e.stdout else "No stdout captured")
            f.write("\n" + "=" * 80 + "\n")
            f.write("STDERR:\n")
            f.write(e.stderr if e.stderr else "No stderr captured")
        
        logging.error(f"Command failed for {test_name} with code {e.returncode}")
        raise e

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors: List[str] = []
        self.start_time = time.time()
    
    def add_pass(self, test_name: str):
        self.passed += 1
        print(f"✅ PASS: {test_name}")
    
    def add_fail(self, test_name: str, error: str):
        self.failed += 1
        self.errors.append(f"{test_name}: {error}")
        print(f"❌ FAIL: {test_name}")
        print(f"   Error: {error}")
        #raise Exception(f"Test failed: {test_name}")
    
    def get_summary(self) -> str:
        duration = time.time() - self.start_time
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0
        
        summary = f"""
{'='*80}
INTEGRATION TEST SUMMARY
{'='*80}
Total Tests: {total}
Passed: {self.passed}
Failed: {self.failed}
Pass Rate: {pass_rate:.1f}%
Duration: {duration:.1f} seconds
{'='*80}
"""
        if self.errors:
            summary += "\nFAILED TESTS:\n"
            for error in self.errors:
                summary += f"❌ {error}\n"
        
        return summary

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Integration test for time encoder system')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with reduced combinations')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS, 
                        help='Test specific models only')
    parser.add_argument('--datasets', nargs='+', choices=ALL_DATASETS,
                        help='Test specific datasets only')
    parser.add_argument('--encoders', nargs='+', choices=ALL_TIME_ENCODERS,
                        help='Test specific encoders only')
    parser.add_argument('--neg_strategies', nargs='+', choices=ALL_NEG_STRATEGIES,
                        help='Test specific negative sampling strategies only')
    parser.add_argument('--test_epochs', type=int, default=2,
                        help='Number of epochs for training tests (default: 2)')
    parser.add_argument('--test_runs', type=int, default=2,
                        help='Number of runs for multi-run test (default: 2)')
    parser.add_argument('--timeout_minutes', type=int, default=10,
                        help='Timeout per test in minutes (default: 10)')
    parser.add_argument('--data_ratio', type=float, default=0.1,
                        help='Fraction of data to use for testing (default: 0.1 = 10%% of data)')
    parser.add_argument('--max_retries', type=int, default=2,
                        help='Maximum number of retries for failed tests (default: 2)')
    parser.add_argument('--cleanup_on_retry', action='store_true', default=True,
                        help='Clean up artifacts before retrying failed tests')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be tested without running')
    parser.add_argument('--verbose', action='store_true',
                        help='Show detailed output for each test')
    parser.add_argument('--log_dir', type=str, default='test_logs',
                        help='Directory to save log files (default: test_logs)')
    
    return parser.parse_args()

def get_test_combinations(args):
    """Get the test combinations based on arguments"""
    if args.quick:
        models = args.models or QUICK_MODELS
        datasets = args.datasets or QUICK_DATASETS  
        encoders = args.encoders or QUICK_ENCODERS
        neg_strategies = ALL_NEG_STRATEGIES  # Only test random for quick
    else:
        models = args.models or ALL_MODELS
        datasets = args.datasets or ALL_DATASETS
        encoders = args.encoders or ALL_TIME_ENCODERS
        neg_strategies = args.neg_strategies or ALL_NEG_STRATEGIES
    
    return models, datasets, encoders, neg_strategies

def run_training_test(model: str, dataset: str, encoder: str, test_epochs: int, 
                     timeout_minutes: int, data_ratio: float = 0.1, max_retries: int = 2, verbose: bool = False) -> Tuple[bool, str]:
    """Test training with specific configuration, with retry on failure"""
    
    for attempt in range(max_retries + 1):  # 0, 1, 2 (total 3 attempts)
        try:
            if attempt > 0:
                print(f"   🔄 Retry attempt {attempt}/{max_retries} for {model}-{dataset}-{encoder}")
                logging.info(f"Retry attempt {attempt}/{max_retries} for {model}-{dataset}-{encoder}")
                
                # Clean up any partial artifacts from previous failed attempt
                cleanup_failed_training_artifacts(model, dataset, encoder)
            
            command = [
                'python', 'experiments/train_link_prediction.py',
                '--model_name', model,
                '--dataset_name', dataset, 
                '--time_encoder_type', encoder,
                '--num_epochs', str(test_epochs),
                '--num_runs', '2',  # Train 2 seeds for testing
                '--data_ratio', str(data_ratio),  # Use small fraction of data
                '--load_best_configs',
                '--save_checkpoints',
                '--checkpoint_strategy', 'minimal'
            ]
            
            # Add encoder-specific arguments
            encoder_args = get_encoder_args(encoder)
            if encoder_args:
                command.extend(encoder_args.split())
            
            test_name = f"Training-{encoder}-{model}-{dataset}"
            
            # Run command with logging
            result = run_command_with_logging(
                command, test_name, timeout_minutes, verbose
            )
            
            # Check if models were saved for all runs
            import glob
            all_models_found = True
            found_models = []
            
            for run in range(2):  # Check both seed0 and seed1
                # Try multiple possible patterns to find the model files
                possible_patterns = [
                    f"./saved_models/{model}/{dataset}/*{encoder}*seed{run}/*.pth",  # .pth files
                    f"./saved_models/{model}/{dataset}/*{encoder}*seed{run}/*.pkl",  # .pkl files  
                    f"./saved_models/{model}/{dataset}/{model}_{encoder}_seed{run}/*.pth",  # Specific pattern with .pth
                    f"./saved_models/{model}/{dataset}/{model}_{encoder}_seed{run}/*.pkl",  # Specific pattern with .pkl
                ]
                
                model_found = False
                for pattern in possible_patterns:
                    model_files = glob.glob(pattern)
                    if model_files:
                        found_models.extend(model_files)
                        model_found = True
                        break
                
                if not model_found:
                    all_models_found = False
                    msg = f"No model file found for seed {run}"
                    print(f"   ⚠️  {msg}")
                    logging.warning(f"{test_name}: {msg}")
                    
                    # Debug: Show what files exist
                    if verbose:
                        debug_pattern = f"./saved_models/{model}/{dataset}/*{encoder}*seed{run}"
                        debug_dirs = glob.glob(debug_pattern)
                        debug_msg = f"Debug - directories matching {debug_pattern}: {debug_dirs}"
                        print(f"   🔍 {debug_msg}")
                        logging.debug(f"{test_name}: {debug_msg}")
                        if debug_dirs:
                            for debug_dir in debug_dirs:
                                files_in_dir = glob.glob(f"{debug_dir}/*")
                                files_msg = f"Files in {debug_dir}: {files_in_dir}"
                                print(f"   🔍 {files_msg}")
                                logging.debug(f"{test_name}: {files_msg}")
            
            if not all_models_found:
                if attempt < max_retries:
                    msg = f"Some model files missing on attempt {attempt + 1}, retrying..."
                    print(f"   ⚠️  {msg}")
                    logging.warning(f"{test_name}: {msg}")
                    continue
                else:
                    error_msg = f"Model files missing after {max_retries + 1} attempts"
                    logging.error(f"{test_name}: {error_msg}")
                    return False, error_msg
            
            # Verify model files are valid
            try:
                for model_file in found_models:
                    if model_file.endswith('.pth'):
                        try:
                            import torch
                            torch.load(model_file, map_location='cpu')
                        except ImportError:
                            warning_msg = "Warning: PyTorch not available, skipping .pth validation"
                            print(f"   ⚠️  {warning_msg}")
                            logging.warning(f"{test_name}: {warning_msg}")
                    elif model_file.endswith('.pkl'):
                        # PyTorch models saved as .pkl still need torch.load()
                        try:
                            import torch
                            torch.load(model_file, map_location='cpu')
                        except ImportError:
                            warning_msg = "Warning: PyTorch not available, trying pickle for .pkl file"
                            print(f"   ⚠️  {warning_msg}")
                            logging.warning(f"{test_name}: {warning_msg}")
                            import pickle
                            with open(model_file, 'rb') as f:
                                pickle.load(f)
                    else:
                        # For any other extension, try torch.load first, then pickle as fallback
                        try:
                            import torch
                            torch.load(model_file, map_location='cpu')
                        except ImportError:
                            import pickle
                            with open(model_file, 'rb') as f:
                                pickle.load(f)
                        except:
                            import pickle
                            with open(model_file, 'rb') as f:
                                pickle.load(f)
                            
                success_msg = f"Training succeeded on attempt {attempt + 1}"
                models_msg = f"Found models: {[os.path.basename(f) for f in found_models]}"
                print(f"   ✅ {success_msg}")
                print(f"   📁 {models_msg}")
                logging.info(f"{test_name}: {success_msg}")
                logging.info(f"{test_name}: {models_msg}")
                return True, f"Training completed successfully (attempt {attempt + 1})"
                
            except Exception as e:
                if attempt < max_retries:
                    error_msg = f"Model file validation failed on attempt {attempt + 1}: {e}"
                    print(f"   ⚠️  {error_msg}")
                    logging.warning(f"{test_name}: {error_msg}")
                    continue
                else:
                    final_error = f"Model file validation failed after {max_retries + 1} attempts: {str(e)}"
                    logging.error(f"{test_name}: {final_error}")
                    return False, final_error
                
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                timeout_msg = f"Training timeout on attempt {attempt + 1}, retrying..."
                print(f"   ⚠️  {timeout_msg}")
                logging.warning(f"{test_name}: {timeout_msg}")
                continue
            else:
                final_timeout = f"Training timeout after {max_retries + 1} attempts ({timeout_minutes}min each)"
                logging.error(f"{test_name}: {final_timeout}")
                return False, final_timeout
                
        except subprocess.CalledProcessError as e:
            if attempt < max_retries:
                process_error = f"Training failed on attempt {attempt + 1} (code {e.returncode}), retrying..."
                print(f"   ⚠️  {process_error}")
                logging.warning(f"{test_name}: {process_error}")
                continue
            else:
                final_process_error = f"Training failed after {max_retries + 1} attempts with code {e.returncode}"
                logging.error(f"{test_name}: {final_process_error}")
                return False, final_process_error
                
        except Exception as e:
            if attempt < max_retries:
                unexpected_error = f"Unexpected error on attempt {attempt + 1}: {str(e)}, retrying..."
                print(f"   ⚠️  {unexpected_error}")
                logging.warning(f"{test_name}: {unexpected_error}")
                continue
            else:
                final_unexpected = f"Unexpected error after {max_retries + 1} attempts: {str(e)}"
                logging.error(f"{test_name}: {final_unexpected}")
                return False, final_unexpected
    
    return False, "Should not reach here"

def run_evaluation_test(model: str, dataset: str, encoder: str, neg_strategy: str,
                       timeout_minutes: int, data_ratio: float = 0.1, max_retries: int = 1, verbose: bool = False) -> Tuple[bool, str]:
    """Test evaluation with specific configuration, with retry on failure"""
    
    test_name = f"Evaluation-{encoder}-{model}-{dataset}-{neg_strategy}"
    
    # First check if model file exists (prerequisite for evaluation)
    import glob
    model_found = False
    possible_patterns = [
        f"./saved_models/{model}/{dataset}/*{encoder}*seed0/*.pth",  # .pth files
        f"./saved_models/{model}/{dataset}/*{encoder}*seed0/*.pkl",  # .pkl files  
        f"./saved_models/{model}/{dataset}/{model}_{encoder}_seed0/*.pth",  # Specific pattern with .pth
        f"./saved_models/{model}/{dataset}/{model}_{encoder}_seed0/*.pkl",  # Specific pattern with .pkl
    ]
    
    for pattern in possible_patterns:
        model_files = glob.glob(pattern)
        if model_files:
            model_found = True
            break
    
    if not model_found:
        error_msg = "No trained model found - training may have failed"
        logging.error(f"{test_name}: {error_msg}")
        return False, error_msg
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                retry_msg = f"Evaluation retry attempt {attempt}/{max_retries} for {model}-{dataset}-{encoder}-{neg_strategy}"
                print(f"   🔄 {retry_msg}")
                logging.info(f"{test_name}: {retry_msg}")
            
            command = [
                'python', 'experiments/evaluate_link_prediction.py',
                '--model_name', model,
                '--dataset_name', dataset,
                '--time_encoder_type', encoder,
                '--negative_sample_strategy', neg_strategy,
                '--num_runs', '2',  # Evaluate both runs that were trained
                '--data_ratio', str(data_ratio),  # Use same data ratio as training
                '--load_best_configs'
            ]
            
            # Run command with logging
            result = run_command_with_logging(
                command, test_name, timeout_minutes, verbose
            )
            
            # Check if results were saved for all runs
            import glob
            all_results_found = True
            found_results = []
            
            for run in range(2):  # Check both seed0 and seed1
                result_pattern = f"./saved_results/{model}/{dataset}/*{neg_strategy}*{encoder}*seed{run}*.json"
                result_files = glob.glob(result_pattern)
                
                if not result_files:
                    all_results_found = False
                    print(f"   ⚠️  No result file found for seed {run}")
                    
                    # Debug: Show what files exist
                    if verbose:
                        debug_pattern = f"./saved_results/{model}/{dataset}/*{neg_strategy}*{encoder}*"
                        debug_files = glob.glob(debug_pattern)
                        print(f"   🔍 Debug - result files matching pattern: {debug_files}")
                else:
                    found_results.extend(result_files)
            
            if not all_results_found:
                if attempt < max_retries:
                    print(f"   ⚠️  Some result files missing on attempt {attempt + 1}, retrying...")
                    continue
                else:
                    return False, f"Result files missing after {max_retries + 1} attempts"
            
            # Verify result files contain expected content
            try:
                import json
                for result_file in found_results:
                    with open(result_file, 'r') as f:
                        data = json.load(f)
                        
                        # Check that all expected metric sections exist
                        required_sections = ['test metrics', 'new node test metrics']
                        # Memory models don't have validation metrics
                        if model not in ['JODIE', 'DyRep', 'TGN']:
                            required_sections.extend(['validate metrics', 'new node validate metrics'])
                        
                        for section in required_sections:
                            if section not in data:
                                raise ValueError(f"Missing section: {section}")
                        
                        # Check that metrics have reasonable values
                        for section in required_sections:
                            metrics = data[section]
                            for metric_name, metric_value in metrics.items():
                                try:
                                    value = float(metric_value)
                                    if not (0.0 <= value <= 1.0):
                                        print(f"   ⚠️  Warning: {metric_name} = {value} outside [0,1] range")
                                except ValueError:
                                    raise ValueError(f"Invalid metric value: {metric_name} = {metric_value}")
                        
                print(f"   ✅ Evaluation succeeded on attempt {attempt + 1}")
                print(f"   📁 Found results: {[os.path.basename(f) for f in found_results]}")
                return True, f"Evaluation completed successfully (attempt {attempt + 1})"
                
            except Exception as e:
                if attempt < max_retries:
                    print(f"   ⚠️  Result file validation failed on attempt {attempt + 1}: {e}")
                    continue
                else:
                    return False, f"Result file validation failed after {max_retries + 1} attempts: {str(e)}"
            
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                print(f"   ⚠️  Evaluation timeout on attempt {attempt + 1}, retrying...")
                continue
            else:
                return False, f"Evaluation timeout after {max_retries + 1} attempts ({timeout_minutes}min each)"
                
        except subprocess.CalledProcessError as e:
            if attempt < max_retries:
                print(f"   ⚠️  Evaluation failed on attempt {attempt + 1} (code {e.returncode}), retrying...")
                continue
            else:
                return False, f"Evaluation failed after {max_retries + 1} attempts with code {e.returncode}"
                
        except Exception as e:
            if attempt < max_retries:
                print(f"   ⚠️  Unexpected error on attempt {attempt + 1}: {str(e)}, retrying...")
                continue
            else:
                return False, f"Unexpected error after {max_retries + 1} attempts: {str(e)}"
    
    return False, "Should not reach here"

def run_multi_run_test(model: str, dataset: str, encoder: str, num_runs: int,
                      test_epochs: int, timeout_minutes: int, data_ratio: float = 0.1, verbose: bool = False) -> Tuple[bool, str]:
    """Test multi-run functionality"""
    try:
        command = [
            'python', 'experiments/train_link_prediction.py',
            '--model_name', model,
            '--dataset_name', dataset,
            '--time_encoder_type', encoder,
            '--num_epochs', str(test_epochs),
            '--num_runs', str(num_runs),
            '--data_ratio', str(data_ratio),  # Use small fraction of data
            '--load_best_configs',
            '--save_checkpoints',
            '--checkpoint_strategy', 'minimal'
        ]
        
        # Add encoder-specific arguments
        encoder_args = get_encoder_args(encoder)
        if encoder_args:
            command.extend(encoder_args.split())
        
        if verbose:
            print(f"   Command: {' '.join(command)}")
        
        result = subprocess.run(
            command,
            capture_output=not verbose,
            text=True,
            timeout=timeout_minutes * 60 * num_runs,  # Scale timeout by number of runs
            check=True
        )
        
        # Check if models for all runs were saved
        import glob
        all_runs_successful = True
        found_models = []
        
        for run in range(num_runs):
            # Try multiple possible patterns to find the model files
            possible_patterns = [
                f"./saved_models/{model}/{dataset}/*{encoder}*seed{run}/*.pth",  # .pth files
                f"./saved_models/{model}/{dataset}/*{encoder}*seed{run}/*.pkl",  # .pkl files  
                f"./saved_models/{model}/{dataset}/{model}_{encoder}_seed{run}/*.pth",  # Specific pattern with .pth
                f"./saved_models/{model}/{dataset}/{model}_{encoder}_seed{run}/*.pkl",  # Specific pattern with .pkl
            ]
            
            model_found = False
            for pattern in possible_patterns:
                model_files = glob.glob(pattern)
                if model_files:
                    found_models.extend(model_files)
                    model_found = True
                    break
            
            if not model_found:
                all_runs_successful = False
                print(f"   ⚠️  No model file found for run {run}")
                
                # Debug: Show what files exist
                if verbose:
                    debug_pattern = f"./saved_models/{model}/{dataset}/*{encoder}*seed{run}"
                    debug_dirs = glob.glob(debug_pattern)
                    print(f"   🔍 Debug - directories matching {debug_pattern}: {debug_dirs}")
                    if debug_dirs:
                        for debug_dir in debug_dirs:
                            files_in_dir = glob.glob(f"{debug_dir}/*")
                            print(f"   🔍 Files in {debug_dir}: {files_in_dir}")
        
        if not all_runs_successful:
            return False, f"Missing model files for some runs"
        
        print(f"   📁 Found models for all {num_runs} runs: {[os.path.basename(f) for f in found_models]}")
        return True, f"Multi-run ({num_runs} runs) completed successfully"
        
    except subprocess.TimeoutExpired:
        return False, f"Multi-run timeout ({timeout_minutes * num_runs}min)"
    except subprocess.CalledProcessError as e:
        return False, f"Multi-run failed with code {e.returncode}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def get_encoder_args(encoder: str) -> str:
    """Get encoder-specific arguments"""
    encoder_args = {
        'kan_mammote': '--num_mixtures 12 --mamba_d_state 16 --mamba_d_conv 4 --mamba_expand 2 --mamba_headdim 64 --sort_neighbors_by_time',
        'kan_mammote_lite': '--num_mixtures 12 --sort_neighbors_by_time',
        'lete': '',
        'mercer': '',
        'bochner': '',
        'time2vec': '',
        'original': ''
    }
    return encoder_args.get(encoder, '')

def cleanup_failed_training_artifacts(model: str, dataset: str, encoder: str):
    """Clean up artifacts from failed training attempts"""
    import shutil
    import glob
    
    try:
        # Remove potentially corrupted model directories for both seeds
        for seed in range(2):  # Clean up both seed0 and seed1
            model_patterns = [
                f"./saved_models/{model}/{dataset}/*{encoder}*seed{seed}",
                f"./saved_models/{model}/{dataset}/{model}_{encoder}_seed{seed}"
            ]
            
            for pattern in model_patterns:
                model_dirs = glob.glob(pattern)
                for model_dir in model_dirs:
                    if os.path.exists(model_dir):
                        shutil.rmtree(model_dir, ignore_errors=True)
                        print(f"   🧹 Cleaned up: {model_dir}")
        
        # Remove potentially corrupted log directories
        log_patterns = [
            f"./logs/{model}/{dataset}/*{encoder}*seed*",
            f"./logs/{model}/{dataset}/{model}_{encoder}_seed*"
        ]
        
        for pattern in log_patterns:
            log_dirs = glob.glob(pattern)
            for log_dir in log_dirs:
                if os.path.exists(log_dir):
                    shutil.rmtree(log_dir, ignore_errors=True)
                    print(f"   🧹 Cleaned up: {log_dir}")
                
    except Exception as e:
        print(f"   ⚠️  Could not clean up artifacts: {e}")

def cleanup_test_artifacts():
    """Clean up test artifacts to save space"""
    import shutil
    import glob
    
    # Remove test model files (keep only the latest few)
    model_dirs = glob.glob("./saved_models/*/*")
    for model_dir in model_dirs:
        if "test" in model_dir.lower() or len(glob.glob(model_dir + "/*")) > 10:
            try:
                shutil.rmtree(model_dir, ignore_errors=True)
            except:
                pass
    
    # Remove old result files
    result_files = glob.glob("./saved_results/*/*/*.json")
    if len(result_files) > 50:  # Keep only recent results
        result_files.sort(key=os.path.getmtime)
        for old_file in result_files[:-50]:
            try:
                os.remove(old_file)
            except:
                pass

def main():
    args = parse_arguments()
    result = TestResult()
    
    # Setup logging
    main_log_file = setup_logging(args.log_dir)
    
    print("🧪 Starting Integration Test for Time Encoder System")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📝 Main log file: {main_log_file}")
    print(f"📂 Command logs directory: {args.log_dir}")
    
    logging.info("Integration test started")
    logging.info(f"Log directory: {args.log_dir}")
    
    models, datasets, encoders, neg_strategies = get_test_combinations(args)
    
    print(f"\nTest Configuration:")
    print(f"  Models: {models}")
    print(f"  Datasets: {datasets}")
    print(f"  Encoders: {encoders}")
    print(f"  Negative Strategies: {neg_strategies}")
    print(f"  Training Epochs: {args.test_epochs}")
    print(f"  Multi-run Count: {args.test_runs}")
    print(f"  Data Ratio: {args.data_ratio} ({args.data_ratio*100:.1f}% of full dataset)")
    print(f"  Timeout: {args.timeout_minutes} minutes per test")
    print(f"  Quick Mode: {args.quick}")
    
    # Calculate total tests
    training_tests = len(models) * len(datasets) * len(encoders)
    evaluation_tests = len(models) * len(datasets) * len(encoders) * len(neg_strategies)
    multirun_tests = len(QUICK_MODELS) * len(QUICK_DATASETS[:1]) * len(QUICK_ENCODERS[:2])  # Limited for speed
    total_tests = training_tests + evaluation_tests + multirun_tests
    
    print(f"\nTotal Tests Planned:")
    print(f"  Training Tests: {training_tests}")
    print(f"  Evaluation Tests: {evaluation_tests}")
    print(f"  Multi-run Tests: {multirun_tests}")
    print(f"  Total: {total_tests}")
    print(f"  Estimated Duration: {total_tests * args.timeout_minutes / 6:.1f} - {total_tests * args.timeout_minutes / 3:.1f} minutes")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No tests will be executed")
        return
    
    print(f"\n{'='*80}")
    print("PHASE 1: TRAINING TESTS")
    print(f"{'='*80}")
    
    # Test 1: Training with all encoder/model/dataset combinations
    test_count = 0
    for encoder, model, dataset in itertools.product(encoders, models, datasets):
        test_count += 1
        test_name = f"Training-{encoder}-{model}-{dataset}"
        print(f"\n[{test_count}/{training_tests}] Testing: {test_name}")
        
        success, message = run_training_test(
            model, dataset, encoder, args.test_epochs, 
            args.timeout_minutes, args.data_ratio, args.max_retries, args.verbose
        )
        
        if success:
            result.add_pass(test_name)
        else:
            result.add_fail(test_name, message)
    
    print(f"\n{'='*80}")
    print("PHASE 2: EVALUATION TESTS")
    print(f"{'='*80}")
    
    # Test 2: Evaluation with all negative sampling strategies
    test_count = 0
    for encoder, model, dataset, neg_strategy in itertools.product(encoders, models, datasets, neg_strategies):
        test_count += 1
        test_name = f"Evaluation-{encoder}-{model}-{dataset}-{neg_strategy}"
        print(f"\n[{test_count}/{evaluation_tests}] Testing: {test_name}")
        
        success, message = run_evaluation_test(
            model, dataset, encoder, neg_strategy,
            args.timeout_minutes, args.data_ratio, args.max_retries, args.verbose
        )
        
        if success:
            result.add_pass(test_name)
        else:
            result.add_fail(test_name, message)
    
    print(f"\n{'='*80}")
    if not True:
        print("PHASE 3: MULTI-RUN TESTS")
        print(f"{'='*80}")
        
        # Test 3: Multi-run functionality (limited combinations for speed)
        test_count = 0
        for encoder, model, dataset in itertools.product(QUICK_ENCODERS[:2], QUICK_MODELS, QUICK_DATASETS[:1]):
            test_count += 1
            test_name = f"MultiRun-{encoder}-{model}-{dataset}-{args.test_runs}runs"
            print(f"\n[{test_count}/{multirun_tests}] Testing: {test_name}")
            
            success, message = run_multi_run_test(
                model, dataset, encoder, args.test_runs,
                args.test_epochs, args.timeout_minutes, args.data_ratio, args.verbose
            )
            
            if success:
                result.add_pass(test_name)
            else:
                result.add_fail(test_name, message)
    
    # Cleanup test artifacts
    print(f"\n🧹 Cleaning up test artifacts...")
    cleanup_test_artifacts()
    
    # Print final summary
    print(result.get_summary())
    
    # Exit with appropriate code
    if result.failed > 0:
        print("❌ Some tests failed. Check the error messages above.")
        sys.exit(1)
    else:
        print("✅ All tests passed!")
        sys.exit(0)

if __name__ == "__main__":
    main()