
import subprocess
import itertools
import os
import glob
import json
import time
import argparse
import sys
from datetime import datetime

# Define experiment parameters
models = ['TGAT', 'JODIE', 'TGN', 'DyGFormer', 'DyGMamba', 'TCL']  # 'CAWN', 'DyRep', 'GraphMixer'
datasets = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci', 
            'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']
time_encoders = ['original', 'lete', 'kan_mammote_dual_kmote','mercer', 'bochner', 'time2vec',"kan_mammote"]

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run time encoder comparison experiments')
    parser.add_argument('--models', nargs='+', choices=models, default=models,
                        help='Models to test (default: all)')
    parser.add_argument('--datasets', nargs='+', choices=datasets, default=datasets,
                        help='Datasets to test (default: all)')
    parser.add_argument('--time_encoders', nargs='+', choices=time_encoders, default=time_encoders,
                        help='Time encoders to test (default: all)')
    parser.add_argument('--single_encoder', type=str, choices=time_encoders, required=True,
                        help='Run experiments for a single time encoder only (REQUIRED for proper organization)')
    parser.add_argument('--resume_only', action='store_true',
                        help='Only resume incomplete experiments')
    parser.add_argument('--generate_report', action='store_true',
                        help='Generate experiment report and exit')
    parser.add_argument('--num_runs', type=int, default=3,
                        help='Number of runs per experiment (default: 1)')
    parser.add_argument('--timeout_hours', type=float, default=600.0,
                        help='Timeout in hours per experiment (default: 12)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing them')
    parser.add_argument('--num_epochs', type=int, default=None,
                        help='Override number of training epochs; if omitted, uses best-config default')
    parser.add_argument("--data_ratio", type=float, default=1.0)
    # Add failure handling arguments
    parser.add_argument('--force_restart', action='store_true',
                        help='Force restart of incomplete/failed experiments from scratch (ignores checkpoints)')
    parser.add_argument('--force_restart_combinations', nargs='+',
                        help='Force restart specific combinations (format: encoder_model_dataset)')
    parser.add_argument('--clear_failed', action='store_true',
                        help='Clear all failed experiment status and restart from scratch')
    parser.add_argument('--retry_failed', action='store_true',
                        help='Automatically retry failed experiments (no checkpoints, fresh start)')
    parser.add_argument('--max_retries', type=int, default=2,
                        help='Maximum number of retries for failed experiments (default: 2)')
    parser.add_argument('--disable_progress_bar', action='store_true', default=False,
                        help='Disable tqdm progress bars (useful for logging to files in batch jobs)')
    # AMP and checkpointing options (forwarded to training script)
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable CUDA Automatic Mixed Precision (forwarded to training script)')
    parser.add_argument('--use_gradient_checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing inside time-encoder models (forwarded to training script)')
    
    # Add hyperparameter arguments to forward to training script
    parser.add_argument('--expert_dim', type=int, default=None,
                        help='dimension of each expert in K-MOTE (for kan_mammote encoder)')
    parser.add_argument('--mamba_d_state', type=int, default=None,
                        help='Mamba state dimension (for kan_mammote encoder)')
    parser.add_argument('--mamba_expand', type=int, default=None,
                        help='Mamba expansion factor (for kan_mammote encoder)')
    parser.add_argument('--encoder_dropout', type=float, default=None,
                        help='dropout rate for GNN backbone')
    parser.add_argument('--num_mixtures', type=int, default=None,
                        help='number of mixture components in SM-Kernel (for kan_mammote encoder)')
    parser.add_argument('--mamba_d_conv', type=int, default=None,
                        help='Mamba convolution dimension (for kan_mammote encoder)')
    parser.add_argument('--mamba_headdim', type=int, default=None,
                        help='Mamba head dimension (for kan_mammote encoder)')
    
    return parser.parse_args()

def get_encoder_log_dir(time_encoder):
    """Get encoder-specific log directory with timestamp organization"""
    base_dir = f"experiment_logs/{time_encoder}"
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def get_log_files(time_encoder):
    """Get encoder-specific log file names with organized directory structure"""
    log_dir = get_encoder_log_dir(time_encoder)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create timestamped subdirectory for this run
    run_dir = os.path.join(log_dir, f"{time_encoder}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    
    # Status files are shared across runs (no timestamp)
    status_file = os.path.join(log_dir, f'experiment_status_{time_encoder}.json')
    completed_file = os.path.join(log_dir, f'completed_experiments_{time_encoder}.txt')
    
    # Progress and lock files are per-run (with timestamp)
    progress_file = os.path.join(run_dir, f'experiment_progress_{time_encoder}_{timestamp}.log')
    lock_file = os.path.join(log_dir, f'experiment_lock_{time_encoder}.lock')
    
    return completed_file, status_file, progress_file, lock_file, run_dir

def get_experiment_status(time_encoder):
    """Read completed and incomplete experiments with seed-level granularity"""
    log_file, status_file, progress_file, lock_file, run_dir = get_log_files(time_encoder)
    
    completed = set()
    incomplete = set()
    seed_progress = {}  # Track progress per seed: {experiment_key: {'completed_seeds': [0,1,2], 'last_incomplete_seed': 3, 'last_incomplete_epoch': 40}}
    
    # Use lock file to prevent concurrent access
    lock_acquired = False
    try:
        # Try to acquire lock (non-blocking)
        if not os.path.exists(lock_file):
            with open(lock_file, 'w') as f:
                f.write(f"locked_by_pid_{os.getpid()}_{datetime.now().isoformat()}")
            lock_acquired = True
        
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                    completed = set(status_data.get('completed', []))
                    incomplete = set(status_data.get('incomplete', []))
                    seed_progress = status_data.get('seed_progress', {})
            except (json.JSONDecodeError, OSError) as e:
                print(f"Warning: Could not read status file {status_file}: {e}")
        
        # Also check legacy log file
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.endswith('_incomplete'):
                            base_name = line.replace('_incomplete', '')
                            incomplete.add(base_name)
                        elif line:
                            completed.add(line)
            except OSError as e:
                print(f"Warning: Could not read legacy log file {log_file}: {e}")
    
    finally:
        # Release lock
        if lock_acquired and os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except OSError:
                pass
    
    return completed, incomplete, seed_progress

def save_experiment_status(completed, incomplete, time_encoder, seed_progress=None):
    """Save experiment status with seed-level tracking"""
    log_file, status_file, progress_file, lock_file, run_dir = get_log_files(time_encoder)
    
    if seed_progress is None:
        seed_progress = {}
    
    status_data = {
        'completed': list(completed),
        'incomplete': list(incomplete),
        'seed_progress': seed_progress,
        'time_encoder': time_encoder,
        'last_updated': datetime.now().isoformat(),
        'pid': os.getpid(),
        'run_directory': run_dir
    }
    
    # Use lock file to prevent concurrent writes
    max_retries = 5
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Try to acquire lock
            if not os.path.exists(lock_file):
                with open(lock_file, 'w') as f:
                    f.write(f"locked_by_pid_{os.getpid()}_{datetime.now().isoformat()}")
                
                # Write status file
                with open(status_file, 'w') as f:
                    json.dump(status_data, f, indent=2)
                
                # Remove lock
                os.remove(lock_file)
                return
            else:
                # Lock exists, wait and retry
                time.sleep(retry_delay)
                retry_delay *= 1.5  # Exponential backoff
                
        except (OSError, IOError) as e:
            print(f"Warning: Could not save status file {status_file} (attempt {attempt+1}): {e}")
            if attempt == max_retries - 1:
                print(f"Failed to save status file after {max_retries} attempts")
                return
            time.sleep(retry_delay)
    
    print(f"Warning: Could not acquire lock for status file {status_file}")

def log_progress(message, time_encoder):
    """Log progress message to encoder-specific log file"""
    _, _, progress_file, _, _ = get_log_files(time_encoder)
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}\n"
    
    try:
        with open(progress_file, 'a') as f:
            f.write(log_message)
        print(message)  # Also print to console
    except OSError as e:
        print(f"Warning: Could not write to progress log {progress_file}: {e}")
        print(message)  # Still print to console

def mark_experiment_incomplete(combo_key, time_encoder):
    """Mark experiment as incomplete"""
    completed, incomplete, seed_progress = get_experiment_status(time_encoder)
    log_file, status_file, _, _, _ = get_log_files(time_encoder)
    
    incomplete.add(combo_key)
    completed.discard(combo_key)  # Remove from completed if it was there
    save_experiment_status(completed, incomplete, time_encoder, seed_progress)
    log_progress(f"Marked as incomplete: {combo_key}", time_encoder)
    
    # Also update legacy log file for backward compatibility
    try:
        with open(log_file, 'a') as f:
            f.write(f'{combo_key}_incomplete\n')
    except OSError as e:
        print(f"Warning: Could not update legacy log file: {e}")

def clear_experiment_artifacts(model_name, dataset_name, time_encoder_type, seed=None):
    """Clear all artifacts for a specific experiment combination"""
    import shutil
    
    print(f"🧹 Clearing artifacts for {time_encoder_type}_{model_name}_{dataset_name}")
    
    try:
        # Clear model files
        if seed is not None:
            model_pattern = f"./saved_models/{model_name}/{dataset_name}/*{time_encoder_type}*seed{seed}"
        else:
            model_pattern = f"./saved_models/{model_name}/{dataset_name}/*{time_encoder_type}*"
        
        model_dirs = glob.glob(model_pattern)
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir, ignore_errors=True)
                print(f"   Removed: {model_dir}")
        
        # Clear result files
        if seed is not None:
            result_pattern = f"./saved_results/{model_name}/{dataset_name}/*{time_encoder_type}*seed{seed}*"
        else:
            result_pattern = f"./saved_results/{model_name}/{dataset_name}/*{time_encoder_type}*"
        
        result_files = glob.glob(result_pattern)
        for result_file in result_files:
            if os.path.exists(result_file):
                os.remove(result_file)
                print(f"   Removed: {result_file}")
        
        # Clear log files
        log_pattern = f"./logs/{model_name}/{dataset_name}/*{time_encoder_type}*"
        log_dirs = glob.glob(log_pattern)
        for log_dir in log_dirs:
            if os.path.exists(log_dir):
                shutil.rmtree(log_dir, ignore_errors=True)
                print(f"   Removed: {log_dir}")
        
        print(f"✅ Cleared artifacts for {time_encoder_type}_{model_name}_{dataset_name}")
        return True
        
    except Exception as e:
        print(f"❌ Error clearing artifacts: {e}")
        return False

def mark_experiment_for_fresh_restart(model_name, dataset_name, time_encoder_type, time_encoder):
    """Mark an experiment to be restarted from scratch (remove from status tracking)"""
    try:
        completed, incomplete, seed_progress = get_experiment_status(time_encoder)
        combo_key = create_experiment_key(model_name, dataset_name, time_encoder_type)
        
        # Remove from all tracking
        if combo_key in completed:
            completed.remove(combo_key)
        if combo_key in incomplete:
            incomplete.remove(combo_key)
        if combo_key in seed_progress:
            del seed_progress[combo_key]
        
        # Save updated status
        save_experiment_status(completed, incomplete, time_encoder, seed_progress)
        
        print(f"🔄 Marked for fresh restart: {combo_key}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating experiment status: {e}")
        return False

def verify_training_success(model_name, dataset_name, time_encoder_type, num_runs):
    """Verify that training actually produced valid model files for all runs"""
    
    for run in range(num_runs):
        # Check for model file
        model_pattern = f"./saved_models/{model_name}/{dataset_name}/*{time_encoder_type}*seed{run}/*.pth"
        model_files = glob.glob(model_pattern)
        
        if not model_files:
            print(f"❌ No model file found for run {run}")
            return False
        
        # Verify model file can be loaded
        try:
            import torch
            torch.load(model_files[0], map_location='cpu')
        except Exception as e:
            print(f"❌ Model file corrupted for run {run}: {e}")
            return False
    
    return True

def run_experiment_with_retry(model, dataset, time_encoder_name, combo_key, command, 
                             args, time_encoder, max_retries=2):
    """Run experiment with retry logic for failed attempts"""
    
    for attempt in range(max_retries + 1):  # 0, 1, 2 (total 3 attempts)
        if attempt > 0:
            print(f"\n🔄 Retry attempt {attempt}/{max_retries} for {combo_key}")
            
            # Clear artifacts from previous failed attempt
            clear_experiment_artifacts(model, dataset, time_encoder_name)
            
            # Remove from status tracking to force fresh start
            mark_experiment_for_fresh_restart(model, dataset, time_encoder_name, time_encoder)
            
            # Wait a bit between retries
            time.sleep(5)
        
        try:
            print(f"🚀 Running: {' '.join(command)}")
            
            # Run the training
            result = subprocess.run(command, check=True, capture_output=False, text=True)
            
            # Verify that training actually succeeded
            if verify_training_success(model, dataset, time_encoder_name, args.num_runs):
                print(f"✅ Training succeeded for {combo_key} (attempt {attempt + 1})")
                return True
            else:
                if attempt < max_retries:
                    print(f"⚠️  Training appeared to complete but no valid models found (attempt {attempt + 1})")
                    continue
                else:
                    print(f"❌ Training failed - no valid models after {max_retries + 1} attempts")
                    return False
                    
        except subprocess.CalledProcessError as e:
            if attempt < max_retries:
                print(f"⚠️  Training failed with exit code {e.returncode} (attempt {attempt + 1})")
                continue
            else:
                print(f"❌ Training failed after {max_retries + 1} attempts with exit code {e.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            if attempt < max_retries:
                print(f"⚠️  Training timeout (attempt {attempt + 1})")
                continue
            else:
                print(f"❌ Training timeout after {max_retries + 1} attempts")
                return False
                
        except Exception as e:
            if attempt < max_retries:
                print(f"⚠️  Unexpected error (attempt {attempt + 1}): {e}")
                continue
            else:
                print(f"❌ Unexpected error after {max_retries + 1} attempts: {e}")
                return False
    
    return False

def mark_experiment_complete(combo_key, time_encoder):
    """Mark experiment as complete and remove incomplete marker"""
    completed, incomplete, seed_progress = get_experiment_status(time_encoder)
    log_file, status_file, _, _, _ = get_log_files(time_encoder)
    
    completed.add(combo_key)
    incomplete.discard(combo_key)  # Remove from incomplete
    save_experiment_status(completed, incomplete, time_encoder, seed_progress)
    log_progress(f"Completed: {combo_key}", time_encoder)
    
    # Also update legacy log file
    try:
        lines = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
        
        incomplete_marker = f'{combo_key}_incomplete'
        if incomplete_marker in lines:
            lines.remove(incomplete_marker)
        
        if combo_key not in lines:
            lines.append(combo_key)
        
        with open(log_file, 'w') as f:
            for line in lines:
                f.write(f'{line}\n')
    except OSError as e:
        print(f"Warning: Could not update legacy log file: {e}")

def create_seed_experiment_key(model, dataset, time_encoder, seed):
    """Create a unique key for the experiment combination including seed"""
    return f'{time_encoder}_{model}_{dataset}_seed{seed}'

def find_incomplete_seed_checkpoint(model_name, dataset_name, time_encoder_type, seed):
    """Find the best available checkpoint for a specific seed with corruption handling"""
    import os
    import sys
    
    # Build possible checkpoint directories
    checkpoint_dirs = [
        f"./saved_models/{model_name}/{dataset_name}/{model_name}_{time_encoder_type}_seed{seed}",
        f"./saved_models/{model_name}/{dataset_name}/{model_name}_seed{seed}"  # Fallback
    ]
    
    for checkpoint_dir in checkpoint_dirs:
        if os.path.exists(checkpoint_dir):
            print(f"🔍 Searching for checkpoints in: {checkpoint_dir}")
            
            # Create a simple logger for checkpoint validation
            class SimpleLogger:
                def info(self, msg): print(f"ℹ️  {msg}")
                def warning(self, msg): print(f"⚠️  {msg}")
                def error(self, msg): print(f"❌ {msg}")
            
            logger = SimpleLogger()
            
            # Use the robust checkpoint finding function
            try:
                # Import the functions from the training script
                sys.path.append('./experiments')
                
                # Try to find checkpoints manually if import fails
                checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "checkpoint_epoch_*.pth"))
                if checkpoint_files:
                    # Sort by epoch and return newest valid checkpoint
                    def extract_epoch(filepath):
                        try:
                            filename = os.path.basename(filepath)
                            epoch_str = filename.split('checkpoint_epoch_')[1].split('.pth')[0]
                            return int(epoch_str)
                        except (IndexError, ValueError):
                            return 0
                    
                    checkpoint_files.sort(key=extract_epoch, reverse=True)
                    
                    # Try each checkpoint from newest to oldest
                    for checkpoint_path in checkpoint_files:
                        epoch = extract_epoch(checkpoint_path)
                        print(f"🔍 Trying checkpoint: {checkpoint_path} (epoch {epoch})")
                        
                        try:
                            # Basic validation - try to load the checkpoint
                            import torch
                            checkpoint = torch.load(checkpoint_path, map_location='cpu')
                            
                            # Check for required fields
                            required_fields = ['epoch', 'model_state_dict', 'optimizer_state_dict']
                            if all(field in checkpoint for field in required_fields):
                                print(f"✅ Found valid checkpoint: {checkpoint_path} (epoch {epoch})")
                                return os.path.abspath(checkpoint_path), epoch
                            else:
                                print(f"⚠️  Checkpoint missing required fields: {checkpoint_path}")
                                continue
                                
                        except Exception as e:
                            print(f"❌ Checkpoint corrupted: {checkpoint_path} - {e}")
                            continue
                    
                    print(f"❌ No valid checkpoints found in {checkpoint_dir}")
                else:
                    print(f"❌ No checkpoint files found in {checkpoint_dir}")
            
            except Exception as e:
                print(f"⚠️  Error checking checkpoints: {e}")
    
    return None, None

def check_seed_completion(model_name, dataset_name, time_encoder_type, seed):
    """Check if a specific seed completed training"""
    result_pattern = f"./saved_results/{model_name}/{dataset_name}/*{time_encoder_type}*_seed{seed}_*.json"
    result_files = glob.glob(result_pattern)
    
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict) and data.get('time_encoder_type') == time_encoder_type:
                    return True
        except Exception:
            continue
    
    return False

def find_checkpoint_file(model_name, dataset_name, time_encoder_type):
    """Find the most recent checkpoint file for resuming"""
    # Prefer encoder-specific run folders if present
    checkpoint_pattern = f"./saved_models/{model_name}/{dataset_name}/*{time_encoder_type}*_seed0/checkpoint*.pth"
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        # Fallback to legacy pattern without encoder in name
        checkpoint_pattern = f"./saved_models/{model_name}/{dataset_name}/*_seed0/checkpoint*.pth"
        checkpoint_files = glob.glob(checkpoint_pattern)
        if not checkpoint_files:
            return None
    
    # Return the most recent checkpoint as ABSOLUTE PATH
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return os.path.abspath(checkpoint_files[0])

def check_training_completion(model_name, dataset_name, time_encoder_type):
    """Check if training was completed by looking for final results (encoder-aware)."""
    # Prefer encoder-specific filenames (save_model_name now includes encoder)
    result_pattern = f"./saved_results/{model_name}/{dataset_name}/*{time_encoder_type}*_seed0_*.json"
    result_files = glob.glob(result_pattern)

    # Fallback to legacy filenames if none match
    if not result_files:
        result_pattern = f"./saved_results/{model_name}/{dataset_name}/*_seed0_*.json"
        result_files = glob.glob(result_pattern)
    
    # Check JSON content for encoder tag
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                # New format includes explicit field
                if isinstance(data, dict) and data.get('time_encoder_type') == time_encoder_type:
                    return True
                # Backward-compat: search raw text if load failed or field missing
        except Exception:
            try:
                with open(result_file, 'r') as f:
                    content = f.read()
                    if time_encoder_type in content:
                        return True
            except Exception:
                continue
    
    return False

def create_experiment_key(model, dataset, time_encoder):
    """Create a unique key for the experiment combination"""
    return f'{time_encoder}_{model}_{dataset}'

def build_training_command(model, dataset, time_encoder_name, args, 
                          resume_checkpoint=None, start_seed=0):
    """Build training command with appropriate arguments"""
    command = [
        'python', 'experiments/train_link_prediction.py',
        '--model_name', model,
        '--dataset_name', dataset,
        '--time_encoder_type', time_encoder_name,
        '--num_runs', str(args.num_runs),
        '--load_best_configs',
        '--save_checkpoints',
        '--checkpoint_strategy', 'smart',
        '--max_checkpoints_to_keep', '3',
        '--validate_checkpoints'
    ]
    
    if args.num_epochs is not None:
        command.extend(['--num_epochs', str(args.num_epochs)])
    
    if resume_checkpoint:
        command.extend(['--resume_from_checkpoint', resume_checkpoint])
    
    if start_seed > 0:
        command.extend(['--start_from_seed', str(start_seed)])
    
    # Add disable_progress_bar if specified
    if args.disable_progress_bar:
        command.append('--disable_progress_bar')
    
    # Add hyperparameters if specified
    if args.expert_dim is not None:
        command.extend(['--expert_dim', str(args.expert_dim)])
    if args.mamba_d_state is not None:
        command.extend(['--mamba_d_state', str(args.mamba_d_state)])
    if args.mamba_expand is not None:
        command.extend(['--mamba_expand', str(args.mamba_expand)])
    if args.encoder_dropout is not None:
        command.extend(['--encoder_dropout', str(args.encoder_dropout)])
    if args.num_mixtures is not None:
        command.extend(['--num_mixtures', str(args.num_mixtures)])
    if args.mamba_d_conv is not None:
        command.extend(['--mamba_d_conv', str(args.mamba_d_conv)])
    if args.mamba_headdim is not None:
        command.extend(['--mamba_headdim', str(args.mamba_headdim)])
    
    # Add encoder-specific arguments
    encoder_args = get_time_encoder_args(time_encoder_name)
    if encoder_args:
        command.extend(encoder_args.split())

    # Forward runtime flags for AMP and gradient checkpointing to the training script
    if hasattr(args, 'use_amp') and args.use_amp:
        command.append('--use_amp')
    if hasattr(args, 'use_gradient_checkpointing') and args.use_gradient_checkpointing:
        command.append('--use_gradient_checkpointing')
    
    return command

def get_time_encoder_args(time_encoder):
    """Get encoder-specific command-line arguments"""
    if time_encoder == 'kan_mammote':
        return '--num_mixtures 12 --mamba_d_state 16 --mamba_d_conv 4 --mamba_expand 2 --mamba_headdim 64 --sort_neighbors_by_time'
    elif time_encoder == 'kan_mammote_lite':
        return '--num_mixtures 12 --sort_neighbors_by_time'
    elif time_encoder == 'lete':
        return ''
    elif time_encoder == 'mercer':
        return ''
    elif time_encoder == 'bochner':
        return ''
    elif time_encoder == 'time2vec':
        return ''
    elif time_encoder == 'original':
        return ''
    else:
        return ''

def generate_experiment_report(time_encoder):
    """Generate a detailed experiment report"""
    completed, incomplete, seed_progress = get_experiment_status(time_encoder)
    log_file, status_file, _, _, run_dir = get_log_files(time_encoder)
    
    # Save report in the encoder's log directory
    log_dir = get_encoder_log_dir(time_encoder)
    report_file = os.path.join(log_dir, f"experiment_report_{time_encoder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    
    try:
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"TIME ENCODER EXPERIMENT REPORT - {time_encoder.upper()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Time Encoder: {time_encoder}\n")
            f.write(f"Status File: {status_file}\n")
            f.write(f"Log Directory: {log_dir}\n")
            f.write(f"Current Run Directory: {run_dir}\n\n")
            
            # Calculate totals for this encoder
            total_experiments = len(models) * len(datasets)
            
            f.write(f"Total possible experiments: {total_experiments}\n")
            f.write(f"Completed experiments: {len(completed)}\n")
            f.write(f"Incomplete experiments: {len(incomplete)}\n")
            f.write(f"Remaining experiments: {total_experiments - len(completed) - len(incomplete)}\n\n")
            
            f.write("COMPLETED EXPERIMENTS BY MODEL:\n")
            f.write("-" * 50 + "\n")
            for model in models:
                model_completed = [exp for exp in completed if f"_{model}_" in exp]
                f.write(f"{model}: {len(model_completed)}\n")
            
            f.write("\nCOMPLETED EXPERIMENTS BY DATASET:\n")
            f.write("-" * 50 + "\n")
            for dataset in datasets:
                dataset_completed = [exp for exp in completed if exp.endswith(f"_{dataset}")]
                f.write(f"{dataset}: {len(dataset_completed)}\n")
            
            if completed:
                f.write(f"\nALL COMPLETED EXPERIMENTS ({len(completed)}):\n")
                f.write("-" * 50 + "\n")
                for exp in sorted(completed):
                    f.write(f"✅ {exp}\n")
            
            if incomplete:
                f.write(f"\nINCOMPLETE EXPERIMENTS ({len(incomplete)}):\n")
                f.write("-" * 50 + "\n")
                for exp in sorted(incomplete):
                    f.write(f"⚠️  {exp}\n")
            
            # Generate missing experiments
            all_possible = set()
            for dataset, model in itertools.product(datasets, models):
                all_possible.add(create_experiment_key(model, dataset, time_encoder))
            
            missing = all_possible - completed - incomplete
            if missing:
                f.write(f"\nMISSING EXPERIMENTS ({len(missing)}):\n")
                f.write("-" * 50 + "\n")
                for exp in sorted(missing):
                    f.write(f"❌ {exp}\n")
        
        print(f"📊 Experiment report generated: {report_file}")
        return report_file
        
    except OSError as e:
        print(f"Warning: Could not generate report: {e}")
        return None

def print_experiment_summary(completed, incomplete, time_encoder):
    """Print a summary of experiment status"""
    print("\n" + "="*80)
    print(f"EXPERIMENT SUMMARY - {time_encoder.upper()}")
    print("="*80)
    
    total_experiments = len(models) * len(datasets)
    
    print(f"Time Encoder: {time_encoder}")
    print(f"Total possible experiments: {total_experiments}")
    print(f"Completed experiments: {len(completed)}")
    print(f"Incomplete experiments: {len(incomplete)}")
    print(f"Remaining experiments: {total_experiments - len(completed) - len(incomplete)}")
    
    if completed:
        print(f"\n✅ Completed ({len(completed)}):")
        for exp in sorted(completed):
            print(f"   {exp}")
    
    if incomplete:
        print(f"\n⚠️  Incomplete ({len(incomplete)}):")
        for exp in sorted(incomplete):
            print(f"   {exp}")
    
    print("="*80)

# Main execution
if __name__ == "__main__":
    args = parse_arguments()
    
    time_encoder = args.single_encoder
    
    # ===== ADD: Create required base directories =====
    print("🔧 Setting up directories...")
    required_base_dirs = [
        './saved_models',
        './saved_results',
        './logs',
        './experiment_logs'
    ]
    
    for directory in required_base_dirs:
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as e:
            print(f"⚠️  Warning: Could not create {directory}: {e}")
    
    # Also create encoder-specific log directory
    encoder_log_dir = get_encoder_log_dir(time_encoder)
    print(f"✅ Directories ready (encoder logs: {encoder_log_dir})")
    # ===== END ADD =====

    # Handle report generation
    if args.generate_report:
        generate_experiment_report(time_encoder)
        sys.exit(0)
    
    # Use arguments to filter experiment parameters
    models_to_run = args.models
    datasets_to_run = args.datasets
    
    print(f"🎯 Running experiments for time encoder: {time_encoder}")
    
    # Get current experiment status with seed tracking
    completed, incomplete, seed_progress = get_experiment_status(time_encoder)
    
    print(f"\n🚀 Starting Time Encoder Experiments")
    print(f"Time Encoder: {time_encoder}")
    print(f"Models: {models_to_run}")
    print(f"Datasets: {datasets_to_run}")
    print(f"Runs per experiment: {args.num_runs}")
    print(f"Timeout: {args.timeout_hours} hours")
    if args.num_epochs is not None:
        print(f"Epochs override: {args.num_epochs} (otherwise uses best-config defaults)")
    
    log_file, status_file, progress_file, _, run_dir = get_log_files(time_encoder)
    print(f"Log Directory: {get_encoder_log_dir(time_encoder)}")
    print(f"Run Directory: {run_dir}")
    print(f"Status File: {status_file}")
    print(f"Progress File: {progress_file}")
    
    if args.dry_run:
        print("🔍 DRY RUN MODE - Commands will be printed but not executed")
    if args.resume_only:
        print("🔄 RESUME ONLY MODE - Only incomplete experiments will be run")
    
    print_experiment_summary(completed, incomplete, time_encoder)
    print("-" * 80)
    
    # Count total experiments to run
    experiments_to_run = []
    for model, dataset in itertools.product(models_to_run, datasets_to_run):
        combo_key = create_experiment_key(model, dataset, time_encoder)
        
        if combo_key in completed:
            continue  # Skip completed
        
        if args.resume_only and combo_key not in incomplete:
            continue  # Skip new experiments if resume_only mode
        
        experiments_to_run.append((model, dataset, time_encoder, combo_key))
    
    print(f"📋 Experiments to run: {len(experiments_to_run)}")
    if args.dry_run:
        print("🔍 Commands that would be executed:")
    print("-" * 80)
    
    # Iterate over all combinations
    for i, (model, dataset, time_encoder_name, combo_key) in enumerate(experiments_to_run, 1):
        print(f"\n[{i}/{len(experiments_to_run)}] Processing: {combo_key}")
        
        # Check if this combination should be force restarted
        force_restart_this = (
            args.force_restart or 
            args.clear_failed or
            (args.force_restart_combinations and combo_key in args.force_restart_combinations)
        )
        
        if force_restart_this:
            print(f"🔄 Force restart requested for {combo_key}")
            clear_experiment_artifacts(model, dataset, time_encoder_name)
            mark_experiment_for_fresh_restart(model, dataset, time_encoder_name, time_encoder)
            # Refresh status after clearing
            completed, incomplete, seed_progress = get_experiment_status(time_encoder)
        
        # Check seed-level progress for smart resuming
        checkpoint_file = None
        resume_from_checkpoint = False
        start_from_seed = 0
        
        if combo_key in seed_progress and not force_restart_this:
            progress = seed_progress[combo_key]
            completed_seeds = progress.get('completed_seeds', [])
            last_incomplete_seed = progress.get('last_incomplete_seed', None)
            last_incomplete_epoch = progress.get('last_incomplete_epoch', None)
            
            if last_incomplete_seed is not None and last_incomplete_epoch is not None:
                print(f"🔄 Found incomplete experiment at seed {last_incomplete_seed}, epoch {last_incomplete_epoch}")
                
                # Find checkpoint for the incomplete seed
                checkpoint_file, checkpoint_epoch = find_incomplete_seed_checkpoint(
                    model, dataset, time_encoder_name, last_incomplete_seed
                )
                
                if checkpoint_file:
                    resume_from_checkpoint = True
                    start_from_seed = last_incomplete_seed
                    print(f"🔄 Will resume from seed {last_incomplete_seed}, epoch {checkpoint_epoch}")
                    print(f"   Checkpoint: {checkpoint_file}")
                else:
                    print(f"⚠️  No checkpoint found for seed {last_incomplete_seed}, restarting from that seed")
                    start_from_seed = last_incomplete_seed
            else:
                # Check which seeds are complete vs remaining
                remaining_seeds = [s for s in range(args.num_runs) if s not in completed_seeds]
                if remaining_seeds:
                    start_from_seed = min(remaining_seeds)
                    print(f"🚀 Starting from seed {start_from_seed} (seeds {completed_seeds} already completed)")
                else:
                    print(f"✅ All seeds completed for {combo_key}")
                    continue
        elif combo_key in incomplete and not force_restart_this:
            # Legacy incomplete detection - try to find any checkpoint
            checkpoint_file = find_checkpoint_file(model, dataset, time_encoder_name)
            if checkpoint_file:
                print(f"🔄 Found checkpoint for incomplete experiment")
                print(f"   Checkpoint: {checkpoint_file}")
                resume_from_checkpoint = True
            else:
                print(f"⚠️  Incomplete experiment found but no checkpoint available")
        
        # Build the command using helper so CLI hyperparameters are forwarded
        resume_ckpt = None
        if resume_from_checkpoint and checkpoint_file:
            # prefer absolute path when forwarding
            resume_ckpt = os.path.abspath(checkpoint_file)

        command = build_training_command(
            model=model,
            dataset=dataset,
            time_encoder_name=time_encoder_name,
            args=args,
            resume_checkpoint=resume_ckpt,
            start_seed=start_from_seed
        )

        # Print checkpoint debugging info (do not append flags here - build_training_command already included them)
        if resume_from_checkpoint and checkpoint_file:
            abs_checkpoint_path = os.path.abspath(checkpoint_file)
            print(f"🔍 Checkpoint debugging:")
            print(f"   Original path: {checkpoint_file}")
            print(f"   Absolute path: {abs_checkpoint_path}")
            print(f"   File exists: {os.path.exists(abs_checkpoint_path)}")
            print(f"   Working dir: {os.getcwd()}")
            print(f"🔄 Resuming from checkpoint at seed {start_from_seed}")
        else:
            if start_from_seed > 0:
                print(f"🚀 Starting from seed {start_from_seed} (fresh training)")
            else:
                print(f"🚀 Starting new training from seed 0")
        
        print(f"Command: {' '.join(command)}")
        print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if args.dry_run:
            print("🔍 DRY RUN - Command not executed")
            print("-" * 80)
            continue
        
        print("-" * 80)
        
        # Mark as incomplete when starting (if not already)
        if combo_key not in incomplete:
            mark_experiment_incomplete(combo_key, time_encoder)
        
        # Run the experiment (with retry if enabled)
        if args.retry_failed:
            success = run_experiment_with_retry(
                model, dataset, time_encoder_name, combo_key, command, 
                args, time_encoder, max_retries=args.max_retries
            )
            if success:
                print(f"✅ Successfully completed: {combo_key}")
                mark_experiment_complete(combo_key, time_encoder)
            else:
                print(f"❌ Failed after all retry attempts: {combo_key}")
                # Leave as incomplete for potential manual investigation
        else:
            # Original single-attempt logic
            try:
                # Run the command with timeout
                start_time = time.time()
                timeout_seconds = int(args.timeout_hours * 3600)
                result = subprocess.run(command, text=True, check=True)
                end_time = time.time()
                
                # Check if training actually completed
                if check_training_completion(model, dataset, time_encoder_name):
                    duration = end_time - start_time
                    print("-" * 80)
                    print(f"✅ Successfully completed: {combo_key}")
                    print(f"   Duration: {duration/3600:.2f} hours")
                    mark_experiment_complete(combo_key, time_encoder)
                else:
                    print("-" * 80)
                    print(f"⚠️  Training finished but no results found: {combo_key}")
                    
            except subprocess.TimeoutExpired:
                print("-" * 80)
                print(f"⏰ Training timeout ({args.timeout_hours}h) for: {combo_key}")
                print("   Experiment marked as incomplete for future resuming")
                
            except subprocess.CalledProcessError as e:
                print("-" * 80)
                print(f"❌ Error occurred while running: {combo_key}")
                print(f"Return code: {e.returncode}")
                
            except KeyboardInterrupt:
                print(f"\n⏹️  Interrupted by user. Last attempted: {combo_key}")
                break
                
            except Exception as e:
                print(f"❌ Unexpected error for {combo_key}: {e}")
        
        # Print updated summary after each experiment
        completed, incomplete, seed_progress = get_experiment_status(time_encoder)
        remaining = len(experiments_to_run) - (i)
        print(f"\n📊 Progress: {len([exp for exp in experiments_to_run[:i] if exp[3] in completed])} completed, {remaining} remaining\n")
    
    # Final summary
    print("\n🏁 Experiment batch completed!")
    completed, incomplete, seed_progress = get_experiment_status(time_encoder)
    print_experiment_summary(completed, incomplete, time_encoder)
    
    # Generate detailed report
    report_file = generate_experiment_report(time_encoder)
    if report_file:
        print(f"📊 Detailed report saved to: {report_file}")
    
    # Save final status
    save_experiment_status(completed, incomplete, time_encoder, seed_progress)
    
    log_file, status_file, progress_file, _, run_dir = get_log_files(time_encoder)
    print(f"\n📁 Log organization:")
    print(f"   - Encoder directory: {get_encoder_log_dir(time_encoder)}")
    print(f"   - Current run: {run_dir}")
    print(f"   - Status file: {status_file}")
    if report_file:
        print(f"   - Report: {report_file}")
    print(f"\n✨ Experiments for {time_encoder} completed successfully!")