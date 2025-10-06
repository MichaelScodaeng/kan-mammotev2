"""
Time Encoder Comparison Experiment Runner

This script runs comprehensive experiments comparing different time encoders
(original, LeTE, KAN-MAMMOTE, Mercer, Bochner, Time2Vec) across multiple 
models and datasets.

Features:
- Automatic experiment tracking and resuming
- Configurable model/dataset/encoder combinations
- Progress monitoring and reporting
- Proper file organization to avoid conflicts
- HPC parallel execution support

Usage Examples:
  # Run all experiments
  python experiment_kanmammote.py

  # Run specific combinations
  python experiment_kanmammote.py --models TGAT JODIE --datasets wikipedia reddit --time_encoders original kan_mammote

  # HPC: Run experiments for a single time encoder (for parallel execution)
  python experiment_kanmammote.py --single_encoder kan_mammote
  python experiment_kanmammote.py --single_encoder lete --models TGAT --datasets wikipedia

  # Resume only incomplete experiments
  python experiment_kanmammote.py --resume_only

  # Resume incomplete experiments for specific encoder
  python experiment_kanmammote.py --single_encoder kan_mammote --resume_only

  # Generate experiment report (all encoders)
  python experiment_kanmammote.py --generate_report
  
  # Generate report for specific encoder
  python experiment_kanmammote.py --single_encoder kan_mammote --generate_report

  # Dry run (show commands without executing)
  python experiment_kanmammote.py --dry_run --models TGAT --datasets wikipedia

HPC Parallel Execution:
  # Submit separate jobs for each time encoder
  qsub -v TIME_ENCODER=kan_mammote job_script.sh
  qsub -v TIME_ENCODER=lete job_script.sh
  qsub -v TIME_ENCODER=original job_script.sh
  # etc.
  
  # In job_script.sh:
  python experiment_kanmammote.py --single_encoder $TIME_ENCODER
"""

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
models = ['TGAT'] #, 'JODIE', 'TGN',  'GraphMixer', 'DyGFormer', 'DyGMamba','TCL'
datasets = ['wikipedia']#, 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci'
time_encoders = ['kan_mammote', 'kan_mammote_lite', 'lete', 'original', 'mercer'] #'original',

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run time encoder comparison experiments')
    parser.add_argument('--models', nargs='+', choices=models, default=models,
                        help='Models to test (default: all)')
    parser.add_argument('--datasets', nargs='+', choices=datasets, default=datasets,
                        help='Datasets to test (default: all)')
    parser.add_argument('--time_encoders', nargs='+', choices=time_encoders, default=time_encoders,
                        help='Time encoders to test (default: all)')
    parser.add_argument('--single_encoder', type=str, choices=time_encoders, default="kan_mammote",
                        help='Run experiments for a single time encoder only (for HPC parallel execution)')
    parser.add_argument('--resume_only', action='store_true',
                        help='Only resume incomplete experiments')
    parser.add_argument('--generate_report', action='store_true',
                        help='Generate experiment report and exit')
    parser.add_argument('--num_runs', type=int, default=1,
                        help='Number of runs per experiment (default: 5)')
    parser.add_argument('--timeout_hours', type=float, default=12.0,
                        help='Timeout in hours per experiment (default: 12)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Print commands without executing them')
    # Optional: override number of epochs; if not set, use best-config defaults
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Override number of training epochs; if omitted, uses best-config default')
    
    return parser.parse_args()

def get_log_files(time_encoder):
    """Get encoder-specific log file names"""
    log_file = f'completed_experiments_{time_encoder}.txt'
    status_file = f'experiment_status_{time_encoder}.json'
    progress_file = f'experiment_progress_{time_encoder}.log'
    lock_file = f'experiment_lock_{time_encoder}.lock'
    
    return log_file, status_file, progress_file, lock_file

def get_experiment_status(time_encoder):
    """Read completed and incomplete experiments from JSON status file"""
    log_file, status_file, progress_file, lock_file = get_log_files(time_encoder)
    
    completed = set()
    incomplete = set()
    
    # Use lock file to prevent concurrent access
    lock_acquired = False
    try:
        # Try to acquire lock (non-blocking)
        if not os.path.exists(lock_file):
            with open(lock_file, 'w') as f:
                f.write(f"locked_by_pid_{os.getpid()}")
            lock_acquired = True
        
        if os.path.exists(status_file):
            try:
                with open(status_file, 'r') as f:
                    status_data = json.load(f)
                    completed = set(status_data.get('completed', []))
                    incomplete = set(status_data.get('incomplete', []))
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
    
    return completed, incomplete

def save_experiment_status(completed, incomplete, time_encoder):
    """Save experiment status to JSON file with file locking for parallel safety"""
    log_file, status_file, progress_file, lock_file = get_log_files(time_encoder)
    
    status_data = {
        'completed': list(completed),
        'incomplete': list(incomplete),
        'time_encoder': time_encoder,
        'last_updated': datetime.now().isoformat(),
        'pid': os.getpid()
    }
    
    # Use lock file to prevent concurrent writes
    max_retries = 5
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            # Try to acquire lock
            if not os.path.exists(lock_file):
                with open(lock_file, 'w') as f:
                    f.write(f"locked_by_pid_{os.getpid()}")
                
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
    _, _, progress_file, _ = get_log_files(time_encoder)
    
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
    completed, incomplete = get_experiment_status(time_encoder)
    log_file, status_file, _, _ = get_log_files(time_encoder)
    
    incomplete.add(combo_key)
    completed.discard(combo_key)  # Remove from completed if it was there
    save_experiment_status(completed, incomplete, time_encoder)
    log_progress(f"Marked as incomplete: {combo_key}", time_encoder)
    
    # Also update legacy log file for backward compatibility
    try:
        with open(log_file, 'a') as f:
            f.write(f'{combo_key}_incomplete\n')
    except OSError as e:
        print(f"Warning: Could not update legacy log file: {e}")

def mark_experiment_complete(combo_key, time_encoder):
    """Mark experiment as complete and remove incomplete marker"""
    completed, incomplete = get_experiment_status(time_encoder)
    log_file, status_file, _, _ = get_log_files(time_encoder)
    
    completed.add(combo_key)
    incomplete.discard(combo_key)  # Remove from incomplete
    save_experiment_status(completed, incomplete, time_encoder)
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
    
    # Return the most recent checkpoint
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0]

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
    completed, incomplete = get_experiment_status(time_encoder)
    log_file, status_file, _, _ = get_log_files(time_encoder)
    
    report_file = f"experiment_report_{time_encoder}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    try:
        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"TIME ENCODER EXPERIMENT REPORT - {time_encoder.upper()}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Time Encoder: {time_encoder}\n")
            f.write(f"Status File: {status_file}\n")
            f.write(f"Log File: {log_file}\n\n")
            
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
    
    # REQUIRE single_encoder to be specified for simplicity
    if not args.single_encoder:
        print("❌ Error: --single_encoder must be specified")
        print("   This script is designed to run ONE encoder per execution for HPC parallel processing")
        print("   Example: python experiment_kanmammote.py --single_encoder kan_mammote")
        print("   Available encoders:", time_encoders)
        sys.exit(1)
    
    time_encoder = args.single_encoder
    
    # Handle report generation
    if args.generate_report:
        generate_experiment_report(time_encoder)
        sys.exit(0)
    
    # Use arguments to filter experiment parameters
    models_to_run = args.models
    datasets_to_run = args.datasets
    
    print(f"🎯 Running experiments for time encoder: {time_encoder}")
    
    # Get current experiment status
    completed, incomplete = get_experiment_status(time_encoder)
    
    print(f"\n🚀 Starting Time Encoder Experiments")
    print(f"Time Encoder: {time_encoder}")
    print(f"Models: {models_to_run}")
    print(f"Datasets: {datasets_to_run}")
    print(f"Runs per experiment: {args.num_runs}")
    print(f"Timeout: {args.timeout_hours} hours")
    if args.num_epochs is not None:
        print(f"Epochs override: {args.num_epochs} (otherwise uses best-config defaults)")
    
    log_file, status_file, progress_file, _ = get_log_files(time_encoder)
    print(f"Status File: {status_file}")
    print(f"Log File: {log_file}")
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
        
        # Check if this experiment was incomplete and has a checkpoint
        checkpoint_file = None
        resume_from_checkpoint = False
        
        if combo_key in incomplete:
            checkpoint_file = find_checkpoint_file(model, dataset, time_encoder_name)
            if checkpoint_file:
                print(f"🔄 Found checkpoint for incomplete experiment")
                print(f"   Checkpoint: {checkpoint_file}")
                resume_from_checkpoint = True
            else:
                print(f"⚠️  Incomplete experiment found but no checkpoint available")
        
        # Build the command using the training script
        command = [
            'python', 'experiments/train_link_prediction.py',
            '--model_name', model,
            '--dataset_name', dataset,
            '--time_encoder_type', time_encoder_name,
            '--num_runs', str(args.num_runs),
            '--load_best_configs'
        ]

        # If epochs override provided, pass it through; otherwise rely on best-config defaults
        if args.num_epochs is not None:
            command.extend(['--num_epochs', str(args.num_epochs)])
        
        # Add time encoder specific arguments
        encoder_specific_args = get_time_encoder_args(time_encoder_name)
        if encoder_specific_args:
            command.extend(encoder_specific_args.split())
        
        # Add checkpoint resuming if available (this would need to be implemented in the training script)
        if resume_from_checkpoint and checkpoint_file:
            # Note: The training script would need to support checkpoint resuming
            # command.extend(['--resume_from_checkpoint', checkpoint_file])
            print(f"🔄 Would resume from checkpoint")
        else:
            print(f"🚀 Starting new training")
        
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
        
        try:
            # Run the command with timeout
            start_time = time.time()
            timeout_seconds = int(args.timeout_hours * 3600)
            result = subprocess.run(command, text=True, check=True, timeout=timeout_seconds)
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
        completed, incomplete = get_experiment_status(time_encoder)
        remaining = len(experiments_to_run) - (i)
        print(f"\n📊 Progress: {len([exp for exp in experiments_to_run[:i] if exp[3] in completed])} completed, {remaining} remaining\n")
    
    # Final summary
    print("\n🏁 Experiment batch completed!")
    completed, incomplete = get_experiment_status(time_encoder)
    print_experiment_summary(completed, incomplete, time_encoder)
    
    # Generate detailed report
    report_file = generate_experiment_report(time_encoder)
    if report_file:
        print(f"📊 Detailed report saved to: {report_file}")
    
    # Save final status
    save_experiment_status(completed, incomplete, time_encoder)
    
    log_file, status_file, progress_file, _ = get_log_files(time_encoder)
    print(f"\n📁 Status files:")
    print(f"   - JSON status: {status_file}")
    print(f"   - Legacy log: {log_file}")
    print(f"   - Progress log: {progress_file}")
    if report_file:
        print(f"   - Report: {report_file}")
    print(f"\n✨ Experiments for {time_encoder} completed successfully!")