import subprocess
import itertools
import os
import glob
from datetime import datetime

# Define model and dataset choices
models = ['JODIE', 'DyRep', 'TGN', 'TCL', 'GraphMixer','DyGFormer', 'DyGMamba'] #'CAWN', 
datasets = ['wikipedia',  'mooc',  'enron', 'SocialEvo', 'uci','reddit'] #,'lastfm',

# File to track completed runs (simple text file)
log_file = 'completed_experiments_noTime.txt'
time_encoder = 'NoTime'

def get_experiment_status():
    """Read completed and incomplete experiments"""
    completed = set()
    incomplete = set()
    
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.endswith('_incomplete'):
                        # Extract the base experiment name
                        base_name = line.replace('_incomplete', '')
                        incomplete.add(base_name)
                    elif line:
                        completed.add(line)
        except OSError as e:
            print(f"Warning: Could not read log file {log_file}: {e}")
    
    return completed, incomplete

def mark_experiment_incomplete(combo_key):
    """Mark experiment as incomplete in the log file"""
    try:
        with open(log_file, 'a') as f:
            f.write(f'{combo_key}_incomplete\n')
    except OSError as e:
        print(f"Warning: Could not mark {combo_key} as incomplete: {e}")

def mark_experiment_complete(combo_key):
    """Mark experiment as complete and remove incomplete marker"""
    try:
        # Read all lines
        lines = []
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
        
        # Remove incomplete marker if it exists
        incomplete_marker = f'{combo_key}_incomplete'
        if incomplete_marker in lines:
            lines.remove(incomplete_marker)
        
        # Add completion marker if not already present
        if combo_key not in lines:
            lines.append(combo_key)
        
        # Write back to file
        with open(log_file, 'w') as f:
            for line in lines:
                f.write(f'{line}\n')
                
    except OSError as e:
        print(f"Warning: Could not mark {combo_key} as complete: {e}")

def find_checkpoint_file(model_name, dataset_name, time_encoder_type):
    """Find the most recent checkpoint file for resuming"""
    checkpoint_pattern = f"./saved_models/{model_name}/{dataset_name}/*/{time_encoder_type}/*checkpoint*.pth"
    checkpoint_files = glob.glob(checkpoint_pattern)
    
    if not checkpoint_files:
        return None
    
    # Return the most recent checkpoint
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    return checkpoint_files[0]

def check_training_completion(model_name, dataset_name, time_encoder_type):
    """Check if training was completed by looking for final results"""
    result_pattern = f"./saved_results/{model_name}/{dataset_name}/{time_encoder_type}/*_seed0_{time_encoder_type}_*.json"
    result_files = glob.glob(result_pattern)
    return len(result_files) > 0

# Get current experiment status
completed, incomplete = get_experiment_status()
print(f"Found {len(completed)} completed and {len(incomplete)} incomplete experiments")

# Iterate over all combinations
for model, dataset in itertools.product(models, datasets):
    combo_key = f'{time_encoder}_{model}__{dataset}'
    
    # Skip if already completed
    if combo_key in completed:
        print(f"⏭️  Skipping already completed: {combo_key}")
        continue
    
    # Check if this experiment was incomplete and has a checkpoint
    checkpoint_file = None
    resume_from_checkpoint = False
    
    if combo_key in incomplete:
        checkpoint_file = find_checkpoint_file(model, dataset, time_encoder)
        if checkpoint_file:
            print(f"🔄 Found checkpoint for incomplete experiment: {combo_key}")
            print(f"   Checkpoint: {checkpoint_file}")
            resume_from_checkpoint = True
        else:
            print(f"⚠️  Incomplete experiment {combo_key} found but no checkpoint available")
    
    # Build the command
    command = [
        'python', '-m', 'DyGMamba.train_link_prediction',
        '--model_name', model,
        '--dataset_name', dataset,
        '--time_encoder_type', time_encoder,
        '--load_best_configs'
    ]
    
    # Add checkpoint resuming if available
    if resume_from_checkpoint and checkpoint_file:
        command.extend(['--resume_from_checkpoint', '--load_model_path', checkpoint_file])
        print(f"🔄 Resuming from checkpoint: {combo_key}")
    else:
        print(f"🚀 Starting new training: {combo_key}")
        # Mark as incomplete when starting
        mark_experiment_incomplete(combo_key)

    print(f"Command: {' '.join(command)}")
    print("-" * 80)
    
    try:
        # Run the command and display output directly in terminal
        result = subprocess.run(command, text=True, check=True, timeout=12*3600)  # 12 hour timeout
        
        # Check if training actually completed
        if check_training_completion(model, dataset, time_encoder):
            print("-" * 80)
            print(f"✅ Successfully completed: {combo_key}")
            mark_experiment_complete(combo_key)
        else:
            print("-" * 80)
            print(f"⚠️  Training finished but no results found: {combo_key}")
            
    except subprocess.TimeoutExpired:
        print("-" * 80)
        print(f"⏰ Training timeout (12h) for: {combo_key}")
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

print("\n🏁 Experiment batch completed!")