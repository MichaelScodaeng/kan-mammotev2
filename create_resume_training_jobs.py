#!/usr/bin/env python3
"""
Resume training script for experiments that stopped at epoch 100 and need to be extended to 200.
This script creates job scripts to resume training from existing checkpoints.
"""

import os
import subprocess
from pathlib import Path

def create_resume_training_script():
    """Create PBS job scripts to resume training for all experiments."""
    
    # List of experiments that need to be resumed
    experiments_to_resume = [
        ("DyGMamba", "lastfm", "time2vec", "seed0"),
        ("DyGMamba", "mooc", "time2vec", "seed0"),
        ("DyGMamba", "uci", "time2vec", "seed0"),
        ("JODIE", "Contacts", "time2vec", "seed0"),
        ("JODIE", "SocialEvo", "time2vec", "seed0"),
        ("JODIE", "uci", "time2vec", "seed0"),
        ("JODIE", "wikipedia", "time2vec", "seed0"),
        ("TCL", "Contacts", "time2vec", "seed0"),
        ("TCL", "lastfm", "time2vec", "seed0"),
        ("TCL", "mooc", "time2vec", "seed0"),
        ("TCL", "reddit", "time2vec", "seed0"),
        ("TGN", "Flights", "time2vec", "seed0")
    ]
    
    # PBS template for resuming training
    pbs_template = """#!/bin/bash
#PBS -N resume_{model}_{dataset}_{time_encoder}
#PBS -l select=1:ncpus=8:mem=60gb:ngpus=1:gpu_model=a100
#PBS -l walltime=24:00:00
#PBS -q gpu
#PBS -o {output_file}
#PBS -e {error_file}

# Load required modules
module load python/3.11.0-ffypltn cuda/12.1.1-y3rfgp6

# Navigate to project directory  
cd /home/s2516027/kan-mammotev2

# Activate environment
source mambaforge/envs/py11_cuda121/bin/activate

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0

# Resume training from checkpoint
echo "===== RESUMING TRAINING ====="
echo "Model: {model}"
echo "Dataset: {dataset}"
echo "Time Encoder: {time_encoder}"
echo "Checkpoint: {checkpoint_path}"
echo "Resuming from epoch: 101"
echo "Target epochs: 200"
echo "=============================="

python -u experiments/train_link_prediction.py \\
    --model_name {model} \\
    --dataset_name {dataset} \\
    --time_encoder {time_encoder} \\
    --num_epochs 200 \\
    --num_runs 1 \\
    --seed 0 \\
    --resume_from_checkpoint {checkpoint_path} \\
    --validate_checkpoints \\
    --save_checkpoints \\
    --checkpoint_interval 10 \\
    --max_checkpoints_to_keep 5 \\
    --disable_progress_bar \\
    --gpu 0

echo "===== TRAINING COMPLETED ====="
"""

    print("=" * 80)
    print("CREATING RESUME TRAINING JOB SCRIPTS")
    print("=" * 80)
    
    job_scripts = []
    
    for model, dataset, time_encoder, seed in experiments_to_resume:
        # Construct checkpoint path
        experiment_name = f"{model}_{time_encoder}_{seed}"
        checkpoint_path = f"/home/s2516027/kan-mammotev2/saved_models/{model}/{dataset}/{experiment_name}/checkpoint_epoch_100.pth"
        
        # Verify checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"❌ SKIP: Checkpoint not found - {checkpoint_path}")
            continue
        
        # Create job script filename
        job_name = f"resume_{model.lower()}_{dataset.lower()}_{time_encoder}"
        job_script_path = f"/home/s2516027/kan-mammotev2/resume_jobs/{job_name}.sh"
        output_file = f"/home/s2516027/kan-mammotev2/resume_jobs/{job_name}.o"
        error_file = f"/home/s2516027/kan-mammotev2/resume_jobs/{job_name}.e"
        
        # Create job directory if it doesn't exist
        os.makedirs("/home/s2516027/kan-mammotev2/resume_jobs", exist_ok=True)
        
        # Generate job script content
        job_content = pbs_template.format(
            model=model,
            dataset=dataset,
            time_encoder=time_encoder,
            checkpoint_path=checkpoint_path,
            output_file=output_file,
            error_file=error_file
        )
        
        # Write job script
        with open(job_script_path, 'w') as f:
            f.write(job_content)
        
        # Make executable
        os.chmod(job_script_path, 0o755)
        
        job_scripts.append((job_script_path, model, dataset, time_encoder))
        print(f"✅ Created: {job_script_path}")
    
    print(f"\n📊 SUMMARY: Created {len(job_scripts)} job scripts")
    
    # Create submission script
    submission_script_path = "/home/s2516027/kan-mammotev2/submit_resume_jobs.sh"
    with open(submission_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Submit all resume training jobs\n\n")
        f.write("echo 'Submitting resume training jobs...'\n\n")
        
        for job_script_path, model, dataset, time_encoder in job_scripts:
            f.write(f"echo 'Submitting: {model}/{dataset}/{time_encoder}'\n")
            f.write(f"qsub {job_script_path}\n")
            f.write("sleep 2  # Small delay between submissions\n\n")
        
        f.write("echo 'All jobs submitted!'\n")
        f.write("echo 'Check status with: qstat -u $USER'\n")
    
    os.chmod(submission_script_path, 0o755)
    print(f"✅ Created submission script: {submission_script_path}")
    
    # Create monitoring script
    monitoring_script_path = "/home/s2516027/kan-mammotev2/monitor_resume_jobs.py"
    with open(monitoring_script_path, 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Monitor the progress of resume training jobs by checking validation metrics.
"""

import os
import glob
import pandas as pd
from pathlib import Path

def monitor_resume_progress():
    """Monitor progress of resumed training jobs."""
    
    experiments = [
        ("DyGMamba", "lastfm", "time2vec", "seed0"),
        ("DyGMamba", "mooc", "time2vec", "seed0"),
        ("DyGMamba", "uci", "time2vec", "seed0"),
        ("JODIE", "Contacts", "time2vec", "seed0"),
        ("JODIE", "SocialEvo", "time2vec", "seed0"),
        ("JODIE", "uci", "time2vec", "seed0"),
        ("JODIE", "wikipedia", "time2vec", "seed0"),
        ("TCL", "Contacts", "time2vec", "seed0"),
        ("TCL", "lastfm", "time2vec", "seed0"),
        ("TCL", "mooc", "time2vec", "seed0"),
        ("TCL", "reddit", "time2vec", "seed0"),
        ("TGN", "Flights", "time2vec", "seed0")
    ]
    
    print("=" * 100)
    print("RESUME TRAINING PROGRESS MONITOR")
    print("=" * 100)
    
    for model, dataset, time_encoder, seed in experiments:
        experiment_name = f"{model}_{time_encoder}_{seed}"
        
        # Check validation metrics file
        val_metrics_pattern = f"saved_metrics/{model}/{dataset}/{experiment_name}/val_metrics_*.csv"
        val_files = glob.glob(val_metrics_pattern)
        
        if val_files:
            val_file = val_files[0]
            try:
                df = pd.read_csv(val_file)
                max_epoch = df['epoch'].max()
                latest_score = df.iloc[-1]['average_precision'] + df.iloc[-1]['roc_auc']
                
                if max_epoch >= 200:
                    status = "✅ COMPLETED"
                elif max_epoch > 100:
                    status = f"🔄 IN PROGRESS (epoch {max_epoch}/200)"
                else:
                    status = "⏸️  NOT STARTED"
                
                print(f"{status:<20} {model:<12} {dataset:<15} latest_score: {latest_score:.4f}")
            except Exception as e:
                print(f"❌ ERROR reading {val_file}: {e}")
        else:
            print(f"❓ NO METRICS      {model:<12} {dataset:<15} - metrics file not found")
    
    print("=" * 100)

if __name__ == "__main__":
    monitor_resume_progress()
''')
    
    os.chmod(monitoring_script_path, 0o755)
    print(f"✅ Created monitoring script: {monitoring_script_path}")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS:")
    print("=" * 80)
    print("1. Review the generated job scripts in: resume_jobs/")
    print("2. Submit all jobs with: ./submit_resume_jobs.sh")
    print("3. Monitor progress with: python monitor_resume_jobs.py")
    print("4. Check job status with: qstat -u $USER")
    print("=" * 80)
    
    return job_scripts

if __name__ == "__main__":
    scripts = create_resume_training_script()