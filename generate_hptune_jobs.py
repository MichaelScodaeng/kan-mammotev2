#!/usr/bin/env python3
"""
Generate PBS array job scripts for parallel hyperparameter tuning.
Creates one job per configuration for maximum parallelization.
"""

import itertools
from pathlib import Path
from datetime import datetime

# Configuration
DATASETS = ['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
            'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']

MODELS = ['JODIE', 'TGAT', 'TGN', 'TCL', 'DyGFormer', 'DyGMamba']

TIME_ENCODERS = ['lete', 'kan_mammote_dual_kmote', 'mercer', 'time2vec']

LEARNING_RATES = [1e-4, 5e-4, 1e-3, 5e-3]
WEIGHT_DECAYS = [0.0, 1e-5, 1e-4, 1e-3]

# Fixed parameters
DATA_RATIO = 0.1
NUM_EPOCHS = 10
PATIENCE = 3
NUM_RUNS = 1
BATCH_SIZE = 200


def generate_config_file():
    """Generate a config file listing all experiments."""
    experiments = list(itertools.product(
        MODELS,
        DATASETS,
        TIME_ENCODERS,
        LEARNING_RATES,
        WEIGHT_DECAYS
    ))
    
    config_dir = Path('./hptune_configs')
    config_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_file = config_dir / f'hptune_jobs_{timestamp}.txt'
    
    with open(config_file, 'w') as f:
        f.write("# job_id\tmodel\tdataset\ttime_encoder\tlr\twd\n")
        for i, (model, dataset, time_encoder, lr, wd) in enumerate(experiments, 1):
            f.write(f"{i}\t{model}\t{dataset}\t{time_encoder}\t{lr}\t{wd}\n")
    
    print(f"Generated config file: {config_file}")
    print(f"Total jobs: {len(experiments)}")
    
    return config_file, len(experiments)


def generate_array_job_script(config_file, num_jobs):
    """Generate PBS array job script."""
    
    pbs_script = f"""#!/bin/bash
#PBS -N hptune_array
#PBS -l select=1:ncpus=2:mem=16gb:ngpus=1
#PBS -l walltime=2:00:00
#PBS -J 1-{num_jobs}
#PBS -j oe
#PBS -o ./hptune_logs/job_${{PBS_ARRAY_INDEX}}.log

# Fast hyperparameter tuning - array job
# Each job runs one configuration

cd $PBS_O_WORKDIR

# Create log directory
mkdir -p hptune_logs

# Activate virtual environment
source .venv/bin/activate

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Read configuration for this job
CONFIG_FILE="{config_file}"
JOB_LINE=$(sed -n "${{PBS_ARRAY_INDEX}}p" $CONFIG_FILE | grep -v '^#')

if [ -z "$JOB_LINE" ]; then
    echo "ERROR: Could not read configuration for job $PBS_ARRAY_INDEX"
    exit 1
fi

# Parse configuration
JOB_ID=$(echo $JOB_LINE | awk '{{print $1}}')
MODEL=$(echo $JOB_LINE | awk '{{print $2}}')
DATASET=$(echo $JOB_LINE | awk '{{print $3}}')
TIME_ENCODER=$(echo $JOB_LINE | awk '{{print $4}}')
LR=$(echo $JOB_LINE | awk '{{print $5}}')
WD=$(echo $JOB_LINE | awk '{{print $6}}')

echo "=========================================="
echo "Job ID: $PBS_ARRAY_INDEX (Config ID: $JOB_ID)"
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Time Encoder: $TIME_ENCODER"
echo "Learning Rate: $LR"
echo "Weight Decay: $WD"
echo "=========================================="

# Determine training script
TRAIN_SCRIPT="experiment_unified.py"
if [ ! -f "$TRAIN_SCRIPT" ]; then
    TRAIN_SCRIPT="train_link_prediction.py"
fi

if [ ! -f "$TRAIN_SCRIPT" ]; then
    echo "ERROR: Training script not found!"
    exit 1
fi

# Create output directory
TIMESTAMP=$(date +%Y%m%d)
SAVE_SUFFIX="_HPTUNE_${{TIMESTAMP}}"
ABLATION_DIR="./hyperparameter_tuning_results/${{MODEL}}_${{DATASET}}_${{TIME_ENCODER}}_lr${{LR}}_wd${{WD}}"
mkdir -p "$ABLATION_DIR"

# Run training
python $TRAIN_SCRIPT \\
    --model_name $MODEL \\
    --dataset_name $DATASET \\
    --time_encoder_type $TIME_ENCODER \\
    --learning_rate $LR \\
    --weight_decay $WD \\
    --data_ratio {DATA_RATIO} \\
    --num_epochs {NUM_EPOCHS} \\
    --patience {PATIENCE} \\
    --num_runs {NUM_RUNS} \\
    --batch_size {BATCH_SIZE} \\
    --gpu 0 \\
    --save_model_name_suffix "$SAVE_SUFFIX" \\
    --ablation_dir "$ABLATION_DIR" \\
    --load_best_configs \\
    --disable_progress_bar

EXIT_CODE=$?

echo "=========================================="
echo "Job completed with exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "=========================================="

exit $EXIT_CODE
"""
    
    script_file = Path('./run_hptune_array.sh')
    with open(script_file, 'w') as f:
        f.write(pbs_script)
    
    print(f"Generated array job script: {script_file}")
    print(f"\nTo submit: qsub {script_file}")
    
    return script_file


def generate_collector_script():
    """Generate script to collect results from all array jobs."""
    
    collector_script = """#!/usr/bin/env python3
\"\"\"
Collect and analyze results from hyperparameter tuning array jobs.
\"\"\"

import json
import glob
from pathlib import Path
from collections import defaultdict

def collect_results():
    results_dir = Path('./hyperparameter_tuning_results')
    all_results = []
    
    # Find all result directories
    for exp_dir in results_dir.glob('*_lr*_wd*'):
        config_name = exp_dir.name
        parts = config_name.split('_')
        
        # Parse config
        config = {
            'config_name': config_name,
            'model': parts[0] if len(parts) > 0 else 'unknown',
            'dataset': parts[1] if len(parts) > 1 else 'unknown',
            'time_encoder': '_'.join(parts[2:-2]) if len(parts) > 4 else 'unknown',
        }
        
        # Extract lr and wd
        for part in parts:
            if part.startswith('lr'):
                config['lr'] = float(part[2:])
            elif part.startswith('wd'):
                config['wd'] = float(part[2:])
        
        # Find result files
        json_files = list(exp_dir.rglob('*.json'))
        if json_files:
            try:
                with open(json_files[0], 'r') as f:
                    metrics = json.load(f)
                    config.update(metrics)
                    config['status'] = 'success'
            except Exception as e:
                config['status'] = 'error'
                config['error'] = str(e)
        else:
            config['status'] = 'no_results'
        
        all_results.append(config)
    
    # Save collected results
    output_file = results_dir / 'collected_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Collected {len(all_results)} results")
    print(f"Saved to: {output_file}")
    
    # Generate summary
    generate_summary(all_results, results_dir / 'summary_report.txt')
    
    return all_results


def generate_summary(results, output_file):
    \"\"\"Generate summary report.\"\"\"
    
    # Group by model, dataset, time_encoder
    grouped = defaultdict(list)
    for r in results:
        if r.get('status') == 'success':
            key = (r.get('model'), r.get('dataset'), r.get('time_encoder'))
            grouped[key].append(r)
    
    with open(output_file, 'w') as f:
        f.write("HYPERPARAMETER TUNING SUMMARY\\n")
        f.write("=" * 80 + "\\n\\n")
        
        for (model, dataset, time_encoder), configs in sorted(grouped.items()):
            f.write(f"\\n{model} + {dataset} + {time_encoder}\\n")
            f.write("-" * 60 + "\\n")
            
            # Find best by validation AP
            best = max(configs, key=lambda x: x.get('validate_ap', 0))
            
            f.write(f"  Best: LR={best.get('lr')}, WD={best.get('wd')}\\n")
            f.write(f"  Val AP: {best.get('validate_ap', 'N/A')}\\n")
            f.write(f"  Val AUC: {best.get('validate_auc', 'N/A')}\\n")
            
            # Top 3
            top3 = sorted(configs, key=lambda x: x.get('validate_ap', 0), reverse=True)[:3]
            f.write(f"\\n  Top 3:\\n")
            for i, cfg in enumerate(top3, 1):
                f.write(f"    {i}. LR={cfg.get('lr')}, WD={cfg.get('wd')}, "
                       f"AP={cfg.get('validate_ap', 'N/A'):.4f}\\n")
    
    print(f"Summary saved to: {output_file}")


if __name__ == '__main__':
    collect_results()
"""
    
    collector_file = Path('./collect_hptune_results.py')
    with open(collector_file, 'w') as f:
        f.write(collector_script)
    
    print(f"Generated results collector: {collector_file}")
    
    return collector_file


def main():
    print("="*80)
    print("HYPERPARAMETER TUNING - ARRAY JOB GENERATOR")
    print("="*80)
    print()
    
    # Generate config file
    config_file, num_jobs = generate_config_file()
    
    print()
    
    # Generate array job script
    script_file = generate_array_job_script(config_file, num_jobs)
    
    print()
    
    # Generate collector script
    collector_file = generate_collector_script()
    
    print()
    print("="*80)
    print("SETUP COMPLETE")
    print("="*80)
    print(f"\nTotal experiments: {num_jobs}")
    print(f"Config file: {config_file}")
    print(f"Job script: {script_file}")
    print(f"Collector: {collector_file}")
    print()
    print("To run:")
    print(f"  1. Submit jobs: qsub {script_file}")
    print(f"  2. Monitor: qstat -u $USER")
    print(f"  3. Collect results: python {collector_file}")
    print()


if __name__ == '__main__':
    main()
