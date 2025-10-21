#!/usr/bin/env python3
"""
Fast Hyperparameter Tuning for KAN-MAMMOTE
==========================================

Efficiently tunes KAN-MAMMOTE hyperparameters across all GNN models and datasets.

Strategy:
- Uses 10% of training data (temporal prefix) for quick validation
- 10 epochs with patience=3 for early feedback
- Focused hyperparameter grid based on dataset size
- Single seed runs for speed
- Generates PBS job scripts for HPC execution

Author: KAN-MAMMOTE Team
Date: October 2025
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from itertools import product
import subprocess

# ============================================================================
# UNIFORM Hyperparameter Search Space (RECOMMENDED FOR FAIR COMPARISON)
# ============================================================================
# All datasets use the SAME hyperparameter grid for scientific rigor.
# This ensures fair comparison and avoids dataset-specific tuning bias.
#
# Rationale:
# - Baseline models (TGAT, TGN, etc.) use same settings across datasets
# - Shows KAN-MAMMOTE's general capability, not cherry-picked performance
# - Simpler analysis and easier to defend in academic papers
#
# Constraint: expert_dim * mamba_expand / mamba_headdim must be multiple of 8
# Valid combinations are automatically validated by is_valid_mamba_config()
# ============================================================================
DATASET_CONFIGS = {
    # Small datasets: Conservative capacity
    'Contacts': {
        'expert_dim': [64, 128],
        'mamba_d_state': [128, 256],
        'mamba_expand': [2],
        'dropout': [0.0, 0.1],
        'mamba_headdim': [64],
        'mamba_d_conv': [4]
    },
    'USLegis': {
        'expert_dim': [64, 128],
        'mamba_d_state': [128, 256],
        'mamba_expand': [2],
        'dropout': [0.0, 0.1],
        'mamba_headdim': [64],
        'mamba_d_conv': [4]
    },
    'Flights': {
        'expert_dim': [64, 128],
        'mamba_d_state': [128, 256],
        'mamba_expand': [2],
        'dropout': [0.0, 0.1],
        'mamba_headdim': [64],
        'mamba_d_conv': [4]
    },
    'UNvote': {
        'expert_dim': [64, 128],
        'mamba_d_state': [128, 256],
        'mamba_expand': [2],
        'dropout': [0.0, 0.1],
        'mamba_headdim': [64],
        'mamba_d_conv': [4]
    },
    
    # Medium datasets: Moderate capacity
    'wikipedia': {
        'expert_dim': [128, 256],
        'mamba_d_state': [256, 512],
        'mamba_expand': [2, 4],
        'dropout': [0.1, 0.2],
        'mamba_headdim': [64],
        'mamba_d_conv': [4]
    },
    'reddit': {
        'expert_dim': [128, 256],
        'mamba_d_state': [256, 512],
        'mamba_expand': [2, 4],
        'dropout': [0.1, 0.2],
        'mamba_headdim': [64],
        'mamba_d_conv': [4]
    },
    'mooc': {
        'expert_dim': [128, 256],
        'mamba_d_state': [256, 512],
        'mamba_expand': [2, 4],
        'dropout': [0.1, 0.2],
        'mamba_headdim': [64],
        'mamba_d_conv': [4]
    },
    'lastfm': {
        'expert_dim': [128, 256],
        'mamba_d_state': [256, 512],
        'mamba_expand': [2, 4],
        'dropout': [0.1, 0.2],
        'mamba_headdim': [64],
        'mamba_d_conv': [4]
    },
    'enron': {
        'expert_dim': [128, 256],
        'mamba_d_state': [256, 512],
        'mamba_expand': [2, 4],
        'dropout': [0.1, 0.2],
        'mamba_headdim': [64],
        'mamba_d_conv': [4]
    },
    'UNtrade': {
        'expert_dim': [128, 256],
        'mamba_d_state': [256, 512],
        'mamba_expand': [2, 4],
        'dropout': [0.1, 0.2],
        'mamba_headdim': [64],
        'mamba_d_conv': [4]
    },
    
    # Large datasets: Higher capacity
    'CanParl': {
        'expert_dim': [256, 512],
        'mamba_d_state': [512, 1024],
        'mamba_expand': [4, 8],
        'dropout': [0.2, 0.3],
        'mamba_headdim': [64, 128],
        'mamba_d_conv': [4]
    },
    'SocialEvo': {
        'expert_dim': [256, 512],
        'mamba_d_state': [512, 1024],
        'mamba_expand': [4, 8],
        'dropout': [0.2, 0.3],
        'mamba_headdim': [64, 128],
        'mamba_d_conv': [4]
    },
    'uci': {
        'expert_dim': [128, 256],
        'mamba_d_state': [256, 512],
        'mamba_expand': [2, 4],
        'dropout': [0.1, 0.2],
        'mamba_headdim': [64],
        'mamba_d_conv': [4]
    }
}

# Uniform search space for ALL datasets
UNIFORM_CONFIG = {
    'expert_dim': [64, 128, 256],           # 3 values: small, medium, large
    'mamba_d_state': [128, 256, 512],       # 3 values: state capacity
    'mamba_expand': [2, 4],                 # 2 values: expansion factor
    'dropout': [0,0.1,0.2,0.3],                       # 1 value: use model-specific default
    'mamba_headdim': [64],                  # 1 value: standard head dimension
    'mamba_d_conv': [4]                     # 1 value: standard convolution
}

# Dataset list (all use UNIFORM_CONFIG)
DATASET_CONFIGS = {
    # Small datasets
    'Contacts': UNIFORM_CONFIG,
    'USLegis': UNIFORM_CONFIG,
    'Flights': UNIFORM_CONFIG,
    'UNvote': UNIFORM_CONFIG,
    
    # Medium datasets
    'wikipedia': UNIFORM_CONFIG,
    'reddit': UNIFORM_CONFIG,
    'mooc': UNIFORM_CONFIG,
    'lastfm': UNIFORM_CONFIG,
    'enron': UNIFORM_CONFIG,
    'UNtrade': UNIFORM_CONFIG,
    'uci': UNIFORM_CONFIG,
    
    # Large datasets
    'CanParl': UNIFORM_CONFIG,
    'SocialEvo': UNIFORM_CONFIG,
}

# ============================================================================
# ALTERNATIVE: Dataset-Specific Configs (if you prefer optimization)
# ============================================================================
# Uncomment this section and comment out UNIFORM_CONFIG above if you want
# dataset-specific tuning (less scientifically rigorous but potentially better results)
# ============================================================================
"""
DATASET_CONFIGS = {
    # Small datasets: Conservative capacity
    'Contacts': {
        'expert_dim': [64, 128],
        'mamba_d_state': [128, 256],
        'mamba_expand': [2],
        'dropout': [0.0, 0.1],
        'mamba_headdim': [64],
        'mamba_d_conv': [4]
    },
    # ... (rest of dataset-specific configs)
}
"""

# GNN models to tune (KAN-MAMMOTE compatible)
GNN_MODELS = [
    'TGAT',
    'TGN',
    'TCL',
    'JODIE',
    'DyGFormer',
    'DyGMamba'
]

# Fixed hyperparameters for fast tuning
FAST_TUNING_PARAMS = {
    'train_only_ratio': 0.1,  # Use 10% of training data
    'num_epochs': 10,   # Quick training
    'patience': 3,      # Early stopping
    'num_runs': 1,      # Single seed for speed
    'seed': 0,          # Fixed seed
    'test_interval_epochs': 1,  # Check validation every 1 epoch
    'checkpoint_strategy': 'minimal',  # Minimal checkpointing overhead
    'disable_progress_bar': True  # Clean logs for batch jobs
}

def is_valid_mamba_config(expert_dim, mamba_expand, mamba_headdim):
    """
    Validate Mamba2 configuration constraint.
    
    Mamba2 requires: expert_dim * mamba_expand / mamba_headdim = ngroups (must be multiple of 8)
    
    Args:
        expert_dim: Dimension of expert output
        mamba_expand: Mamba expansion factor
        mamba_headdim: Mamba head dimension
    
    Returns:
        bool: True if configuration is valid
    """
    # Calculate ngroups
    inner_dim = expert_dim * mamba_expand
    
    # Check if divisible by headdim
    if inner_dim % mamba_headdim != 0:
        return False
    
    ngroups = inner_dim // mamba_headdim
    
    # Check if ngroups is multiple of 8
    if ngroups % 8 != 0:
        return False
    
    return True

def generate_config_grid(dataset):
    """
    Generate hyperparameter configurations for a dataset.
    Filters out invalid configurations that violate Mamba2 constraints.
    """
    if dataset not in DATASET_CONFIGS:
        print(f"⚠️  Warning: No config for {dataset}, using default medium config")
        config_space = DATASET_CONFIGS['wikipedia']
    else:
        config_space = DATASET_CONFIGS[dataset]
    
    # Generate Cartesian product of all hyperparameters
    all_configs = []
    invalid_configs = []
    
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
        
        # Validate Mamba2 constraint
        if is_valid_mamba_config(config['expert_dim'], config['mamba_expand'], config['mamba_headdim']):
            all_configs.append(config)
        else:
            invalid_configs.append(config)
    
    # Print validation summary
    if invalid_configs:
        print(f"   ⚠️  Filtered out {len(invalid_configs)} invalid configurations (Mamba2 constraint violation)")
        print(f"   ✓  {len(all_configs)} valid configurations remain")
    
    return all_configs

def create_experiment_command(dataset, model, config, config_idx):
    """Create command to run single experiment"""
    
    cmd = [
        'python', 'experiments/train_link_prediction.py',
        '--dataset_name', dataset,
        '--model_name', model,
        '--time_encoder_type', 'kan_mammote_dual_kmote',
        
        # KAN-MAMMOTE hyperparameters
        '--expert_dim', str(config['expert_dim']),
        '--mamba_d_state', str(config['mamba_d_state']),
        '--mamba_expand', str(config['mamba_expand']),
        '--dropout', str(config['dropout']),
        '--mamba_headdim', str(config['mamba_headdim']),
        '--mamba_d_conv', str(config['mamba_d_conv']),
        
        # Fast tuning parameters
        '--data_ratio', str(1.0) if model in ['JODIE', 'TCL'] else str(FAST_TUNING_PARAMS['train_only_ratio']),
        '--num_epochs', str(FAST_TUNING_PARAMS['num_epochs']),
        '--patience', str(FAST_TUNING_PARAMS['patience']),
        '--num_runs', str(FAST_TUNING_PARAMS['num_runs']),
        '--seed', str(FAST_TUNING_PARAMS['seed']),
        '--test_interval_epochs', str(FAST_TUNING_PARAMS['test_interval_epochs']),
        '--checkpoint_strategy', FAST_TUNING_PARAMS['checkpoint_strategy'],
        
        # Experiment tracking
        '--save_model_name_suffix', f'hptune_c{config_idx:03d}_ed{config["expert_dim"]}_ds{config["mamba_d_state"]}_ex{config["mamba_expand"]}',
        '--ablation_dir', f'./hptune_results/{dataset}/{model}'
    ]
    
    # Add --train_only_ratio flag (no value) for JODIE and TCL
    if model in ['JODIE', 'TCL']:
        cmd.append('--train_only_ratio')
    
    # Add progress bar flag conditionally
    if FAST_TUNING_PARAMS['disable_progress_bar']:
        cmd.append('--disable_progress_bar')
    
    return ' '.join(cmd)

def create_pbs_job_script(dataset, model, config_idx, config, output_dir):
    """Create PBS job script for HPC execution"""
    
    job_name = f"hptune_{dataset}_{model}_c{config_idx:03d}"
    script_path = output_dir / f"{job_name}.sh"
    
    # Create command
    cmd = create_experiment_command(dataset, model, config, config_idx)
    
    # Calculate ngroups for validation info
    inner_dim = config['expert_dim'] * config['mamba_expand']
    ngroups = inner_dim // config['mamba_headdim']
    
    # PBS script content
    pbs_script = f"""#!/bin/bash
#PBS -N {job_name}
#PBS -l select=1:ncpus=4:mem=16gb:ngpus=1:gpu_type=RTX6000
#PBS -l walltime=02:00:00
#PBS -j oe
#PBS -o {output_dir}/logs/{job_name}.log

# Load environment
cd $PBS_O_WORKDIR
source /home/s2516027/kan-mammotev2/.venv/bin/activate

# Print job info
echo "========================================="
echo "Job: {job_name}"
echo "Dataset: {dataset}"
echo "Model: {model}"
echo "Config Index: {config_idx}"
echo "Config: {json.dumps(config, indent=2)}"
echo ""
echo "Mamba2 Validation:"
echo "  expert_dim × mamba_expand = {config['expert_dim']} × {config['mamba_expand']} = {inner_dim}"
echo "  inner_dim / mamba_headdim = {inner_dim} / {config['mamba_headdim']} = {ngroups} (ngroups)"
echo "  ngroups % 8 = {ngroups % 8} ✓ (valid)" 
echo ""
echo "Start Time: $(date)"
echo "========================================="

# Run experiment
{cmd}

# Print completion
echo "========================================="
echo "End Time: $(date)"
echo "========================================="
"""
    
    # Write script
    with open(script_path, 'w') as f:
        f.write(pbs_script)
    
    # Make executable
    os.chmod(script_path, 0o755)
    
    return script_path

def create_summary_file(output_dir, dataset, model, configs):
    """Create summary of all configurations"""
    
    summary_path = output_dir / f"summary_{dataset}_{model}.json"
    
    summary = {
        'dataset': dataset,
        'model': model,
        'num_configs': len(configs),
        'fast_tuning_params': FAST_TUNING_PARAMS,
        'configs': configs,
        'generated_at': datetime.now().isoformat()
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary_path

def create_submit_all_script(job_scripts, output_dir):
    """Create script to submit all PBS jobs"""
    
    submit_script_path = output_dir / "submit_all_jobs.sh"
    
    script_content = "#!/bin/bash\n\n"
    script_content += "# Auto-generated script to submit all hyperparameter tuning jobs\n"
    script_content += f"# Generated: {datetime.now().isoformat()}\n"
    script_content += f"# Total jobs: {len(job_scripts)}\n\n"
    
    for job_script in job_scripts:
        script_content += f"qsub {job_script}\n"
        script_content += "sleep 1  # Avoid overwhelming scheduler\n"
    
    with open(submit_script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(submit_script_path, 0o755)
    
    return submit_script_path

def main():
    parser = argparse.ArgumentParser(description='Generate KAN-MAMMOTE hyperparameter tuning jobs')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Datasets to tune (default: all)')
    parser.add_argument('--models', type=str, nargs='+', default=None,
                        help='Models to tune (default: all)')
    parser.add_argument('--output_dir', type=str, default='./hptune_jobs',
                        help='Output directory for job scripts')
    parser.add_argument('--dry_run', action='store_true',
                        help='Generate scripts without submitting')
    parser.add_argument('--max_configs', type=int, default=None,
                        help='Maximum configs per dataset/model (for testing)')
    parser.add_argument('--auto_submit', action='store_true',
                        help='Automatically submit jobs without prompting')
    
    args = parser.parse_args()
    
    # Select datasets and models
    datasets = args.datasets if args.datasets else list(DATASET_CONFIGS.keys())
    models = args.models if args.models else GNN_MODELS
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'logs').mkdir(exist_ok=True)
    
    print("=" * 80)
    print("KAN-MAMMOTE Hyperparameter Tuning Job Generator")
    print("=" * 80)
    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Output directory: {output_dir}")
    print(f"Fast tuning params: {json.dumps(FAST_TUNING_PARAMS, indent=2)}")
    print("=" * 80)
    
    all_job_scripts = []
    total_configs = 0
    
    # Generate jobs for each dataset/model combination
    for dataset in datasets:
        for model in models:
            print(f"\n📊 Generating jobs for {dataset} / {model}...")
            
            # Generate configurations
            configs = generate_config_grid(dataset)
            
            # Limit configs if specified
            if args.max_configs:
                configs = configs[:args.max_configs]
            
            print(f"   Total configurations: {len(configs)}")
            total_configs += len(configs)
            
            # Create job scripts
            job_scripts = []
            for idx, config in enumerate(configs):
                script_path = create_pbs_job_script(
                    dataset, model, idx, config, output_dir
                )
                job_scripts.append(script_path)
            
            all_job_scripts.extend(job_scripts)
            
            # Create summary file
            summary_path = create_summary_file(output_dir, dataset, model, configs)
            print(f"   ✅ Created {len(job_scripts)} job scripts")
            print(f"   📄 Summary saved to: {summary_path}")
    
    # Create submit all script
    submit_script = create_submit_all_script(all_job_scripts, output_dir)
    
    print("\n" + "=" * 80)
    print(f"✅ Generated {len(all_job_scripts)} total job scripts")
    print(f"   Total configurations to test: {total_configs}")
    print(f"   Estimated time per job: ~30-60 minutes")
    print(f"   Total estimated time (sequential): ~{len(all_job_scripts) * 0.75:.1f} hours")
    print(f"   Total estimated time (parallel): ~1-2 hours (with sufficient GPUs)")
    print("=" * 80)
    
    print(f"\n📜 Submit all jobs with:")
    print(f"   bash {submit_script}")
    print(f"\n   Or submit individual jobs from: {output_dir}/")
    
    # Create analysis script
    create_analysis_script(output_dir)
    
    print(f"\n📊 Analyze results with:")
    print(f"   python {output_dir}/analyze_results.py")
    
    if not args.dry_run:
        if sys.stdin.isatty() and not args.auto_submit:
            try:
                response = input("\n🚀 Submit all jobs now? [y/N]: ")
                if response.lower() == 'y':
                    subprocess.run(['bash', str(submit_script)])
                    print("✅ Jobs submitted!")
                else:
                    print("Jobs not submitted. Run the submit script manually when ready.")
            except (EOFError, KeyboardInterrupt):
                print("\nJobs not submitted. Run the submit script manually when ready.")
        elif args.auto_submit:
            print("\n🚀 Auto-submitting jobs...")
            subprocess.run(['bash', str(submit_script)])
            print("✅ Jobs submitted!")
        else:
            print("\n💡 Running in non-interactive mode. Jobs not submitted.")
            print(f"   To submit, run: bash {submit_script}")

def create_analysis_script(output_dir):
    """Create script to analyze tuning results"""
    
    analysis_script = output_dir / "analyze_results.py"
    
    content = '''#!/usr/bin/env python3
"""
Analyze hyperparameter tuning results and find best configurations.
"""

import json
import pandas as pd
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='./hptune_results')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    all_results = []
    
    # Collect all results
    for result_file in results_dir.rglob('*.json'):
        if 'summary' not in result_file.name:
            try:
                with open(result_file) as f:
                    data = json.load(f)
                
                # Extract key metrics
                result = {
                    'dataset': result_file.parts[-3],
                    'model': result_file.parts[-2],
                    'config': result_file.stem,
                    'test_ap': data.get('test metrics', {}).get('average_precision', 0.0),
                    'test_auc': data.get('test metrics', {}).get('roc_auc', 0.0),
                    'val_ap': data.get('validate metrics', {}).get('average_precision', 0.0),
                }
                
                all_results.append(result)
            except Exception as e:
                print(f"Error reading {result_file}: {e}")
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    if len(df) == 0:
        print("No results found!")
        return
    
    # Find best configs per dataset/model
    print("\\n" + "="*80)
    print("Best Configurations per Dataset/Model")
    print("="*80)
    
    for (dataset, model), group in df.groupby(['dataset', 'model']):
        best = group.loc[group['test_ap'].idxmax()]
        print(f"\\n{dataset} / {model}:")
        print(f"  Config: {best['config']}")
        print(f"  Test AP: {best['test_ap']:.4f}")
        print(f"  Test AUC: {best['test_auc']:.4f}")
    
    # Save summary
    summary_file = results_dir / "best_configs_summary.csv"
    best_configs = df.loc[df.groupby(['dataset', 'model'])['test_ap'].idxmax()]
    best_configs.to_csv(summary_file, index=False)
    
    print(f"\\n✅ Summary saved to: {summary_file}")

if __name__ == '__main__':
    main()
'''
    
    with open(analysis_script, 'w') as f:
        f.write(content)
    
    os.chmod(analysis_script, 0o755)

if __name__ == '__main__':
    main()
