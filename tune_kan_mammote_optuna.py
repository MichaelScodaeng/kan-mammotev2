#!/usr/bin/env python3
"""
KAN-MAMMOTE Optuna Hyperparameter Tuning
=========================================

Uses Optuna with Hyperband pruning to efficiently tune KAN-MAMMOTE hyperparameters.
This approach provides a professional-grade alternative to manual Hyperband implementation.

Key Features:
- Optuna's battle-tested Hyperband implementation
- Automatic state management and resume capability
- Built-in constraint handling for Mamba2 architecture
- Web dashboard for monitoring progress
- SQLite database storage for persistence

Usage:
    python tune_kan_mammote_optuna.py --dataset wikipedia --model TGAT
    python tune_kan_mammote_optuna.py --dataset wikipedia --model TGAT --n_trials 50
    python tune_kan_mammote_optuna.py --multi_dataset  # Tune across multiple datasets
    python tune_kan_mammote_optuna.py --study_name "kan_mammote_final" --storage "sqlite:///my_tuning.db"
"""

import optuna
import sys
import os
import argparse
import json
from pathlib import Path
from datetime import datetime
import time

# Add parent directory to path to import the training script's functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the refactored training function and arg parser
from experiments.train_link_prediction_tune import run_training_session
from utils.load_configs import get_link_prediction_args

# Import constraint validation
def is_valid_mamba_config(expert_dim, mamba_expand, mamba_headdim):
    """Validate Mamba2 configuration constraint"""
    inner_dim = expert_dim * mamba_expand
    if inner_dim % mamba_headdim != 0:
        return False
    ngroups = inner_dim // mamba_headdim
    return ngroups % 8 == 0

def create_training_command_from_config(config_file, additional_args=None):
    """
    Create a complete training command from saved configuration file.
    
    Args:
        config_file (str): Path to the saved best config JSON file
        additional_args (list): Additional command line arguments to append
        
    Returns:
        list: Complete command line arguments for training
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    dataset = config['dataset']
    model = config['model']
    best_params = config['best_params']
    fixed_params = config.get('fixed_params', {})
    
    # Build complete command
    cmd = [
        'python', 'experiments/train_link_prediction.py',
        '--dataset_name', dataset,
        '--model_name', model,
    ]
    
    # Add all tuned parameters
    for param, value in best_params.items():
        cmd.extend([f'--{param}', str(value)])
    
    # Add fixed parameters
    for param, value in fixed_params.items():
        cmd.extend([f'--{param}', str(value)])
    
    # Add any additional arguments
    if additional_args:
        cmd.extend(additional_args)
    
    return cmd

def print_reproduction_info(config_file):
    """Print information for reproducing the best configuration"""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    print(f"\n📋 REPRODUCTION INFORMATION")
    print(f"=" * 50)
    print(f"Config file: {config_file}")
    print(f"Best validation AP: {config['best_validation_ap']:.4f}")
    print(f"Trial number: {config['best_trial_number']}")
    print(f"Total trials: {config['total_trials']}")
    
    print(f"\n🔧 COMPLETE HYPERPARAMETERS:")
    print(f"Architecture:")
    for param, value in config['best_params'].items():
        if param in ['expert_dim', 'mamba_d_state', 'mamba_expand', 'time_feat_dim', 'num_mixtures', 'dropout']:
            print(f"  {param}: {value}")
    
    print(f"Training:")
    for param, value in config['best_params'].items():
        if param in ['learning_rate', 'weight_decay', 'batch_size', 'max_grad_norm', 'optimizer']:
            print(f"  {param}: {value}")
    
    print(f"Fixed:")
    for param, value in config.get('fixed_params', {}).items():
        print(f"  {param}: {value}")
    
    # Generate reproduction command
    cmd = create_training_command_from_config(config_file, ['--save_model_name_suffix', 'reproduced'])
    print(f"\n🚀 REPRODUCTION COMMAND:")
    print(" ".join(cmd))
    print("")

def create_objective(dataset='wikipedia', model='TGAT', num_epochs=15, ablation_dir='./optuna_results'):
    """
    Create an objective function for a specific dataset/model combination.
    
    Args:
        dataset: Dataset name to tune on
        model: Model name to tune
        num_epochs: Maximum epochs for training
        ablation_dir: Directory to save results
        
    Returns:
        objective: Function that Optuna will optimize
    """
    
    def objective(trial: optuna.Trial) -> float:
        """
        The objective function that Optuna will optimize.
        A "trial" represents a single run with a specific set of hyperparameters.
        """
        try:
            # 1. Define the COMPLETE hyperparameter search space
            # Architecture parameters
            expert_dim = trial.suggest_categorical("expert_dim", [64, 128, 256])
            mamba_d_state = trial.suggest_categorical("mamba_d_state", [128, 256, 512])
            mamba_expand = trial.suggest_categorical("mamba_expand", [2, 4])

            dropout = trial.suggest_float("dropout", 0.0, 0.3, step=0.1)
            encoder_dropout = trial.suggest_float("encoder_dropout", 0.0, 0.3, step=0.1)  # Tie encoder dropout to overall dropout
            mamba_headdim = trial.suggest_categorical("mamba_headdim", [16, 32, 64,128])  # Fixed as in original config
            mamba_d_conv = 4    # Fixed as in original config
            
            
            # Training parameters - Make it like baseline first!
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [200])

            # Training parameters - CRITICAL for reproducibility!
            # Allow disabling weight decay while keeping log-sampling for positive values
            weight_decay = trial.suggest_float("weight_decay", 1e-12, 1e-2, log=True)

            max_grad_norm = trial.suggest_categorical("max_grad_norm", [1.0])
            
            
            # Handle architectural constraints - prune invalid combinations early
            if not is_valid_mamba_config(expert_dim, mamba_expand, mamba_headdim):
                trial.set_user_attr("pruning_reason", "Invalid Mamba2 architecture constraint")
                raise optuna.exceptions.TrialPruned()

            # 2. Build command line arguments for this trial
            trial_suffix = f"trial_{trial.number}"
            
            # Create sys.argv with ALL the parameters we want to set
            trial_argv = [
                'train_link_prediction_tune.py',
                '--dataset_name', dataset,
                '--model_name', model,
                '--time_encoder_type', 'kan_mammote_dual_kmote',
                # Architecture parameters
                '--expert_dim', str(expert_dim),
                '--mamba_d_state', str(mamba_d_state),
                '--mamba_expand', str(mamba_expand),
                '--encoder_dropout', str(encoder_dropout),
                '--mamba_headdim', str(mamba_headdim),
                '--mamba_d_conv', str(mamba_d_conv),
                '--dropout', str(dropout),
                '--batch_size', str(batch_size),
                # Training parameters
                '--learning_rate', str(learning_rate),
                '--weight_decay', str(weight_decay),
                '--max_grad_norm', str(max_grad_norm),
                '--num_epochs', str(num_epochs),
                # Fixed training parameters
                '--patience', '5',
                '--disable_progress_bar',
                '--num_runs', '1',
                '--seed', '0',
                '--save_model_name_suffix', trial_suffix,
                '--ablation_dir', ablation_dir
            ]
            
            # 3. Temporarily replace sys.argv and get args
            original_argv = sys.argv.copy()
            sys.argv = trial_argv
            args = get_link_prediction_args(is_evaluation=False)
            sys.argv = original_argv  # Restore original
            
            # 4. Debug: Print what we actually got
            # Use getattr with defaults so objective can run when only a subset of
            # parameters are provided via `trial_argv`. This avoids AttributeError
            # if get_link_prediction_args() doesn't set every optional attr.
            print(f"🐛 DEBUG Trial {trial.number} - Final parameters:")
            print(f"   model_name: {getattr(args, 'model_name', '<unset>')}")
            print(f"   dataset_name: {getattr(args, 'dataset_name', '<unset>')}")
            print(f"   time_encoder_type: {getattr(args, 'time_encoder_type', '<unset>')}")
            print(f"   Architecture:")
            print(f"     expert_dim: {getattr(args, 'expert_dim', '<unset>')}")
            print(f"     mamba_d_state: {getattr(args, 'mamba_d_state', '<unset>')}")
            print(f"     mamba_expand: {getattr(args, 'mamba_expand', '<unset>')}")
            print(f"     mamba_headdim: {getattr(args, 'mamba_headdim', '<unset>')}")
            print(f"     time_feat_dim: {getattr(args, 'time_feat_dim', '<unset>')}")
            print(f"     num_mixtures: {getattr(args, 'num_mixtures', '<unset>')}")
            print(f"     encoder_dropout: {getattr(args, 'encoder_dropout', '<unset>')}")
            print(f"   Training:")
            print(f"     learning_rate: {getattr(args, 'learning_rate', '<unset>')}")
            print(f"     weight_decay: {getattr(args, 'weight_decay', '<unset>')}")
            print(f"     batch_size: {getattr(args, 'batch_size', '<unset>')}")
            print(f"     max_grad_norm: {getattr(args, 'max_grad_norm', '<unset>')}")
            print(f"     optimizer: {getattr(args, 'optimizer', '<unset>')}")
            print(f"     num_epochs: {getattr(args, 'num_epochs', '<unset>')}")

            # 5. Execute the training session and return the metric
            validation_ap = run_training_session(args=args, trial=trial)
            
            if validation_ap is None:
                # Handle case where training fails
                trial.set_user_attr("error_reason", "Training returned None")
                return 0.0
            # Save trial parameters and argparse args for reproducibility
            try:
                # Create results directory mirroring saved_results layout
                results_dir = os.path.join(ablation_dir, 'saved_results', model, dataset)
                os.makedirs(results_dir, exist_ok=True)

                # Prepare serializable args dict
                def sanitize(obj):
                    if obj is None or isinstance(obj, (str, bool, int, float)):
                        return obj
                    if isinstance(obj, (list, tuple)):
                        return [sanitize(x) for x in obj]
                    if isinstance(obj, dict):
                        return {k: sanitize(v) for k, v in obj.items()}
                    try:
                        return str(obj)
                    except Exception:
                        return None

                args_dict = vars(args).copy() if hasattr(args, '__dict__') else dict(args)
                args_dict = sanitize(args_dict)

                payload = {
                    'timestamp': datetime.now().isoformat(),
                    'dataset': dataset,
                    'model': model,
                    'trial_number': trial.number,
                    'trial_params': sanitize(trial.params),
                    'args': args_dict,
                    'validation_ap': validation_ap,
                }

                fname = f"{model}_{dataset}_trial_{trial.number}_{int(time.time())}.json"
                with open(os.path.join(results_dir, fname), 'w') as wf:
                    json.dump(payload, wf, indent=2)
            except Exception as e:
                print(f"Warning: failed to save trial params for trial {trial.number}: {e}")

            return validation_ap
            
        except optuna.exceptions.TrialPruned:
            # Re-raise pruning exceptions
            raise
        except Exception as e:
            # Handle potential errors during training (e.g., OOM, CUDA errors)
            print(f"Trial {trial.number} failed with error: {e}")
            trial.set_user_attr("error_reason", str(e))
            # Return a low value indicating failure
            return 0.0
    
    return objective

def run_single_dataset_tuning(dataset, model, n_trials=100, num_epochs=15, 
                            study_name=None, storage=None, resume=True):
    """
    Run Optuna tuning for a single dataset/model combination.
    
    Args:
        dataset: Dataset name
        model: Model name  
        n_trials: Number of trials to run
        num_epochs: Maximum epochs per trial
        study_name: Name for the Optuna study
        storage: Storage URL (e.g., "sqlite:///tuning.db")
        resume: Whether to resume existing study
        
    Returns:
        study: Completed Optuna study
    """
    
    # Create study name if not provided
    if study_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        study_name = f"kan_mammote_{dataset}_{model}_{timestamp}"
    
    # Create storage URL if not provided
    if storage is None:
        storage = f"sqlite:///optuna_results/{study_name}.db"
    
    # Ensure storage directory exists
    storage_dir = Path(storage.split('///')[-1]).parent
    storage_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting Optuna tuning for {dataset}/{model}")
    print(f"📋 Study name: {study_name}")
    print(f"💾 Storage: {storage}")
    print(f"🎯 Target: {n_trials} trials, {num_epochs} epochs max")
    
    # 1. Create the pruner: Hyperband algorithm
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=3,          # Minimum epochs (r)
        max_resource=num_epochs, # Maximum epochs (R)  
        reduction_factor=3,      # Halving factor (eta)
    )

    # 2. Create a study: This manages the optimization process
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",    # Maximize validation average precision
        pruner=pruner,
        storage=storage,
        load_if_exists=resume,   # Resume if study exists
    )

    # 3. Create objective function for this dataset/model
    ablation_dir = f'./optuna_results/{dataset}/{model}'
    objective = create_objective(
        dataset=dataset,
        model=model, 
        num_epochs=num_epochs,
        ablation_dir=ablation_dir
    )

    # 4. Start the optimization
    print(f"🔥 Starting optimization...")
    study.optimize(objective, n_trials=n_trials)

    # 5. Print results
    print(f"\n{'='*80}")
    print(f"OPTUNA TUNING RESULTS: {dataset}/{model}")
    print(f"{'='*80}")
    print(f"Study statistics:")
    print(f"  ├─ Number of finished trials: {len(study.trials)}")
    print(f"  ├─ Number of pruned trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")
    print(f"  ├─ Number of complete trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}")
    print(f"  └─ Number of failed trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}")
    
    if study.best_trial:
        print(f"\nBest trial:")
        print(f"  ├─ Value (Validation AP): {study.best_trial.value:.4f}")
        print(f"  ├─ Trial number: {study.best_trial.number}")
        print(f"  └─ Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"      ├─ {key}: {value}")
        
        # Normalize and save the best configuration with COMPLETE parameter set
        best_params = study.best_trial.params.copy()

        best_config = {
            'dataset': dataset,
            'model': model,
            'best_validation_ap': study.best_trial.value,
            'best_trial_number': study.best_trial.number,
            # All tuned parameters (normalized)
            'best_params': best_params,
            # Additional fixed parameters for complete reproducibility
            'fixed_params': {
                'time_encoder_type': 'kan_mammote_dual_kmote',
                'mamba_headdim': 64,
                'mamba_d_conv': 4,
                'patience': 5,
                'num_runs': 1,
                'seed': 0,
                'num_epochs': num_epochs
            },
            'study_name': study_name,
            'total_trials': len(study.trials),
            'timestamp': datetime.now().isoformat(),
            'optuna_version': optuna.__version__
        }
        
        config_file = f'./optuna_results/{dataset}_{model}_best_config.json'
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(best_config, f, indent=2)
        print(f"💾 Best config saved to: {config_file}")
    else:
        print(f"\n❌ No successful trials found!")
    
    return study

def run_multi_dataset_tuning(datasets, models, n_trials_per_combo=50, num_epochs=15,
                           base_study_name="kan_mammote_multi", storage_dir="./optuna_results/studies"):
    """
    Run Optuna tuning across multiple datasets and models.
    
    Args:
        datasets: List of dataset names
        models: List of model names
        n_trials_per_combo: Number of trials per dataset/model combination
        num_epochs: Maximum epochs per trial
        base_study_name: Base name for studies
        storage_dir: Directory to store SQLite databases
        
    Returns:
        dict: Results summary
    """
    
    os.makedirs(storage_dir, exist_ok=True)
    
    total_combinations = len(datasets) * len(models)
    print(f"🚀 Starting multi-dataset tuning")
    print(f"📊 Combinations: {len(datasets)} datasets × {len(models)} models = {total_combinations}")
    print(f"🎯 Trials per combination: {n_trials_per_combo}")
    print(f"⏱️  Total trials: {total_combinations * n_trials_per_combo}")
    
    results = {}
    
    for i, (dataset, model) in enumerate([(d, m) for d in datasets for m in models]):
        print(f"\n[{i+1}/{total_combinations}] Processing {dataset}/{model}...")
        
        # Create unique study name and storage for this combination
        study_name = f"{base_study_name}_{dataset}_{model}"
        storage = f"sqlite:///{storage_dir}/{study_name}.db"
        
        try:
            study = run_single_dataset_tuning(
                dataset=dataset,
                model=model,
                n_trials=n_trials_per_combo,
                num_epochs=num_epochs,
                study_name=study_name,
                storage=storage,
                resume=True
            )
            
            results[f"{dataset}_{model}"] = {
                'best_value': study.best_trial.value if study.best_trial else 0.0,
                'best_params': study.best_trial.params if study.best_trial else {},
                'total_trials': len(study.trials),
                'study_name': study_name
            }
            
        except Exception as e:
            print(f"❌ Failed to tune {dataset}/{model}: {e}")
            results[f"{dataset}_{model}"] = {
                'error': str(e),
                'best_value': 0.0,
                'best_params': {},
                'total_trials': 0
            }
    
    # Save overall results summary
    summary_file = f"{storage_dir}/multi_tuning_summary.json"
    summary = {
        'datasets': datasets,
        'models': models,
        'n_trials_per_combo': n_trials_per_combo,
        'num_epochs': num_epochs,
        'total_combinations': total_combinations,
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"MULTI-DATASET TUNING SUMMARY")
    print(f"{'='*80}")
    print(f"📊 Results saved to: {summary_file}")
    
    # Show top results
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if 'error' not in v],
        key=lambda x: x[1]['best_value'],
        reverse=True
    )
    
    print(f"\n🏆 Top 5 Results:")
    for i, (combo, result) in enumerate(sorted_results[:5]):
        print(f"  {i+1}. {combo}: {result['best_value']:.4f} AP")
        print(f"     Config: {result['best_params']}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Run KAN-MAMMOTE Optuna hyperparameter tuning')
    parser.add_argument('--dataset', type=str, default='wikipedia',
                        help='Dataset to tune on (default: wikipedia)')
    parser.add_argument('--model', type=str, default='TCL', 
                        choices=['TGAT', 'TGN', 'TCL', 'JODIE', 'DyGFormer', 'DyGMamba'],
                        help='Model to tune (default: TCL)')
    parser.add_argument('--n_trials', type=int, default=30,
                        help='Number of trials to run (default: 30)')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Maximum epochs per trial (default: 20)')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Name for the Optuna study (default: auto-generated)')
    parser.add_argument('--storage', type=str, default=None,
                        help='Storage URL (default: SQLite in optuna_results/)')
    parser.add_argument('--no_resume', action='store_true', default=False,
                        help='Do not resume existing study (default: resume if exists)')
    parser.add_argument('--multi_dataset', action='store_true', default=False,
                        help='Run tuning across multiple datasets/models')
    parser.add_argument('--datasets', nargs='+', 
                        default=['wikipedia', 'reddit', 'mooc'],
                        help='Datasets for multi-dataset tuning')
    parser.add_argument('--models', nargs='+',
                        default=['TGAT', 'TGN', 'DyGMamba'],
                        help='Models for multi-dataset tuning')
    parser.add_argument('--trials_per_combo', type=int, default=50,
                        help='Trials per combination in multi-dataset mode (default: 50)')
    
    # New option for analyzing saved configurations
    parser.add_argument('--analyze_config', type=str, default=None,
                        help='Path to best config JSON file to analyze and show reproduction info')
    
    args = parser.parse_args()
    
    # Handle config analysis mode
    if args.analyze_config:
        if os.path.exists(args.analyze_config):
            print_reproduction_info(args.analyze_config)
        else:
            print(f"❌ Config file not found: {args.analyze_config}")
        return
    
    print("="*80)
    print("🔥 KAN-MAMMOTE OPTUNA HYPERPARAMETER TUNING")
    print("="*80)
    print(f"Optuna version: {optuna.__version__}")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.multi_dataset:
        # Multi-dataset tuning mode
        print(f"🔄 Multi-dataset mode enabled")
        print(f"📊 Datasets: {args.datasets}")
        print(f"📊 Models: {args.models}")
        
        results = run_multi_dataset_tuning(
            datasets=args.datasets,
            models=args.models,
            n_trials_per_combo=args.trials_per_combo,
            num_epochs=args.num_epochs
        )
        
    else:
        # Single dataset/model tuning mode
        print(f"🎯 Single combination mode")
        print(f"📊 Dataset: {args.dataset}")
        print(f"📊 Model: {args.model}")
        
        study = run_single_dataset_tuning(
            dataset=args.dataset,
            model=args.model,
            n_trials=args.n_trials,
            num_epochs=args.num_epochs,
            study_name=args.study_name,
            storage=args.storage,
            resume=not args.no_resume
        )
    
    print(f"\n✅ Tuning completed!")
    print(f"💡 To analyze results:")
    print(f"  📊 Compare all studies: python analyze_optuna_results.py --compare_all")
    print(f"  🔍 View specific study: python analyze_optuna_results.py --study_name <name> --storage <db_url>")
    print(f"  🚀 Export best config: python analyze_optuna_results.py --export_config {args.dataset if not args.multi_dataset else 'DATASET'} {args.model if not args.multi_dataset else 'MODEL'}")
    print(f"  📈 Web dashboard: optuna-dashboard sqlite:///optuna_results/<study_name>.db")

if __name__ == '__main__':
    main()