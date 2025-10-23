#!/usr/bin/env python3
"""
Optuna Results Analyzer
=======================

This script helps you analyze and retrieve the best hyperparameters from completed Optuna studies.
It provides multiple ways to access and compare results across different experiments.

Usage:
    python analyze_optuna_results.py --study_name "kan_mammote_wikipedia_TGAT_20241023_143000"
    python analyze_optuna_results.py --storage "sqlite:///optuna_results/my_study.db"
    python analyze_optuna_results.py --best_config "optuna_results/wikipedia_TGAT_best_config.json"
    python analyze_optuna_results.py --compare_all  # Compare all studies in optuna_results/
    python analyze_optuna_results.py --export_config wikipedia TGAT  # Export config for retraining
"""

import optuna
import json
import os
import argparse
import glob
from pathlib import Path
from datetime import datetime
import pandas as pd

def load_study_from_storage(storage_url, study_name):
    """Load an Optuna study from storage."""
    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
        return study
    except Exception as e:
        print(f"❌ Failed to load study '{study_name}' from {storage_url}: {e}")
        return None

def analyze_single_study(study, show_details=True):
    """Analyze a single Optuna study and return detailed results."""
    if not study:
        return None
    
    analysis = {
        'study_name': study.study_name,
        'total_trials': len(study.trials),
        'complete_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
        'best_trial': None,
        'best_value': None,
        'best_params': None,
        'trial_history': []
    }
    
    if study.best_trial:
        analysis['best_trial'] = study.best_trial.number
        analysis['best_value'] = study.best_trial.value
        analysis['best_params'] = study.best_trial.params
        
        # Get trial history for plotting/analysis
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                analysis['trial_history'].append({
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'duration': trial.duration.total_seconds() if trial.duration else None
                })
    
    if show_details:
        print(f"\n{'='*80}")
        print(f"📊 STUDY ANALYSIS: {study.study_name}")
        print(f"{'='*80}")
        print(f"Trial Statistics:")
        print(f"  ├─ Total trials: {analysis['total_trials']}")
        print(f"  ├─ Complete trials: {analysis['complete_trials']}")
        print(f"  ├─ Pruned trials: {analysis['pruned_trials']}")
        print(f"  └─ Failed trials: {analysis['failed_trials']}")
        
        if analysis['best_trial'] is not None:
            print(f"\n🏆 Best Result:")
            print(f"  ├─ Trial #{analysis['best_trial']}")
            print(f"  ├─ Validation AP: {analysis['best_value']:.4f}")
            print(f"  └─ Best Parameters:")
            for key, value in analysis['best_params'].items():
                print(f"      ├─ {key}: {value}")
        else:
            print(f"\n❌ No successful trials found!")
        
        # Show top 5 trials
        if len(analysis['trial_history']) > 1:
            sorted_trials = sorted(analysis['trial_history'], key=lambda x: x['value'], reverse=True)
            print(f"\n📈 Top 5 Trials:")
            for i, trial in enumerate(sorted_trials[:5]):
                print(f"  {i+1}. Trial #{trial['trial_number']}: {trial['value']:.4f} AP")
                
        # Show parameter importance if enough trials
        if len(analysis['trial_history']) >= 10:
            try:
                importance = optuna.importance.get_param_importances(study)
                print(f"\n🎯 Parameter Importance:")
                for param, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
                    print(f"  ├─ {param}: {score:.3f}")
            except Exception:
                pass  # Skip if importance calculation fails
    
    return analysis

def load_best_config_json(config_file):
    """Load best configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        print(f"\n{'='*80}")
        print(f"📄 BEST CONFIG FROM JSON: {os.path.basename(config_file)}")
        print(f"{'='*80}")
        print(f"Dataset: {config.get('dataset', 'N/A')}")
        print(f"Model: {config.get('model', 'N/A')}")
        print(f"Best Validation AP: {config.get('best_validation_ap', 'N/A'):.4f}")
        print(f"Best Trial: #{config.get('best_trial_number', 'N/A')}")
        print(f"Total Trials: {config.get('total_trials', 'N/A')}")
        print(f"Timestamp: {config.get('timestamp', 'N/A')}")
        
        print(f"\n🏆 Best Hyperparameters:")
        best_params = config.get('best_params', {})
        for key, value in best_params.items():
            print(f"  ├─ {key}: {value}")
        
        return config
        
    except Exception as e:
        print(f"❌ Failed to load config from {config_file}: {e}")
        return None

def compare_all_studies(results_dir="./optuna_results"):
    """Compare all studies found in the results directory."""
    print(f"\n{'='*80}")
    print(f"🔍 COMPARING ALL STUDIES IN: {results_dir}")
    print(f"{'='*80}")
    
    # Find all SQLite databases
    db_files = glob.glob(f"{results_dir}/**/*.db", recursive=True)
    json_files = glob.glob(f"{results_dir}/**/*_best_config.json", recursive=True)
    
    print(f"Found {len(db_files)} database files and {len(json_files)} config files")
    
    all_results = []
    
    # Load from JSON files (faster and contains summary info)
    for json_file in json_files:
        config = load_best_config_json(json_file)
        if config and config.get('best_validation_ap'):
            all_results.append({
                'source': os.path.basename(json_file),
                'dataset': config.get('dataset', 'Unknown'),
                'model': config.get('model', 'Unknown'),
                'best_ap': config.get('best_validation_ap', 0.0),
                'best_params': config.get('best_params', {}),
                'total_trials': config.get('total_trials', 0),
                'study_name': config.get('study_name', 'Unknown')
            })
    
    if not all_results:
        print("❌ No results found!")
        return None
    
    # Sort by performance
    all_results.sort(key=lambda x: x['best_ap'], reverse=True)
    
    print(f"\n🏆 RANKING (Top {min(10, len(all_results))} Results):")
    print(f"{'Rank':<4} {'Dataset':<12} {'Model':<10} {'Best AP':<8} {'Trials':<7} {'Best Config'}")
    print("-" * 80)
    
    for i, result in enumerate(all_results[:10]):
        config_str = ", ".join([f"{k}={v}" for k, v in list(result['best_params'].items())[:3]])
        if len(result['best_params']) > 3:
            config_str += "..."
        
        print(f"{i+1:<4} {result['dataset']:<12} {result['model']:<10} {result['best_ap']:<8.4f} "
              f"{result['total_trials']:<7} {config_str}")
    
    return all_results

def export_training_config(dataset, model, results_dir="./optuna_results", 
                          output_file=None):
    """Export the best configuration for retraining."""
    
    # Look for the best config JSON file
    pattern = f"{results_dir}/**/{dataset}_{model}_best_config.json"
    matches = glob.glob(pattern, recursive=True)
    
    if not matches:
        print(f"❌ No results found for {dataset}/{model} in {results_dir}")
        print(f"   Searched for: {pattern}")
        return None
    
    # Use the most recent if multiple matches
    config_file = max(matches, key=os.path.getmtime)
    config = load_best_config_json(config_file)
    
    if not config:
        return None
    
    # Create training command
    best_params = config['best_params']
    training_config = {
        'dataset_name': dataset,
        'model_name': model,
        'time_encoder_type': 'kan_mammote_dual_kmote',
        'expert_dim': best_params.get('expert_dim', 128),
        'mamba_d_state': best_params.get('mamba_d_state', 256),
        'mamba_expand': best_params.get('mamba_expand', 2),
        'encoder_dropout': best_params.get('dropout', 0.1),
        'mamba_headdim': 64,  # Fixed value
        'mamba_d_conv': 4,    # Fixed value
        'num_epochs': 50,     # Full training epochs
        'patience': 10,       # More patience for final training
        'num_runs': 3,        # Multiple runs for statistics
        'seed': 0
    }
    
    # Generate command
    cmd_parts = ['python', 'experiments/train_link_prediction_tune.py']
    for key, value in training_config.items():
        cmd_parts.extend([f'--{key}', str(value)])
    
    training_command = ' '.join(cmd_parts)
    
    # Save to file if requested
    if output_file:
        export_data = {
            'source_study': config.get('study_name', 'Unknown'),
            'best_validation_ap': config.get('best_validation_ap', 0.0),
            'best_trial_number': config.get('best_trial_number', 0),
            'training_config': training_config,
            'training_command': training_command,
            'exported_at': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"💾 Training config exported to: {output_file}")
    
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING COMMAND FOR {dataset}/{model}")
    print(f"{'='*80}")
    print(f"Best Validation AP from tuning: {config.get('best_validation_ap', 0.0):.4f}")
    print(f"Source study: {config.get('study_name', 'Unknown')}")
    print(f"\n📋 Training Configuration:")
    for key, value in training_config.items():
        print(f"  ├─ {key}: {value}")
    
    print(f"\n🔥 Command to run:")
    print(f"{training_command}")
    
    # Also create a shell script
    script_file = f"retrain_{dataset}_{model}.sh"
    with open(script_file, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Retrain {dataset}/{model} with best hyperparameters from Optuna\n")
        f.write(f"# Best validation AP: {config.get('best_validation_ap', 0.0):.4f}\n")
        f.write(f"# Source: {config.get('study_name', 'Unknown')}\n\n")
        f.write(training_command + "\n")
    
    os.chmod(script_file, 0o755)  # Make executable
    print(f"📝 Executable script saved to: {script_file}")
    
    return training_config

def find_all_studies(results_dir="./optuna_results"):
    """Find all available studies."""
    db_files = glob.glob(f"{results_dir}/**/*.db", recursive=True)
    json_files = glob.glob(f"{results_dir}/**/*_best_config.json", recursive=True)
    
    print(f"\n📚 AVAILABLE STUDIES:")
    print(f"Database files ({len(db_files)}):")
    for db in db_files:
        rel_path = os.path.relpath(db)
        study_name = os.path.splitext(os.path.basename(db))[0]
        print(f"  ├─ {rel_path} (study: {study_name})")
    
    print(f"\nJSON configs ({len(json_files)}):")
    for json_file in json_files:
        rel_path = os.path.relpath(json_file)
        print(f"  ├─ {rel_path}")
    
    return db_files, json_files

def main():
    parser = argparse.ArgumentParser(description='Analyze Optuna hyperparameter tuning results')
    parser.add_argument('--study_name', type=str, help='Name of the Optuna study to analyze')
    parser.add_argument('--storage', type=str, help='Storage URL (e.g., sqlite:///path/to/study.db)')
    parser.add_argument('--best_config', type=str, help='Path to best_config.json file')
    parser.add_argument('--compare_all', action='store_true', 
                        help='Compare all studies in optuna_results/')
    parser.add_argument('--results_dir', type=str, default='./optuna_results',
                        help='Directory containing results (default: ./optuna_results)')
    parser.add_argument('--export_config', nargs=2, metavar=('DATASET', 'MODEL'),
                        help='Export training config for dataset/model (e.g., wikipedia TGAT)')
    parser.add_argument('--output_file', type=str, 
                        help='Output file for exported config (default: auto-generated)')
    parser.add_argument('--list_studies', action='store_true',
                        help='List all available studies')
    
    args = parser.parse_args()
    
    print("="*80)
    print("📊 OPTUNA RESULTS ANALYZER")
    print("="*80)
    print(f"Optuna version: {optuna.__version__}")
    print(f"Analysis time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.list_studies:
        find_all_studies(args.results_dir)
        return
    
    if args.compare_all:
        compare_all_studies(args.results_dir)
        return
    
    if args.export_config:
        dataset, model = args.export_config
        export_training_config(
            dataset=dataset, 
            model=model,
            results_dir=args.results_dir,
            output_file=args.output_file
        )
        return
    
    if args.best_config:
        load_best_config_json(args.best_config)
        return
    
    if args.study_name and args.storage:
        study = load_study_from_storage(args.storage, args.study_name)
        if study:
            analyze_single_study(study)
        return
    
    if args.study_name or args.storage:
        print("❌ Both --study_name and --storage are required to load a study from database")
        return
    
    # Default: show help and list available studies
    print("No specific analysis requested. Here's what's available:")
    find_all_studies(args.results_dir)
    print(f"\n💡 Usage examples:")
    print(f"  python {__file__} --compare_all")
    print(f"  python {__file__} --export_config wikipedia TGAT")
    print(f"  python {__file__} --best_config optuna_results/wikipedia_TGAT_best_config.json")

if __name__ == '__main__':
    main()