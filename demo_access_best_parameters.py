#!/usr/bin/env python3
"""
Quick Demo: How to Access Best Parameters from Optuna
=====================================================

This script demonstrates the main ways to get the best hyperparameters
after your Optuna tuning is complete.
"""

import json
import os

def demo_access_best_parameters():
    """Demonstrate different ways to access best parameters."""
    
    print("=" * 80)
    print("🏆 HOW TO GET BEST PARAMETERS AFTER OPTUNA TUNING")
    print("=" * 80)
    
    print("\n1️⃣ METHOD 1: JSON Files (Easiest & Fastest)")
    print("-" * 50)
    print("📄 Optuna automatically saves best configs to JSON files:")
    print("   📁 Location: ./optuna_results/{dataset}_{model}_best_config.json")
    print("   📋 Example: ./optuna_results/wikipedia_TGAT_best_config.json")
    print()
    print("🔍 To view the best config:")
    print("   python analyze_optuna_results.py --best_config optuna_results/wikipedia_TGAT_best_config.json")
    print()
    print("📝 JSON structure:")
    example_json = {
        "dataset": "wikipedia",
        "model": "TGAT", 
        "best_validation_ap": 0.8234,
        "best_trial_number": 42,
        "best_params": {
            "expert_dim": 256,
            "mamba_d_state": 512,
            "mamba_expand": 2,
            "dropout": 0.1
        },
        "study_name": "kan_mammote_wikipedia_TGAT_20241023_143000",
        "total_trials": 100,
        "timestamp": "2024-10-23T14:45:30"
    }
    print(json.dumps(example_json, indent=2))
    
    print("\n2️⃣ METHOD 2: Direct Study Access (Programmatic)")
    print("-" * 50)
    print("🐍 Access Optuna study object directly:")
    print("""
import optuna

# Load study from database
study = optuna.load_study(
    study_name="kan_mammote_wikipedia_TGAT_20241023_143000",
    storage="sqlite:///optuna_results/kan_mammote_wikipedia_TGAT_20241023_143000.db"
)

# Get best parameters
best_params = study.best_trial.params
best_value = study.best_trial.value
best_trial_number = study.best_trial.number

print(f"Best Validation AP: {best_value:.4f}")
print(f"Best Trial: #{best_trial_number}")
print(f"Best Parameters: {best_params}")

# Access individual parameters
expert_dim = best_params['expert_dim']
mamba_d_state = best_params['mamba_d_state']
dropout = best_params['dropout']
""")
    
    print("\n3️⃣ METHOD 3: Analysis Script (Comprehensive)")
    print("-" * 50)
    print("📊 Use the analysis script for detailed insights:")
    print()
    print("🔍 Compare all your experiments:")
    print("   python analyze_optuna_results.py --compare_all")
    print()
    print("🚀 Export best config for retraining:")
    print("   python analyze_optuna_results.py --export_config wikipedia TGAT")
    print()
    print("📈 This will generate:")
    print("   ├─ retrain_wikipedia_TGAT.sh (executable script)")
    print("   ├─ Training command with best hyperparameters")
    print("   └─ JSON export with all details")
    
    print("\n4️⃣ METHOD 4: Web Dashboard (Visual)")
    print("-" * 50)
    print("🌐 Launch Optuna's web dashboard:")
    print("   optuna-dashboard sqlite:///optuna_results/your_study.db")
    print()
    print("🖥️  Then open: http://localhost:8080")
    print("📊 Features:")
    print("   ├─ Interactive plots")
    print("   ├─ Parameter importance")
    print("   ├─ Trial history")
    print("   └─ Hyperparameter correlations")
    
    print("\n5️⃣ PRACTICAL WORKFLOW")
    print("-" * 50)
    print("🔄 Typical workflow after tuning:")
    print()
    print("1. Compare all results:")
    print("   python analyze_optuna_results.py --compare_all")
    print()
    print("2. Export best config for your dataset/model:")
    print("   python analyze_optuna_results.py --export_config wikipedia TGAT")
    print()
    print("3. Run the generated script for final training:")
    print("   ./retrain_wikipedia_TGAT.sh")
    print()
    print("4. Or use the config in your own training script")
    
    print("\n6️⃣ EXAMPLE: Loading Best Config in Python")
    print("-" * 50)
    print("🐍 Integrate best parameters into your training:")
    print("""
import json

# Load best config
with open('optuna_results/wikipedia_TGAT_best_config.json', 'r') as f:
    config = json.load(f)

best_params = config['best_params']

# Use in your training script
args.expert_dim = best_params['expert_dim']
args.mamba_d_state = best_params['mamba_d_state']  
args.mamba_expand = best_params['mamba_expand']
args.encoder_dropout = best_params['dropout']

print(f"Training with best config (AP: {config['best_validation_ap']:.4f})")
""")
    
    print("\n" + "=" * 80)
    print("✅ SUMMARY: Multiple Easy Ways to Access Best Parameters!")
    print("=" * 80)
    print("🏆 Fastest: JSON files in optuna_results/")
    print("📊 Most detailed: analyze_optuna_results.py --compare_all")
    print("🚀 Ready to train: analyze_optuna_results.py --export_config")
    print("🌐 Most visual: optuna-dashboard")
    print()

if __name__ == '__main__':
    demo_access_best_parameters()