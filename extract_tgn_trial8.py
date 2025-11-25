#!/usr/bin/env python3
"""
Extract TGN TRIAL 5 Parameters from Optuna Database
"""
import optuna
import json
from pathlib import Path

def extract_trial_24_params():
    # Try each TGN database
    databases = [
        "optuna_results/studies/kan_mammote_multi_mooc_TGN.db", 
    ]
    
    for db_path in databases:
        if not Path(db_path).exists():
            print(f"Database not found: {db_path}")
            continue
            
        print(f"\n🔍 Checking database: {db_path}")
        storage = f"sqlite:///{db_path}"
        
        try:
            # List all studies in this database
            study_summaries = optuna.get_all_study_summaries(storage=storage)
            print(f"Found {len(study_summaries)} studies:")
            
            for summary in study_summaries:
                print(f"  - Study: {summary.study_name}")
                print(f"    Trials: {summary.n_trials}")
                print(f"    Direction: {summary.direction}")
                
                # Try to load this study
                try:
                    study = optuna.load_study(
                        study_name=summary.study_name,
                        storage=storage
                    )
                    
                    # Look for TRIAL 5
                    trial_24 = None
                    for trial in study.trials:
                        if trial.number == 24:
                            trial_24 = trial
                            break
                    
                    if trial_24:
                        print(f"\n🎯 FOUND TRIAL 5 in {summary.study_name}!")
                        print(f"Dataset: {db_path.split('_')[-2] if '_' in db_path else 'unknown'}")
                        print(f"Model: TGN")
                        print(f"Trial state: {trial_24.state}")
                        print(f"Trial value (validation_ap): {trial_24.value}")
                        print(f"Trial datetime: {trial_24.datetime_start}")
                        
                        print(f"\n🔧 HYPERPARAMETERS:")
                        for param, value in sorted(trial_24.params.items()):
                            print(f"  {param}: {value}")
                        
                        # Save to JSON file
                        dataset = db_path.split('_')[-2] if '_' in db_path else 'unknown'
                        output_file = f"TGN_{dataset}_trial_24_parameters.json"
                        
                        trial_data = {
                            "dataset": dataset,
                            "model": "TGN",
                            "trial_number": trial_24.number,
                            "trial_state": str(trial_24.state),
                            "validation_ap": trial_24.value,
                            "datetime_start": str(trial_24.datetime_start),
                            "datetime_complete": str(trial_24.datetime_complete),
                            "trial_params": trial_24.params,
                            "study_name": summary.study_name,
                            "database_path": db_path
                        }
                        
                        with open(output_file, 'w') as f:
                            json.dump(trial_data, f, indent=2)
                        
                        print(f"\n💾 Saved parameters to: {output_file}")
                        return trial_data
                    else:
                        print(f"    No TRIAL 5 found in this study")
                        
                except Exception as e:
                    print(f"    Error loading study {summary.study_name}: {e}")
                    
        except Exception as e:
            print(f"Error accessing database {db_path}: {e}")
    
    print("\n❌ TRIAL 5 not found in any TGN database")
    return None

def list_all_trials():
    """List all trials in all TGN databases for reference"""
    databases = [
        "optuna_results/studies/kan_mammote_multi_enron_TGN.db",
        "optuna_results/studies/kan_mammote_multi_mooc_TGN.db", 
        "optuna_results/studies/kan_mammote_multi_UNvote_TGN.db"
    ]
    
    for db_path in databases:
        if not Path(db_path).exists():
            continue
            
        print(f"\n📊 All trials in {db_path}:")
        storage = f"sqlite:///{db_path}"
        
        try:
            study_summaries = optuna.get_all_study_summaries(storage=storage)
            
            for summary in study_summaries:
                try:
                    study = optuna.load_study(
                        study_name=summary.study_name,
                        storage=storage
                    )
                    
                    print(f"\nStudy: {summary.study_name}")
                    print(f"Trials: {[t.number for t in study.trials]}")
                    print(f"Best trial: {study.best_trial.number if study.best_trial else 'None'}")
                    
                except Exception as e:
                    print(f"Error loading study: {e}")
                    
        except Exception as e:
            print(f"Error accessing database: {e}")

if __name__ == "__main__":
    print("🔍 Searching for TGN TRIAL 5 parameters...")
    result = extract_trial_24_params()
    
    if not result:
        print("\n📊 Listing all available trials...")
        list_all_trials()