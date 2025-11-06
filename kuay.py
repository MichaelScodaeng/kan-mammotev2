import optuna
import json

# Find and load the TGN study (replace with actual path)
study_name = "kan_mammote_multi_enron_TGN.db"  # You'll need to find this
storage = "sqlite:///optuna_results/studies/YOUR_TGN_STUDY.db"

try:
    study = optuna.load_study(study_name=study_name, storage=storage)
    
    # Find trial 8
    for trial in study.trials:
        if trial.number == 8:
            print(f"TGN Enron Trial 8 Parameters:")
            print(f"Trial state: {trial.state}")
            print(f"Trial value (validation_ap): {trial.value}")
            print(f"Parameters:")
            for param, value in trial.params.items():
                print(f"  {param}: {value}")
            break
    else:
        print("Trial 8 not found")
        
except Exception as e:
    print(f"Error loading study: {e}")