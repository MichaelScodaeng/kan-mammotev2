#!/usr/bin/env python3
"""
Create a clean summary of unique model/dataset/time_encoder combinations that need to be rerun to 200 epochs.
Based on validation analysis - experiments that stopped at 100 epochs but were still improving.
"""

import pandas as pd

def create_clean_summary():
    """Create a clean summary table of combinations that need to be rerun."""
    
    # Read the validation experiments to rerun
    df = pd.read_csv('/home/s2516027/kan-mammotev2/validation_experiments_to_rerun.csv')
    
    # Group by model, dataset, time_encoder to get unique combinations
    unique_combinations = df.groupby(['model', 'dataset', 'time_encoder']).agg({
        'seed': lambda x: sorted(list(set(x))),  # Unique seeds
        'max_epoch': 'max',  # Maximum epoch reached across all runs
        'epoch_of_max_score': lambda x: list(x),  # Epochs where best performance occurred
        'max_combined_score': lambda x: list(x)  # Best scores achieved
    }).reset_index()
    
    # Sort by model, then dataset, then time_encoder
    unique_combinations = unique_combinations.sort_values(['model', 'dataset', 'time_encoder'])
    
    print("=" * 120)
    print("UNIQUE MODEL/DATASET/TIME_ENCODER COMBINATIONS THAT NEED TO BE RERUN TO 200 EPOCHS")
    print("(Based on validation analysis: stopped at 100 epochs, best performance at epoch > 80)")
    print("=" * 120)
    print(f"{'Model':<12} | {'Dataset':<12} | {'Time Encoder':<25} | {'Seeds':<15} | {'Best At Epochs':<20} | {'Scores'}")
    print("-" * 120)
    
    # Group by model for cleaner output
    current_model = None
    for _, row in unique_combinations.iterrows():
        if current_model != row['model']:
            if current_model is not None:
                print("-" * 120)
            current_model = row['model']
        
        seeds_str = ', '.join(row['seed'])
        best_epochs_str = ', '.join([f"ep{ep}" for ep in row['epoch_of_max_score']])
        scores_str = ', '.join([f"{score:.3f}" for score in row['max_combined_score']])
        print(f"{row['model']:<12} | {row['dataset']:<12} | {row['time_encoder']:<25} | {seeds_str:<15} | {best_epochs_str:<20} | {scores_str}")
    
    print("-" * 120)
    print(f"TOTAL UNIQUE COMBINATIONS: {len(unique_combinations)}")
    
    # Create summary by model and time_encoder
    print("\n" + "=" * 80)
    print("SUMMARY BY MODEL AND TIME ENCODER")
    print("=" * 80)
    
    model_te_summary = unique_combinations.groupby(['model', 'time_encoder']).size().reset_index(name='count')
    print(f"{'Model':<12} | {'Time Encoder':<25} | {'Count'}")
    print("-" * 50)
    for _, row in model_te_summary.iterrows():
        print(f"{row['model']:<12} | {row['time_encoder']:<25} | {row['count']}")
    
    # Save the unique combinations to CSV
    unique_combinations.to_csv('/home/s2516027/kan-mammotev2/validation_unique_combinations_to_rerun.csv', index=False)
    print(f"\nUnique combinations saved to: validation_unique_combinations_to_rerun.csv")
    
    # Create a simple list for scripting
    print("\n" + "=" * 80)
    print("SIMPLE LIST FOR RERUN SCRIPTS:")
    print("=" * 80)
    for _, row in unique_combinations.iterrows():
        for seed in row['seed']:
            print(f"{row['model']},{row['dataset']},{row['time_encoder']},{seed}")

if __name__ == "__main__":
    create_clean_summary()