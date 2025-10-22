#!/usr/bin/env python3
"""
Script to analyze validation metrics files and identify which experiments need to be rerun to 200 epochs.
Criteria: stopped at 100 epochs, best performance at epoch > 80 (still improving), seed0 only, specific time encoders.
"""

import os
import pandas as pd
import glob
from pathlib import Path

def analyze_training_metrics():
    """Analyze all validation metrics files to find which ones ended at 100 epochs with good performance."""
    
    # Find all validation metrics files
    metrics_files = glob.glob("./saved_metrics/*/*/*/val_metrics_*.csv")
    
    results = []
    experiments_to_rerun = []
    
    # Valid time encoders to consider
    valid_time_encoders = ['lete', 'time2vec', 'mercer', 'original']
    
    print("Analyzing validation metrics files...")
    print("=" * 80)
    
    for file_path in sorted(metrics_files):
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get the maximum epoch number
            max_epoch = df['epoch'].max() if 'epoch' in df.columns else 0
            
            # Parse the path to extract model, dataset, and time_encoder information
            path_parts = Path(file_path).parts
            if len(path_parts) >= 6:
                model = path_parts[-4]  # e.g., 'DyGFormer'
                dataset = path_parts[-3]  # e.g., 'Contacts'
                experiment_dir = path_parts[-2]  # e.g., 'DyGFormer_kan_mammote_dual_kmote_seed0'
                
                # Extract time encoder from experiment directory name
                if 'kan_mammote_dual_kmote' in experiment_dir:
                    time_encoder = 'kan_mammote_dual_kmote'
                elif 'time2vec' in experiment_dir:
                    time_encoder = 'time2vec'
                elif 'lete' in experiment_dir:
                    time_encoder = 'lete'
                elif 'mercer' in experiment_dir:
                    time_encoder = 'mercer'
                elif 'original' in experiment_dir:
                    time_encoder = 'original'
                else:
                    time_encoder = 'unknown'
                
                # Extract seed information - only consider seed0
                seed = 'unknown'
                if 'seed0' in experiment_dir:
                    seed = 'seed0'
                
                # Skip if not seed0 or not a valid time encoder
                if seed != 'seed0' or time_encoder not in valid_time_encoders:
                    continue
                
                # Calculate performance metrics
                max_combined_score = 0
                avg_precision_max = 0
                roc_auc_max = 0
                epoch_of_max_score = 0
                
                if 'average_precision' in df.columns and 'roc_auc' in df.columns:
                    # Calculate combined score (average_precision + roc_auc)
                    df['combined_score'] = df['average_precision'] + df['roc_auc']
                    max_combined_score = df['combined_score'].max()
                    avg_precision_max = df['average_precision'].max()
                    roc_auc_max = df['roc_auc'].max()
                    
                    # Find the epoch where the maximum combined score occurred
                    max_score_idx = df['combined_score'].idxmax()
                    epoch_of_max_score = df.loc[max_score_idx, 'epoch']
                
                result = {
                    'model': model,
                    'dataset': dataset,
                    'time_encoder': time_encoder,
                    'seed': seed,
                    'max_epoch': max_epoch,
                    'max_combined_score': max_combined_score,
                    'epoch_of_max_score': epoch_of_max_score,
                    'avg_precision_max': avg_precision_max,
                    'roc_auc_max': roc_auc_max,
                    'file_path': file_path,
                    'experiment_dir': experiment_dir
                }
                
                results.append(result)
                
                # Check if needs to be rerun: 
                # 1. Ended at exactly 100 epochs
                # 2. Max(average_precision + roc_auc) occurred at epoch > 80
                if max_epoch == 100 and epoch_of_max_score > 80:
                    experiments_to_rerun.append(result)
                    print(f"NEEDS RERUN: {model}/{dataset}/{time_encoder}/{seed} - Max epoch: {max_epoch}, Best score at epoch: {epoch_of_max_score} (score: {max_combined_score:.4f})")
                else:
                    status = "OK"
                    reasons = []
                    if max_epoch != 100:
                        reasons.append(f"epochs={max_epoch}")
                    if epoch_of_max_score <= 80:
                        reasons.append(f"best_at_epoch={epoch_of_max_score}<=80")
                    if reasons:
                        status += f" ({', '.join(reasons)})"
                    print(f"{status}: {model}/{dataset}/{time_encoder}/{seed}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    print("\n" + "=" * 80)
    print("SUMMARY OF EXPERIMENTS THAT NEED TO BE RERUN TO 200 EPOCHS:")
    print("(Criteria: stopped at 100 epochs, best performance at epoch > 80, seed0 only)")
    print("=" * 80)
    
    if not experiments_to_rerun:
        print("No experiments need to be rerun! All either completed 200+ epochs or converged early.")
    else:
        # Group by model, dataset, time_encoder
        rerun_groups = {}
        for exp in experiments_to_rerun:
            key = (exp['model'], exp['dataset'], exp['time_encoder'])
            if key not in rerun_groups:
                rerun_groups[key] = []
            rerun_groups[key].append(exp)
        
        print(f"Total experiments to rerun: {len(experiments_to_rerun)}")
        print(f"Unique (model, dataset, time_encoder) combinations: {len(rerun_groups)}")
        print()
        
        for (model, dataset, time_encoder), exps in sorted(rerun_groups.items()):
            seeds = [exp['seed'] for exp in exps]
            best_epochs = [f"ep{exp['epoch_of_max_score']}" for exp in exps]
            scores = [f"{exp['max_combined_score']:.4f}" for exp in exps]
            print(f"{model:12} | {dataset:12} | {time_encoder:25} | Seeds: {', '.join(seeds)} | Best at: {best_epochs} | Scores: {scores}")
    
    print("\n" + "=" * 80)
    print("FULL SUMMARY BY MODEL:")
    print("=" * 80)
    
    # Create summary by model
    model_summary = {}
    for result in results:
        model = result['model']
        if model not in model_summary:
            model_summary[model] = {'total': 0, 'stopped_at_100_still_improving': 0, 'other': 0}
        
        model_summary[model]['total'] += 1
        if result['max_epoch'] == 100 and result['epoch_of_max_score'] > 80:
            model_summary[model]['stopped_at_100_still_improving'] += 1
        else:
            model_summary[model]['other'] += 1
    
    for model, stats in sorted(model_summary.items()):
        print(f"{model:12}: Total: {stats['total']:3d}, Stopped at 100 (still improving): {stats['stopped_at_100_still_improving']:3d}, Other: {stats['other']:3d}")
    
    # Save detailed results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('./validation_analysis_results.csv', index=False)
    
    df_rerun = pd.DataFrame(experiments_to_rerun)
    if not df_rerun.empty:
        df_rerun.to_csv('./validation_experiments_to_rerun.csv', index=False)
    
    print(f"\nDetailed results saved to: validation_analysis_results.csv")
    if not df_rerun.empty:
        print(f"Experiments to rerun saved to: validation_experiments_to_rerun.csv")
        print(f"\nSimple list for scripts:")
        for _, exp in df_rerun.iterrows():
            print(f"{exp['model']},{exp['dataset']},{exp['time_encoder']},{exp['seed']}")
    else:
        print("No experiments to rerun found.")

if __name__ == "__main__":
    analyze_training_metrics()