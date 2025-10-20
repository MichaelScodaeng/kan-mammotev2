#!/usr/bin/env python3
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
    print("\n" + "="*80)
    print("Best Configurations per Dataset/Model")
    print("="*80)
    
    for (dataset, model), group in df.groupby(['dataset', 'model']):
        best = group.loc[group['test_ap'].idxmax()]
        print(f"\n{dataset} / {model}:")
        print(f"  Config: {best['config']}")
        print(f"  Test AP: {best['test_ap']:.4f}")
        print(f"  Test AUC: {best['test_auc']:.4f}")
    
    # Save summary
    summary_file = results_dir / "best_configs_summary.csv"
    best_configs = df.loc[df.groupby(['dataset', 'model'])['test_ap'].idxmax()]
    best_configs.to_csv(summary_file, index=False)
    
    print(f"\n✅ Summary saved to: {summary_file}")

if __name__ == '__main__':
    main()
