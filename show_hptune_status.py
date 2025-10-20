#!/usr/bin/env python3
"""
Show status of hyperparameter tuning experiments.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime

def show_status():
    """Display current status of hyperparameter tuning."""
    
    results_dir = Path('./hyperparameter_tuning_results')
    
    print("="*80)
    print("HYPERPARAMETER TUNING STATUS")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    print()
    
    if not results_dir.exists():
        print("❌ No results directory found.")
        print("   Run experiments first: python tune_hyperparams_fast.py")
        return
    
    # Count experiment directories
    exp_dirs = list(results_dir.glob('*_lr*_wd*'))
    
    if not exp_dirs:
        print("❌ No experiment results found.")
        print("   Run experiments first: python tune_hyperparams_fast.py")
        return
    
    print(f"📁 Found {len(exp_dirs)} experiment directories")
    print()
    
    # Analyze completion status
    completed = 0
    with_results = 0
    with_models = 0
    
    model_count = Counter()
    dataset_count = Counter()
    encoder_count = Counter()
    
    for exp_dir in exp_dirs:
        # Parse config name
        parts = exp_dir.name.split('_')
        if len(parts) >= 2:
            model = parts[0]
            dataset = parts[1]
            model_count[model] += 1
            dataset_count[dataset] += 1
        
        # Check for results
        json_files = list(exp_dir.rglob('*.json'))
        log_files = list(exp_dir.glob('*.log'))
        model_files = list(exp_dir.rglob('*.pth'))
        
        if json_files or log_files:
            completed += 1
        if json_files:
            with_results += 1
        if model_files:
            with_models += 1
    
    # Display summary
    print("📊 Completion Summary:")
    print(f"   Total experiments:     {len(exp_dirs)}")
    print(f"   Completed (has logs):  {completed}")
    print(f"   With results (JSON):   {with_results}")
    print(f"   With saved models:     {with_models}")
    print()
    
    completion_rate = (completed / len(exp_dirs) * 100) if exp_dirs else 0
    print(f"   Completion rate: {completion_rate:.1f}%")
    print()
    
    # Show breakdown by model
    if model_count:
        print("📈 By Model:")
        for model, count in sorted(model_count.items()):
            print(f"   {model:15s}: {count:4d} experiments")
        print()
    
    # Show breakdown by dataset
    if dataset_count:
        print("📈 By Dataset:")
        for dataset, count in sorted(dataset_count.items()):
            print(f"   {dataset:15s}: {count:4d} experiments")
        print()
    
    # Check for collected results
    collected_file = results_dir / 'collected_results.json'
    if collected_file.exists():
        print("✅ Collected results found!")
        
        try:
            with open(collected_file) as f:
                collected = json.load(f)
            
            success_count = sum(1 for r in collected if r.get('status') == 'success')
            fail_count = sum(1 for r in collected if r.get('status') in ['failed', 'error'])
            
            print(f"   Total collected: {len(collected)}")
            print(f"   Successful:      {success_count}")
            print(f"   Failed:          {fail_count}")
            print()
            
            # Show best result
            successful = [r for r in collected if r.get('status') == 'success']
            if successful and any('validate_ap' in r for r in successful):
                best = max(successful, key=lambda x: x.get('validate_ap', 0))
                print("🏆 Best Configuration So Far:")
                print(f"   Model:       {best.get('model')}")
                print(f"   Dataset:     {best.get('dataset')}")
                print(f"   Time Encoder: {best.get('time_encoder')}")
                print(f"   LR:          {best.get('lr')}")
                print(f"   WD:          {best.get('wd')}")
                print(f"   Val AP:      {best.get('validate_ap', 'N/A')}")
                print()
        except Exception as e:
            print(f"   ⚠️  Error reading collected results: {e}")
            print()
    else:
        print("ℹ️  Results not yet collected.")
        print("   Run: python collect_hptune_results.py")
        print()
    
    # Check for summary report
    summary_files = list(results_dir.glob('summary_*.txt'))
    if summary_files:
        latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
        print(f"📄 Summary report: {latest_summary.name}")
        print(f"   View with: cat {latest_summary}")
        print()
    
    # Disk usage
    try:
        import subprocess
        result = subprocess.run(
            ['du', '-sh', str(results_dir)],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            size = result.stdout.split()[0]
            print(f"💾 Disk usage: {size}")
            print()
    except:
        pass
    
    # Next steps
    print("="*80)
    print("NEXT STEPS:")
    print("="*80)
    
    if completed < len(exp_dirs):
        print("⏳ Experiments still running or incomplete")
        print("   Monitor: qstat -u $USER")
        print()
    
    if completed == len(exp_dirs) and not collected_file.exists():
        print("✅ All experiments complete!")
        print("   Collect results: python collect_hptune_results.py")
        print()
    
    if collected_file.exists():
        print("✅ Results collected!")
        print("   View summary: cat hyperparameter_tuning_results/summary_*.txt")
        print("   Analyze: python -c 'import json; ...'")
        print()
    
    print()


if __name__ == '__main__':
    show_status()
