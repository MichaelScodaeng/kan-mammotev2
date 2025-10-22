#!/usr/bin/env python3
"""
Final verification script before submitting resume training jobs.
Checks all checkpoints, job scripts, and validates everything is ready.
"""

import os
import torch
import pandas as pd
from pathlib import Path

def verify_pre_submission():
    """Comprehensive verification before job submission."""
    
    experiments = [
        ("DyGMamba", "lastfm", "time2vec", "seed0"),
        ("DyGMamba", "mooc", "time2vec", "seed0"),
        ("DyGMamba", "uci", "time2vec", "seed0"),
        ("JODIE", "Contacts", "time2vec", "seed0"),
        ("JODIE", "SocialEvo", "time2vec", "seed0"),
        ("JODIE", "uci", "time2vec", "seed0"),
        ("JODIE", "wikipedia", "time2vec", "seed0"),
        ("TCL", "Contacts", "time2vec", "seed0"),
        ("TCL", "lastfm", "time2vec", "seed0"),
        ("TCL", "mooc", "time2vec", "seed0"),
        ("TCL", "reddit", "time2vec", "seed0"),
        ("TGN", "Flights", "time2vec", "seed0")
    ]
    
    print("=" * 100)
    print("PRE-SUBMISSION VERIFICATION")
    print("=" * 100)
    
    all_checks_passed = True
    
    for i, (model, dataset, time_encoder, seed) in enumerate(experiments, 1):
        print(f"\n{i:2d}. {model}/{dataset}/{time_encoder}/{seed}")
        experiment_name = f"{model}_{time_encoder}_{seed}"
        
        # Check 1: Checkpoint exists and is loadable
        checkpoint_path = f"/home/s2516027/kan-mammotev2/saved_models/{model}/{dataset}/{experiment_name}/checkpoint_epoch_100.pth"
        if os.path.exists(checkpoint_path):
            try:
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                epoch = checkpoint['epoch']
                print(f"    ✅ Checkpoint: epoch {epoch} - {os.path.getsize(checkpoint_path)/1024/1024:.1f}MB")
            except Exception as e:
                print(f"    ❌ Checkpoint CORRUPTED: {e}")
                all_checks_passed = False
        else:
            print(f"    ❌ Checkpoint MISSING: {checkpoint_path}")
            all_checks_passed = False
        
        # Check 2: Current validation metrics show training stopped at 100
        val_metrics_pattern = f"/home/s2516027/kan-mammotev2/saved_metrics/{model}/{dataset}/{experiment_name}/val_metrics_*.csv"
        import glob
        val_files = glob.glob(val_metrics_pattern)
        if val_files:
            try:
                df = pd.read_csv(val_files[0])
                max_epoch = df['epoch'].max()
                if max_epoch == 100:
                    latest_score = df.iloc[-1]['average_precision'] + df.iloc[-1]['roc_auc']
                    print(f"    ✅ Metrics: stopped at epoch {max_epoch}, last score: {latest_score:.4f}")
                else:
                    print(f"    ❓ Metrics: training already extended to epoch {max_epoch}")
            except Exception as e:
                print(f"    ❌ Metrics ERROR: {e}")
                all_checks_passed = False
        else:
            print(f"    ❌ Metrics MISSING: {val_metrics_pattern}")
            all_checks_passed = False
        
        # Check 3: Job script exists
        job_script = f"/home/s2516027/kan-mammotev2/resume_jobs/resume_{model.lower()}_{dataset.lower()}_{time_encoder}.sh"
        if os.path.exists(job_script):
            print(f"    ✅ Job script: {job_script}")
        else:
            print(f"    ❌ Job script MISSING: {job_script}")
            all_checks_passed = False
    
    print("\n" + "=" * 100)
    if all_checks_passed:
        print("🎉 ALL CHECKS PASSED - READY TO SUBMIT!")
        print("=" * 100)
        print("To submit all jobs, run:")
        print("    ./submit_resume_jobs.sh")
        print()
        print("To monitor progress, run:")
        print("    python monitor_resume_jobs.py")
        print()
        print("Expected completion time: ~24 hours per job (100 additional epochs)")
        print("Total GPU hours: ~288 hours (12 jobs × 24 hours)")
    else:
        print("❌ SOME CHECKS FAILED - REVIEW ERRORS ABOVE")
        print("Fix issues before submitting jobs.")
    
    print("=" * 100)
    
    return all_checks_passed

if __name__ == "__main__":
    verify_pre_submission()