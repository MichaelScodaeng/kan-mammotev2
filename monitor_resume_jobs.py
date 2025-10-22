#!/usr/bin/env python3
"""
Monitor the progress of resume training jobs by checking validation metrics.
"""

import os
import glob
import pandas as pd
from pathlib import Path

def monitor_resume_progress():
    """Monitor progress of resumed training jobs."""
    
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
    print("RESUME TRAINING PROGRESS MONITOR")
    print("=" * 100)
    
    for model, dataset, time_encoder, seed in experiments:
        experiment_name = f"{model}_{time_encoder}_{seed}"
        
        # Check validation metrics file
        val_metrics_pattern = f"saved_metrics/{model}/{dataset}/{experiment_name}/val_metrics_*.csv"
        val_files = glob.glob(val_metrics_pattern)
        
        if val_files:
            val_file = val_files[0]
            try:
                df = pd.read_csv(val_file)
                max_epoch = df['epoch'].max()
                latest_score = df.iloc[-1]['average_precision'] + df.iloc[-1]['roc_auc']
                
                if max_epoch >= 200:
                    status = "✅ COMPLETED"
                elif max_epoch > 100:
                    status = f"🔄 IN PROGRESS (epoch {max_epoch}/200)"
                else:
                    status = "⏸️  NOT STARTED"
                
                print(f"{status:<20} {model:<12} {dataset:<15} latest_score: {latest_score:.4f}")
            except Exception as e:
                print(f"❌ ERROR reading {val_file}: {e}")
        else:
            print(f"❓ NO METRICS      {model:<12} {dataset:<15} - metrics file not found")
    
    print("=" * 100)

if __name__ == "__main__":
    monitor_resume_progress()
