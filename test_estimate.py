#!/usr/bin/env python3
"""
Quick test script for training time estimation
"""

import subprocess
import sys

def test_estimation():
    """Test the estimation script with a simple configuration"""
    #kan_mammote_dual_kmote
    #lete
    cmd = [
        "python", "experiments/train_link_prediction_estimate.py",
        "--dataset_name", "CanParl", 
        "--model_name", "DyGMamba",
        "--time_encoder_type", "kan_mammote_dual_kmote",
        "--num_epochs", "200",
        "--batch_size", "200",
        "--data_ratio", "1",  # Use small data subset for quick test
        "--num_runs", "1",
        "--disable_progress_bar"
    ]
    
    print("🔄 Testing training time estimation...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            print("✅ Estimation completed successfully!")
            print("\n📊 Output:")
            print(result.stdout)
        else:
            print("❌ Estimation failed!")
            print("\n🔍 Error output:")
            print(result.stderr)
            print("\n🔍 Standard output:")
            print(result.stdout)
            
    except subprocess.TimeoutExpired:
        print("⏰ Estimation timed out (took longer than 5 minutes)")
    except Exception as e:
        print(f"💥 Error running estimation: {e}")

if __name__ == "__main__":
    test_estimation()