#!/usr/bin/env python3
"""
Quick test script for the comprehensive benchmarking
Tests a small subset to verify everything works
"""

import subprocess
import sys

def test_benchmark():
    """Test the benchmark script with a small subset"""
    
    cmd = [
        "python", "benchmark_training_time.py",
        "--output", "test_benchmark_results.csv",
        "--timeout", "300",  # 5 minutes timeout
        "--datasets", "wikipedia", "reddit",  # Only 2 datasets
        "--models", "TGAT", "TGN",  # Only 2 models  
        "--encoders", "lete", "mercer"  # Only 2 encoders
    ]
    
    print("🔄 Testing benchmark script with small subset...")
    print(f"Command: {' '.join(cmd)}")
    print("Expected combinations: 2 datasets × 2 models × 2 encoders = 8 tests")
    
    try:
        result = subprocess.run(cmd, timeout=1800)  # 30 min total timeout
        
        if result.returncode == 0:
            print("✅ Benchmark test completed successfully!")
            print("Check test_benchmark_results.csv for results")
        else:
            print("❌ Benchmark test failed!")
            
    except subprocess.TimeoutExpired:
        print("⏰ Benchmark test timed out (took longer than 30 minutes)")
    except Exception as e:
        print(f"💥 Error running benchmark test: {e}")

if __name__ == "__main__":
    test_benchmark()