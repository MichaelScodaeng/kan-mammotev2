#!/usr/bin/env python3
"""
Full benchmark script with sensible defaults and error recovery
"""

import subprocess
import sys
import os

def run_full_benchmark():
    """Run the full benchmark with error recovery and resume capability"""
    
    output_file = "full_training_time_benchmark.csv"
    
    cmd = [
        "python", "benchmark_training_time.py",
        "--output", output_file,
        "--timeout", "9009",  # 15 minutes per test
        "--skip_existing"  # Resume from where we left off
    ]
    
    print("🚀 Starting FULL training time benchmark")
    print("This will test ALL combinations:")
    print("   📊 13 datasets × 6 models × 4 encoders = 312 total combinations")
    print("   ⏱️  15 minutes timeout per test")
    print("   💾 Results saved to:", output_file)
    print("   🔄 Will resume from existing results if interrupted")
    print("\nThis may take several hours to complete!")
    
    # Check if results file already exists
    if os.path.exists(output_file):
        print(f"\n📋 Found existing results file: {output_file}")
        print("   The benchmark will skip completed combinations and resume from where it left off")
    
    response = input("\nDo you want to continue? (y/N): ")
    if response.lower() != 'y':
        print("Benchmark cancelled.")
        return
    
    print(f"\n🎯 Running command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print("\n✅ Full benchmark completed successfully!")
            print(f"📊 Results saved to: {output_file}")
            print("📈 You can now analyze the results with pandas:")
            print(f"   import pandas as pd")
            print(f"   df = pd.read_csv('{output_file}')")
            print(f"   df[df['status'] == 'success'].groupby(['model', 'encoder'])['estimated_total_time_hours'].mean()")
        else:
            print("\n❌ Benchmark failed or was interrupted!")
            print(f"📋 Check {output_file} for partial results")
            
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        print(f"📋 Partial results saved to: {output_file}")
        print("🔄 Run this script again to resume from where you left off")
    except Exception as e:
        print(f"\n💥 Error running benchmark: {e}")

if __name__ == "__main__":
    run_full_benchmark()