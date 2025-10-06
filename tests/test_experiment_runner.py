#!/usr/bin/env python3
"""
Test script for the experiment runner
"""

import subprocess
import os

def test_experiment_runner():
    """Test the experiment runner with various options"""
    
    print("🧪 Testing Time Encoder Experiment Runner")
    print("=" * 50)
    
    # Test 1: Generate report (should work even with no experiments)
    print("\n1. Testing report generation...")
    try:
        result = subprocess.run([
            'python', 'experiment_kanmammote.py', 
            '--generate_report'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Report generation works")
        else:
            print(f"❌ Report generation failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Report generation error: {e}")
    
    # Test 2: Dry run with specific parameters
    print("\n2. Testing dry run mode...")
    try:
        result = subprocess.run([
            'python', 'experiment_kanmammote.py',
            '--dry_run',
            '--models', 'TGAT',
            '--datasets', 'wikipedia', 
            '--time_encoders', 'original', 'kan_mammote',
            '--num_runs', '1'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Dry run mode works")
            print("   Commands that would be executed:")
            lines = result.stdout.split('\n')
            for line in lines:
                if line.startswith('Command:'):
                    print(f"   {line}")
        else:
            print(f"❌ Dry run failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Dry run error: {e}")
    
    # Test 3: Check argument parsing
    print("\n3. Testing help output...")
    try:
        result = subprocess.run([
            'python', 'experiment_kanmammote.py', 
            '--help'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Help output works")
        else:
            print(f"❌ Help failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Help error: {e}")
    
    # Test 4: Resume only mode (should show no experiments to run initially)
    print("\n4. Testing resume-only mode...")
    try:
        result = subprocess.run([
            'python', 'experiment_kanmammote.py',
            '--resume_only',
            '--dry_run'
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ Resume-only mode works")
        else:
            print(f"❌ Resume-only failed: {result.stderr}")
    except Exception as e:
        print(f"❌ Resume-only error: {e}")
    
    print("\n🎉 Experiment runner testing completed!")
    print("\nNext steps:")
    print("1. Run: python experiment_kanmammote.py --dry_run (to see all commands)")
    print("2. Run: python experiment_kanmammote.py --models TGAT --datasets wikipedia --time_encoders original (small test)")
    print("3. Run: python experiment_kanmammote.py (full experiment suite)")

if __name__ == "__main__":
    test_experiment_runner()
