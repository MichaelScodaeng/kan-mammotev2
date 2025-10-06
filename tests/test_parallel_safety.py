#!/usr/bin/env python3
"""
Test parallel safety of the experiment runner
"""

import os
import subprocess
import time
import json
import tempfile
from pathlib import Path

def test_parallel_execution():
    """Test that multiple instances can run simultaneously without I/O conflicts"""
    
    print("🧪 Testing parallel execution safety...")
    
    # Test different time encoders in parallel
    test_encoders = ['original', 'lete', 'kan_mammote']
    processes = []
    
    print(f"📝 Starting {len(test_encoders)} parallel test processes...")
    
    for encoder in test_encoders:
        # Start process for each encoder
        cmd = [
            'python', 'experiment_kanmammote.py',
            '--single_encoder', encoder,
            '--models', 'TGAT',  # Use minimal config for fast test
            '--datasets', 'wikipedia', 
            '--dry_run',  # Don't actually run experiments
            '--num_runs', '1'
        ]
        
        print(f"  Starting process for {encoder}...")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        processes.append((encoder, proc))
        time.sleep(0.1)  # Small delay to stagger starts
    
    # Wait for all processes and collect results
    results = {}
    for encoder, proc in processes:
        stdout, stderr = proc.communicate()
        results[encoder] = {
            'returncode': proc.returncode,
            'stdout': stdout.decode(),
            'stderr': stderr.decode()
        }
        print(f"  ✓ Process for {encoder} completed with code {proc.returncode}")
    
    # Check for conflicts
    print("\n🔍 Checking for file conflicts...")
    
    # Check that separate files were created
    expected_files = []
    for encoder in test_encoders:
        expected_files.extend([
            f'experiment_status_{encoder}.json',
            f'completed_experiments_{encoder}.txt',
            f'experiment_progress_{encoder}.log'
        ])
    
    conflicts_found = False
    for filename in expected_files:
        if os.path.exists(filename):
            print(f"  ✓ Found expected file: {filename}")
        else:
            print(f"  ⚠️  Missing expected file: {filename}")
    
    # Check that no shared files were created (should not exist)
    shared_files = [
        'experiment_status_time_encoders.json',
        'completed_experiments_time_encoders.txt',
        'experiment_progress_time_encoders.log'
    ]
    
    for filename in shared_files:
        if os.path.exists(filename):
            print(f"  ❌ Unexpected shared file found: {filename}")
            conflicts_found = True
        else:
            print(f"  ✓ No unexpected shared file: {filename}")
    
    # Check process outputs
    print("\n📊 Process Results:")
    for encoder, result in results.items():
        print(f"  {encoder}:")
        print(f"    Return code: {result['returncode']}")
        if result['stderr']:
            print(f"    Stderr: {result['stderr']}")
    
    # Cleanup test files
    print("\n🧹 Cleaning up test files...")
    for filename in expected_files:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"  Removed: {filename}")
    
    if conflicts_found:
        print("\n❌ PARALLEL SAFETY TEST FAILED: File conflicts detected!")
        return False
    else:
        print("\n✅ PARALLEL SAFETY TEST PASSED: No conflicts detected!")
        return True

def test_file_isolation():
    """Test that each encoder uses completely separate files"""
    
    print("\n🔒 Testing file isolation...")
    
    # Import the functions we want to test
    import sys
    sys.path.append('.')
    from experiment_kanmammote import get_log_files
    
    encoders = ['original', 'lete', 'kan_mammote', 'mercer']
    
    all_files = {}
    for encoder in encoders:
        files = get_log_files(encoder)
        all_files[encoder] = files
        print(f"  {encoder}: {files}")
    
    # Check for overlaps
    all_filenames = []
    for encoder, files in all_files.items():
        all_filenames.extend(files)
    
    unique_filenames = set(all_filenames)
    
    if len(all_filenames) == len(unique_filenames):
        print("  ✅ All file names are unique - no conflicts possible!")
        return True
    else:
        print("  ❌ Duplicate file names found - conflicts possible!")
        duplicates = [f for f in all_filenames if all_filenames.count(f) > 1]
        print(f"  Duplicates: {set(duplicates)}")
        return False

if __name__ == '__main__':
    print("🚀 Testing KAN-MAMMOTE experiment runner parallel safety...")
    
    # Test 1: File isolation
    isolation_ok = test_file_isolation()
    
    # Test 2: Parallel execution
    parallel_ok = test_parallel_execution()
    
    print("\n" + "="*60)
    if isolation_ok and parallel_ok:
        print("🎉 ALL TESTS PASSED! Parallel execution is safe.")
        print("\n✅ You can safely run multiple instances like:")
        print("   python experiment_kanmammote.py --single_encoder kan_mammote &")
        print("   python experiment_kanmammote.py --single_encoder lete &")
        print("   python experiment_kanmammote.py --single_encoder original &")
        exit(0)
    else:
        print("💥 TESTS FAILED! Parallel execution may have conflicts.")
        print("\n❌ Do NOT run multiple instances simultaneously until issues are fixed.")
        exit(1)
