#!/usr/bin/env python3
"""
Mamba-SSM GLIBC Issue Tester
============================

This script specifically tests why mamba-ssm and causal_conv1d fail with GLIBC_2.34 error.
"""

import subprocess
import sys
import os
import traceback

def run_cmd(cmd):
    """Run command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_system_glibc():
    """Check what GLIBC version is available on the system"""
    print("=== System GLIBC Check ===")
    
    # Check ldd version (shows GLIBC version)
    ret, stdout, stderr = run_cmd("ldd --version")
    print(f"ldd version output:\n{stdout}")
    
    # Check available GLIBC symbols in libc.so.6
    libc_path = "/lib/x86_64-linux-gnu/libc.so.6"
    if os.path.exists(libc_path):
        print(f"\nChecking GLIBC symbols in {libc_path}:")
        ret, stdout, stderr = run_cmd(f"strings {libc_path} | grep 'GLIBC_2\\.3[0-9]' | sort -V")
        if stdout:
            versions = stdout.strip().split('\n')
            print(f"Available GLIBC 2.3x versions:")
            for v in versions:
                print(f"  {v}")
            
            # Check specifically for 2.34
            has_2_34 = any('GLIBC_2.34' in v for v in versions)
            print(f"\nGLIBC_2.34 available: {has_2_34}")
        else:
            print("Could not extract GLIBC versions")
    else:
        print(f"libc.so.6 not found at {libc_path}")

def find_problematic_library():
    """Find exactly which .so file is causing the GLIBC_2.34 error"""
    print("\n=== Finding Problematic Library ===")
    
    # Try to import and catch the exact error
    try:
        print("Testing: import causal_conv1d")
        import causal_conv1d
        print("✓ causal_conv1d imported successfully")
    except ImportError as e:
        print(f"✗ causal_conv1d failed: {e}")
        if "GLIBC_2.34" in str(e):
            print("🎯 Found GLIBC_2.34 error in causal_conv1d!")
            
            # Extract the problematic file path
            import re
            match = re.search(r'(/[^:]+\.so)', str(e))
            if match:
                problematic_file = match.group(1)
                print(f"Problematic file: {problematic_file}")
                
                # Analyze this file
                analyze_so_file(problematic_file)
    
    try:
        print("\nTesting: from mamba_ssm.ops import selective_scan_interface")
        from mamba_ssm.ops import selective_scan_interface
        print("✓ mamba_ssm.ops imported successfully")
    except ImportError as e:
        print(f"✗ mamba_ssm.ops failed: {e}")
        if "GLIBC" in str(e):
            print("🎯 Found GLIBC error in mamba_ssm!")

def analyze_so_file(so_file_path):
    """Analyze a specific .so file for GLIBC dependencies"""
    print(f"\n=== Analyzing {so_file_path} ===")
    
    if not os.path.exists(so_file_path):
        print(f"File does not exist: {so_file_path}")
        return
    
    # Check file type
    ret, stdout, stderr = run_cmd(f"file {so_file_path}")
    print(f"File type: {stdout.strip()}")
    
    # Check library dependencies
    print("\nLibrary dependencies:")
    ret, stdout, stderr = run_cmd(f"ldd {so_file_path}")
    if ret == 0:
        for line in stdout.split('\n'):
            if 'libc.so' in line or 'not found' in line:
                print(f"  {line.strip()}")
    
    # Check required GLIBC symbols
    print("\nRequired GLIBC symbols:")
    ret, stdout, stderr = run_cmd(f"objdump -T {so_file_path} | grep GLIBC_2")
    if ret == 0:
        lines = stdout.split('\n')
        glibc_versions = set()
        for line in lines:
            if 'GLIBC_2' in line:
                import re
                match = re.search(r'GLIBC_(\d+\.\d+)', line)
                if match:
                    glibc_versions.add(match.group(1))
        
        if glibc_versions:
            sorted_versions = sorted(glibc_versions, key=lambda x: tuple(map(float, x.split('.'))))
            print(f"Required GLIBC versions: {sorted_versions}")
            max_version = sorted_versions[-1]
            print(f"Maximum required version: GLIBC_{max_version}")
            
            if float(max_version) >= 2.34:
                print(f"⚠️  This library requires GLIBC_{max_version} or higher!")
        else:
            print("No GLIBC version requirements found")

def check_package_info():
    """Check information about installed packages"""
    print("\n=== Package Information ===")
    
    packages = ['causal-conv1d', 'mamba-ssm', 'triton']
    
    for package in packages:
        print(f"\n[{package}]")
        ret, stdout, stderr = run_cmd(f"pip show {package}")
        if ret == 0:
            lines = stdout.split('\n')
            for line in lines:
                if line.startswith(('Version:', 'Location:')):
                    print(f"  {line}")
        else:
            print(f"  Package not found")

def test_individual_imports():
    """Test imports one by one to isolate the issue"""
    print("\n=== Individual Import Tests ===")
    
    tests = [
        "import torch",
        "import causal_conv1d",
        "from causal_conv1d import causal_conv1d_fn", 
        "import causal_conv1d_cuda",
        "import mamba_ssm",
        "from mamba_ssm.modules.mamba2 import Mamba2",
        "from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined"
    ]
    
    for test in tests:
        print(f"\nTesting: {test}")
        try:
            exec(test)
            print("  ✓ Success")
        except ImportError as e:
            print(f"  ✗ ImportError: {e}")
            if "GLIBC" in str(e):
                print("  🎯 GLIBC Error detected!")
        except Exception as e:
            print(f"  ✗ Other error: {e}")

def suggest_fixes():
    """Suggest specific fixes based on the diagnosis"""
    print("\n=== SUGGESTED FIXES ===")
    
    print("\n1. Install compatible versions:")
    print("   pip uninstall causal-conv1d mamba-ssm -y")
    print("   pip install causal-conv1d==1.0.0 mamba-ssm==1.0.1 --no-cache-dir")
    
    print("\n2. Build from source (if you have build tools):")
    print("   pip uninstall causal-conv1d -y")
    print("   cd /tmp")
    print("   git clone https://github.com/Dao-AILab/causal-conv1d.git")
    print("   cd causal-conv1d")
    print("   pip install -e .")
    
    print("\n3. Use conda-forge versions:")
    print("   conda install -c conda-forge causal-conv1d")
    
    print("\n4. Check if newer system available:")
    print("   # This error often occurs on older systems")
    print("   # Consider using Docker/Singularity with newer base image")

def main():
    print("Mamba-SSM GLIBC Diagnosis Tool")
    print("=" * 40)
    
    # System info
    print(f"Python: {sys.version}")
    print(f"OS: {os.uname().sysname} {os.uname().release}")
    
    # Run diagnosis steps
    check_system_glibc()
    check_package_info()
    find_problematic_library()
    test_individual_imports()
    suggest_fixes()
    
    print("\n" + "=" * 40)
    print("Diagnosis complete!")

if __name__ == "__main__":
    main()
