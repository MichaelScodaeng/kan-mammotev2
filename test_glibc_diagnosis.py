#!/usr/bin/env python3
"""
GLIBC Diagnosis Test Script
===========================

This script diagnoses why GLIBC_2.34 not found error occurs with mamba-ssm and causal_conv1d.
It checks system GLIBC version, library dependencies, and provides solutions.
"""

import os
import sys
import subprocess
import platform
import importlib.util
from pathlib import Path
import traceback

def run_command(cmd, capture_output=True, text=True):
    """Run shell command safely"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=text)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)

def check_glibc_version():
    """Check system GLIBC version"""
    print("=== GLIBC Version Analysis ===")
    
    # Method 1: ldd --version
    print("\n[Method 1] Using ldd --version:")
    ret, stdout, stderr = run_command("ldd --version")
    if ret == 0:
        lines = stdout.split('\n')
        if lines:
            print(f"  {lines[0]}")
            # Extract version number
            import re
            version_match = re.search(r'(\d+\.\d+)', lines[0])
            if version_match:
                glibc_version = version_match.group(1)
                print(f"  Detected GLIBC version: {glibc_version}")
                return float(glibc_version)
    else:
        print(f"  Error: {stderr}")
    
    # Method 2: getconf GNU_LIBC_VERSION
    print("\n[Method 2] Using getconf:")
    ret, stdout, stderr = run_command("getconf GNU_LIBC_VERSION")
    if ret == 0:
        print(f"  {stdout.strip()}")
    else:
        print(f"  Error: {stderr}")
    
    # Method 3: Check /lib/x86_64-linux-gnu/libc.so.6
    print("\n[Method 3] Checking libc.so.6 directly:")
    libc_paths = [
        "/lib/x86_64-linux-gnu/libc.so.6",
        "/lib64/libc.so.6",
        "/usr/lib/x86_64-linux-gnu/libc.so.6"
    ]
    
    for libc_path in libc_paths:
        if os.path.exists(libc_path):
            print(f"  Found: {libc_path}")
            ret, stdout, stderr = run_command(f"strings {libc_path} | grep GLIBC_")
            if ret == 0:
                versions = [line for line in stdout.split('\n') if 'GLIBC_' in line]
                if versions:
                    print(f"  Available GLIBC versions: {len(versions)} found")
                    # Show first few and last few
                    for v in sorted(versions)[:5]:
                        print(f"    {v}")
                    if len(versions) > 10:
                        print("    ...")
                        for v in sorted(versions)[-5:]:
                            print(f"    {v}")
                    
                    # Check if GLIBC_2.34 is available
                    has_2_34 = any('GLIBC_2.34' in v for v in versions)
                    print(f"  Has GLIBC_2.34: {has_2_34}")
                    return None
            break
    else:
        print("  Could not find libc.so.6")
    
    return None

def analyze_problematic_libraries():
    """Analyze the specific libraries causing GLIBC issues"""
    print("\n=== Problematic Library Analysis ===")
    
    # Check installed packages
    packages_to_check = ['causal-conv1d', 'mamba-ssm', 'triton']
    
    for package in packages_to_check:
        print(f"\n[Package] {package}")
        
        # Check if installed
        ret, stdout, stderr = run_command(f"pip show {package}")
        if ret == 0:
            lines = stdout.split('\n')
            version_line = [l for l in lines if l.startswith('Version:')]
            location_line = [l for l in lines if l.startswith('Location:')]
            
            if version_line:
                print(f"  Version: {version_line[0].split(':', 1)[1].strip()}")
            if location_line:
                location = location_line[0].split(':', 1)[1].strip()
                print(f"  Location: {location}")
                
                # Find .so files
                package_path = Path(location) / package.replace('-', '_')
                if package_path.exists():
                    so_files = list(package_path.rglob("*.so"))
                    print(f"  Found {len(so_files)} .so files")
                    
                    for so_file in so_files[:3]:  # Check first 3
                        print(f"    Analyzing: {so_file.name}")
                        ret, stdout, stderr = run_command(f"ldd {so_file}")
                        if ret == 0:
                            glibc_deps = [line for line in stdout.split('\n') if 'libc.so' in line]
                            for dep in glibc_deps:
                                print(f"      {dep.strip()}")
                        
                        # Check required GLIBC symbols
                        ret, stdout, stderr = run_command(f"objdump -T {so_file} | grep GLIBC")
                        if ret == 0:
                            glibc_symbols = stdout.split('\n')
                            glibc_versions = set()
                            for symbol in glibc_symbols:
                                if 'GLIBC_' in symbol:
                                    import re
                                    match = re.search(r'GLIBC_(\d+\.\d+)', symbol)
                                    if match:
                                        glibc_versions.add(match.group(1))
                            
                            if glibc_versions:
                                sorted_versions = sorted(glibc_versions, key=lambda x: tuple(map(float, x.split('.'))))
                                print(f"      Required GLIBC versions: {sorted_versions}")
                                max_version = sorted_versions[-1]
                                print(f"      Maximum required: GLIBC_{max_version}")
        else:
            print(f"  Not installed")

def test_import_hierarchy():
    """Test imports to see exactly where GLIBC error occurs"""
    print("\n=== Import Hierarchy Test ===")
    
    import_tests = [
        ("torch", "import torch"),
        ("causal_conv1d_interface", "from causal_conv1d import causal_conv1d_fn"),
        ("causal_conv1d_cuda", "import causal_conv1d_cuda"),
        ("mamba_ssm", "import mamba_ssm"),
        ("mamba_ssm.ops", "from mamba_ssm.ops import selective_scan_interface"),
        ("triton", "import triton"),
        ("triton.backends.nvidia.driver", "from triton.backends.nvidia.driver import CudaUtils"),
    ]
    
    for test_name, import_cmd in import_tests:
        print(f"\n[Import Test] {test_name}")
        print(f"  Command: {import_cmd}")
        
        try:
            exec(import_cmd)
            print(f"  ✓ Success")
        except ImportError as e:
            print(f"  ✗ ImportError: {e}")
            if "GLIBC" in str(e):
                print(f"  🎯 GLIBC Error Found!")
                
                # Extract the problematic file
                import re
                file_match = re.search(r'(/[^:]+\.so)', str(e))
                if file_match:
                    problematic_file = file_match.group(1)
                    print(f"  Problematic file: {problematic_file}")
                    
                    # Analyze this specific file
                    if os.path.exists(problematic_file):
                        print(f"  Analyzing {problematic_file}:")
                        
                        # Check file info
                        ret, stdout, stderr = run_command(f"file {problematic_file}")
                        if ret == 0:
                            print(f"    File type: {stdout.strip()}")
                        
                        # Check dependencies
                        ret, stdout, stderr = run_command(f"ldd {problematic_file}")
                        if ret == 0:
                            deps = stdout.split('\n')
                            glibc_deps = [d for d in deps if 'libc.so' in d]
                            print(f"    GLIBC dependencies:")
                            for dep in glibc_deps:
                                print(f"      {dep.strip()}")
                        
                        # Check required symbols
                        ret, stdout, stderr = run_command(f"nm -D {problematic_file} | grep GLIBC")
                        if ret == 0:
                            symbols = stdout.split('\n')
                            glibc_symbols = [s for s in symbols if 'GLIBC_' in s]
                            if glibc_symbols:
                                print(f"    Required GLIBC symbols (first 10):")
                                for symbol in glibc_symbols[:10]:
                                    print(f"      {symbol.strip()}")
        except Exception as e:
            print(f"  ✗ Other Error: {e}")

def check_conda_environment():
    """Check conda environment details"""
    print("\n=== Conda Environment Analysis ===")
    
    # Check if we're in conda
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        print(f"Active conda environment: {conda_env}")
        
        # Check conda info
        ret, stdout, stderr = run_command("conda info")
        if ret == 0:
            lines = stdout.split('\n')
            for line in lines:
                if 'platform' in line.lower() or 'python version' in line.lower():
                    print(f"  {line.strip()}")
        
        # Check installed packages related to compilation
        print("\nChecking compilation-related packages:")
        packages = ['gcc', 'gxx', 'binutils', 'glibc']
        for pkg in packages:
            ret, stdout, stderr = run_command(f"conda list | grep {pkg}")
            if ret == 0 and stdout.strip():
                print(f"  {pkg}: installed")
                for line in stdout.strip().split('\n'):
                    print(f"    {line}")
            else:
                print(f"  {pkg}: not found")
    else:
        print("Not in a conda environment")

def provide_solutions():
    """Provide specific solutions based on diagnosis"""
    print("\n=== SOLUTIONS ===")
    
    print("\n[Solution 1] Reinstall with compatible versions:")
    print("  # Uninstall problematic packages")
    print("  pip uninstall causal-conv1d mamba-ssm triton -y")
    print("  ")
    print("  # Install older, more compatible versions")
    print("  pip install causal-conv1d==1.0.0 --no-cache-dir")
    print("  pip install mamba-ssm==1.0.1 --no-cache-dir")
    print("  pip install triton==2.0.0 --no-cache-dir")
    
    print("\n[Solution 2] Build from source:")
    print("  # Install build dependencies")
    print("  sudo apt-get install build-essential")
    print("  # OR in conda:")
    print("  conda install -c conda-forge gcc_linux-64 gxx_linux-64")
    print("  ")
    print("  # Build causal-conv1d from source")
    print("  pip uninstall causal-conv1d -y")
    print("  git clone https://github.com/Dao-AILab/causal-conv1d.git")
    print("  cd causal-conv1d && pip install -e .")
    
    print("\n[Solution 3] Use conda-forge (most compatible):")
    print("  conda install -c conda-forge causal-conv1d mamba-ssm")
    
    print("\n[Solution 4] Use Docker/Singularity:")
    print("  # Use a container with compatible GLIBC")
    print("  singularity exec --nv pytorch_2.0.1_cu118.sif python your_script.py")
    
    print("\n[Solution 5] Upgrade system GLIBC (advanced):")
    print("  # This is system-dependent and risky")
    print("  # Check your OS documentation for GLIBC upgrade")

def main():
    """Main diagnostic function"""
    print("GLIBC Compatibility Diagnosis for mamba-ssm and causal_conv1d")
    print("=" * 60)
    
    print(f"Python: {sys.version}")
    print(f"Platform: {platform.platform()}")
    print(f"Architecture: {platform.machine()}")
    
    # Check GLIBC version
    glibc_version = check_glibc_version()
    
    # Analyze problematic libraries
    analyze_problematic_libraries()
    
    # Test import hierarchy
    test_import_hierarchy()
    
    # Check conda environment
    check_conda_environment()
    
    # Provide solutions
    provide_solutions()
    
    print(f"\n{'=' * 60}")
    print("Diagnosis complete!")
    
    if glibc_version and glibc_version < 2.34:
        print(f"⚠️  Your GLIBC {glibc_version} is older than required 2.34")
        print("   Recommend: Use Solution 1 (compatible versions) or Solution 3 (conda-forge)")
    else:
        print("ℹ️  Your GLIBC version should be compatible")
        print("   The issue might be with specific compiled binaries")
        print("   Recommend: Use Solution 2 (build from source)")

if __name__ == "__main__":
    main()
