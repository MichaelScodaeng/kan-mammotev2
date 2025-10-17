"""
Simple script to demonstrate the warm-up effect on Mamba2 and ControllableMamba2.
This shows the compilation overhead on first run vs subsequent runs.
"""

import torch
import time
from mamba_ssm.modules.mamba2 import Mamba2
from models.time_encoders.controllable_mamba2 import ControllableMamba2

def test_warmup_effect():
    """Test the warm-up effect on both vanilla and controllable Mamba2"""
    
    print("="*80)
    print("Warm-up Effect Demonstration")
    print("="*80)
    
    # Configuration
    batch_size = 2
    seq_len = 128
    expert_dim = 256
    d_state = 64
    d_conv = 4
    expand = 2
    headdim = 16
    
    # Calculate nheads based on Mamba2 internals
    # d_inner = expand * d_model = 2 * 256 = 512
    # d_ssm = d_inner (by default) = 512
    # nheads = d_ssm // headdim = 512 // 16 = 32
    d_inner = expand * expert_dim
    d_ssm = d_inner  # Default behavior when d_ssm is None
    nheads = d_ssm // headdim
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Expert dimension (d_model): {expert_dim}")
    print(f"  d_inner (expand * d_model): {d_inner}")
    print(f"  d_ssm: {d_ssm}")
    print(f"  headdim: {headdim}")
    print(f"  nheads (d_ssm // headdim): {nheads}")
    
    # Test data
    u = torch.randn(batch_size, seq_len, expert_dim).to("cuda")
    
    # Create temporal modulators with CORRECT dimensions
    # gamma and beta should have shape (batch, seq_len, nheads)
    gamma = torch.sigmoid(torch.randn(batch_size, seq_len, nheads).to("cuda")) + 0.5
    beta = torch.randn(batch_size, seq_len, nheads).to("cuda")
    temporal_modulators = (gamma, beta)
    
    print(f"\nTemporal modulators shape:")
    print(f"  gamma: {gamma.shape}")
    print(f"  beta: {beta.shape}")
    
    # ========================================
    # Test 1: Vanilla Mamba2
    # ========================================
    print(f"\n{'='*80}")
    print("TEST 1: Vanilla Mamba2")
    print(f"{'='*80}")
    
    model_vanilla = Mamba2(
        d_model=expert_dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim
    ).to("cuda")
    
    print(f"\nVanilla Mamba2 architecture:")
    print(f"  d_model: {model_vanilla.d_model}")
    print(f"  d_inner: {model_vanilla.d_inner}")
    print(f"  d_ssm: {model_vanilla.d_ssm}")
    print(f"  nheads: {model_vanilla.nheads}")
    print(f"  headdim: {model_vanilla.headdim}")
    
    # First run (with compilation)
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        y1 = model_vanilla(u)
    torch.cuda.synchronize()
    first_run_time = time.time() - t1
    
    print(f"\n  First run (with compilation): {first_run_time:.3f} s")
    
    # Second run (cached)
    torch.cuda.synchronize()
    t2 = time.time()
    with torch.no_grad():
        y2 = model_vanilla(u)
    torch.cuda.synchronize()
    second_run_time = time.time() - t2
    
    print(f"  Second run (cached):          {second_run_time:.3f} s")
    print(f"  Speedup: {first_run_time/second_run_time:.1f}x")
    
    # ========================================
    # Test 2: ControllableMamba2
    # ========================================
    print(f"\n{'='*80}")
    print("TEST 2: ControllableMamba2")
    print(f"{'='*80}")
    
    model_controllable = ControllableMamba2(
        d_model=expert_dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim
    ).to("cuda")
    
    print(f"\nControllableMamba2 architecture:")
    print(f"  d_model: {model_controllable.d_model}")
    print(f"  d_inner: {model_controllable.d_inner}")
    print(f"  d_ssm: {model_controllable.d_ssm}")
    print(f"  nheads: {model_controllable.nheads}")
    print(f"  headdim: {model_controllable.headdim}")
    
    # Verify modulators match nheads
    assert gamma.shape[-1] == model_controllable.nheads, \
        f"Modulator nheads mismatch: {gamma.shape[-1]} vs {model_controllable.nheads}"
    
    # First run (with compilation)
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        y1 = model_controllable(u, temporal_modulators=temporal_modulators)
    torch.cuda.synchronize()
    first_run_time = time.time() - t1
    
    print(f"\n  First run (with compilation): {first_run_time:.3f} s")
    
    # Second run (cached)
    torch.cuda.synchronize()
    t2 = time.time()
    with torch.no_grad():
        y2 = model_controllable(u, temporal_modulators=temporal_modulators)
    torch.cuda.synchronize()
    second_run_time = time.time() - t2
    
    print(f"  Second run (cached):          {second_run_time:.3f} s")
    print(f"  Speedup: {first_run_time/second_run_time:.1f}x")
    
    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print("\nWarm-up is CRITICAL for accurate benchmarking!")
    print("Always run a few iterations before timing to compile CUDA kernels.")
    print("\nRecommended warm-up pattern:")
    print("```python")
    print("# Warm-up phase")
    print("for _ in range(3):")
    print("    with torch.no_grad():")
    print("        _ = model(input)")
    print("")
    print("# Now measure actual performance")
    print("torch.cuda.synchronize()")
    print("t1 = time.time()")
    print("output = model(input)")
    print("torch.cuda.synchronize()")
    print("elapsed = time.time() - t1")
    print("```")

def test_multiple_runs():
    """Test performance consistency across multiple runs"""
    print(f"\n\n{'='*80}")
    print("Multiple Runs Test (after warm-up)")
    print(f"{'='*80}")
    
    batch_size = 2
    seq_len = 128
    expert_dim = 256
    d_state = 64
    d_conv = 4
    expand = 2
    headdim = 16
    
    # Calculate correct nheads
    d_inner = expand * expert_dim
    d_ssm = d_inner
    nheads = d_ssm // headdim
    
    u = torch.randn(batch_size, seq_len, expert_dim).to("cuda")
    gamma = torch.sigmoid(torch.randn(batch_size, seq_len, nheads).to("cuda")) + 0.5
    beta = torch.randn(batch_size, seq_len, nheads).to("cuda")
    temporal_modulators = (gamma, beta)
    
    model = ControllableMamba2(
        d_model=expert_dim,
        d_state=d_state,
        d_conv=d_conv,
        expand=expand,
        headdim=headdim
    ).to("cuda")
    
    print(f"\nModel nheads: {model.nheads}")
    print(f"Modulator nheads: {gamma.shape[-1]}")
    
    # Warm-up
    print("\nWarming up (3 iterations)...")
    for _ in range(3):
        with torch.no_grad():
            _ = model(u, temporal_modulators=temporal_modulators)
    
    # Test 10 runs
    print("\nRunning 10 iterations...")
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        t1 = time.time()
        with torch.no_grad():
            _ = model(u, temporal_modulators=temporal_modulators)
        torch.cuda.synchronize()
        elapsed = time.time() - t1
        times.append(elapsed)
        print(f"  Run {i+1}: {elapsed*1000:.3f} ms")
    
    # Statistics
    times = torch.tensor(times)
    print(f"\nStatistics:")
    print(f"  Mean: {times.mean()*1000:.3f} ms")
    print(f"  Std:  {times.std()*1000:.3f} ms")
    print(f"  Min:  {times.min()*1000:.3f} ms")
    print(f"  Max:  {times.max()*1000:.3f} ms")

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Run tests
    test_warmup_effect()
    test_multiple_runs()
    
    print(f"\n{'='*80}")
    print("Tests complete!")
    print(f"{'='*80}")