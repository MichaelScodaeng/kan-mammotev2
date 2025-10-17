"""
Test script to verify KAN-MAMMOTE warm-up functionality.
This demonstrates the compilation overhead reduction from warm-up.
"""

import torch
import time
from models.time_encoders.kan_mammote import KAN_MAMMOTE

def test_warmup_benefit():
    """Test the benefit of warm-up on KAN-MAMMOTE"""
    
    print("="*80)
    print("KAN-MAMMOTE Warm-up Test")
    print("="*80)
    
    # Configuration
    embedding_dim = 256
    expert_dim = 256
    num_mixtures = 8
    batch_size = 2
    seq_len = 128
    
    print(f"\nConfiguration:")
    print(f"  Embedding dim: {embedding_dim}")
    print(f"  Expert dim: {expert_dim}")
    print(f"  Num mixtures: {num_mixtures}")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    
    # Initialize model
    print(f"\nInitializing KAN-MAMMOTE...")
    model = KAN_MAMMOTE(
        embedding_dim=embedding_dim,
        expert_dim=expert_dim,
        num_mixtures=num_mixtures,
        mamba_d_state=256,
        mamba_d_conv=4,
        mamba_expand=4,
        wavelet_type='shock',
        mamba_headdim=16
    ).to("cuda")
    
    # Initialize SM-Kernel
    print(f"\nInitializing SM-Kernel...")
    delta_t_sample = torch.randn(1, 100, 1).abs().to("cuda")
    model.initialize_sm_kernel(delta_t_sample)
    
    # Create test data
    t_abs = torch.randn(batch_size, seq_len, 1).to("cuda")
    t_rel = torch.randn(batch_size, seq_len, 1).abs().to("cuda")
    
    # ========================================
    # Test WITHOUT warm-up (new process simulation)
    # ========================================
    print(f"\n{'='*80}")
    print("TEST 1: Running WITHOUT explicit warm-up")
    print(f"{'='*80}")
    print("(Simulates first run in a new process)")
    
    torch.cuda.synchronize()
    t1 = time.time()
    with torch.no_grad():
        output1 = model(t_abs, t_rel)
    torch.cuda.synchronize()
    first_cold_time = time.time() - t1
    
    print(f"  First run (cold): {first_cold_time:.3f}s")
    
    # Second run (now cached)
    torch.cuda.synchronize()
    t2 = time.time()
    with torch.no_grad():
        output2 = model(t_abs, t_rel)
    torch.cuda.synchronize()
    second_cached_time = time.time() - t2
    
    print(f"  Second run (cached): {second_cached_time:.3f}s")
    print(f"  Speedup: {first_cold_time/second_cached_time:.1f}x")
    
    # ========================================
    # Test WITH warm-up (recommended approach)
    # ========================================
    print(f"\n{'='*80}")
    print("TEST 2: Using warmup() method (RECOMMENDED)")
    print(f"{'='*80}")
    
    # Create fresh model to simulate new process
    print(f"\nCreating fresh model...")
    model2 = KAN_MAMMOTE(
        embedding_dim=embedding_dim,
        expert_dim=expert_dim,
        num_mixtures=num_mixtures,
        mamba_d_state=256,
        mamba_d_conv=4,
        mamba_expand=4,
        wavelet_type='shock',
        mamba_headdim=16
    ).to("cuda")
    model2.initialize_sm_kernel(delta_t_sample)
    
    # Call warmup
    model2.warmup(device='cuda', num_iterations=3)
    
    # Now measure performance (should be fast)
    print(f"\nMeasuring performance after warm-up...")
    times = []
    for i in range(10):
        torch.cuda.synchronize()
        t_start = time.time()
        with torch.no_grad():
            _ = model2(t_abs, t_rel)
        torch.cuda.synchronize()
        elapsed = time.time() - t_start
        times.append(elapsed)
    
    import numpy as np
    mean_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"  Average time (10 runs): {mean_time*1000:.3f} ms (± {std_time*1000:.3f} ms)")
    print(f"  Throughput: {1/mean_time:.2f} iterations/sec")
    
    # ========================================
    # Summary
    # ========================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    
    print(f"\n✅ Warm-up Impact:")
    print(f"   - Cold start (1st run): {first_cold_time:.3f}s")
    print(f"   - After warm-up: {mean_time*1000:.3f}ms")
    print(f"   - Speedup: {first_cold_time/(mean_time):.0f}x")
    
    print(f"\n💡 Recommendation:")
    print(f"   Always call model.warmup() ONCE before training:")
    print(f"   ```python")
    print(f"   # After model initialization")
    print(f"   kan_mammote.warmup(device='cuda', num_iterations=3)")
    print(f"   ")
    print(f"   # Then start training loop")
    print(f"   for epoch in range(num_epochs):")
    print(f"       ...  # No need to warm up again!")
    print(f"   ```")
    
    print(f"\n🚀 This saves ~{first_cold_time:.1f} seconds at the start of training!")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Run test
    test_warmup_benefit()
