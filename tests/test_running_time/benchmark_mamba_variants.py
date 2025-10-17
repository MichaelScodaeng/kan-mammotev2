"""
Benchmark script to compare vanilla Mamba2 vs ControllableMamba2 performance.
This will help identify if the FiLM modulation introduces significant overhead.
"""

import torch
import torch.nn as nn
import time
from einops import rearrange

from mamba_ssm.modules.mamba2 import Mamba2
from models.time_encoders.controllable_mamba2 import ControllableMamba2

def benchmark_model(model, input_data, num_warmup=5, num_runs=50, model_name="Model"):
    """Benchmark a model with proper warm-up."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*60}")
    
    # Warm-up phase
    print(f"Warming up ({num_warmup} iterations)...")
    for i in range(num_warmup):
        with torch.no_grad():
            if isinstance(input_data, dict):
                _ = model(**input_data)
            else:
                _ = model(input_data)
        if i == 0:
            print(f"  First warm-up iteration completed (compilation done)")
    
    # Synchronize CUDA before timing
    torch.cuda.synchronize()
    
    # Benchmark phase
    print(f"Running benchmark ({num_runs} iterations)...")
    times = []
    
    for i in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        
        with torch.no_grad():
            if isinstance(input_data, dict):
                output = model(**input_data)
            else:
                output = model(input_data)
        
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
        
        if (i + 1) % 10 == 0:
            print(f"  Completed {i+1}/{num_runs} iterations")
    
    # Statistics
    times = torch.tensor(times)
    mean_time = times.mean().item()
    std_time = times.std().item()
    min_time = times.min().item()
    max_time = times.max().item()
    
    print(f"\nResults for {model_name}:")
    print(f"  Mean time: {mean_time*1000:.3f} ms (± {std_time*1000:.3f} ms)")
    print(f"  Min time:  {min_time*1000:.3f} ms")
    print(f"  Max time:  {max_time*1000:.3f} ms")
    print(f"  Throughput: {1/mean_time:.2f} iterations/sec")
    
    return mean_time, output

def create_baseline_model(expert_dim, d_state, d_conv, expand):
    """Create baseline: concatenation + MLP + vanilla Mamba2"""
    class BaselineMambaModel(nn.Module):
        def __init__(self, expert_dim, d_state, d_conv, expand):
            super().__init__()
            # Fusion MLP (simulating your fusion)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(8, expert_dim),  # num_mixtures=8
                nn.LayerNorm(expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, expert_dim)
            )
            
            # Vanilla Mamba2
            self.mamba2 = Mamba2(
                d_model=expert_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=16
            )
        
        def forward(self, u, v):
            # Concatenate and fuse
            fused = self.fusion_mlp(v)
            # Add to input (residual-like)
            combined = u + fused
            # Pass through vanilla Mamba2
            return self.mamba2(combined)
    
    return BaselineMambaModel(expert_dim, d_state, d_conv, expand).to("cuda")

def create_controllable_model(expert_dim, d_state, d_conv, expand):
    """Create your ControllableMamba2 model"""
    class ControllableMambaModel(nn.Module):
        def __init__(self, expert_dim, d_state, d_conv, expand):
            super().__init__()
            self.expert_dim = expert_dim
            
            # Fusion MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(8, expert_dim),  # num_mixtures=8
                nn.LayerNorm(expert_dim),
                nn.GELU(),
                nn.Linear(expert_dim, (expert_dim // 16) * 2)  # nheads * 2
            )
            
            # ControllableMamba2
            self.mamba2 = ControllableMamba2(
                d_model=expert_dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=16
            )
        
        def forward(self, u, v):
            # Generate temporal modulators from v
            modulator_logits = self.fusion_mlp(v)
            gamma_logits, beta = modulator_logits.chunk(2, dim=-1)
            gamma = torch.sigmoid(gamma_logits) + 0.5  # Range: [0.5, 1.5]
            temporal_modulators = (gamma, beta)
            
            return self.mamba2(u=u, temporal_modulators=temporal_modulators)
    
    return ControllableMambaModel(expert_dim, d_state, d_conv, expand).to("cuda")

def main():
    print("="*60)
    print("Mamba2 Variants Performance Benchmark")
    print("="*60)
    
    # Configuration
    batch_size = 2
    seq_len = 512
    expert_dim = 256
    d_state = 64
    d_conv = 4
    expand = 2
    num_mixtures = 8
    nheads = expert_dim // 16
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Expert dimension: {expert_dim}")
    print(f"  d_state: {d_state}")
    print(f"  d_conv: {d_conv}")
    print(f"  expand: {expand}")
    print(f"  nheads: {nheads}")
    
    # Create test data
    print(f"\nCreating test data...")
    u = torch.randn(batch_size, seq_len, expert_dim).to("cuda")
    v = torch.randn(batch_size, seq_len, num_mixtures).to("cuda")
    
    # ========================================
    # Test 1: Vanilla Mamba2 (Baseline)
    # ========================================
    print(f"\n{'#'*60}")
    print("TEST 1: Baseline (Concatenation + MLP + Vanilla Mamba2)")
    print(f"{'#'*60}")
    
    baseline_model = create_baseline_model(expert_dim, d_state, d_conv, expand)
    baseline_input = {'u': u, 'v': v}
    baseline_time, baseline_output = benchmark_model(
        baseline_model, 
        baseline_input,
        num_warmup=5,
        num_runs=50,
        model_name="Baseline Mamba2"
    )
    
    # ========================================
    # Test 2: ControllableMamba2
    # ========================================
    print(f"\n{'#'*60}")
    print("TEST 2: ControllableMamba2 (with FiLM modulation)")
    print(f"{'#'*60}")
    
    controllable_model = create_controllable_model(expert_dim, d_state, d_conv, expand)
    controllable_input = {'u': u, 'v': v}
    controllable_time, controllable_output = benchmark_model(
        controllable_model,
        controllable_input,
        num_warmup=5,
        num_runs=50,
        model_name="ControllableMamba2"
    )
    
    # ========================================
    # Comparison
    # ========================================
    print(f"\n{'='*60}")
    print("PERFORMANCE COMPARISON")
    print(f"{'='*60}")
    
    speedup_ratio = controllable_time / baseline_time
    overhead_pct = (speedup_ratio - 1.0) * 100
    
    print(f"\nBaseline Mamba2:       {baseline_time*1000:.3f} ms")
    print(f"ControllableMamba2:    {controllable_time*1000:.3f} ms")
    print(f"\nRelative performance:")
    
    if speedup_ratio > 1.0:
        print(f"  Overhead: +{overhead_pct:.2f}%")
        print(f"  ControllableMamba2 is {speedup_ratio:.2f}x SLOWER")
    else:
        print(f"  Speedup: {-overhead_pct:.2f}%")
        print(f"  ControllableMamba2 is {1/speedup_ratio:.2f}x FASTER")
    
    print(f"\n{'='*60}")
    print("Shape verification:")
    print(f"  Baseline output shape: {baseline_output.shape}")
    print(f"  Controllable output shape: {controllable_output.shape}")
    print(f"{'='*60}")
    
    # Memory usage
    print(f"\n{'='*60}")
    print("Memory Usage:")
    print(f"{'='*60}")
    print(f"  GPU memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"  GPU memory reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Run benchmark
    main()
