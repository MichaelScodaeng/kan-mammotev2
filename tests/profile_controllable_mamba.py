"""
Profiling script to identify performance bottlenecks in ControllableMamba2.
This will generate a detailed trace that can be viewed in Chrome.
"""

import torch
import torch.profiler as profiler
from models.time_encoders.controllable_mamba2 import ControllableMamba2
from mamba_ssm.modules.mamba2 import Mamba2
import torch.nn as nn

def profile_vanilla_mamba():
    """Profile vanilla Mamba2 as baseline"""
    print("="*80)
    print("Profiling Vanilla Mamba2 (Baseline)")
    print("="*80)
    
    # Setup
    batch_size = 2
    seq_len = 512
    expert_dim = 256
    
    model = Mamba2(
        d_model=expert_dim,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=16
    ).to("cuda")
    
    u = torch.randn(batch_size, seq_len, expert_dim).to("cuda")
    
    # Warm-up
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = model(u)
    
    torch.cuda.synchronize()
    
    # Profile
    print("\nProfiling vanilla Mamba2...")
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for _ in range(10):  # Run multiple iterations for better statistics
                output = model(u)
    
    # Print results
    print("\n" + "="*80)
    print("Top 15 operations by CUDA time:")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=15
    ))
    
    # Export trace
    prof.export_chrome_trace("vanilla_mamba_trace.json")
    print("\nTrace exported to: vanilla_mamba_trace.json")

def profile_controllable_mamba():
    """Profile ControllableMamba2 to identify bottlenecks"""
    print("\n" + "="*80)
    print("Profiling ControllableMamba2")
    print("="*80)
    
    # Setup
    batch_size = 2
    seq_len = 512
    expert_dim = 256
    nheads = expert_dim // 16
    
    model = ControllableMamba2(
        d_model=expert_dim,
        d_state=64,
        d_conv=4,
        expand=2,
        headdim=16
    ).to("cuda")
    
    u = torch.randn(batch_size, seq_len, expert_dim).to("cuda")
    gamma = torch.sigmoid(torch.randn(batch_size, seq_len, nheads).to("cuda")) + 0.5
    beta = torch.randn(batch_size, seq_len, nheads).to("cuda")
    temporal_modulators = (gamma, beta)
    
    # Warm-up
    print("Warming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = model(u, temporal_modulators=temporal_modulators)
    
    torch.cuda.synchronize()
    
    # Profile
    print("\nProfiling ControllableMamba2...")
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        with torch.no_grad():
            for _ in range(10):  # Run multiple iterations for better statistics
                output = model(u, temporal_modulators=temporal_modulators)
    
    # Print results
    print("\n" + "="*80)
    print("Top 15 operations by CUDA time:")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=15
    ))
    
    print("\n" + "="*80)
    print("Top 15 operations by CPU time:")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total", 
        row_limit=15
    ))
    
    # Export trace
    prof.export_chrome_trace("controllable_mamba_trace.json")
    print("\nTrace exported to: controllable_mamba_trace.json")

def profile_film_overhead():
    """Profile just the FiLM modulation overhead"""
    print("\n" + "="*80)
    print("Profiling FiLM Modulation Overhead")
    print("="*80)
    
    batch_size = 2
    seq_len = 512
    expert_dim = 256
    nheads = expert_dim // 16
    
    # Create dummy tensors
    dt_content = torch.randn(batch_size, seq_len, nheads).to("cuda")
    gamma = torch.sigmoid(torch.randn(batch_size, seq_len, nheads).to("cuda")) + 0.5
    beta = torch.randn(batch_size, seq_len, nheads).to("cuda")
    
    # Dummy splits for concatenation
    z0 = torch.randn(batch_size, seq_len, 100).to("cuda")
    x0 = torch.randn(batch_size, seq_len, 100).to("cuda")
    z = torch.randn(batch_size, seq_len, expert_dim).to("cuda")
    xBC = torch.randn(batch_size, seq_len, expert_dim + 128).to("cuda")
    
    # Warm-up
    for _ in range(100):
        dt_fused = gamma * dt_content + beta
        zxbcdt_modified = torch.cat([z0, x0, z, xBC, dt_fused], dim=-1).contiguous()
    
    torch.cuda.synchronize()
    
    # Profile
    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for _ in range(1000):
            dt_fused = gamma * dt_content + beta
            zxbcdt_modified = torch.cat([z0, x0, z, xBC, dt_fused], dim=-1).contiguous()
    
    print("\n" + "="*80)
    print("FiLM operations breakdown:")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cuda_time_total", 
        row_limit=10
    ))
    
    prof.export_chrome_trace("film_overhead_trace.json")
    print("\nTrace exported to: film_overhead_trace.json")

def main():
    print("="*80)
    print("Mamba2 Performance Profiling Suite")
    print("="*80)
    print("\nThis will generate 3 Chrome trace files:")
    print("  1. vanilla_mamba_trace.json - Baseline Mamba2")
    print("  2. controllable_mamba_trace.json - ControllableMamba2")
    print("  3. film_overhead_trace.json - FiLM modulation only")
    print("\nView traces in Chrome at: chrome://tracing")
    print("="*80)
    
    # Run all profiles
    profile_vanilla_mamba()
    profile_controllable_mamba()
    profile_film_overhead()
    
    print("\n" + "="*80)
    print("Profiling complete!")
    print("="*80)
    print("\nTo view the traces:")
    print("  1. Open Chrome browser")
    print("  2. Navigate to: chrome://tracing")
    print("  3. Click 'Load' and select one of the trace files")
    print("  4. Use W/A/S/D to navigate the timeline")

if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Run profiling
    main()
