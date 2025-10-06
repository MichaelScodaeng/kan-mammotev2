"""
Diagnostic script to identify training bottlenecks.
"""

import torch
import time
from models.time_encoders.kan_mammote import KAN_MAMMOTE

def diagnose_training_speed():
    """Diagnose where training time is spent"""
    
    print("="*80)
    print("Training Speed Diagnosis")
    print("="*80)
    
    # Your actual training configuration
    batch_size = 32  # ADJUST TO YOUR CONFIG
    seq_len = 512    # ADJUST TO YOUR CONFIG
    embedding_dim = 256
    expert_dim = 256
    num_mixtures = 8
    device = 'cuda'
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Embedding dim: {embedding_dim}")
    
    # Initialize model
    print("\nInitializing model...")
    model = KAN_MAMMOTE(
        embedding_dim=embedding_dim,
        expert_dim=expert_dim,
        num_mixtures=num_mixtures,
        mamba_d_state=256,
        mamba_d_conv=4,
        mamba_expand=4,
        wavelet_type='shock',
        mamba_headdim=16
    ).to(device)
    
    # Initialize SM-Kernel
    delta_t_sample = torch.randn(100, 1).abs().to(device)
    model.initialize_sm_kernel(delta_t_sample)
    
    # Warm up
    print("\nWarming up...")
    model.warmup(device=device, num_iterations=3)
    
    # Create sample data
    t_abs = torch.randn(batch_size, seq_len, 1).to(device)
    t_rel = torch.randn(batch_size, seq_len, 1).abs().to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # ========================================
    # Time one training iteration
    # ========================================
    print("\n" + "="*80)
    print("Timing ONE training iteration (forward + backward + optimizer step)")
    print("="*80)
    
    num_iterations = 10
    times = {
        'forward': [],
        'backward': [],
        'optimizer': [],
        'total': []
    }
    
    model.train()
    for i in range(num_iterations):
        torch.cuda.synchronize()
        iter_start = time.time()
        
        # Forward pass
        torch.cuda.synchronize()
        forward_start = time.time()
        output = model(t_abs, t_rel)
        loss = output.mean()  # Dummy loss
        torch.cuda.synchronize()
        forward_time = time.time() - forward_start
        
        # Backward pass
        torch.cuda.synchronize()
        backward_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - backward_start
        
        # Optimizer step
        torch.cuda.synchronize()
        optimizer_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        optimizer_time = time.time() - optimizer_start
        
        torch.cuda.synchronize()
        iter_time = time.time() - iter_start
        
        times['forward'].append(forward_time)
        times['backward'].append(backward_time)
        times['optimizer'].append(optimizer_time)
        times['total'].append(iter_time)
        
        print(f"  Iter {i+1}: {iter_time*1000:.1f}ms (fwd: {forward_time*1000:.1f}ms, bwd: {backward_time*1000:.1f}ms, opt: {optimizer_time*1000:.1f}ms)")
    
    # Statistics
    import numpy as np
    print("\n" + "="*80)
    print("Statistics (averaged over 10 iterations):")
    print("="*80)
    
    for key in ['forward', 'backward', 'optimizer', 'total']:
        avg = np.mean(times[key]) * 1000
        std = np.std(times[key]) * 1000
        print(f"  {key.capitalize():12s}: {avg:.2f} ms (±{std:.2f} ms)")
    
    # Estimate time per epoch
    print("\n" + "="*80)
    print("Estimated time per epoch:")
    print("="*80)
    
    avg_iteration = np.mean(times['total'])
    
    # Ask user for number of batches
    print("\nHow many batches per epoch do you have?")
    print("(Check your dataloader: len(train_loader))")
    num_batches = int(input("Enter number of batches: "))
    
    estimated_epoch_time = avg_iteration * num_batches
    
    print(f"\nWith {num_batches} batches per epoch:")
    print(f"  Time per iteration: {avg_iteration*1000:.1f} ms")
    print(f"  Estimated epoch time: {estimated_epoch_time:.1f} seconds ({estimated_epoch_time/60:.1f} minutes)")
    
    if estimated_epoch_time > 300:  # > 5 minutes
        print("\n⚠️  Training is slow! Possible optimizations:")
        print("  1. Increase batch size (if GPU memory allows)")
        print("  2. Use gradient accumulation for larger effective batch size")
        print("  3. Use mixed precision training (torch.cuda.amp)")
        print("  4. Check if data loading is the bottleneck")
        print("  5. Profile with PyTorch profiler to find bottlenecks")
    else:
        print("\n✓ Training speed looks reasonable!")

if __name__ == "__main__":
    diagnose_training_speed()