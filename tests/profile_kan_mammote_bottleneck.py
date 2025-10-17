"""
Detailed profiling script to identify bottlenecks in KAN-MAMMOTE training.
This will help identify if MLPs, Mamba2, or other components are causing slowness.
"""

import torch
import torch.nn as nn
import time
import numpy as np
from models.time_encoders.kan_mammote import KAN_MAMMOTE

def profile_component(name, func, num_runs=50):
    """Profile a single component"""
    times = []
    
    # Warm up
    for _ in range(5):
        func()
    
    # Profile
    for _ in range(num_runs):
        torch.cuda.synchronize()
        start = time.time()
        func()
        torch.cuda.synchronize()
        elapsed = time.time() - start
        times.append(elapsed)
    
    times = np.array(times)
    return {
        'mean': times.mean() * 1000,  # ms
        'std': times.std() * 1000,
        'min': times.min() * 1000,
        'max': times.max() * 1000
    }

def detailed_profiling(use_controllable_mamba=True):
    """
    Detailed profiling of all KAN-MAMMOTE components to identify bottlenecks.
    """
    variant = "ControllableMamba2" if use_controllable_mamba else "Vanilla Mamba2"
    
    print(f"\n{'='*80}")
    print(f"Detailed Profiling: KAN-MAMMOTE with {variant}")
    print(f"{'='*80}")
    
    # Configuration matching your training setup
    batch_size = 32  # ← ADJUST THIS to match your training
    seq_len = 512    # ← ADJUST THIS to match your training
    expert_dim = 256
    embedding_dim = 256
    num_mixtures = 8
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Expert dim: {expert_dim}")
    print(f"  Embedding dim: {embedding_dim}")
    
    # Initialize model
    print(f"\nInitializing model...")
    model = KAN_MAMMOTE(
        embedding_dim=embedding_dim,
        expert_dim=expert_dim,
        num_mixtures=num_mixtures,
        mamba_d_state=256,
        mamba_d_conv=4,
        mamba_expand=4,
        wavelet_type='shock',
        mamba_headdim=16,
        use_controllable_mamba=use_controllable_mamba
    ).to("cuda")
    
    # Initialize SM-Kernel (needs 3D tensor: batch, seq, 1)
    delta_t_sample = torch.randn(1, 100, 1).abs().to("cuda")
    model.initialize_sm_kernel(delta_t_sample)
    
    # Create test data
    t_abs = torch.randn(batch_size, seq_len, 1).to("cuda")
    t_rel = torch.randn(batch_size, seq_len, 1).abs().to("cuda")
    
    # Warm up the full model
    print(f"\nWarming up full model...")
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            _ = model(t_abs, t_rel)
    
    # ========================================
    # Profile Individual Components
    # ========================================
    print(f"\n{'='*80}")
    print("Component-wise Profiling (50 iterations each)")
    print(f"{'='*80}\n")
    
    results = {}
    
    # 1. K-MOTE (Wavelet transforms)
    print("1. Profiling K-MOTE (wavelet transforms)...")
    with torch.no_grad():
        stats = profile_component(
            "K-MOTE",
            lambda: model.k_mote(t_abs),
            num_runs=50
        )
    results['k_mote'] = stats
    print(f"   Mean: {stats['mean']:.3f} ms (±{stats['std']:.3f} ms)")
    
    # 2. SM-Kernel
    print("2. Profiling SM-Kernel...")
    with torch.no_grad():
        stats = profile_component(
            "SM-Kernel",
            lambda: model.sm_kernel(t_rel),
            num_runs=50
        )
    results['sm_kernel'] = stats
    print(f"   Mean: {stats['mean']:.3f} ms (±{stats['std']:.3f} ms)")
    
    # 3. Fusion MLP Base
    print("3. Profiling Fusion MLP Base...")
    v_k = model.sm_kernel(t_rel)
    with torch.no_grad():
        stats = profile_component(
            "Fusion MLP",
            lambda: model.fusion_mlp_base(v_k),
            num_runs=50
        )
    results['fusion_mlp'] = stats
    print(f"   Mean: {stats['mean']:.3f} ms (±{stats['std']:.3f} ms)")
    
    # 4. Modulator Head (if ControllableMamba2)
    if use_controllable_mamba:
        print("4. Profiling Modulator Head...")
        fusion_features = model.fusion_mlp_base(v_k)
        with torch.no_grad():
            stats = profile_component(
                "Modulator Head",
                lambda: model.modulator_head(fusion_features),
                num_runs=50
            )
        results['modulator_head'] = stats
        print(f"   Mean: {stats['mean']:.3f} ms (±{stats['std']:.3f} ms)")
    
    # 5. Mamba2 Forward
    print(f"5. Profiling Mamba2 Forward...")
    u_k = model.k_mote(t_abs)
    fusion_features = model.fusion_mlp_base(v_k)
    combined_input = u_k + fusion_features
    
    if use_controllable_mamba:
        modulator_logits = model.modulator_head(fusion_features)
        gamma_logits, beta = modulator_logits.chunk(2, dim=-1)
        gamma = torch.sigmoid(gamma_logits) + 0.5
        temporal_modulators = (gamma, beta)
        
        with torch.no_grad():
            stats = profile_component(
                "Mamba2",
                lambda: model.mamba2(u=combined_input, temporal_modulators=temporal_modulators),
                num_runs=50
            )
    else:
        with torch.no_grad():
            stats = profile_component(
                "Mamba2",
                lambda: model.mamba2(combined_input),
                num_runs=50
            )
    results['mamba2'] = stats
    print(f"   Mean: {stats['mean']:.3f} ms (±{stats['std']:.3f} ms)")
    
    # 6. Output Projection
    print("6. Profiling Output Projection...")
    mamba_output = model.mamba2(combined_input) if not use_controllable_mamba else \
                   model.mamba2(u=combined_input, temporal_modulators=temporal_modulators)
    with torch.no_grad():
        stats = profile_component(
            "Output Projection",
            lambda: model.output_projection(mamba_output),
            num_runs=50
        )
    results['output_proj'] = stats
    print(f"   Mean: {stats['mean']:.3f} ms (±{stats['std']:.3f} ms)")
    
    # 7. Full Forward Pass
    print("7. Profiling Full Forward Pass...")
    with torch.no_grad():
        stats = profile_component(
            "Full Forward",
            lambda: model(t_abs, t_rel),
            num_runs=50
        )
    results['full_forward'] = stats
    print(f"   Mean: {stats['mean']:.3f} ms (±{stats['std']:.3f} ms)")
    
    # ========================================
    # Profile Training Step (Forward + Backward + Optimizer)
    # ========================================
    print(f"\n{'='*80}")
    print("Full Training Step Profiling (Forward + Backward + Optimizer)")
    print(f"{'='*80}\n")
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    train_times = []
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    for i in range(50):
        torch.cuda.synchronize()
        iter_start = time.time()
        
        # Forward
        torch.cuda.synchronize()
        fwd_start = time.time()
        output = model(t_abs, t_rel)
        loss = output.mean()  # Dummy loss
        torch.cuda.synchronize()
        fwd_time = time.time() - fwd_start
        
        # Backward
        torch.cuda.synchronize()
        bwd_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        bwd_time = time.time() - bwd_start
        
        # Optimizer
        torch.cuda.synchronize()
        opt_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        opt_time = time.time() - opt_start
        
        torch.cuda.synchronize()
        total_time = time.time() - iter_start
        
        train_times.append(total_time)
        forward_times.append(fwd_time)
        backward_times.append(bwd_time)
        optimizer_times.append(opt_time)
    
    train_times = np.array(train_times) * 1000
    forward_times = np.array(forward_times) * 1000
    backward_times = np.array(backward_times) * 1000
    optimizer_times = np.array(optimizer_times) * 1000
    
    print(f"Training Step Breakdown:")
    print(f"  Forward pass:   {forward_times.mean():.3f} ms (±{forward_times.std():.3f} ms)")
    print(f"  Backward pass:  {backward_times.mean():.3f} ms (±{backward_times.std():.3f} ms)")
    print(f"  Optimizer step: {optimizer_times.mean():.3f} ms (±{optimizer_times.std():.3f} ms)")
    print(f"  Total:          {train_times.mean():.3f} ms (±{train_times.std():.3f} ms)")
    
    # ========================================
    # Analysis & Recommendations
    # ========================================
    print(f"\n{'='*80}")
    print("BOTTLENECK ANALYSIS")
    print(f"{'='*80}\n")
    
    # Component breakdown
    total_components = sum([
        results['k_mote']['mean'],
        results['sm_kernel']['mean'],
        results['fusion_mlp']['mean'],
        results.get('modulator_head', {'mean': 0})['mean'],
        results['mamba2']['mean'],
        results['output_proj']['mean']
    ])
    
    print("Component Breakdown (% of total):")
    components = [
        ('K-MOTE', results['k_mote']['mean']),
        ('SM-Kernel', results['sm_kernel']['mean']),
        ('Fusion MLP', results['fusion_mlp']['mean']),
    ]
    if use_controllable_mamba:
        components.append(('Modulator Head', results['modulator_head']['mean']))
    components.extend([
        ('Mamba2', results['mamba2']['mean']),
        ('Output Proj', results['output_proj']['mean'])
    ])
    
    for name, time_ms in components:
        pct = (time_ms / total_components) * 100
        bar = '█' * int(pct / 2)
        print(f"  {name:20s}: {time_ms:6.3f} ms ({pct:5.1f}%) {bar}")
    
    # Find bottleneck
    bottleneck_name, bottleneck_time = max(components, key=lambda x: x[1])
    print(f"\n🔍 Bottleneck: {bottleneck_name} ({bottleneck_time:.3f} ms)")
    
    # Estimate epoch time
    print(f"\n{'='*80}")
    print("EPOCH TIME ESTIMATION")
    print(f"{'='*80}\n")
    
    avg_iter_time = train_times.mean() / 1000  # Convert to seconds
    
    print(f"Average training iteration: {avg_iter_time*1000:.1f} ms")
    print(f"\nEstimated time per epoch:")
    
    for num_batches in [100, 500, 1000, 2000, 5000]:
        epoch_time = avg_iter_time * num_batches
        print(f"  {num_batches:5d} batches: {epoch_time:6.1f}s ({epoch_time/60:5.1f} min)")
    
    # Recommendations
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}\n")
    
    if bottleneck_name == 'Mamba2':
        print("✓ Mamba2 is the bottleneck - this is EXPECTED and optimal.")
        print("  The model is compute-bound on the main operation.")
    elif bottleneck_name in ['K-MOTE', 'SM-Kernel']:
        print(f"⚠️  {bottleneck_name} is the bottleneck - preprocessing is slow.")
        print("  Possible optimizations:")
        print("    - Cache wavelet/kernel computations if inputs repeat")
        print("    - Use smaller num_mixtures for SM-Kernel")
        print("    - Profile the wavelet computation implementation")
    elif 'MLP' in bottleneck_name:
        print(f"⚠️  {bottleneck_name} is the bottleneck - MLPs are slow.")
        print("  Possible optimizations:")
        print("    - Reduce hidden dimensions in fusion_mlp_base")
        print("    - Use fewer layers in MLP")
        print("    - This suggests the MLP might be oversized")
    
    # MLP-specific analysis
    mlp_time = results['fusion_mlp']['mean']
    if use_controllable_mamba:
        mlp_time += results['modulator_head']['mean']
    mlp_pct = (mlp_time / total_components) * 100
    
    print(f"\nMLP Time Analysis:")
    print(f"  Total MLP time: {mlp_time:.3f} ms ({mlp_pct:.1f}% of forward pass)")
    
    if mlp_pct > 30:
        print(f"  ⚠️  MLPs take {mlp_pct:.1f}% of time - this is HIGH!")
        print(f"      Current MLP architecture:")
        print(f"        fusion_mlp_base: 8 → 256 → 256 → 256")
        if use_controllable_mamba:
            print(f"        modulator_head:  256 → 128 → {model.mamba2.nheads * 2}")
        print(f"      Consider:")
        print(f"        - Reducing expert_dim (256 → 128)")
        print(f"        - Removing one MLP layer")
        print(f"        - Using smaller hidden dimensions")
    else:
        print(f"  ✓ MLP time is reasonable ({mlp_pct:.1f}%)")
    
    return results

def compare_batch_sizes():
    """Test how batch size affects performance"""
    print(f"\n{'='*80}")
    print("Batch Size Impact Analysis")
    print(f"{'='*80}\n")
    
    seq_len = 512
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    
    model = KAN_MAMMOTE(
        embedding_dim=256,
        expert_dim=256,
        num_mixtures=8,
        mamba_d_state=256,
        mamba_d_conv=4,
        mamba_expand=4,
        wavelet_type='shock',
        mamba_headdim=16,
        use_controllable_mamba=True
    ).to("cuda")
    
    delta_t_sample = torch.randn(1, 100, 1).abs().to("cuda")
    model.initialize_sm_kernel(delta_t_sample)
    model.eval()
    
    print(f"Testing batch sizes: {batch_sizes}")
    print(f"Sequence length: {seq_len}\n")
    
    for batch_size in batch_sizes:
        t_abs = torch.randn(batch_size, seq_len, 1).to("cuda")
        t_rel = torch.randn(batch_size, seq_len, 1).abs().to("cuda")
        
        # Warm up
        with torch.no_grad():
            for _ in range(5):
                _ = model(t_abs, t_rel)
        
        # Profile
        times = []
        with torch.no_grad():
            for _ in range(20):
                torch.cuda.synchronize()
                start = time.time()
                _ = model(t_abs, t_rel)
                torch.cuda.synchronize()
                times.append(time.time() - start)
        
        avg_time = np.mean(times) * 1000
        throughput = batch_size / (avg_time / 1000)
        
        print(f"  Batch {batch_size:3d}: {avg_time:6.2f} ms  |  Throughput: {throughput:6.1f} samples/sec")

if __name__ == "__main__":
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Detailed profiling for both variants
    print("\n" + "="*80)
    print("DETAILED PROFILING OF KAN-MAMMOTE")
    print("="*80)
    
    # Profile ControllableMamba2
    results_controllable = detailed_profiling(use_controllable_mamba=True)
    
    # Profile Vanilla Mamba2
    results_vanilla = detailed_profiling(use_controllable_mamba=False)
    
    # Batch size analysis
    compare_batch_sizes()
    
    print(f"\n{'='*80}")
    print("Profiling Complete!")
    print(f"{'='*80}\n")