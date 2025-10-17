"""
Benchmark script to test if torch.compile() speeds up KAN-MAMMOTE.
Tests both vanilla and controllable Mamba2 variants with and without torch.compile().
"""

import torch
import time
import numpy as np
import sys

# Check PyTorch version
pytorch_version = tuple(int(x) for x in torch.__version__.split('.')[:2])
if pytorch_version < (2, 0):
    print(f"⚠️  torch.compile() requires PyTorch 2.0+. You have {torch.__version__}")
    print("Skipping torch.compile() tests...")
    HAS_COMPILE = False
else:
    print(f"✅ PyTorch {torch.__version__} supports torch.compile()")
    HAS_COMPILE = True

from models.time_encoders.kan_mammote import KAN_MAMMOTE

def benchmark_model(model, t_abs, t_rel, optimizer=None, num_warmup=5, num_runs=50, 
                   test_training=False, model_name="Model"):
    """
    Benchmark a model with proper warm-up.
    
    Args:
        model: The model to benchmark
        t_abs, t_rel: Input tensors
        optimizer: Optimizer for training benchmark (optional)
        num_warmup: Number of warm-up iterations
        num_runs: Number of benchmark iterations
        test_training: If True, test full training step (fwd+bwd+opt)
        model_name: Name for display
    """
    print(f"\n{'='*70}")
    print(f"Benchmarking: {model_name}")
    print(f"{'='*70}")
    
    model.eval()
    
    # ===== WARM UP =====
    print(f"🔥 Warming up ({num_warmup} iterations)...")
    with torch.no_grad():
        torch.cuda.synchronize()
        warm_start = time.time()
        for i in range(num_warmup):
            _ = model(t_abs, t_rel)
        torch.cuda.synchronize()
        warm_time = time.time() - warm_start
    
    print(f"   Warm-up time: {warm_time:.3f}s")
    
    # ===== INFERENCE BENCHMARK =====
    print(f"\n⏱️  Inference benchmark ({num_runs} iterations)...")
    times_inference = []
    
    with torch.no_grad():
        for i in range(num_runs):
            torch.cuda.synchronize()
            start = time.time()
            output = model(t_abs, t_rel)
            torch.cuda.synchronize()
            elapsed = time.time() - start
            times_inference.append(elapsed)
    
    times_inference = np.array(times_inference) * 1000  # Convert to ms
    
    print(f"   Mean: {times_inference.mean():.3f} ms (±{times_inference.std():.3f} ms)")
    print(f"   Min:  {times_inference.min():.3f} ms")
    print(f"   Max:  {times_inference.max():.3f} ms")
    
    results = {
        'inference_mean': times_inference.mean(),
        'inference_std': times_inference.std(),
        'inference_min': times_inference.min(),
        'inference_max': times_inference.max()
    }
    
    # ===== TRAINING BENCHMARK (if requested) =====
    if test_training and optimizer is not None:
        print(f"\n⏱️  Training benchmark ({num_runs} iterations)...")
        model.train()
        
        times_forward = []
        times_backward = []
        times_optimizer = []
        times_total = []
        
        for i in range(num_runs):
            # Forward
            torch.cuda.synchronize()
            fwd_start = time.time()
            output = model(t_abs, t_rel)
            loss = output.mean()
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
            
            total_time = fwd_time + bwd_time + opt_time
            
            times_forward.append(fwd_time)
            times_backward.append(bwd_time)
            times_optimizer.append(opt_time)
            times_total.append(total_time)
        
        times_forward = np.array(times_forward) * 1000
        times_backward = np.array(times_backward) * 1000
        times_optimizer = np.array(times_optimizer) * 1000
        times_total = np.array(times_total) * 1000
        
        print(f"   Forward:   {times_forward.mean():.3f} ms (±{times_forward.std():.3f} ms)")
        print(f"   Backward:  {times_backward.mean():.3f} ms (±{times_backward.std():.3f} ms)")
        print(f"   Optimizer: {times_optimizer.mean():.3f} ms (±{times_optimizer.std():.3f} ms)")
        print(f"   Total:     {times_total.mean():.3f} ms (±{times_total.std():.3f} ms)")
        
        results.update({
            'training_forward': times_forward.mean(),
            'training_backward': times_backward.mean(),
            'training_optimizer': times_optimizer.mean(),
            'training_total': times_total.mean()
        })
    
    return results

def run_comprehensive_benchmark(use_controllable_mamba=False, batch_size=2, seq_len=512):
    """
    Run comprehensive benchmark comparing regular model vs torch.compile().
    
    Args:
        use_controllable_mamba: Test ControllableMamba2 (True) or Vanilla (False)
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
    """
    variant_name = "ControllableMamba2" if use_controllable_mamba else "Vanilla Mamba2"
    
    print(f"\n{'#'*80}")
    print(f"# BENCHMARK: KAN-MAMMOTE with {variant_name}")
    print(f"# Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"{'#'*80}")
    
    # Configuration
    expert_dim = 256
    embedding_dim = 256
    num_mixtures = 8
    device = "cuda"
    
    # Create input tensors
    print(f"\nCreating input tensors...")
    t_abs = torch.randn(batch_size, seq_len, 1).to(device)
    t_rel = torch.randn(batch_size, seq_len, 1).abs().to(device)
    
    # ========================================
    # Test 1: Regular Model (No Compile)
    # ========================================
    print(f"\n{'='*80}")
    print("TEST 1: Regular Model (No torch.compile)")
    print(f"{'='*80}")
    
    model_regular = KAN_MAMMOTE(
        embedding_dim=embedding_dim,
        expert_dim=expert_dim,
        num_mixtures=num_mixtures,
        mamba_d_state=256,
        mamba_d_conv=4,
        mamba_expand=4,
        wavelet_type='shock',
        mamba_headdim=16,
        use_controllable_mamba=use_controllable_mamba
    ).to(device)
    
    # Initialize SM-Kernel
    delta_t_sample = torch.randn(1, 100, 1).abs().to(device)
    model_regular.initialize_sm_kernel(delta_t_sample)
    
    # Create optimizer for training tests
    optimizer_regular = torch.optim.AdamW(model_regular.parameters(), lr=1e-4)
    
    # Benchmark
    results_regular = benchmark_model(
        model=model_regular,
        t_abs=t_abs,
        t_rel=t_rel,
        optimizer=optimizer_regular,
        num_warmup=5,
        num_runs=50,
        test_training=True,
        model_name="Regular Model"
    )
    
    # ========================================
    # Test 2: torch.compile() Model
    # ========================================
    if HAS_COMPILE:
        print(f"\n{'='*80}")
        print("TEST 2: torch.compile(model, mode='reduce-overhead')")
        print(f"{'='*80}")
        
        model_compiled = KAN_MAMMOTE(
            embedding_dim=embedding_dim,
            expert_dim=expert_dim,
            num_mixtures=num_mixtures,
            mamba_d_state=256,
            mamba_d_conv=4,
            mamba_expand=4,
            wavelet_type='shock',
            mamba_headdim=16,
            use_controllable_mamba=use_controllable_mamba
        ).to(device)
        
        # Initialize SM-Kernel
        model_compiled.initialize_sm_kernel(delta_t_sample)
        
        # Apply torch.compile()
        print(f"\n🚀 Applying torch.compile(mode='reduce-overhead')...")
        try:
            model_compiled = torch.compile(model_compiled, mode="reduce-overhead")
            print(f"   ✅ Model compiled successfully!")
        except Exception as e:
            print(f"   ❌ torch.compile() failed: {e}")
            print(f"   Skipping compiled model test.")
            return results_regular, None
        
        # Create optimizer
        optimizer_compiled = torch.optim.AdamW(model_compiled.parameters(), lr=1e-4)
        
        # Benchmark
        results_compiled = benchmark_model(
            model=model_compiled,
            t_abs=t_abs,
            t_rel=t_rel,
            optimizer=optimizer_compiled,
            num_warmup=5,
            num_runs=50,
            test_training=True,
            model_name="Compiled Model (torch.compile)"
        )
    else:
        results_compiled = None
    
    # ========================================
    # Test 3: torch.compile() with different mode
    # ========================================
    if HAS_COMPILE:
        print(f"\n{'='*80}")
        print("TEST 3: torch.compile(model, mode='max-autotune')")
        print(f"{'='*80}")
        
        model_maxautotune = KAN_MAMMOTE(
            embedding_dim=embedding_dim,
            expert_dim=expert_dim,
            num_mixtures=num_mixtures,
            mamba_d_state=256,
            mamba_d_conv=4,
            mamba_expand=4,
            wavelet_type='shock',
            mamba_headdim=16,
            use_controllable_mamba=use_controllable_mamba
        ).to(device)
        
        # Initialize SM-Kernel
        model_maxautotune.initialize_sm_kernel(delta_t_sample)
        
        # Apply torch.compile()
        print(f"\n🚀 Applying torch.compile(mode='max-autotune')...")
        try:
            model_maxautotune = torch.compile(model_maxautotune, mode="max-autotune")
            print(f"   ✅ Model compiled successfully!")
        except Exception as e:
            print(f"   ❌ torch.compile() failed: {e}")
            print(f"   Skipping max-autotune test.")
            results_maxautotune = None
        else:
            # Create optimizer
            optimizer_maxautotune = torch.optim.AdamW(model_maxautotune.parameters(), lr=1e-4)
            
            # Benchmark (inference only - max-autotune takes long to compile)
            results_maxautotune = benchmark_model(
                model=model_maxautotune,
                t_abs=t_abs,
                t_rel=t_rel,
                optimizer=optimizer_maxautotune,
                num_warmup=5,
                num_runs=30,  # Fewer runs since it's slower to compile
                test_training=False,  # Skip training for max-autotune
                model_name="Compiled Model (max-autotune)"
            )
    else:
        results_maxautotune = None
    
    return results_regular, results_compiled, results_maxautotune

def print_comparison(results_regular, results_compiled, results_maxautotune, variant_name):
    """Print comparison of results"""
    
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY - {variant_name}")
    print(f"{'='*80}")
    
    print(f"\n📊 Inference Performance:")
    print(f"  Regular Model:              {results_regular['inference_mean']:.3f} ms")
    
    if results_compiled:
        print(f"  torch.compile (reduce-overhead): {results_compiled['inference_mean']:.3f} ms")
        
        speedup = results_regular['inference_mean'] / results_compiled['inference_mean']
        diff_pct = ((results_compiled['inference_mean'] - results_regular['inference_mean']) / 
                    results_regular['inference_mean']) * 100
        
        if speedup > 1.1:
            print(f"  🚀 Speedup: {speedup:.2f}x faster ({-diff_pct:.1f}% reduction)")
        elif speedup < 0.9:
            print(f"  🐌 Slowdown: {1/speedup:.2f}x slower ({diff_pct:+.1f}% increase)")
        else:
            print(f"  ≈ Similar performance ({diff_pct:+.1f}%)")
    
    if results_maxautotune:
        print(f"  torch.compile (max-autotune):    {results_maxautotune['inference_mean']:.3f} ms")
        
        speedup_max = results_regular['inference_mean'] / results_maxautotune['inference_mean']
        if speedup_max > 1.1:
            print(f"  🚀 Speedup: {speedup_max:.2f}x faster")
    
    # Training comparison
    if 'training_total' in results_regular and results_compiled and 'training_total' in results_compiled:
        print(f"\n🎯 Training Performance (Forward + Backward + Optimizer):")
        print(f"  Regular Model:              {results_regular['training_total']:.3f} ms")
        print(f"  torch.compile:              {results_compiled['training_total']:.3f} ms")
        
        speedup_train = results_regular['training_total'] / results_compiled['training_total']
        diff_pct_train = ((results_compiled['training_total'] - results_regular['training_total']) / 
                         results_regular['training_total']) * 100
        
        if speedup_train > 1.1:
            print(f"  🚀 Speedup: {speedup_train:.2f}x faster ({-diff_pct_train:.1f}% reduction)")
            
            # Estimate epoch time savings
            batches_per_epoch = 1000  # Typical
            regular_epoch = (results_regular['training_total'] / 1000) * batches_per_epoch
            compiled_epoch = (results_compiled['training_total'] / 1000) * batches_per_epoch
            time_saved = regular_epoch - compiled_epoch
            
            print(f"\n  💾 Estimated savings per epoch (1000 batches):")
            print(f"     Regular:  {regular_epoch:.1f}s ({regular_epoch/60:.2f} min)")
            print(f"     Compiled: {compiled_epoch:.1f}s ({compiled_epoch/60:.2f} min)")
            print(f"     Saved:    {time_saved:.1f}s ({time_saved/60:.2f} min) per epoch!")
        elif speedup_train < 0.9:
            print(f"  🐌 Slowdown: {1/speedup_train:.2f}x slower ({diff_pct_train:+.1f}%)")
            print(f"     ⚠️  torch.compile() makes training SLOWER - don't use it!")
        else:
            print(f"  ≈ Similar performance ({diff_pct_train:+.1f}%)")

def main():
    """Run all benchmarks"""
    print("="*80)
    print("KAN-MAMMOTE torch.compile() Benchmark")
    print("="*80)
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    # Test configurations
    configs = [
        {"use_controllable_mamba": False, "batch_size": 2, "seq_len": 512},
        {"use_controllable_mamba": True, "batch_size": 2, "seq_len": 512},
    ]
    
    all_results = []
    
    for config in configs:
        variant_name = "ControllableMamba2" if config["use_controllable_mamba"] else "Vanilla Mamba2"
        
        results = run_comprehensive_benchmark(**config)
        all_results.append((variant_name, results))
    
    # Print all comparisons
    print(f"\n\n{'#'*80}")
    print("# FINAL COMPARISON")
    print(f"{'#'*80}")
    
    for variant_name, (results_regular, results_compiled, results_maxautotune) in all_results:
        print_comparison(results_regular, results_compiled, results_maxautotune, variant_name)
    
    # Final recommendation
    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}\n")
    
    if not HAS_COMPILE:
        print("❌ torch.compile() not available (need PyTorch 2.0+)")
        print("   Upgrade PyTorch to use this optimization")
    else:
        print("Based on the benchmarks above:")
        print("")
        print("✅ If torch.compile() shows >10% speedup:")
        print("   Add this to your training script:")
        print("   ```python")
        print("   model = torch.compile(model, mode='reduce-overhead')")
        print("   ```")
        print("")
        print("❌ If torch.compile() shows <5% speedup or slowdown:")
        print("   Don't use it - the compilation overhead isn't worth it")
        print("")
        print("⚠️  Note: torch.compile() won't fix the backward pass issue")
        print("   in ControllableMamba2. For that, use vanilla Mamba2!")

if __name__ == "__main__":
    main()
