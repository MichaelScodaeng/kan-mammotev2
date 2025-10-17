# KAN-MAMMOTE Warm-up Implementation Guide

## 🔥 What is Warm-up?

Warm-up is the process of **pre-compiling CUDA kernels** used by Mamba2 before actual training begins. Without warm-up, the **first forward pass** can take **5-40 seconds** due to JIT (Just-In-Time) compilation of Triton kernels.

## 📊 Performance Impact

Based on benchmarks:

- **First run (cold)**: 5-40 seconds (compilation + execution)
- **After warm-up**: ~1-5 milliseconds (cached kernels)
- **Speedup**: **1000-40,000x** for subsequent runs! 🚀

## ✅ Implementation

### 1. Added `warmup()` Method to KAN_MAMMOTE

**Location**: `models/time_encoders/kan_mammote.py`

```python
def warmup(self, device='cuda', num_iterations=3):
    """
    Warm up the model by running a few forward passes.
    This compiles the CUDA kernels (especially for Mamba2) and caches them 
    for the entire training session.
    
    Args:
        device: Device to run warm-up on (default: 'cuda')
        num_iterations: Number of warm-up iterations (default: 3)
    """
    # Creates dummy inputs and runs forward passes
    # Tracks compilation time on first iteration
    # Verifies caching on subsequent iterations
```

### 2. Integrated into Training Script

**Location**: `experiments/train_link_prediction.py`

The warm-up is automatically called after model initialization but **before the training loop**:

```python
# After model creation
model = convert_to_gpu(model, device=args.device)

# Warm up KAN-MAMMOTE (if applicable)
if args.model_name == 'TGAT' and hasattr(time_encoder, 'encoder'):
    actual_encoder = time_encoder.encoder
    if isinstance(actual_encoder, (KAN_MAMMOTE, KAN_MAMMOTE_Lite)):
        logger.info(f"Warming up {actual_encoder.__class__.__name__}...")
        actual_encoder.warmup(device=args.device, num_iterations=3)

# Then start training loop
for epoch in range(args.num_epochs):
    # Training happens here - no warm-up needed!
    ...
```

## 🎯 Key Points

### ✅ Do This

1. **Warm up ONCE per Python process** (at the start of training)
2. **After model initialization** but **before training loop**
3. Use **3-5 iterations** for thorough compilation

### ❌ Don't Do This

1. ❌ Don't warm up **every epoch** (wastes time)
2. ❌ Don't warm up **every batch** (unnecessary)
3. ❌ Don't skip warm-up for benchmarking (misleading results)

## 🧪 Testing

Run the test scripts to verify warm-up works:

```bash
# Test 1: Simple Mamba2 warm-up demo
python test_warmup_effect.py

# Test 2: Full KAN-MAMMOTE warm-up test
python test_kan_mammote_warmup.py

# Test 3: Benchmark vanilla vs controllable Mamba2
python benchmark_mamba_variants.py
```

## 📈 Expected Output

When warm-up runs successfully, you'll see:

```
============================================================
🔥 Warming up KAN-MAMMOTE (compiling CUDA kernels)...
============================================================
  Iteration 1/3: 5.234s (compilation)
  Iteration 2/3: 0.003s (cached)
  Iteration 3/3: 0.002s (cached)
✅ Warm-up complete! CUDA kernels cached for this session.
============================================================
```

## 🐛 Troubleshooting

### Issue: Warm-up doesn't speed things up

**Cause**: You're creating a new Python process each time
**Solution**: Warm-up only works within the same process. Each `python script.py` run is a new process.

### Issue: Training is still slow (7 min/epoch)

**Cause**: Warm-up only saves 5-40 seconds **at the start**. If epochs are still slow, the bottleneck is elsewhere:
- Large dataset (many batches)
- Data loading bottleneck
- Model complexity

**Solution**: Run the diagnostic script to identify the real bottleneck:
```bash
python diagnose_training_speed.py
```

## 🔬 Why This Matters

### The Compilation Process

Mamba2 uses **Triton kernels** for efficient GPU computation:

1. **First call**: Triton compiles kernels for your specific:
   - GPU architecture
   - Tensor shapes
   - Data types
   - Autotuning configurations

2. **Subsequent calls**: Uses cached, optimized kernels

### The Cost Without Warm-up

Every time you run a new script:
- ❌ Wait 5-40 seconds for compilation
- ❌ First epoch appears artificially slow
- ❌ Benchmarks are inaccurate

### The Benefit With Warm-up

- ✅ Compilation happens once, upfront
- ✅ Consistent timing from first epoch
- ✅ Accurate performance metrics
- ✅ Faster development iteration

## 📚 References

- Original Mamba2 issue: https://github.com/state-spaces/mamba/issues/324
- Triton documentation: https://triton-lang.org/
- Test files in this repo:
  - `test_warmup_effect.py`
  - `test_kan_mammote_warmup.py`
  - `benchmark_mamba_variants.py`

## 🎓 Best Practices

### For Development

```python
# Quick iteration during development
model = KAN_MAMMOTE(...)
model.initialize_sm_kernel(delta_t_sample)
model.warmup(device='cuda', num_iterations=3)  # Do this once!

# Now experiment freely
for trial in range(100):
    output = model(t_abs, t_rel)  # Fast!
```

### For Training

```python
# In your training script
model = create_model(...)

# Warm up before training loop
model.warmup(device='cuda', num_iterations=3)

# Train normally
for epoch in range(num_epochs):
    for batch in dataloader:
        # All forward passes are fast
        output = model(batch)
```

### For Benchmarking

```python
# ALWAYS warm up before benchmarking
model = create_model(...)
model.warmup(num_iterations=5)  # Extra iterations for stability

# Now measure true performance
torch.cuda.synchronize()
start = time.time()
output = model(input)
torch.cuda.synchronize()
true_inference_time = time.time() - start
```

## ✨ Summary

**One line to save 5-40 seconds**: 
```python
kan_mammote.warmup(device='cuda', num_iterations=3)
```

That's it! Add this after model initialization, and enjoy fast, consistent performance throughout your training! 🚀
