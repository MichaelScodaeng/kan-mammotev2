# Mamba2 Performance Benchmarking Suite

This suite tests the performance differences between vanilla Mamba2 and ControllableMamba2 to identify if the FiLM modulation introduces overhead.

## Files Created

1. **`test_warmup_effect.py`** - Demonstrates the warm-up effect
   - Shows compilation overhead on first run
   - Compares first run vs subsequent runs
   - Tests both vanilla and controllable Mamba2

2. **`benchmark_mamba_variants.py`** - Comprehensive performance comparison
   - Benchmarks vanilla Mamba2 (baseline)
   - Benchmarks ControllableMamba2 (with FiLM modulation)
   - Provides statistical analysis (mean, std, min, max)
   - Calculates overhead percentage

3. **`profile_controllable_mamba.py`** - Detailed profiling
   - Profiles vanilla Mamba2
   - Profiles ControllableMamba2
   - Profiles FiLM modulation overhead only
   - Generates Chrome trace files for visualization

4. **`run_mamba_benchmarks.sh`** - Run all tests at once

## Quick Start

### Run All Tests
```bash
./run_mamba_benchmarks.sh
```

### Run Individual Tests

**Test warm-up effect:**
```bash
python test_warmup_effect.py
```

**Run benchmark comparison:**
```bash
python benchmark_mamba_variants.py
```

**Run detailed profiling:**
```bash
python profile_controllable_mamba.py
```

## Understanding the Results

### Expected Findings

1. **Warm-up Effect**
   - First run: ~10-45 seconds (CUDA kernel compilation)
   - Subsequent runs: ~1-10 milliseconds (cached kernels)
   - **Speedup: 1000x-10000x after warm-up**

2. **Performance Overhead**
   - If ControllableMamba2 is <5% slower: **negligible overhead**
   - If 5-15% slower: **acceptable overhead** for added functionality
   - If >15% slower: **significant bottleneck** needs optimization

3. **Bottleneck Analysis**
   - Check profiling output for time-consuming operations
   - Look for `torch.cat`, `.contiguous()`, element-wise ops
   - Compare CUDA kernel execution times

### Interpreting Profiling Traces

The profiling scripts generate `.json` files that can be viewed in Chrome:

1. Open Chrome browser
2. Navigate to: `chrome://tracing`
3. Click "Load" and select a trace file
4. Use WASD keys to navigate:
   - W/S: Zoom in/out
   - A/D: Pan left/right

**What to look for:**
- Long-running operations (wide bars)
- Repeated operations (patterns)
- Memory operations (copies, allocations)
- Kernel launch overhead

## Common Issues and Solutions

### Issue 1: Slow Performance Without Warm-up

**Problem:** Model takes 30+ seconds on first run

**Solution:** Always warm up before benchmarking:
```python
# Warm-up
for _ in range(3):
    with torch.no_grad():
        _ = model(input)

# Now time actual performance
torch.cuda.synchronize()
t1 = time.time()
output = model(input)
torch.cuda.synchronize()
print(f"Time: {time.time() - t1:.3f}s")
```

### Issue 2: ControllableMamba2 Much Slower

**Possible causes:**
1. `torch.cat()` + `.contiguous()` overhead
2. FiLM modulation computation
3. Tensor reconstruction instead of in-place modification

**Solution:** Check profiling output to identify bottleneck

### Issue 3: Inconsistent Timings

**Problem:** Large variance in execution times

**Possible causes:**
1. GPU thermal throttling
2. Other processes using GPU
3. Insufficient warm-up

**Solution:**
- Check GPU temperature: `nvidia-smi`
- Kill other GPU processes
- Increase warm-up iterations

## Integration with KAN-MAMMOTE

To add warm-up to your training script:

```python
# In your training script, before the training loop:
print("Warming up KAN-MAMMOTE...")
dummy_t_abs = torch.randn(2, 10, 1).to(device)
dummy_t_rel = torch.randn(2, 10, 1).to(device)

for _ in range(3):
    with torch.no_grad():
        _ = kan_mammote(dummy_t_abs, dummy_t_rel)

print("Warm-up complete! Starting training...")
torch.cuda.synchronize()
```

## Performance Optimization Tips

1. **Always warm up** before benchmarking or timing
2. Use `torch.cuda.synchronize()` before/after timing
3. Run multiple iterations and take the mean
4. Profile to identify actual bottlenecks
5. Consider batch size effects on performance
6. Monitor GPU memory usage

## Expected Results

Based on the analysis, here's what you should expect:

### Scenario 1: Warm-up is the Issue
- **Before warm-up:** 30-45 seconds per forward pass
- **After warm-up:** 1-5 milliseconds per forward pass
- **Conclusion:** Your model is fine, just needs warm-up

### Scenario 2: ControllableMamba2 Overhead
- **Vanilla Mamba2:** 1-3 milliseconds
- **ControllableMamba2:** 1.5-4 milliseconds
- **Overhead:** 20-50% (acceptable for added control)

### Scenario 3: Other Bottlenecks
- **K-MOTE:** Check wavelet transform time
- **SM-Kernel:** Check kernel computations
- **Fusion MLP:** Usually negligible
- **Memory alignment:** Already handled in your code

## Next Steps

1. Run the benchmark suite
2. Analyze the results
3. If overhead is high, check profiling traces
4. Optimize identified bottlenecks
5. Add warm-up to training scripts

## Contact

For questions or issues with the benchmarking suite, check:
- Mamba2 documentation: https://github.com/state-spaces/mamba
- PyTorch profiler: https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html
