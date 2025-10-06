#!/bin/bash
# Run all Mamba2 benchmark and profiling scripts

echo "=========================================="
echo "Mamba2 Performance Testing Suite"
echo "=========================================="
echo ""

# Check if CUDA is available
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'Using GPU: {torch.cuda.get_device_name(0)}')"

echo ""
echo "This script will run 3 tests:"
echo "  1. Warm-up effect demonstration"
echo "  2. Benchmark comparison (vanilla vs controllable)"
echo "  3. Detailed profiling"
echo ""
read -p "Press Enter to continue..."

# Test 1: Warm-up Effect
echo ""
echo "=========================================="
echo "Test 1: Warm-up Effect Demonstration"
echo "=========================================="
python test_warmup_effect.py

# Test 2: Benchmark
echo ""
echo ""
echo "=========================================="
echo "Test 2: Performance Benchmark"
echo "=========================================="
python benchmark_mamba_variants.py

# Test 3: Profiling
echo ""
echo ""
echo "=========================================="
echo "Test 3: Detailed Profiling"
echo "=========================================="
echo "This may take a few minutes..."
python profile_controllable_mamba.py

echo ""
echo "=========================================="
echo "All tests complete!"
echo "=========================================="
echo ""
echo "Generated files:"
echo "  - vanilla_mamba_trace.json"
echo "  - controllable_mamba_trace.json"
echo "  - film_overhead_trace.json"
echo ""
echo "To view traces:"
echo "  1. Open Chrome browser"
echo "  2. Navigate to: chrome://tracing"
echo "  3. Load the trace files"
