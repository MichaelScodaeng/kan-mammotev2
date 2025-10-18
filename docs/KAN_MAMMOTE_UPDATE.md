# KAN-MAMMOTE Update: Flexible Fusion Strategies

## Summary of Changes

We've updated `KAN_MAMMOTE` to support multiple fusion strategies and configuration options, making it more flexible for ablation studies and fair comparisons.

## Key Changes

### 1. Default Configuration Updated ✅

**New Defaults:**
- ✅ **K-MOTE for relative time** (was: SM-Kernel)
- ✅ **ControllableMamba2** with FiLM modulation (unchanged)
- ✅ **Mamba fusion** strategy (unchanged)

**Rationale:**
- K-MOTE is more powerful and consistent than SM-Kernel
- SM-Kernel is now "legacy" and kept only for ablation studies

### 2. New Fusion Strategies Added 🎨

KAN-MAMMOTE now supports **4 fusion strategies**:

| Strategy | Architecture | Use Case |
|----------|-------------|----------|
| `'mamba'` (default) | Mamba2 temporal modeling | Best performance, complex patterns |
| `'concat'` | Concatenate + MLP | Lightweight, fast inference |
| `'weighted'` | Learnable weighted sum | Interpretable, minimal parameters |
| `'attention'` | Cross-attention | Expressive, complex interactions |

### 3. New Encoder Types in Experiment Script 🧪

```python
# KAN-MAMMOTE Full variants (all use K-MOTE for relative time by default)
'kan_mammote_full'           # Default: K-MOTE + ControllableMamba2 + mamba fusion
'kan_mammote_concat'         # K-MOTE + concat fusion
'kan_mammote_weighted'       # K-MOTE + weighted fusion
'kan_mammote_attention'      # K-MOTE + attention fusion
'kan_mammote_vanilla_mamba'  # K-MOTE + vanilla Mamba2 (no FiLM)
'kan_mammote_sm_kernel'      # SM-kernel (legacy) + ControllableMamba2
```

## Usage Examples

### Basic Usage (Default Configuration)

```python
from models.time_encoders.kan_mammote import KAN_MAMMOTE

# Default: K-MOTE + ControllableMamba2 + mamba fusion
model = KAN_MAMMOTE(
    embedding_dim=128,
    expert_dim=64
)
```

### Use Different Fusion Strategy

```python
# Lightweight concat fusion
model = KAN_MAMMOTE(
    embedding_dim=128,
    expert_dim=64,
    fusion_strategy='concat'  # or 'weighted', 'attention'
)
```

### Use Vanilla Mamba2 (no FiLM modulation)

```python
model = KAN_MAMMOTE(
    embedding_dim=128,
    expert_dim=64,
    use_controllable_mamba=False  # Use vanilla Mamba2
)
```

### Use SM-Kernel for relative time (legacy)

```python
model = KAN_MAMMOTE(
    embedding_dim=128,
    expert_dim=64,
    use_kmote_for_relative=False,  # Use SM-kernel instead
    num_mixtures=64
)
```

## Running Experiments

### Test All Variants

```bash
# Quick test to verify all variants work
python tests/test_kan_mammote_variants.py
```

### Run Comparison on MNIST

```bash
# Compare all fusion strategies
python experiments/run_kan_mammote_comparison.py \
    --epochs 10 \
    --batch_size 64 \
    --embedding_dim 128 \
    --expert_dim 64 \
    --hidden_dim 256
```

### Run Single Variant

```bash
# Run specific encoder type
python experiments/event_based_mnist_experiment.py \
    --encoder kan_mammote_concat \
    --epochs 10 \
    --batch_size 64
```

## Architectural Improvements

### 1. Fair Comparison Between Fusion Strategies

All fusion strategies now receive the **same input dimensions**:
- Absolute time: `K-MOTE` → `expert_dim`
- Relative time: `K-MOTE` → `expert_dim` (when `use_kmote_for_relative=True`)

This ensures fair comparison since all strategies process equal-dimensional inputs.

### 2. Consistent Relative Time Encoding

**Before:**
- Inconsistent: SM-kernel outputs `num_mixtures` (e.g., 12)
- K-MOTE outputs `expert_dim` (e.g., 64)

**After (Default):**
- Consistent: K-MOTE for both absolute and relative
- Both output `expert_dim`
- Fair comparison between fusion strategies

### 3. Clear Separation of Concerns

```
KAN-MAMMOTE
├── Absolute Time Encoding: K-MOTE (always)
├── Relative Time Encoding: K-MOTE (default) or SM-kernel (legacy)
├── Fusion Strategy: mamba (default), concat, weighted, or attention
└── Output Projection: expert_dim → embedding_dim
```

## Ablation Study Guide

### Test Relative Time Encoding

```python
# Compare K-MOTE vs SM-kernel for relative time
encoders = [
    'kan_mammote_full',      # K-MOTE (default)
    'kan_mammote_sm_kernel'  # SM-kernel (legacy)
]
```

### Test Fusion Strategies

```python
# Compare all fusion strategies (all use K-MOTE)
encoders = [
    'kan_mammote_full',       # Mamba + ControllableMamba2
    'kan_mammote_vanilla_mamba',  # Mamba + Vanilla Mamba2
    'kan_mammote_concat',     # Concat fusion
    'kan_mammote_weighted',   # Weighted fusion
    'kan_mammote_attention'   # Attention fusion
]
```

### Test Component Isolation

```python
# Test individual components
encoders = [
    'kmote_abs_only',        # Only absolute time
    'kmote_rel_only',        # Only relative time
    'dual_stream_baseline',  # Simple fusion
    'kan_mammote_full'       # Full architecture
]
```

## Performance Expectations

Based on architectural complexity:

1. **Best Performance:** `kan_mammote_full` (K-MOTE + ControllableMamba2)
2. **High Performance:** `kan_mammote_vanilla_mamba`, `kan_mammote_attention`
3. **Medium-High:** `kan_mammote_concat`, `kan_mammote_sm_kernel`
4. **Medium:** `kan_mammote_weighted`

Actual results depend on dataset characteristics!

## Fixed Issues

### Issue: K-MOTE Training Collapse ✅

**Problem:** K-MOTE encoders were failing with vanishing gradients and accuracy dropping to random chance (~11% for MNIST).

**Root Cause:** We were normalizing inputs, but K-MOTE has internal LeTE-style frequency initialization that expects **raw time values**.

**Solution:**
- ✅ Removed normalization from ablation encoders
- ✅ K-MOTE's scale-invariance comes from learnable frequencies (w·t + b), not input normalization
- ✅ Updated `ablation_encoders.py` to use raw values

### Issue: Inconsistent Relative Time Encoding ✅

**Problem:** SM-kernel and K-MOTE had different output dimensions, making fusion strategy comparison unfair.

**Solution:**
- ✅ Made K-MOTE the default for relative time
- ✅ All fusion strategies now receive consistent input dimensions
- ✅ SM-kernel kept as legacy option for ablation studies

## Documentation

- **Full guide:** `docs/KAN_MAMMOTE_VARIANTS.md`
- **Quick reference:** This file
- **Code examples:** `tests/test_kan_mammote_variants.py`
- **Experiment script:** `experiments/run_kan_mammote_comparison.py`

## Backward Compatibility

All existing code continues to work:

```python
# Old code (still works, uses new defaults)
model = KAN_MAMMOTE(embedding_dim=128, expert_dim=64)

# This now means:
# - use_kmote_for_relative=True (NEW DEFAULT)
# - fusion_strategy='mamba' (unchanged)
# - use_controllable_mamba=True (unchanged)
```

To get the old SM-kernel behavior:

```python
model = KAN_MAMMOTE(
    embedding_dim=128,
    expert_dim=64,
    use_kmote_for_relative=False,  # Explicitly use SM-kernel
    num_mixtures=64
)
```

## Next Steps

1. ✅ Run `test_kan_mammote_variants.py` to verify all variants work
2. ✅ Run `run_kan_mammote_comparison.py` to compare on MNIST
3. ✅ Analyze results to determine best fusion strategy for your data
4. ✅ Use ablation studies to understand component contributions

## Questions?

See the full documentation in `docs/KAN_MAMMOTE_VARIANTS.md` or check the code comments in:
- `models/time_encoders/kan_mammote.py`
- `models/time_encoders/ablation_encoders.py`
- `experiments/event_based_mnist_experiment.py`
