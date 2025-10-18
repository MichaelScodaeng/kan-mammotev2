# KAN-MAMMOTE Architecture Variants

## Overview

KAN-MAMMOTE now supports multiple fusion strategies and configuration options for comprehensive ablation studies and fair comparisons.

## Architecture Components

### 1. Absolute Time Encoding
- **K-MOTE (Absolute)**: Always used for absolute time encoding
- Multi-expert architecture with B-splines, Fourier, and Wavelet experts
- Configurable wavelet type: `'shock'` (default), `'morlet'`, `'mexican_hat'`, `'haar'`

### 2. Relative Time Encoding (Two Options)

#### Option A: K-MOTE (Default) ✅
```python
KAN_MAMMOTE(embedding_dim=128, use_kmote_for_relative=True)  # Default
```
- **Recommended**: Dual K-MOTE architecture
- Both absolute and relative time use the same powerful K-MOTE experts
- Better pattern detection in relative time differences
- Fair comparison between fusion strategies

#### Option B: SM-Kernel (Legacy) 🔧
```python
KAN_MAMMOTE(embedding_dim=128, use_kmote_for_relative=False)  # Legacy
```
- **For ablation studies only**: Uses spectral mixture kernel
- Gaussian mixture model for relative time patterns
- Kept for backward compatibility and ablation comparisons

### 3. Fusion Strategies (Four Options)

#### Strategy 1: Mamba Fusion (Default) 🚀
```python
KAN_MAMMOTE(embedding_dim=128, fusion_strategy='mamba')  # Default
```

**Original KAN-MAMMOTE architecture:**
1. Fuse relative features through MLP → `expert_dim`
2. Residual addition: `combined = u_abs + fusion_features`
3. Pass through Mamba2 for temporal modeling
4. Project to `embedding_dim`

**Two Mamba variants:**

##### A. ControllableMamba2 (Default) ⚙️
```python
KAN_MAMMOTE(embedding_dim=128, use_controllable_mamba=True)  # Default
```
- **FiLM modulation**: Temporal modulators (γ, β) modify Mamba's `dt` parameter
- Adaptive temporal granularity based on relative time patterns
- Best for data with varying temporal scales

##### B. Vanilla Mamba2 🏃
```python
KAN_MAMMOTE(embedding_dim=128, use_controllable_mamba=False)
```
- Standard Mamba2 without temporal modulation
- Faster inference, simpler architecture
- Good baseline for ablation studies

#### Strategy 2: Concat Fusion 🔗
```python
KAN_MAMMOTE(embedding_dim=128, fusion_strategy='concat')
```

**Lightweight fusion without Mamba:**
1. Concatenate: `[u_abs, v_rel]`
2. MLP projection: `concat_dim → embedding_dim`
3. LayerNorm + GELU + Dropout

**Advantages:**
- Faster training and inference
- Lower memory footprint
- Good for simpler temporal patterns

#### Strategy 3: Weighted Fusion ⚖️
```python
KAN_MAMMOTE(embedding_dim=128, fusion_strategy='weighted')
```

**Learnable weighted sum:**
1. Project relative to match absolute dimension
2. Learnable weights: `w_abs`, `w_rel`
3. Normalized weighted sum: `output = w_abs·u_abs + w_rel·v_rel`
4. Project to `embedding_dim`

**Advantages:**
- Automatically learns importance of each stream
- Very lightweight (only 2 learnable weights)
- Interpretable: can inspect learned weights

#### Strategy 4: Attention Fusion 🎯
```python
KAN_MAMMOTE(embedding_dim=128, fusion_strategy='attention')
```

**Cross-attention between streams:**
1. Project relative to match absolute dimension
2. Cross-attention: absolute (query) attends to relative (key/value)
3. Residual connection + LayerNorm
4. Project to `embedding_dim`

**Advantages:**
- Most expressive fusion mechanism
- Captures complex interactions between streams
- Good for data with intricate temporal dependencies

## Experiment Usage

### Available Encoder Types

```python
# Ablation study encoders
'lstm_only'              # Baseline (no time encoding)
'sm_kernel_only'         # SM-kernel only
'kmote_abs_only'         # K-MOTE absolute only
'kmote_rel_only'         # K-MOTE relative only
'dual_stream_baseline'   # K-MOTE + SM-kernel, simple fusion

# KAN-MAMMOTE Lite (without Mamba)
'kan_mammote_lite'       # Original lite version
'kan_mammote_lite_concat'
'kan_mammote_lite_weighted'
'kan_mammote_lite_attention'

# KAN-MAMMOTE Full (with Mamba)
'kan_mammote_full'       # Default: K-MOTE + ControllableMamba2 + mamba fusion
'kan_mammote_concat'     # K-MOTE + concat fusion
'kan_mammote_weighted'   # K-MOTE + weighted fusion
'kan_mammote_attention'  # K-MOTE + attention fusion
'kan_mammote_vanilla_mamba'  # K-MOTE + vanilla Mamba2 + mamba fusion
'kan_mammote_sm_kernel'  # SM-kernel (legacy) + ControllableMamba2 + mamba fusion

# Optional (if available)
'lete', 'lete_relative'
'time2vec', 'time2vec_relative'
'mercer', 'mercer_relative'
```

### Example Usage

```python
from experiments.event_based_mnist_experiment import TimeEncoderClassifier

# Default KAN-MAMMOTE (best performance)
model = TimeEncoderClassifier(
    encoder_type='kan_mammote_full',
    embedding_dim=128,
    hidden_dim=256
)

# Lightweight alternative (concat fusion)
model = TimeEncoderClassifier(
    encoder_type='kan_mammote_concat',
    embedding_dim=128,
    hidden_dim=256
)

# Ablation: vanilla Mamba2 (no FiLM modulation)
model = TimeEncoderClassifier(
    encoder_type='kan_mammote_vanilla_mamba',
    embedding_dim=128,
    hidden_dim=256
)

# Ablation: SM-kernel instead of K-MOTE for relative time
model = TimeEncoderClassifier(
    encoder_type='kan_mammote_sm_kernel',
    embedding_dim=128,
    hidden_dim=256
)
```

## Architectural Comparisons

### Fusion Strategy Comparison

| Strategy | Parameters | Speed | Expressiveness | Use Case |
|----------|-----------|-------|----------------|----------|
| **Mamba** (ControllableMamba2) | High | Medium | Very High | Complex temporal patterns, varying scales |
| **Mamba** (Vanilla) | High | Medium | High | Standard temporal patterns |
| **Concat** | Medium | Fast | Medium | Simple fusion, fast inference |
| **Weighted** | Very Low | Very Fast | Low | Interpretable, lightweight |
| **Attention** | Medium | Medium | High | Complex interactions between streams |

### Relative Time Encoding Comparison

| Method | Dimension | Learnable | Pattern Detection | Use Case |
|--------|-----------|-----------|-------------------|----------|
| **K-MOTE** (Default) | `expert_dim` | ✅ Multi-expert | Excellent | General purpose, best performance |
| **SM-Kernel** (Legacy) | `num_mixtures` | ✅ Gaussian mixtures | Good | Ablation studies, legacy comparison |

## Training Recommendations

### For Best Performance
```python
encoder_type='kan_mammote_full'
embedding_dim=128
expert_dim=64  # Must be multiple of 16
num_mixtures=64  # Ignored when use_kmote_for_relative=True
```

### For Fast Prototyping
```python
encoder_type='kan_mammote_concat'
embedding_dim=64
expert_dim=32  # Must be multiple of 16
```

### For Ablation Studies
1. **Test fusion strategies:**
   - `kan_mammote_full` (mamba + ControllableMamba2)
   - `kan_mammote_vanilla_mamba` (mamba + vanilla Mamba2)
   - `kan_mammote_concat` (concat)
   - `kan_mammote_weighted` (weighted)
   - `kan_mammote_attention` (attention)

2. **Test relative encoding:**
   - `kan_mammote_full` (K-MOTE, default)
   - `kan_mammote_sm_kernel` (SM-kernel, legacy)

3. **Test components:**
   - `kmote_abs_only` (absolute only)
   - `kmote_rel_only` (relative only)
   - `dual_stream_baseline` (simple fusion)

## Implementation Details

### Input Requirements
- `t_abs`: Absolute timestamps `(batch, seq_len, 1)` - raw pixel positions
- `t_rel`: Relative time differences `(batch, seq_len, 1)` - consecutive differences
- **No normalization required** - K-MOTE has internal LeTE-style frequency initialization

### Dimension Requirements
- `expert_dim` must be multiple of 16 (for Mamba2 hardware compatibility)
- `mamba_d_state` must be multiple of 16
- `embedding_dim` can be any value (auto-projected)

### Debug Mode
```python
model.enable_debug_mode()  # Enable detailed forward pass logging
model.disable_debug_mode()  # Disable debug logging
```

### Warm-up (for Mamba variants)
```python
model.warmup(device='cuda', num_iterations=3)  # Compile CUDA kernels once
```

## Design Philosophy

1. **K-MOTE as Default**: More powerful and consistent than SM-kernel
2. **Multiple Fusion Strategies**: Fair comparison between different approaches
3. **Backward Compatibility**: SM-kernel kept for legacy comparisons
4. **Flexibility**: Easy to switch between variants for ablation studies
5. **Clear Naming**: Encoder types clearly indicate their configuration

## Citation

If you use KAN-MAMMOTE in your research, please cite:
```bibtex
@article{kanmammote2025,
  title={KAN-MAMMOTE: Kolmogorov-Arnold Networks Meet Mamba for Temporal Event Modeling},
  author={Your Name},
  journal={arXiv preprint},
  year={2025}
}
```
