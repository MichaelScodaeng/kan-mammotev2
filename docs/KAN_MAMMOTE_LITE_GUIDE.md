# 🚀 KAN-MAMMOTE Lite User Guide

## Overview

**KAN-MAMMOTE Lite** is a lightweight, stateless version of KAN-MAMMOTE designed for attention-based temporal graph models (TGAT, JODIE, TGN) where temporal encodings are computed independently per timestamp.

---

## 🎯 **When to Use Which Version?**

| Model Architecture | Recommended Version | Reasoning |
|-------------------|---------------------|-----------|
| **TGAT** | KAN-MAMMOTE Lite ✅ | Stateless attention mechanism |
| **JODIE** | KAN-MAMMOTE Lite ✅ | Per-event encoding |
| **TGN** | KAN-MAMMOTE Lite ✅ | Memory-based but stateless encoding |
| **CAWN** | KAN-MAMMOTE Lite ✅ | Walks aggregate independently |
| **TCL** | KAN-MAMMOTE Lite ✅ | Contrastive learning per timestamp |
| **DyGFormer** | KAN-MAMMOTE (Full) ✅ | Processes event sequences |
| **GraphMixer** | KAN-MAMMOTE (Full) ✅ | MLP-Mixer on temporal sequences |
| **DyGMamba** | KAN-MAMMOTE (Full) ✅ | Already uses Mamba |

---

## 📊 **Architecture Comparison**

### **KAN-MAMMOTE (Full)**
```
Input: (t_abs, t_rel) sequences
  ↓
K-MOTE (wavelets + B-splines)
  ↓
SM-Kernel (spectral mixture)
  ↓
Mamba2 (sequence modeling) ← KEY DIFFERENCE
  ↓
Output: Context-aware embeddings

Parameters: ~50,000
Use case: Sequence-based models
```

### **KAN-MAMMOTE Lite**
```
Input: (t_abs, t_rel) individual timestamps
  ↓
K-MOTE (wavelets + B-splines)
  ↓
SM-Kernel (spectral mixture)
  ↓
Simple Fusion MLP ← KEY DIFFERENCE
  ↓
Output: Stateless embeddings

Parameters: ~10,000
Use case: Attention-based models
```

---

## 🛠️ **Usage Examples**

### **1. Basic Usage in Python**

```python
from models.time_encoders import KAN_MAMMOTE_Lite

# Create encoder
encoder = KAN_MAMMOTE_Lite(
    embedding_dim=128,
    num_mixtures=12,
    wavelet_type='shock',
    use_dual_stream=True
)

# Initialize SM-Kernel from data (optional but recommended)
encoder.initialize_sm_kernel(delta_t_samples)

# Encode timestamps
t_abs = torch.tensor([[0.0], [1.5], [3.2]])  # Absolute times
t_rel = torch.tensor([[0.5], [1.0], [0.8]])  # Relative times (delta_t)

embeddings = encoder(t_abs=t_abs, t_rel=t_rel)
# Output shape: (3, 128)
```

### **2. Using in Experiments**

```bash
# Test KAN-MAMMOTE Lite on TGAT + Wikipedia
python experiment_test.py \
  --single_encoder kan_mammote_lite \
  --models TGAT \
  --datasets wikipedia

# Compare all encoders including Lite
python experiment_test.py \
  --time_encoders kan_mammote kan_mammote_lite lete original \
  --models TGAT \
  --datasets wikipedia reddit
```

### **3. Direct Training Script**

```bash
# Train TGAT with KAN-MAMMOTE Lite
python experiments/train_link_prediction.py \
  --dataset_name wikipedia \
  --model_name TGAT \
  --time_encoder kan_mammote_lite \
  --num_mixtures 12 \
  --wavelet_type shock \
  --num_epochs 100
```

### **4. Single-Stream Mode (Only Relative Time)**

```python
# For models that only use delta_t (most common)
encoder = KAN_MAMMOTE_Lite(
    embedding_dim=128,
    num_mixtures=12,
    wavelet_type='shock',
    use_dual_stream=False  # Only use t_rel
)

# Only need t_rel
embeddings = encoder(t_rel=t_rel)
```

---

## ⚙️ **Configuration Options**

### **Parameters**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_dim` | int | 128 | Output dimension |
| `num_mixtures` | int | 12 | Number of Gaussian mixtures in SM-Kernel |
| `wavelet_type` | str | 'shock' | Wavelet type ('shock', 'haar', 'db4', etc.) |
| `use_dual_stream` | bool | True | Use both t_abs and t_rel (False = only t_rel) |

### **Recommended Settings**

#### **For TGAT/JODIE/TGN:**
```python
embedding_dim = 128      # Match node feature dimension
num_mixtures = 12        # Good balance
wavelet_type = 'shock'   # Detect abrupt changes
use_dual_stream = False  # Only need delta_t
```

#### **For Complex Datasets (Reddit, Wikipedia):**
```python
embedding_dim = 172      # Higher capacity
num_mixtures = 16        # More expressive
wavelet_type = 'shock'
use_dual_stream = True   # Use full information
```

#### **For Small/Fast Datasets (MOOC, LastFM):**
```python
embedding_dim = 100
num_mixtures = 8
wavelet_type = 'haar'    # Simpler, faster
use_dual_stream = False
```

---

## 📈 **Expected Performance**

### **Wikipedia Link Prediction (TGAT)**

| Encoder | Val AP | Test AP | Parameters | Notes |
|---------|--------|---------|------------|-------|
| Original | 97.22% | 96.36% | 0 | Fixed frequencies |
| LeTE | 98.04% | 97.59% | ~5K | Learnable Fourier |
| KAN-MAMMOTE (Full) | 95.64% | 94.63% | ~50K | ❌ Overparameterized for TGAT |
| **KAN-MAMMOTE Lite** | **97.5%** | **96.8%** | ~10K | ✅ **Expected improvement!** |

**Prediction:** KAN-MAMMOTE Lite should match or exceed LeTE performance on TGAT!

---

## 🔧 **Implementation Details**

### **Key Differences from Full Version**

1. **No Mamba2**: Removed sequence modeling layer
2. **Simpler Fusion**: MLP instead of temporal modulation
3. **Lower Memory**: 5x fewer parameters
4. **Faster**: No sequential dependencies
5. **Stateless**: Each timestamp encoded independently

### **What's Preserved**

1. ✅ **K-MOTE**: Wavelet decomposition + B-splines
2. ✅ **SM-Kernel**: Spectral mixture for delta_t
3. ✅ **Dual-stream**: Optional t_abs + t_rel encoding
4. ✅ **Initialization**: Data-driven SM-Kernel setup

---

## 🐛 **Troubleshooting**

### **Issue: "requires t_rel" Error**

```python
# Problem:
embeddings = encoder(t_abs=t_abs)  # ❌ Missing t_rel

# Solution:
embeddings = encoder(t_abs=t_abs, t_rel=t_rel)  # ✅
```

### **Issue: SM-Kernel Not Initialized**

```bash
Warning: SM-Kernel not initialized from data

# This happens if train_data is not passed to factory
# It still works (random init) but less optimal

# Solution: Ensure factory gets train_data
time_encoder = create_time_encoder(
    'kan_mammote_lite',
    time_dim=128,
    train_data=train_data,              # ← Pass this
    train_neighbor_sampler=sampler      # ← And this
)
```

### **Issue: Dimension Mismatch**

```python
# Problem: fusion_input_dim mismatch
# This happens if embedding_dim is not divisible by 2

# Solution: Use even embedding dimensions
embedding_dim = 128  # ✅ Even
embedding_dim = 127  # ❌ Odd (causes issues in dual-stream)
```

---

## 📊 **Performance Benchmarking**

### **Test Command**

```bash
# Benchmark all encoders on TGAT
python experiment_test.py \
  --models TGAT \
  --datasets wikipedia \
  --time_encoders original lete kan_mammote kan_mammote_lite \
  --num_runs 3

# Check results
cat experiment_report_*_$(date +%Y%m%d)*.txt
```

### **Expected Results**

```
Model: TGAT, Dataset: wikipedia
----------------------------------------
Encoder              Val AP    Test AP   Params
----------------------------------------
LeTE                 98.04%    97.59%    ~5K
KAN-MAMMOTE Lite     97.50%    96.80%    ~10K   ← New!
Original             97.22%    96.36%    0
Mercer               96.66%    95.63%    ~3K
KAN-MAMMOTE (Full)   95.64%    94.63%    ~50K
```

---

## 🎓 **Tips for Best Performance**

### **1. Always Initialize SM-Kernel**
```python
# Sample training data to get delta_t statistics
encoder.initialize_sm_kernel(delta_t_samples)
```

### **2. Use Appropriate Dual-Stream Setting**
```python
# TGAT only uses delta_t internally → use_dual_stream=False
# DyGFormer uses both → use_dual_stream=True
```

### **3. Tune num_mixtures**
```python
# Small datasets (MOOC, LastFM): num_mixtures=8
# Medium datasets (Wikipedia): num_mixtures=12
# Large datasets (Reddit): num_mixtures=16
```

### **4. Choose Right Wavelet**
```python
# 'shock': Best for abrupt changes (Wikipedia edits, Reddit posts)
# 'haar': Simpler, faster, good for smooth data
# 'db4': Daubechies, good balance
```

---

## 📚 **Further Reading**

- **Full KAN-MAMMOTE**: `models/time_encoders/kan_mammote.py`
- **K-MOTE Details**: `models/time_encoders/k_mote.py`
- **SM-Kernel**: `models/time_encoders/sm_kernel.py`
- **Factory Functions**: `models/time_encoders/factory.py`
- **Training Scripts**: `experiments/train_link_prediction.py`

---

## 🚀 **Quick Start Checklist**

- [ ] Install dependencies: `pip install -r requirements.txt`
- [ ] Verify encoder available: `python -c "from models.time_encoders import KAN_MAMMOTE_Lite; print('OK')"`
- [ ] Test on small dataset: `python experiment_test.py --single_encoder kan_mammote_lite --datasets mooc`
- [ ] Run full comparison: `python experiment_test.py --time_encoders all`
- [ ] Check results: `cat experiment_report_kan_mammote_lite_*.txt`

---

**Last Updated:** October 3, 2025  
**Version:** 1.0  
**Author:** KAN-MAMMOTE Team
