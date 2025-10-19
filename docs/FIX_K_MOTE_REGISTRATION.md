# Fix: "Unknown encoder type 'k_mote'" Error - Complete Explanation

## 🔍 Chain of Thought Analysis

### **What Happened:**

```
User requests: k_mote_shared_abs encoder
       ↓
Creates: KMOTESharedAbsoluteTime wrapper class
       ↓
Wrapper calls: create_time_encoder('k_mote', ...)
       ↓
Factory checks: Is 'k_mote' in available encoders list?
       ↓
Result: ❌ NOT FOUND
       ↓
Factory falls back to: 'original' (OriginalTimeEncoder)
       ↓
Forward pass: Tries to call encoder(t_abs=..., t_rel=...)
       ↓
OriginalTimeEncoder: Only expects t_rel, gets t_abs
       ↓
Result: ⚠️ WARNING: "OriginalTimeEncoder expects t_rel but got t_abs"
```

---

## 🐛 Root Cause

The issue has **TWO parts**:

### **Part 1: Missing Registration**
`'k_mote'` was **NOT registered** as an available encoder type in `factory.py`:
- ❌ Not in `get_available_encoders()` list
- ❌ Not handled in `create_time_encoder()` function

### **Part 2: Fallback Behavior**
When an unknown encoder is requested, the factory falls back to `OriginalTimeEncoder`:
```python
else:
    print(f"WARNING: Unknown encoder type '{encoder_type}'...")
    encoder = OriginalTimeEncoder(time_dim=time_dim)
    time_encoder = TimeEncoderWrapper(encoder)
```

This fallback encoder doesn't match what the user requested!

---

## ✅ The Complete Fix

### **Step 1: Register 'k_mote' in Available Encoders List**

**File:** `models/time_encoders/factory.py`

```python
def get_available_encoders():
    # ...existing code...
    
    # Add ablation study encoders (always available since they're local)
    encoders.extend([
        'sm_kernel_only',
        'kmote_abs_only',
        'kmote_rel_only',
        'k_mote',  # ✅ ADD THIS LINE - Standalone K-MOTE
    ])
    
    return encoders
```

**Why:** This tells the system that `'k_mote'` is a valid encoder type.

---

### **Step 2: Add Handler in create_time_encoder()**

**File:** `models/time_encoders/factory.py`

```python
def create_time_encoder(...):
    # ...existing code...
    
    elif encoder_type == 'k_mote':
        # ✅ Standalone K-MOTE (without Mamba) for MNIST-style experiments
        from .k_mote import KMOTE
        
        print("INFO: Creating standalone K-MOTE encoder.")
        print("Time Embedding dim:", time_dim)
        
        # Get K-MOTE parameters
        if args is not None:
            wavelet_type = getattr(args, 'wavelet_type', kwargs.get('wavelet_type', 'shock'))
            transform_mode = getattr(args, 'transform_mode', kwargs.get('transform_mode', 'adapter'))
            adapter_type = getattr(args, 'adapter_type', kwargs.get('adapter_type', 'affine'))
        else:
            wavelet_type = kwargs.get('wavelet_type', 'shock')
            transform_mode = kwargs.get('transform_mode', 'adapter')
            adapter_type = kwargs.get('adapter_type', 'affine')
        
        print(f"K-MOTE parameters:")
        print(f"  - output_dim: {time_dim}")
        print(f"  - wavelet_type: {wavelet_type}")
        print(f"  - transform_mode: {transform_mode}")
        print(f"  - adapter_type: {adapter_type if transform_mode == 'adapter' else 'N/A'}")
        
        time_encoder = KMOTE(
            input_dim=1,
            output_dim=time_dim,
            wavelet_type=wavelet_type,
            transform_mode=transform_mode,
            adapter_type=adapter_type if transform_mode == 'adapter' else None,
            use_scale=True,
            use_layernorm=True
        )
    
    elif encoder_type in ['original', 'time_encoder', 'default']:
        # ...existing code...
```

**Why:** This actually creates the K-MOTE encoder when requested.

---

## 📊 Architecture Overview

### **How K-MOTE Wrappers Work**

```
MNIST Experiment
       ↓
requests: 'k_mote_shared_abs'
       ↓
Creates wrapper: KMOTESharedAbsoluteTime(time_dim=32)
       ↓
Wrapper.__init__() calls:
       create_time_encoder('k_mote', time_dim=32, transform_mode='shared')
       ↓
Factory creates: KMOTE(input_dim=1, output_dim=32, transform_mode='shared')
       ↓
Wrapper stores: self.kmote = <KMOTE instance>
       ↓
Wrapper.forward(x) calls: self.kmote(x.float())
       ↓
K-MOTE processes: Single tensor input (absolute positions)
       ↓
Returns: Temporal embeddings (batch, seq_len, 32)
```

### **Key Insight: K-MOTE Interface**

K-MOTE's `forward()` signature is:
```python
def forward(self, t: torch.Tensor) -> torch.Tensor:
    # t: (batch, seq_len, 1) or (batch, seq_len)
```

**It only takes ONE argument** (unlike KAN-MAMMOTE which takes `t_abs` and `t_rel`).

So the wrapper classes correctly just pass:
```python
def forward(self, x):
    return self.kmote(x.float())  # ✅ Correct: single input
```

---

## 🎯 Why The Previous Fix Attempts Failed

### **Attempt 1: Adding `**kwargs` to wrapper**
```python
def forward(self, x, **kwargs):
    if 't_abs' in kwargs:
        x = kwargs['t_abs']
    return self.kmote(x.float())
```

**Problem:** K-MOTE doesn't accept `t_abs` and `t_rel` as separate arguments. It only takes `t`.

### **Attempt 2: Passing both to K-MOTE**
```python
embedded = self.time_encoder(t_abs=t_abs, t_rel=t_rel)
```

**Problem:** K-MOTE's signature is `forward(self, t)`, not `forward(self, t_abs, t_rel)`. This would cause a `TypeError`.

### **The Correct Approach:**

K-MOTE wrappers should:
1. ✅ Receive single input `x` (pixel positions)
2. ✅ Optionally transform it (absolute → relative for `*Relative` variants)
3. ✅ Pass single tensor to K-MOTE: `self.kmote(x.float())`

---

## 📝 Summary of Changes

### **Changes Made:**

1. ✅ **Added `'k_mote'` to available encoders list** in `factory.py`
   - Line ~227: Added to `encoders.extend([...])`

2. ✅ **Added handler for `'k_mote'` in `create_time_encoder()`** in `factory.py`  
   - Lines ~640-660: New `elif encoder_type == 'k_mote':` block
   - Imports `KMOTE` from `k_mote.py`
   - Creates instance with proper parameters
   - Supports `transform_mode` and `adapter_type` kwargs

3. ✅ **Wrapper classes already correct** in `event_based_mnist_experiment.py`
   - No changes needed (they were reset to correct version)
   - Already pass single tensor: `self.kmote(x.float())`

---

## 🧪 Expected Behavior After Fix

### **Before Fix:**
```
Requested encoder: k_mote
WARNING: Unknown encoder type 'k_mote' or encoder not available. Using default TimeEncoder.
Time Embedding dim: 32
```
Uses wrong encoder → OriginalTimeEncoder instead of K-MOTE

### **After Fix:**
```
Requested encoder: k_mote
INFO: Creating standalone K-MOTE encoder.
Time Embedding dim: 32
K-MOTE parameters:
  - output_dim: 32
  - wavelet_type: shock
  - transform_mode: shared  
  - adapter_type: N/A
```
Uses correct encoder → Actual K-MOTE with requested configuration

---

## 🔬 Testing Validity

### **Will this break K-MOTE absolute-only and relative-only variants?**

**Answer: NO! ✅**

#### **For Absolute Variants** (e.g., `k_mote_shared_abs`):
```python
class KMOTESharedAbsoluteTime(nn.Module):
    def forward(self, x):
        return self.kmote(x.float())  # ✅ Passes absolute positions directly
```

#### **For Relative Variants** (e.g., `k_mote_shared_rel`):
```python
class KMOTESharedRelativeTime(nn.Module):
    def forward(self, x):
        # Convert absolute → relative
        rel_times = torch.zeros_like(x, dtype=torch.float32)
        rel_times[:, 0] = 0.0
        if seq_len > 1:
            rel_times[:, 1:] = x[:, 1:].float() - x[:, :-1].float()
        return self.kmote(rel_times)  # ✅ Passes relative differences
```

Both work correctly because:
1. ✅ K-MOTE only expects single tensor input
2. ✅ Wrappers handle transformation (absolute vs relative)
3. ✅ No dual-stream interface confusion

---

## ✨ Final Result

After applying this fix:
- ✅ `k_mote_shared_abs` → Uses K-MOTE with shared transform mode (absolute positions)
- ✅ `k_mote_shared_rel` → Uses K-MOTE with shared transform mode (relative differences)
- ✅ `k_mote_abs` → Uses K-MOTE with adapter mode (absolute positions)
- ✅ `k_mote_rel` → Uses K-MOTE with adapter mode (relative differences)
- ✅ All other variants work correctly
- ✅ No more fallback to OriginalTimeEncoder
- ✅ No more warnings about `t_abs` vs `t_rel`

---

## 🎓 Key Takeaways

1. **Registration is required**: Any encoder must be in BOTH:
   - `get_available_encoders()` list
   - `create_time_encoder()` handler

2. **Interface matters**: K-MOTE uses single-input interface `forward(t)`, not dual-stream `forward(t_abs, t_rel)`

3. **Wrappers handle transformation**: The wrapper class decides what to pass to K-MOTE (absolute vs relative)

4. **Factory pattern**: Always check the factory handles your encoder type before using it!
