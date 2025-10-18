# Modulation Pathway Variants in KAN-MAMMOTE

## Overview

KAN-MAMMOTE with ControllableMamba2 now supports two architectural variants for how relative time information flows through the model. This document explains the differences and provides guidance on when to use each variant.

## The Two Variants

### Variant 1: Separate Modulation Pathway (DEFAULT) ✅

**Configuration:**
```python
model = KAN_MAMMOTE(
    embedding_dim=64,
    expert_dim=64,
    fusion_strategy='mamba',
    use_controllable_mamba=True,
    separate_modulation_pathway=True  # DEFAULT
)
```

**Architecture:**
```
Absolute Time (u_k) ──────────────────────┐
                                           ├──► Mamba2 Input
                                           │
Relative Time (v_k) ──► fusion_mlp ───────┼──► FiLM Gates (γ, β)
                                           │    (controls dynamics)
                                           │
                                           └──► Modulates Mamba2
```

**Key Characteristics:**
- **Content Pathway**: Pure absolute time features (`u_k`)
- **Modulation Pathway**: Relative time controls dynamics via FiLM gates
- **Cleaner Separation**: "What" (absolute) vs "How" (relative)
- **Conceptual Clarity**: Absolute = temporal position, Relative = temporal dynamics

**Advantages:**
- ✅ Architecturally cleaner and more interpretable
- ✅ Clear separation of concerns (FiLM modulation principle)
- ✅ Relative time explicitly controls "how" to process absolute time
- ✅ May reduce risk of over-fitting to relative patterns

**When to Use:**
- Default choice for most applications
- When you want clearer architectural interpretation
- When relative time patterns are meant to modulate/control processing
- For research papers where architectural clarity matters

---

### Variant 2: Combined Pathway (LEGACY)

**Configuration:**
```python
model = KAN_MAMMOTE(
    embedding_dim=64,
    expert_dim=64,
    fusion_strategy='mamba',
    use_controllable_mamba=True,
    separate_modulation_pathway=False  # LEGACY
)
```

**Architecture:**
```
Absolute Time (u_k) ──────────────────────┐
                                           ├──► (u_k + fusion_features)
Relative Time (v_k) ──► fusion_mlp ───────┤
                         │                 │
                         │                 └──► Mamba2 Input
                         │
                         └──────────────────► FiLM Gates (γ, β)
                                              (also controls dynamics)
```

**Key Characteristics:**
- **Content Pathway**: Mixed absolute + relative features (`u_k + fusion_features`)
- **Modulation Pathway**: Relative time ALSO controls via FiLM gates
- **Dual Information Flow**: Relative time affects both input and gates
- **Richer Representation**: More relative time information in data flow

**Advantages:**
- ✅ Richer information flow (relative time in two pathways)
- ✅ May capture more complex interactions
- ✅ Backward compatible with earlier experiments
- ✅ Potentially better for tasks where relative time is highly informative

**Potential Drawbacks:**
- ⚠️ Less architecturally clean
- ⚠️ May "double-count" relative information
- ⚠️ Harder to interpret what each component does
- ⚠️ Could lead to over-reliance on relative patterns

**When to Use:**
- When reproducing earlier experiments
- When you need maximum information flow
- When empirical results show it performs better
- For ablation studies comparing both variants

---

## Vanilla Mamba2 Behavior

**Important:** For vanilla Mamba2 (`use_controllable_mamba=False`), the `separate_modulation_pathway` parameter is **ignored**, and the model **always uses the combined pathway** (`u_k + fusion_features`).

**Rationale:** Without FiLM modulation gates, there's no separate modulation pathway, so both streams must be combined in the input.

```python
model = KAN_MAMMOTE(
    fusion_strategy='mamba',
    use_controllable_mamba=False,  # Vanilla Mamba2
    separate_modulation_pathway=True  # IGNORED - always uses combined
)
```

---

## Experimental Results

### Expected Behaviors

**Separate Modulation Pathway:**
- May show **slower initial convergence** (less information in input)
- Should have **better generalization** (cleaner architecture)
- More **interpretable** learned representations
- Better for **out-of-distribution** temporal patterns

**Combined Pathway:**
- May show **faster initial convergence** (richer input)
- Could **overfit** to training temporal patterns
- Less clear what each component learns
- Better for **in-distribution** tasks

### Testing Both Variants

Run the comparison script:
```bash
python tests/test_modulation_pathway_variants.py
```

Or in your experiment:
```python
# Test separate pathway
results_separate = train_model(
    KAN_MAMMOTE(..., separate_modulation_pathway=True),
    ...
)

# Test combined pathway
results_combined = train_model(
    KAN_MAMMOTE(..., separate_modulation_pathway=False),
    ...
)
```

---

## Recommendations

### For Most Users
**Use the default** (`separate_modulation_pathway=True`):
- Cleaner architecture
- Better interpretability
- Follows FiLM modulation principles

### For Ablation Studies
**Test both variants** and report results:
```python
for separate in [True, False]:
    model = KAN_MAMMOTE(
        ...,
        separate_modulation_pathway=separate
    )
    # Train and evaluate
    variant_name = "separate" if separate else "combined"
    print(f"{variant_name}: acc={accuracy:.2f}%")
```

### For Paper Experiments
If you're writing a paper, consider:
1. Using **separate pathway** as your main model (cleaner)
2. Including **combined pathway** in ablation studies
3. Discussing the architectural trade-offs
4. Reporting which works better for your specific task

---

## Implementation Details

The key difference is in the forward pass:

```python
# Get features from both encoders
u_k = self.k_mote_abs(t_abs)  # Absolute time
v_k = self.k_mote_rel(t_rel)  # Relative time
fusion_features = self.fusion_mlp_base(v_k)

# Choose pathway based on configuration
if self.separate_modulation_pathway:
    # VARIANT 1: Pure absolute
    combined_input = u_k
else:
    # VARIANT 2: Mixed
    combined_input = u_k + fusion_features

# Generate FiLM gates (always from relative time)
gamma, beta = self.modulator_head(fusion_features)

# Apply Mamba2 with modulation
output = self.mamba2(
    u=combined_input,
    temporal_modulators=(gamma, beta)
)
```

---

## Debugging

Enable debug mode to see which pathway is being used:

```python
model.enable_debug_mode()
output = model(t_abs, t_rel, debug=True)
```

Output will show:
```
🔗 SEPARATE MODULATION PATHWAY (ControllableMamba2 default):
   combined_input = u_k (pure absolute)
   ...
```

or

```
🔗 COMBINED INPUT PATHWAY (vanilla Mamba2 or legacy):
   combined_input = u_k + fusion_features
   ...
```

---

## References

- **FiLM**: [Feature-wise Linear Modulation](https://arxiv.org/abs/1709.07871)
- **Mamba2**: [Transformers are SSMs](https://arxiv.org/abs/2312.00752)
- **K-MOTE**: Our KAN-based Mixture of Temporal Experts

---

## Summary Table

| Feature | Separate Pathway | Combined Pathway | Vanilla Mamba2 |
|---------|-----------------|------------------|----------------|
| **Default** | ✅ Yes | ❌ No | N/A |
| **Input to Mamba** | `u_k` | `u_k + fusion_features` | `u_k + fusion_features` |
| **FiLM Gates** | ✅ Yes | ✅ Yes | ❌ No |
| **Architectural Clarity** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |
| **Information Richness** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Recommended For** | Research, Production | Ablation, Compatibility | Baseline |

---

**Last Updated:** October 19, 2025
