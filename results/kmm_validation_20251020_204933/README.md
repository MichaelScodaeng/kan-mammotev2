# KAN-MAMMOTE Validation Results

**Experiment Date**: 2025-10-20 20:49:49

## Purpose
Validate that hyperparameter fixes resolve KAN-MAMMOTE's underperformance vs LeTE baseline.

## Fixes Tested
1. ✅ **Gradient Clipping** (max_norm=1.0) - Prevents gradient explosions
2. ✅ **Lower Learning Rate** (0.00001-0.00005) - Appropriate for model complexity
3. ✅ **Weight Decay** (0.0003-0.001) - Regularization for high-capacity model

## Experiments

### 1. uci (TCL)
- **Purpose**: Validate no regression on success case
- **Baseline KMM**: 0.9345 AP (+1.57% vs LeTE)
- **Hyperparameters**: LR=0.00005, WD=0.0005
- **Target**: Maintain AP ≥ 0.93

### 2. lastfm (JODIE)
- **Purpose**: Fix catastrophic failure
- **Baseline KMM**: 0.6519 AP (-10.17% vs LeTE 0.7256)
- **Hyperparameters**: LR=0.00001, WD=0.001 (AGGRESSIVE)
- **Target**: AP ≥ 0.70 (+5-8% improvement)

### 3. mooc (JODIE)
- **Purpose**: Fix moderate gap
- **Baseline KMM**: 0.7662 AP (-5.20% vs LeTE 0.8082)
- **Hyperparameters**: LR=0.00003, WD=0.0005
- **Target**: AP ≥ 0.80

### 4. Contacts (JODIE)
- **Purpose**: Fix training instability
- **Baseline KMM**: 4.01x loss spikes (UNSTABLE)
- **Hyperparameters**: LR=0.00002, WD=0.0003
- **Target**: Spike ratio < 2.0x

## Decision Criteria

**SUCCESS** (≥3/4 improve): Proceed with full evaluation  
**PARTIAL** (2/4 improve): More tuning needed  
**FAILURE** (<2 improve): Architectural changes needed  

## Results
Check individual result files in this directory.

