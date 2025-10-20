# 🔍 Comprehensive Negative Sampling Evaluation

## Overview

After training completes, `train_link_prediction.py` now **automatically evaluates the best model on ALL negative sampling strategies**:

1. **Random** - Random negative edge sampling (baseline)
2. **Historical** - Sample from historically observed edges
3. **Inductive** - Sample from new, unseen edges

This provides a complete picture of model robustness across different evaluation scenarios.

---

## 📊 What Gets Evaluated

For each negative sampling strategy, we measure:

### **Transductive Test Set** (seen nodes)
- Average Precision (AP)
- ROC-AUC
- Loss

### **Inductive Test Set** (new nodes)
- Average Precision (AP)
- ROC-AUC
- Loss

---

## 📁 Output Files

After training, you'll get **two** result files:

### 1. **Standard Results** (Original)
```
saved_results/{model}/{dataset}/{model}_{encoder}_seed{X}_{timestamp}.json
```

Contains results for the **primary** negative sampling strategy only.

### 2. **Comprehensive Results** (New)
```
saved_results/{model}/{dataset}/{model}_{encoder}_seed{X}_comprehensive_{timestamp}.json
```

Contains results for **all three** negative sampling strategies.

**Example:**
```json
{
  "primary_strategy": "random",
  "time_encoder_type": "kan_mammote_dual_kmote",
  "seed": 0,
  "strategies": {
    "random": {
      "transductive_test": {
        "loss": 0.4523,
        "metrics": {
          "average_precision": 0.9345,
          "roc_auc": 0.9567
        }
      },
      "inductive_test": {
        "loss": 0.4892,
        "metrics": {
          "average_precision": 0.9123,
          "roc_auc": 0.9345
        }
      }
    },
    "historical": { ... },
    "inductive": { ... }
  }
}
```

---

## 📊 Example Output During Training

```
================================================================================
🔍 COMPREHENSIVE EVALUATION: Testing all negative sampling strategies
================================================================================

────────────────────────────────────────────────────────────────────────────────
📊 Testing negative sampling strategy: random
────────────────────────────────────────────────────────────────────────────────

RANDOM Negative Sampling Results:
  Transductive Test Loss: 0.4523
  Transductive Test average_precision: 0.9345
  Transductive Test roc_auc: 0.9567
  Inductive Test Loss: 0.4892
  Inductive Test average_precision: 0.9123
  Inductive Test roc_auc: 0.9345

────────────────────────────────────────────────────────────────────────────────
📊 Testing negative sampling strategy: historical
────────────────────────────────────────────────────────────────────────────────

HISTORICAL Negative Sampling Results:
  ...

────────────────────────────────────────────────────────────────────────────────
📊 Testing negative sampling strategy: inductive
────────────────────────────────────────────────────────────────────────────────

INDUCTIVE Negative Sampling Results:
  ...

================================================================================
✅ Comprehensive evaluation results saved to:
   ./saved_results/TGAT/wikipedia/TGAT_kan_mammote_dual_kmote_seed0_comprehensive_1729512345.json
================================================================================

================================================================================
📊 COMPARISON TABLE: Negative Sampling Strategies
================================================================================
Strategy         Trans AP     Trans AUC    Ind AP       Ind AUC     
────────────────────────────────────────────────────────────────────────────────
random           0.9345       0.9567       0.9123       0.9345       ⭐
historical       0.9234       0.9456       0.9012       0.9234      
inductive        0.9123       0.9345       0.8901       0.9123      
================================================================================
```

---

## 🔧 How to Analyze Results

### **Option 1: Automatic Analysis Script**

```bash
# Analyze all comprehensive results
python analyze_comprehensive_results.py --results_dir ./saved_results

# Filter by model
python analyze_comprehensive_results.py --model TGAT

# Filter by dataset
python analyze_comprehensive_results.py --dataset wikipedia

# Save to CSV
python analyze_comprehensive_results.py --output_csv comprehensive_summary.csv
```

**Output:**
```
================================================================================
📊 COMPREHENSIVE EVALUATION SUMMARY
================================================================================

────────────────────────────────────────────────────────────────────────────────
Model: TGAT | Dataset: wikipedia
────────────────────────────────────────────────────────────────────────────────

Strategy         Trans AP             Trans AUC            Ind AP               Ind AUC             
────────────────────────────────────────────────────────────────────────────────
random           0.9345 ± 0.0023      0.9567 ± 0.0019      0.9123 ± 0.0034      0.9345 ± 0.0028     
historical       0.9234 ± 0.0031      0.9456 ± 0.0025      0.9012 ± 0.0041      0.9234 ± 0.0037     
inductive        0.9123 ± 0.0028      0.9345 ± 0.0022      0.8901 ± 0.0045      0.9123 ± 0.0039     

✅ Best strategy for transductive test: random
✅ Best strategy for inductive test: random
```

### **Option 2: Manual Analysis**

```python
import json

# Load comprehensive results
with open('saved_results/TGAT/wikipedia/TGAT_kan_mammote_dual_kmote_seed0_comprehensive_*.json') as f:
    results = json.load(f)

# Access specific metrics
random_trans_ap = results['strategies']['random']['transductive_test']['metrics']['average_precision']
historical_ind_auc = results['strategies']['historical']['inductive_test']['metrics']['roc_auc']

print(f"Random Trans AP: {random_trans_ap:.4f}")
print(f"Historical Ind AUC: {historical_ind_auc:.4f}")
```

---

## 🎯 Use Cases

### **1. Model Robustness Analysis**
Check if your model generalizes across different negative sampling strategies:
- **Robust model**: Similar performance across all strategies
- **Brittle model**: Large variance across strategies

### **2. Hyperparameter Tuning**
When tuning, compare comprehensive results to select configs that:
- Perform well on **all** strategies (not just the primary one)
- Show consistent performance (low variance)

### **3. Publication / Reporting**
Report full results for transparency:
```
Our model achieves:
- Random: 0.9345 AP (trans), 0.9123 AP (ind)
- Historical: 0.9234 AP (trans), 0.9012 AP (ind)
- Inductive: 0.9123 AP (trans), 0.8901 AP (ind)
```

---

## ⚙️ Configuration

### **Enable/Disable Comprehensive Evaluation**

The comprehensive evaluation is **enabled by default**. To disable it (if you only care about the primary strategy):

```python
# In train_link_prediction.py, comment out the comprehensive evaluation section
# Lines ~1150-1290
```

### **Change Evaluation Strategies**

To test different strategies, modify:

```python
# In train_link_prediction.py
all_strategies = ['random', 'historical', 'inductive']  # Modify this list
```

---

## 📈 Performance Impact

**Time overhead per run:**
- **+30-60 seconds** (2-3x evaluation time)
- Minimal compared to training time (5-30 minutes)

**Disk overhead:**
- **~5-10 KB** extra per comprehensive results file
- Negligible

**Worth it?**
- ✅ **YES** - Comprehensive insights for minimal cost
- ✅ Helps identify overfitting to specific negative sampling strategy
- ✅ Required for rigorous academic evaluation

---

## 🔬 Best Practices

### **1. Always Check Comprehensive Results**
Don't just look at the primary strategy - check all three!

### **2. Report All Strategies**
In papers/reports, include comprehensive results for transparency.

### **3. Use for Model Selection**
When comparing models, prefer those with:
- High average performance across all strategies
- Low variance across strategies

### **4. Hyperparameter Tuning**
Use comprehensive results to validate that tuned hyperparameters generalize.

---

## 🚀 Integration with Hyperparameter Tuning

The hyperparameter tuning script (`run_hptune_sequential.py`) automatically benefits from this feature:

```bash
# Run tuning - comprehensive evaluation happens automatically
python run_hptune_sequential.py \
    --datasets wikipedia \
    --models TGAT \
    --max_configs 5

# Analyze comprehensive results
python analyze_comprehensive_results.py \
    --results_dir ./hptune_results/wikipedia/TGAT \
    --output_csv hptune_comprehensive.csv
```

This gives you a **complete picture** of which hyperparameters work best across all evaluation scenarios!

---

## 📝 Summary

✅ **Automatic** - No extra flags needed  
✅ **Comprehensive** - All 3 negative sampling strategies  
✅ **Fast** - Only ~1 minute overhead  
✅ **Actionable** - Clear comparison tables  
✅ **Reproducible** - Consistent seeds across strategies  

Happy evaluating! 🎯
