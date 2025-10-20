# Hyperparameter Tuning Scripts - Quick Reference

## 📋 What Was Created

I've created a complete hyperparameter tuning system for your temporal graph learning project:

### Core Scripts

1. **`tune_hyperparams_fast.py`** - Main sequential tuning script
2. **`generate_hptune_jobs.py`** - Generate parallel array job configurations
3. **`tune_hyperparams_fast.sh`** - PBS submission script for single job
4. **`test_hptune_setup.py`** - Quick test to verify setup works

### Generated Files (after running `generate_hptune_jobs.py`)

- `run_hptune_array.sh` - PBS array job script
- `hptune_configs/hptune_jobs_*.txt` - Job configuration file
- `collect_hptune_results.py` - Results collection script

## 🚀 Quick Start (3 Steps)

### Option A: Fast Parallel Execution (RECOMMENDED)

```bash
# 1. Generate job scripts (takes 1 second)
python generate_hptune_jobs.py

# 2. Submit to HPC cluster (runs all in parallel!)
qsub run_hptune_array.sh

# 3. Monitor and collect results
qstat -u $USER
python collect_hptune_results.py  # Run after jobs complete
```

### Option B: Simple Sequential Execution

```bash
# Test with small subset first
python tune_hyperparams_fast.py --subset 5 --dry_run

# Run on specific configurations
python tune_hyperparams_fast.py \
    --models TGAT DyGMamba \
    --datasets wikipedia mooc \
    --time_encoders lete kan_mammote_dual_kmote \
    --gpu 0
```

## ⚙️ Configuration

### Fixed Parameters (for speed)
- **Data ratio**: 10% (temporal prefix strategy)
- **Epochs**: 10 max
- **Patience**: 3 (early stopping)
- **Runs**: 1 (for initial search)
- **Batch size**: 200

### Hyperparameter Search Grid
- **Learning Rates**: [1e-4, 5e-4, 1e-3, 5e-3]
- **Weight Decays**: [0.0, 1e-5, 1e-4, 1e-3]

### Models to Tune
```python
['JODIE', 'TGAT', 'TGN', 'TCL', 'DyGFormer', 'DyGMamba']
```

### Datasets
```python
['wikipedia', 'reddit', 'mooc', 'lastfm', 'enron', 'SocialEvo', 'uci',
 'CanParl', 'Contacts', 'Flights', 'UNtrade', 'UNvote', 'USLegis']
```

### Time Encoders
```python
['lete', 'kan_mammote_dual_kmote', 'mercer', 'time2vec']
```

**Total experiments**: 6 × 13 × 4 × 4 × 4 = **4,992 configurations**

## 📊 Expected Performance

### Time Estimates (with 10% data, 10 epochs)

| Dataset Type | Time per Config | Example Datasets |
|--------------|----------------|------------------|
| Small        | 2-5 min        | Contacts, Flights |
| Medium       | 5-15 min       | wikipedia, reddit, mooc |
| Large        | 15-30 min      | CanParl |

### Execution Time Comparison

| Method | Total Time | Requirements |
|--------|------------|--------------|
| Sequential (1 GPU) | ~300-500 hours | 1 GPU |
| Array Jobs (100 GPUs) | ~3-5 hours | HPC cluster |
| Array Jobs (50 GPUs) | ~6-10 hours | HPC cluster |

## 🎯 Key Features

### 1. **No Conflicts with Main Training**
- All outputs use `_HPTUNE_YYYYMMDD` suffix
- Separate output directory: `hyperparameter_tuning_results/`
- Safe to run alongside production experiments

### 2. **Fast Temporal Prefix Strategy**
- Uses only first 10% of training data
- Maintains temporal ordering (no random sampling)
- Sufficient for hyperparameter selection

### 3. **Early Stopping**
- Max 10 epochs per config
- Patience of 3 epochs
- Saves time on poor configurations

### 4. **Automatic Result Collection**
- Parses all experiment outputs
- Ranks by validation AP
- Generates human-readable summary

## 📁 Output Structure

```
hyperparameter_tuning_results/
├── MODEL_DATASET_ENCODER_lrX_wdY/
│   ├── saved_models/
│   │   └── *_HPTUNE_20231021.pth
│   ├── saved_results/
│   │   └── results.json
│   ├── saved_metrics/
│   └── training.log
├── collected_results.json
├── summary_report.txt
└── results_final_TIMESTAMP.json
```

## 🔍 Analyzing Results

### Quick View
```bash
# See best configs for each combination
cat hyperparameter_tuning_results/summary_report.txt
```

### Python Analysis
```python
import json
import pandas as pd

# Load results
with open('hyperparameter_tuning_results/collected_results.json') as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results)

# Filter successful runs
df_success = df[df['status'] == 'success']

# Find best overall
best = df_success.loc[df_success['validate_ap'].idxmax()]
print(f"Best config: {best['model']} + {best['dataset']} + {best['time_encoder']}")
print(f"LR: {best['lr']}, WD: {best['wd']}")
print(f"Val AP: {best['validate_ap']:.4f}")

# Best per model
best_per_model = df_success.groupby('model').apply(
    lambda x: x.loc[x['validate_ap'].idxmax()]
)
print("\nBest per model:")
print(best_per_model[['model', 'lr', 'wd', 'validate_ap']])
```

## 🧪 Testing Before Full Run

```bash
# Test environment setup
python test_hptune_setup.py

# Dry run (see commands without executing)
python tune_hyperparams_fast.py --dry_run --subset 3

# Run one quick experiment manually
python tune_hyperparams_fast.py \
    --models TGAT \
    --datasets Contacts \
    --time_encoders lete \
    --subset 1
```

## 📝 Customization Examples

### Example 1: Focus on Specific Models
```bash
python tune_hyperparams_fast.py \
    --models DyGMamba DyGFormer \
    --datasets wikipedia reddit mooc \
    --time_encoders kan_mammote_dual_kmote lete
```

### Example 2: Different Learning Rate Range
Edit `tune_hyperparams_fast.py`:
```python
LEARNING_RATES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]  # Wider range
WEIGHT_DECAYS = [0.0, 1e-6, 1e-5, 1e-4]         # Finer granularity
```

### Example 3: Longer Training for Final Validation
```python
DATA_RATIO = 0.2      # 20% of data
NUM_EPOCHS = 20       # More epochs
PATIENCE = 5          # More patience
```

## 🛠️ Troubleshooting

### Jobs Failing?
```bash
# Check individual job logs
cat hptune_logs/job_1.log

# Check GPU availability
nvidia-smi

# Verify data files exist
ls data/
```

### Out of Memory?
Edit the scripts to reduce:
```python
BATCH_SIZE = 100  # Reduce from 200
```

### Missing Dependencies?
```bash
# Activate environment
source .venv/bin/activate

# Install requirements
pip install -r requirement.txt
```

## 📈 Next Steps After Tuning

1. **Identify Best Configurations**
   ```bash
   python collect_hptune_results.py
   cat hyperparameter_tuning_results/summary_report.txt
   ```

2. **Update Config Files**
   Edit `utils/load_configs.py` with best LR and WD for each model/dataset

3. **Run Full Training**
   ```bash
   python experiment_unified.py \
       --model_name DyGMamba \
       --dataset_name wikipedia \
       --time_encoder_type kan_mammote_dual_kmote \
       --learning_rate 0.001 \
       --weight_decay 1e-5 \
       --data_ratio 1.0 \
       --num_epochs 100 \
       --num_runs 3
   ```

4. **Compare with Baselines**
   - Run with original configs
   - Compare performance improvement
   - Document in paper/report

## 📚 Files Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `tune_hyperparams_fast.py` | Sequential tuning | Single GPU, small search |
| `generate_hptune_jobs.py` | Generate array jobs | HPC cluster, large search |
| `test_hptune_setup.py` | Verify setup | Before starting tuning |
| `collect_hptune_results.py` | Gather results | After jobs complete |
| `HYPERPARAMETER_TUNING_README.md` | Detailed docs | Full documentation |

## 💡 Pro Tips

1. **Start Small**: Test with `--subset 10` first
2. **Use Array Jobs**: 100x speedup with HPC resources
3. **Monitor Disk**: Each config uses ~100-500MB
4. **Clean Up**: Delete results after collecting best configs
5. **Document**: Save `summary_report.txt` for your records

## ❓ Common Questions

**Q: Will this interfere with my main experiments?**  
A: No! Everything uses `_HPTUNE_` suffix and separate directories.

**Q: How long will this take?**  
A: With array jobs on HPC: 3-5 hours. Sequential: several days.

**Q: Can I resume if interrupted?**  
A: Yes for array jobs (resubmit missing jobs). Sequential requires restart.

**Q: What if I only care about some combinations?**  
A: Use the `--models`, `--datasets`, `--time_encoders` arguments to filter.

**Q: How do I know which configs worked?**  
A: Run `collect_hptune_results.py` - it shows success/fail for each.

---

**Ready to start?**
```bash
python test_hptune_setup.py  # Test first!
python generate_hptune_jobs.py  # Then generate jobs
qsub run_hptune_array.sh  # Submit!
```
