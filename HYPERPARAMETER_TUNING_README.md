# Fast Hyperparameter Tuning Guide

This directory contains scripts for fast hyperparameter tuning of learning rate and weight decay across all models, datasets, and time encoders.

## Key Features

- **10% temporal prefix data** - Uses only the first 10% of training data for speed
- **10 epochs max** with **patience 3** - Early stopping for efficiency
- **Special naming suffix** - Results saved with `_HPTUNE_YYYYMMDD` suffix to avoid conflicts
- **Parallel execution** - Can run as array jobs on HPC clusters

## Configuration

### Models
- JODIE, TGAT, TGN, TCL, DyGFormer, DyGMamba

### Datasets
- wikipedia, reddit, mooc, lastfm, enron, SocialEvo, uci
- CanParl, Contacts, Flights, UNtrade, UNvote, USLegis

### Time Encoders
- lete, kan_mammote_dual_kmote, mercer, time2vec

### Hyperparameter Search Space
- **Learning Rates**: [1e-4, 5e-4, 1e-3, 5e-3]
- **Weight Decays**: [0.0, 1e-5, 1e-4, 1e-3]

**Total combinations**: 6 models × 13 datasets × 4 time encoders × 4 LRs × 4 WDs = **4,992 experiments**

## Usage Options

### Option 1: Sequential Execution (Simple)

Run all experiments sequentially on a single GPU:

```bash
# Interactive testing (dry run)
python tune_hyperparams_fast.py --dry_run --subset 5

# Run on subset
python tune_hyperparams_fast.py --models TGAT TGN --datasets wikipedia reddit --subset 10

# Full run (will take a long time!)
python tune_hyperparams_fast.py --gpu 0
```

### Option 2: HPC Single Job

Submit as a single long-running PBS job:

```bash
qsub tune_hyperparams_fast.sh
```

Edit the script to customize which models/datasets/encoders to run.

### Option 3: HPC Array Jobs (RECOMMENDED - Fastest!)

Generate and submit parallel array jobs:

```bash
# Step 1: Generate job configuration and scripts
python generate_hptune_jobs.py

# Step 2: Submit array job (runs all configs in parallel!)
qsub run_hptune_array.sh

# Step 3: Monitor progress
qstat -u $USER
watch -n 30 'qstat -u $USER | grep hptune'

# Step 4: Collect results when done
python collect_hptune_results.py
```

## Output Structure

```
hyperparameter_tuning_results/
├── MODEL_DATASET_ENCODER_lrX_wdY/
│   ├── saved_results/
│   ├── saved_models/
│   └── training.log
├── collected_results.json       # All results in one file
├── summary_report.txt           # Human-readable summary
└── results_final_TIMESTAMP.json # Timestamped backup
```

## Results Analysis

After running experiments:

```bash
# Collect all results
python collect_hptune_results.py

# View summary
cat hyperparameter_tuning_results/summary_report.txt

# Analyze in Python
python -c "
import json
with open('hyperparameter_tuning_results/collected_results.json') as f:
    results = json.load(f)
    
# Find best configurations
best = max(results, key=lambda x: x.get('validate_ap', 0))
print(f'Best config: {best}')
"
```

## Customization

### Modify Search Space

Edit the constants in `tune_hyperparams_fast.py` or `generate_hptune_jobs.py`:

```python
LEARNING_RATES = [1e-4, 5e-4, 1e-3]  # Customize
WEIGHT_DECAYS = [0.0, 1e-5, 1e-4]    # Customize
DATA_RATIO = 0.1                     # 10% of data
NUM_EPOCHS = 10                      # Max epochs
PATIENCE = 3                         # Early stopping
```

### Filter Models/Datasets

```bash
# Only specific combinations
python tune_hyperparams_fast.py \
    --models DyGMamba DyGFormer \
    --datasets wikipedia mooc \
    --time_encoders kan_mammote_dual_kmote lete \
    --gpu 0
```

## Performance Estimates

With 10% data and 10 epochs:
- Small datasets (Contacts, Flights): ~2-5 min per config
- Medium datasets (wikipedia, reddit): ~5-15 min per config  
- Large datasets (CanParl): ~15-30 min per config

**Array job approach**: All 4,992 configs complete in ~2-4 hours (with enough GPUs)
**Sequential approach**: ~300-500 hours total

## Tips

1. **Start small**: Test with `--subset 10` first
2. **Use array jobs**: 100x faster than sequential for large searches
3. **Monitor disk space**: Each config saves models (~100MB each)
4. **Check logs**: Failed jobs are logged in `hptune_logs/`
5. **Resume failed jobs**: Re-run specific configs by editing the config file

## File Naming Convention

All tuning outputs use the suffix `_HPTUNE_YYYYMMDD` to ensure they don't interfere with your actual training runs. You can safely delete the entire `hyperparameter_tuning_results/` directory without affecting production models.

## Troubleshooting

**Problem**: Jobs timeout  
**Solution**: Increase walltime in PBS script or reduce dataset size

**Problem**: Out of GPU memory  
**Solution**: Reduce batch size in the script (currently 200)

**Problem**: Missing results  
**Solution**: Check `hptune_logs/job_N.log` for errors

**Problem**: Can't find training script  
**Solution**: The script auto-detects. Ensure `experiment_unified.py` or `train_link_prediction.py` exists

## Next Steps

After finding best hyperparameters:
1. Update `utils/load_configs.py` with best values
2. Run full training with 100% data and more epochs
3. Compare against baseline configurations
