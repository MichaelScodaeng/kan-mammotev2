# KAN-MAMMOTE Fast Hyperparameter Tuning

Efficient hyperparameter tuning strategy for KAN-MAMMOTE across all GNN models and datasets.

## 🎯 Overview

This tuning framework uses a **temporal prefix strategy** for fast validation:
- **10% of training data** (temporal prefix - first 10% chronologically)
- **10 epochs** with **patience=3** for early stopping
- **Single seed** (seed=0) for speed
- Dataset-specific hyperparameter grids

## 📊 Hyperparameters Being Tuned

### Priority 1: Core Architecture
- `expert_dim`: K-MOTE expert dimension [64, 128, 256, 512]
- `mamba_d_state`: Mamba state dimension [128, 256, 512, 1024]
- `mamba_expand`: Mamba expansion factor [2, 4, 8]
- `dropout`: Dropout rate [0.0, 0.1, 0.2, 0.3]

### Priority 2: Mamba Configuration
- `mamba_headdim`: Mamba head dimension [64, 128]
- `mamba_d_conv`: Mamba convolution dimension [4]

### Fixed (Not Tuned)
- `learning_rate`: 0.0001 (same as baseline models)
- `weight_decay`: 0.0 (same as baseline models)
- `optimizer`: Adam
- `num_layers`: 2 (dataset-specific)
- `num_neighbors`: Dataset-specific (from best configs)

## 🚀 Quick Start

### 1. Test on Single Dataset/Model (Recommended First)

```bash
# Test with 2 configs on wikipedia/TGAT
bash test_hptune.sh

# Check generated scripts
ls ./hptune_test/

# Submit test jobs
bash ./hptune_test/submit_all_jobs.sh
```

### 2. Generate Jobs for Specific Datasets/Models

```bash
# Small datasets only
python tune_kan_mammote_fast.py \
    --datasets Contacts USLegis \
    --models TGAT TGN

# Medium datasets, all models
python tune_kan_mammote_fast.py \
    --datasets wikipedia reddit \
    --models TGAT TGN CAWN GraphMixer DyGFormer DyGMamba

# Limit configs for testing (first 3 per dataset/model)
python tune_kan_mammote_fast.py \
    --datasets wikipedia \
    --max_configs 3
```

### 3. Generate Jobs for All Datasets and Models

```bash
# Full hyperparameter search (recommended)
bash run_fast_hptune.sh

# Or with Python directly
python tune_kan_mammote_fast.py --output_dir ./hptune_jobs
```

### 4. Submit Jobs

```bash
# Submit all generated jobs
bash ./hptune_jobs/submit_all_jobs.sh

# Or submit individual jobs
qsub ./hptune_jobs/hptune_wikipedia_TGAT_c000.sh
qsub ./hptune_jobs/hptune_wikipedia_TGAT_c001.sh
# ...
```

### 5. Monitor Jobs

```bash
# Check job status
qstat -u s2516027

# Check job logs
tail -f ./hptune_jobs/logs/hptune_wikipedia_TGAT_c000.log
```

## 📈 Analyze Results

### Extract Best Configurations

```bash
# Analyze all tuning results
python ./hptune_jobs/analyze_results.py

# Results saved to:
# ./hptune_results/best_configs_summary.csv
```

### Manual Analysis

```bash
# Check results for specific dataset/model
ls ./hptune_results/wikipedia/TGAT/saved_results/

# View specific config result
cat ./hptune_results/wikipedia/TGAT/saved_results/TGAT/wikipedia/*.json
```

## 📁 Output Structure

```
hptune_jobs/
├── hptune_wikipedia_TGAT_c000.sh      # PBS job script for config 0
├── hptune_wikipedia_TGAT_c001.sh      # PBS job script for config 1
├── ...
├── summary_wikipedia_TGAT.json        # All configs for wikipedia/TGAT
├── submit_all_jobs.sh                 # Submit all jobs
├── analyze_results.py                 # Analysis script
└── logs/
    ├── hptune_wikipedia_TGAT_c000.log
    └── ...

hptune_results/
├── wikipedia/
│   ├── TGAT/
│   │   ├── saved_models/
│   │   ├── saved_results/
│   │   └── saved_metrics/
│   └── TGN/
│       └── ...
├── reddit/
│   └── ...
└── best_configs_summary.csv           # Best configs per dataset/model
```

## 🎛️ Dataset-Specific Search Spaces

### Small Datasets (Contacts, USLegis, Flights, UNvote)
- Expert dim: [64, 128]
- Mamba state: [128, 256]
- Expand: [2]
- Dropout: [0.0, 0.1]
- **Total configs: 8 per model**

### Medium Datasets (wikipedia, reddit, mooc, lastfm, enron, UNtrade)
- Expert dim: [128, 256]
- Mamba state: [256, 512]
- Expand: [2, 4]
- Dropout: [0.1, 0.2]
- **Total configs: 16 per model**

### Large Datasets (CanParl, SocialEvo)
- Expert dim: [256, 512]
- Mamba state: [512, 1024]
- Expand: [4, 8]
- Dropout: [0.2, 0.3]
- **Total configs: 32 per model**

## ⏱️ Time Estimates

### Per Configuration
- Small datasets: ~15-30 minutes
- Medium datasets: ~30-60 minutes
- Large datasets: ~60-120 minutes

### Full Tuning (All Datasets, All Models)
- Total configs: ~1000-1500
- Sequential time: ~30-50 hours
- Parallel time (with 10 GPUs): ~3-5 hours
- Parallel time (with 30 GPUs): ~1-2 hours

## 🔧 Customization

### Modify Hyperparameter Grid

Edit `tune_kan_mammote_fast.py`:

```python
DATASET_CONFIGS = {
    'wikipedia': {
        'expert_dim': [64, 128, 256],  # Add more values
        'mamba_d_state': [128, 256, 512, 1024],
        # ...
    }
}
```

### Adjust Fast Tuning Parameters

```python
FAST_TUNING_PARAMS = {
    'data_ratio': 0.2,  # Use 20% instead of 10%
    'num_epochs': 20,   # More epochs
    'patience': 5,      # More patience
}
```

### Change Resource Requirements

Edit PBS header in `create_pbs_job_script()`:

```bash
#PBS -l select=1:ncpus=8:mem=32gb:ngpus=1
#PBS -l walltime=04:00:00
```

## 📊 Best Practices

1. **Start Small**: Test on 1-2 datasets first
2. **Incremental Scaling**: Run small → medium → large datasets
3. **Monitor Early**: Check first few jobs complete successfully
4. **Resource Planning**: Estimate total GPU-hours needed
5. **Result Validation**: Verify results make sense before full runs

## 🎓 Interpreting Results

### What to Look For
- **Consistency**: Best configs should show stable performance
- **Trends**: Larger datasets may need larger `expert_dim` and `mamba_d_state`
- **Overfitting**: High dropout helps on small datasets
- **Convergence**: Jobs should complete in < 10 epochs with patience=3

### Red Flags
- All configs perform equally → Grid too narrow
- High variance → Need more seeds (increase `num_runs`)
- No convergence → Increase `num_epochs` or `patience`
- OOM errors → Reduce `mamba_d_state` or `expert_dim`

## 🔄 Next Steps After Tuning

1. **Select Best Configs**: Use `analyze_results.py` to find winners
2. **Full Training**: Re-train with best configs on 100% data
3. **Multiple Seeds**: Run 3-5 seeds for statistical significance
4. **Fine-tuning**: Optionally tune K-MOTE internals (n_harmonics, n_wavelets, etc.)

## 📝 Example Workflow

```bash
# Day 1: Test and validate pipeline
bash test_hptune.sh
bash ./hptune_test/submit_all_jobs.sh
# Wait ~1 hour, verify results

# Day 2: Run small datasets
python tune_kan_mammote_fast.py \
    --datasets Contacts USLegis Flights UNvote \
    --models TGAT TGN
bash ./hptune_jobs/submit_all_jobs.sh
# Wait ~4-6 hours

# Day 3: Run medium datasets
python tune_kan_mammote_fast.py \
    --datasets wikipedia reddit mooc lastfm \
    --models TGAT TGN CAWN GraphMixer
bash ./hptune_jobs/submit_all_jobs.sh
# Wait ~6-8 hours

# Day 4: Run large datasets
python tune_kan_mammote_fast.py \
    --datasets CanParl SocialEvo \
    --models TGAT DyGFormer DyGMamba
bash ./hptune_jobs/submit_all_jobs.sh
# Wait ~8-12 hours

# Day 5: Analyze and select best configs
python ./hptune_jobs/analyze_results.py

# Day 6+: Full training with best configs
# Use best configs from summary for final experiments
```

## 🆘 Troubleshooting

### Jobs Fail Immediately
- Check PBS logs: `cat ./hptune_jobs/logs/*.log`
- Verify Python environment: `source .venv/bin/activate`
- Test command manually: Copy command from job script

### Out of Memory (OOM)
- Reduce `mamba_d_state` or `expert_dim`
- Reduce `batch_size` (edit train script)
- Request more memory: `#PBS -l mem=32gb`

### Slow Convergence
- Increase `num_epochs` to 20
- Increase `patience` to 5
- Use more training data: `data_ratio=0.2`

### No Clear Winner
- Grid might be too narrow → Expand search space
- Try different seed: Change `seed` in `FAST_TUNING_PARAMS`
- Check if baseline models also struggle on this dataset

## 📚 References

- Main experiment script: `experiments/train_link_prediction.py`
- Config loader: `utils/load_configs.py`
- KAN-MAMMOTE implementation: `models/time_encoders/kan_mammote.py`
- K-MOTE implementation: `models/time_encoders/k_mote.py`
