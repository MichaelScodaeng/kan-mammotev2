# Ablation Study with Multiple Models and Datasets

## Overview

The `experiment_ablation.py` script now supports running comprehensive ablation studies across:
- **Multiple GNN models**: TGAT, TGN, CAWN, TCL, GraphMixer, DyGFormer, DyGMamba
- **Multiple datasets**: 13 datasets total (wikipedia, reddit, mooc, lastfm, enron, SocialEvo, uci, CanParl, Contacts, Flights, UNtrade, UNvote, USLegis)
- **Multiple time encoders**: 10 encoder configurations (LeTE, Bochner, Mercer, Original, SM-Kernel only, K-MOTE abs/rel only, Dual Stream, KAN-MAMMOTE Lite/Full)

## Key Features

### 1. **Multi-Model Support**
Run experiments on multiple GNN backbones automatically:

```bash
# Run ALL models on a single dataset
python experiment_ablation.py --all_models --dataset wikipedia --data_ratio 0.1

# Run specific model on all datasets  
python experiment_ablation.py --model TGAT --all_datasets --data_ratio 0.1

# Run ALL models on ALL datasets (comprehensive study)
python experiment_ablation.py --all_models --all_datasets --data_ratio 0.1
```

### 2. **Organized Output Structure**
All experiment outputs (models, metrics, results) are saved in a timestamped directory:

```
ablation_20251006_143022/
├── saved_models/
│   ├── TGAT/
│   │   ├── wikipedia/
│   │   ├── reddit/
│   │   └── ...
│   ├── TGN/
│   └── ...
├── saved_metrics/
│   ├── TGAT/
│   ├── TGN/
│   └── ...
├── saved_results/
│   ├── TGAT/
│   ├── TGN/
│   └── ...
└── ablation_study_summary_TGAT_TGN_TCL_wikipedia_reddit_20251006_143022.json
```

### 3. **Comprehensive Summary Reports**
JSON summary includes:
- All model names tested
- All datasets used
- Per-experiment results with model, dataset, encoder, and performance metrics
- Execution times and success/failure status

## Usage Examples

### Example 1: Compare All Encoders on TGAT + Wikipedia
```bash
python experiment_ablation.py \
  --model TGAT \
  --dataset wikipedia \
  --data_ratio 0.1 \
  --num_epochs 50
```

### Example 2: Compare TGAT vs TGN vs TCL on Multiple Datasets
```bash
python experiment_ablation.py \
  --model TGAT \
  --model TGN \
  --model TCL \
  --all_datasets \
  --data_ratio 0.1 \
  --num_epochs 50
```
**Note:** Currently you need to specify `--all_models` or a single model. Multiple specific models need to be run separately.

### Example 3: Full Comprehensive Study (All Models × All Datasets × All Encoders)
```bash
python experiment_ablation.py \
  --all_models \
  --all_datasets \
  --data_ratio 0.05 \
  --num_epochs 30
```
**Warning:** This will run **910 experiments** (7 models × 13 datasets × 10 encoders)!

### Example 4: Focused Study (Specific Encoders on Multiple Models)
```bash
python experiment_ablation.py \
  --all_models \
  --dataset wikipedia \
  --encoders kan_mammote_full lete original \
  --data_ratio 0.1 \
  --num_epochs 50
```
This runs only 3 encoders × 7 models = **21 experiments** on Wikipedia dataset.

### Example 5: Quick Validation Test
```bash
python experiment_ablation.py \
  --model TGAT \
  --dataset wikipedia \
  --encoders lete \
  --data_ratio 0.02 \
  --num_epochs 2 \
  --dry_run
```

## Arguments

### Model Selection
- `--model <NAME>`: Specify a single model (TGAT, TGN, CAWN, TCL, GraphMixer, DyGFormer, DyGMamba)
- `--all_models`: Run on all 7 available GNN models

### Dataset Selection
- `--dataset <NAME>`: Specify a single dataset
- `--all_datasets`: Run on all 13 available datasets

### Encoder Selection
- `--encoders <NAME1> <NAME2> ...`: Specify specific encoders to test
- If omitted, all 10 encoders are tested

### Training Parameters
- `--data_ratio <FLOAT>`: Fraction of data to use (default: 0.1 = 10%)
- `--num_epochs <INT>`: Number of training epochs (default: 10)
- `--batch_size <INT>`: Batch size (default: 200)
- `--num_neighbors <INT>`: Number of temporal neighbors (default: 20)
- `--learning_rate <FLOAT>`: Learning rate (default: 0.0001)

### Execution Control
- `--dry_run`: Print commands without executing
- `--verbose`: Enable verbose output
- `--timeout <INT>`: Timeout per experiment in seconds (default: 3600)

## Experiment Calculation

**Total experiments = Models × Datasets × Encoders**

Examples:
- 1 model × 1 dataset × 10 encoders = **10 experiments**
- 7 models × 1 dataset × 10 encoders = **70 experiments**
- 1 model × 13 datasets × 10 encoders = **130 experiments**
- 7 models × 13 datasets × 10 encoders = **910 experiments**
- 7 models × 13 datasets × 3 encoders = **273 experiments**

## Output Summary

The summary JSON file includes:

```json
{
  "experiment_info": {
    "models": ["TGAT", "TGN", "TCL"],
    "datasets": ["wikipedia", "reddit", "mooc"],
    "data_ratio": 0.1,
    "num_epochs": 50,
    "total_experiments": 90
  },
  "results": [
    {
      "model": "TGAT",
      "dataset": "wikipedia",
      "encoder_name": "kan_mammote_full",
      "success": true,
      "duration": 1932.5,
      "config": {...}
    },
    ...
  ]
}
```

The console output shows a formatted table:

```
================================================================================
ABLATION STUDY RESULTS SUMMARY
================================================================================
Model        Dataset         Encoder                   Status     Duration     Description
-------------------------------------------------------------------------------------------
TGAT         wikipedia       kan_mammote_full          ✅ SUCCESS  1932.5s     Full KAN-MAMMOTE for ref
TGAT         wikipedia       lete                      ✅ SUCCESS  1845.2s     LeTE encoder for compari
TGN          wikipedia       kan_mammote_full          ✅ SUCCESS  1672.8s     Full KAN-MAMMOTE for ref
...
-------------------------------------------------------------------------------------------
Total: 90 experiments, 87 successful, 3 failed
================================================================================
```

## Tips for Large-Scale Studies

1. **Start small**: Test with `--data_ratio 0.02` and `--num_epochs 2` first
2. **Use dry-run**: Always verify with `--dry_run` before large experiments
3. **Monitor resources**: Large studies may take days to complete
4. **Check intermediate results**: Results are saved after each experiment
5. **Use PBS/SLURM**: Submit as array jobs for parallel execution on HPC

## Parallel Execution on HPC

For large-scale studies, consider splitting into multiple jobs:

```bash
# Job 1: Models 1-3 on all datasets
python experiment_ablation.py --model TGAT TGN CAWN --all_datasets

# Job 2: Models 4-6 on all datasets  
python experiment_ablation.py --model TCL GraphMixer DyGFormer --all_datasets

# Job 3: Model 7 on all datasets
python experiment_ablation.py --model DyGMamba --all_datasets
```

Each job will create its own `ablation_*` directory with timestamped results.
