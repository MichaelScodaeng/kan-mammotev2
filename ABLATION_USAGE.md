# Ablation Study Usage Guide

## Overview

The `experiment_ablation.py` script now supports running experiments on **all encoders** and **all datasets** with flexible configuration options.

## Key Features

### 1. **Run All Encoders (Default)**
By default, the script runs all 10 encoder configurations:
- LeTE
- Bochner (Gaussian Fourier features)
- Mercer (harmonic eigenfunction expansion)  
- Original (cosine-based from DyGMamba)
- SM-Kernel Only
- K-MOTE Absolute Only
- K-MOTE Relative Only
- Dual Stream Baseline
- KAN-MAMMOTE Lite
- Full KAN-MAMMOTE

### 2. **Run All Datasets**
Use `--all_datasets` to run experiments on all 13 available datasets:
- wikipedia, reddit, mooc, lastfm, enron, SocialEvo, uci
- CanParl, Contacts, Flights, UNtrade, UNvote, USLegis

### 3. **Organized Output**
All outputs (models, metrics, results) are saved in a timestamped folder:
```
ablation_20251006_143022/
├── saved_models/
├── saved_metrics/
├── saved_results/
└── ablation_study_summary_TGAT_13datasets_20251006_143022.json
```

## Usage Examples

### Example 1: Single Dataset, All Encoders
```bash
python experiment_ablation.py --dataset wikipedia --data_ratio 0.1
```
Runs all 10 encoders on Wikipedia dataset (10% of data).

### Example 2: All Datasets, All Encoders
```bash
python experiment_ablation.py --all_datasets --data_ratio 0.1
```
Runs all 10 encoders on all 13 datasets = **130 total experiments**

### Example 3: Specific Encoders on One Dataset
```bash
python experiment_ablation.py --dataset reddit \
    --encoders kan_mammote_full lete bochner mercer \
    --data_ratio 0.1
```
Runs only 4 specific encoders on Reddit dataset.

### Example 4: All Encoders on Specific Datasets
You can run multiple datasets by calling the script separately:
```bash
for dataset in wikipedia reddit mooc; do
    python experiment_ablation.py --dataset $dataset --data_ratio 0.1
done
```

### Example 5: Different Model on All Datasets
```bash
python experiment_ablation.py --model DyGFormer --all_datasets --data_ratio 0.05
```
Tests DyGFormer backbone with all encoders on all datasets.

### Example 6: Dry Run (Preview Commands)
```bash
python experiment_ablation.py --all_datasets --data_ratio 0.1 --dry_run
```
Shows all commands that would be executed without actually running them.

### Example 7: Fast Testing
```bash
python experiment_ablation.py --dataset wikipedia \
    --encoders lete original \
    --num_epochs 2 \
    --data_ratio 0.02
```
Quick test with 2 encoders, 2 epochs, and 2% of data.

## Command-Line Arguments

### Required (pick one):
- `--dataset <name>`: Run on a specific dataset
- `--all_datasets`: Run on all 13 datasets

### Optional:
- `--model <name>`: Model backbone (default: TGAT)
  - Choices: TGAT, CAWN, TCL, GraphMixer, DyGFormer, DyGMamba
- `--encoders <list>`: Specific encoders to test (default: all)
  - Choices: lete, bochner, mercer, original, sm_kernel_only, kmote_abs_only, kmote_rel_only, dual_stream_baseline, kan_mammote_lite, kan_mammote_full
- `--data_ratio <float>`: Portion of data to use (default: 0.1)
- `--num_epochs <int>`: Training epochs (default: 10)
- `--batch_size <int>`: Batch size (default: 200)
- `--num_neighbors <int>`: Temporal neighbors (default: 20)
- `--learning_rate <float>`: Learning rate (default: 0.0001)
- `--dropout <float>`: Dropout rate (default: 0.1)
- `--tolerance <int>`: Early stopping patience (default: 5)
- `--verbose`: Enable verbose output
- `--dry_run`: Preview commands without execution

## Output Structure

After running, you'll get:

1. **Timestamped Ablation Directory**: `ablation_YYYYMMDD_HHMMSS/`
2. **Organized Subdirectories**:
   - `saved_models/`: Model checkpoints
   - `saved_metrics/`: Training/validation metrics CSVs
   - `saved_results/`: Final test results JSONs
3. **Summary JSON**: Complete experiment log with all results
4. **Terminal Summary**: Formatted table showing success/failure and duration

## Pro Tips

### For Quick Exploration:
```bash
python experiment_ablation.py --dataset wikipedia \
    --encoders lete original kan_mammote_lite \
    --data_ratio 0.05 \
    --num_epochs 5
```

### For Comprehensive Study:
```bash
python experiment_ablation.py --all_datasets \
    --data_ratio 0.1 \
    --num_epochs 50 \
    --tolerance 10
```

### For Production Encoders Only:
```bash
python experiment_ablation.py --all_datasets \
    --encoders kan_mammote_lite dual_stream_baseline lete \
    --data_ratio 1.0
```

## Monitoring Progress

During execution, you'll see:
```
[15/130] Running bochner on reddit...
[16/130] Running mercer on reddit...
...
```

At the end:
```
================================================================================
ABLATION STUDY RESULTS SUMMARY
================================================================================
Encoder                   Dataset         Status     Duration     Description
--------------------------------------------------------------------------------
lete                      wikipedia       ✅ SUCCESS  1234.5s      LeTE encoder
bochner                   wikipedia       ✅ SUCCESS  1156.2s      Bochner encoder
...
--------------------------------------------------------------------------------
Total: 130 experiments, 128 successful, 2 failed
================================================================================
```

## Troubleshooting

### Error: "You must specify either --dataset <name> or --all_datasets"
**Solution**: Provide either a specific dataset or use `--all_datasets`.

### Running out of memory?
**Solution**: Reduce `--data_ratio` or `--batch_size`.

### Too many experiments?
**Solution**: Use `--encoders` to select specific encoders or `--dataset` for a single dataset.

### Want to continue interrupted experiments?
**Solution**: Currently, each run is independent. Check the summary JSON to see which succeeded, then manually re-run failed ones.

## Performance Estimates

Approximate times (may vary):
- 1 experiment (1 encoder, 1 dataset, 10% data, 10 epochs): ~20-30 minutes
- 10 encoders on 1 dataset: ~3-5 hours
- 10 encoders on 13 datasets (130 total): ~40-65 hours

**Recommendation**: Start with `--data_ratio 0.1` and `--num_epochs 10` for initial exploration, then increase for final results.
