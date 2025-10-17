# Spectral Entropy Analysis for Dynamic Graphs

This analysis replicates and extends Figure 8 from your paper, computing spectral entropy for temporal interaction patterns across all 13 datasets.

## 📊 What it does

The analysis computes spectral entropy using Fast Fourier Transform (FFT) on:
1. **Normalized timestamps** of interactions for each node
2. **Normalized time differences** between consecutive interactions

**Spectral entropy formula**: `H(P) = -Σf P(f) log P(f)`
- Lower entropy = more periodic/regular patterns  
- Higher entropy = more random/irregular patterns

## 🗂️ Datasets Analyzed

All 13 datasets from your configuration:
`wikipedia`, `reddit`, `mooc`, `lastfm`, `enron`, `SocialEvo`, `uci`, `CanParl`, `Contacts`, `Flights`, `UNtrade`, `UNvote`, `USLegis`

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_spectral_entropy.txt
```

### 2. Run Complete Analysis
```bash
# Run analysis for all datasets
python run_spectral_entropy_analysis.py --data_root ./data --output_dir ./spectral_entropy_results

# Run for specific datasets only
python run_spectral_entropy_analysis.py --datasets wikipedia reddit mooc --data_root ./data

# Skip analysis and only create plots (if results exist)
python run_spectral_entropy_analysis.py --skip_analysis --output_dir ./spectral_entropy_results
```

### 3. Run Individual Components

**Analysis only:**
```bash
python spectral_entropy_analysis.py --data_root ./data --output_dir ./results
```

**Visualization only:**
```bash
python spectral_entropy_visualizer.py --results_file ./results/spectral_entropy_results.pkl
```

## 📈 Generated Outputs

### Files Created:
- `spectral_entropy_results.pkl` - Raw analysis results
- `spectral_entropy_density_plots.pdf/png` - Main Figure 8 style plots
- `spectral_entropy_summary_statistics.png` - Dataset comparison charts
- `spectral_entropy_heatmap.png` - Cross-dataset entropy comparison
- `spectral_entropy_statistics.csv` - Numerical results table
- `spectral_entropy_analysis_report.txt` - Comprehensive text report

### Main Visualizations:

1. **Density Plots** (Figure 8 style)
   - Left: Spectral entropy of interaction timestamps
   - Right: Spectral entropy of time differences
   - Shows all 13 datasets with different colors

2. **Summary Statistics**
   - Bar charts comparing datasets
   - Node counts, interaction counts, average entropies

3. **Comparison Heatmap**
   - Matrix view of entropy statistics across datasets
   - Easy identification of patterns

## ⚙️ Configuration Options

```bash
--data_root          # Directory containing dataset CSV files (default: ./data)
--output_dir         # Where to save results (default: ./spectral_entropy_results)  
--min_interactions   # Min interactions per node to include (default: 5)
--datasets          # Specific datasets to analyze (default: all 13)
--skip_analysis     # Only create visualizations from existing results
--skip_visualization # Only run analysis, skip plots
```

## 📁 Expected Data Format

The script looks for CSV files in this priority order:
1. `ml_{dataset_name}.csv` (primary format)
2. `{dataset_name}/ml_{dataset_name}.csv` 
3. `{dataset_name}.csv`
4. `{dataset_name}/{dataset_name}.csv`
5. `{dataset_name}/edges.csv`

Expected columns:
- `u` or `source`: Source node ID
- `i` or `target`: Target node ID  
- `ts` or `timestamp`: Interaction timestamp
- `label`: Edge label (optional, defaults to 0)

Alternative column names are automatically mapped:
- `src`, `from`, `node1` → `u`
- `dst`, `to`, `node2` → `i`
- `time`, `t` → `ts`

## 🔍 Key Insights

The analysis reveals:
- **Most nodes show high entropy** → non-periodic temporal patterns
- **Small fraction exhibits periodicity** → regular interaction timing
- **Time differences are often less predictable** than absolute timestamps
- **Dataset-specific patterns** in temporal regularity

## 📊 Interpretation

- **Low entropy (< 2.0)**: Periodic, predictable interaction patterns
- **High entropy (> 6.0)**: Random, irregular interaction timing
- **Medium entropy (2.0-6.0)**: Semi-regular patterns with some randomness

This analysis helps understand whether time encoding methods should focus on:
- **Periodic pattern capture** (for low-entropy datasets)
- **General temporal modeling** (for high-entropy datasets)
- **Hybrid approaches** (for mixed patterns)

## 🛠️ Troubleshooting

**Data loading issues:**
- Check file paths and CSV format
- Ensure timestamp columns contain numeric values
- Verify column names match expected format

**Memory issues:**
- Large datasets may require more RAM
- Consider reducing `min_interactions` threshold
- Process datasets individually if needed

**Visualization issues:**
- Install latest matplotlib/seaborn versions
- Check display backend for remote servers
- Save plots will work even if display fails