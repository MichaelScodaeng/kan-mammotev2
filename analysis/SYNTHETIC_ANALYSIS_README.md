# Synthetic Pattern Analysis for Time Encoders

This analysis comprehensively tests different time encoders on three types of synthetic data patterns, similar to the reference paper analysis.

## What it Tests

### 🏗️ Models Tested:
- **KAN-MAMMOTE** (main model)
- **K-MOTE** (expert mixture system)
- **K-MOTE Individual Experts**:
  - B-Spline KAN Expert
  - Fourier KAN Expert  
  - Wavelet KAN Expert
- **Baseline Time Encoders**:
  - Original Time Encoder
  - Mercer Time Encoder
  - Time2Vec Encoder
  - LeTE (Learnable Time Encoder)

### 📊 Data Patterns:
1. **Synthetic Periodic Data**: Multiple harmonic components with varying frequencies
2. **Synthetic Non-Periodic Data**: Exponential decays, step functions, spikes, random walks
3. **Synthetic Mixed Data**: Combination of periodic (60%) and non-periodic (40%) components

### 🎯 Training Strategy:
- **Convergence-based training** with shared hyperparameters
- **Shared config**: `lr=5e-4, patience=300, max_epochs=8000`
- **Fair comparison**: Each model trains until convergence
- **Reproducible**: Fixed random seeds

## Generated Outputs

### 📈 Visualizations:
- **Main comparison plot**: 3x3 grid showing patterns, reconstructions, and performance
- **Detailed pattern analysis**: Individual plots for each pattern type
- **Performance comparisons**: Bar charts and metrics
- **Training convergence**: Loss curves for convergence analysis

### 📊 Results Files:
- `synthetic_analysis_[timestamp].csv`: Detailed results for all experiments
- `synthetic_summary_[timestamp].csv`: Summary statistics by pattern
- Performance metrics: MSE, MAE, RMSE, R² scores
- Training info: Convergence epochs, training success

### 🎨 Key Features:
- **Pattern visualization**: Shows how each synthetic pattern is generated
- **Reconstruction quality**: Original vs predicted comparisons
- **Performance ranking**: Which models work best for which patterns
- **Expert analysis**: How K-MOTE experts specialize for different patterns
- **Statistical summary**: Comprehensive performance analysis

## Usage

```bash
cd /home/s2516027/kan-mammotev2/analysis
python synthetic_pattern_analysis.py
```

## Expected Insights

1. **Pattern Specialization**: Which models excel at periodic vs non-periodic patterns
2. **Expert Utilization**: How K-MOTE experts specialize for different data types
3. **Baseline Comparison**: How KAN-MAMMOTE/K-MOTE compare to traditional encoders
4. **Reconstruction Quality**: Quantitative performance across pattern types
5. **Training Efficiency**: Which models converge faster and more reliably

The analysis will help determine the best time encoder for different temporal pattern types and validate the effectiveness of the K-MOTE expert mixture approach.