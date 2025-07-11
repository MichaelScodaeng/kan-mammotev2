#!/usr/bin/env python3
"""
🔬 Advanced LSTM Time Embedding Analysis
=======================================

This script provides detailed analysis and ablation studies for the LSTM time embedding comparison.
It includes:

1. **Detailed Performance Analysis**: Loss curves, convergence analysis, training dynamics
2. **Ablation Studies**: Effect of different hyperparameters and components
3. **Temporal Pattern Analysis**: How well each method captures temporal dependencies
4. **Statistical Significance Testing**: Rigorous comparison of model performances
5. **Visualization**: Comprehensive plots and analysis dashboards

Author: Generated for KAN-MAMMOTE Project
Date: July 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import classification_report, confusion_matrix
import json
import pandas as pd
from tqdm import tqdm
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = '/home/s2516027/kan-mammote'
if project_root not in sys.path:
    sys.path.append(project_root)

from lstm_embedding_comparison import *

print("🔬 Advanced LSTM Time Embedding Analysis")
print("=" * 60)

# ============================================================================
# 📊 ANALYSIS FUNCTIONS
# ============================================================================

def analyze_convergence(results):
    """Analyze convergence properties of different models."""
    convergence_analysis = {}
    
    for model_name, history in results.items():
        test_acc = np.array(history['test_acc'])
        train_acc = np.array(history['train_acc'])
        
        # Find convergence epoch (when test accuracy stabilizes)
        # Use sliding window to detect when improvement becomes minimal
        window_size = 5
        convergence_epoch = NUM_EPOCHS
        
        for i in range(window_size, len(test_acc)):
            window = test_acc[i-window_size:i]
            if np.std(window) < 0.005:  # Very small variation
                convergence_epoch = i
                break
        
        # Calculate overfitting (train acc - test acc)
        overfitting = train_acc - test_acc
        max_overfitting = np.max(overfitting)
        final_overfitting = overfitting[-1]
        
        # Training stability (variance in last 10 epochs)
        late_stability = np.std(test_acc[-10:])
        
        convergence_analysis[model_name] = {
            'convergence_epoch': convergence_epoch,
            'max_overfitting': max_overfitting,
            'final_overfitting': final_overfitting,
            'late_stability': late_stability,
            'final_test_acc': test_acc[-1],
            'best_test_acc': np.max(test_acc),
            'epoch_of_best': np.argmax(test_acc) + 1
        }
    
    return convergence_analysis

def analyze_temporal_patterns(model, test_loader, model_name):
    """Analyze how well models capture temporal patterns."""
    model.eval()
    
    # Collect sequence lengths and accuracies
    length_accuracies = {}
    
    with torch.no_grad():
        for events, features, lengths, labels in tqdm(test_loader, desc=f"Analyzing {model_name}"):
            events = events.to(device)
            features = features.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            outputs = model(events, features, lengths)
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == labels).cpu().numpy()
            
            # Group by sequence length
            for length, is_correct in zip(lengths.cpu().numpy(), correct):
                if length not in length_accuracies:
                    length_accuracies[length] = []
                length_accuracies[length].append(is_correct)
    
    # Calculate accuracy by sequence length
    length_stats = {}
    for length, correct_list in length_accuracies.items():
        length_stats[length] = {
            'accuracy': np.mean(correct_list),
            'count': len(correct_list),
            'std': np.std(correct_list)
        }
    
    return length_stats

def statistical_significance_test(results):
    """Perform statistical significance testing between models."""
    model_names = list(results.keys())
    n_models = len(model_names)
    
    # Extract final 10 epochs for each model (more stable comparison)
    final_accuracies = {}
    for name in model_names:
        final_accuracies[name] = results[name]['test_acc'][-10:]
    
    # Pairwise t-tests
    p_values = np.zeros((n_models, n_models))
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i != j:
                t_stat, p_val = stats.ttest_ind(
                    final_accuracies[name1], 
                    final_accuracies[name2]
                )
                p_values[i, j] = p_val
    
    return p_values, model_names

def detailed_error_analysis(models, test_loader):
    """Perform detailed error analysis for each model."""
    error_analysis = {}
    
    for model_name, model in models.items():
        model.eval()
        
        all_predictions = []
        all_labels = []
        all_lengths = []
        
        with torch.no_grad():
            for events, features, lengths, labels in tqdm(test_loader, desc=f"Error analysis {model_name}"):
                events = events.to(device)
                features = features.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                
                outputs = model(events, features, lengths)
                predictions = torch.argmax(outputs, dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_lengths.extend(lengths.cpu().numpy())
        
        # Classification report
        class_report = classification_report(all_labels, all_predictions, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_predictions)
        
        # Error by sequence length
        df = pd.DataFrame({
            'predictions': all_predictions,
            'labels': all_labels,
            'lengths': all_lengths
        })
        
        # Group by length and calculate accuracy
        length_accuracy = df.groupby('lengths').apply(
            lambda x: (x['predictions'] == x['labels']).mean()
        ).to_dict()
        
        error_analysis[model_name] = {
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'length_accuracy': length_accuracy,
            'overall_accuracy': np.mean(np.array(all_predictions) == np.array(all_labels))
        }
    
    return error_analysis

# ============================================================================
# 🎨 VISUALIZATION FUNCTIONS
# ============================================================================

def plot_convergence_analysis(convergence_analysis):
    """Plot convergence analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    models = list(convergence_analysis.keys())
    
    # Convergence epochs
    conv_epochs = [convergence_analysis[m]['convergence_epoch'] for m in models]
    axes[0, 0].bar(models, conv_epochs, color=['blue', 'green', 'red', 'orange'])
    axes[0, 0].set_title('Convergence Epochs')
    axes[0, 0].set_ylabel('Epoch')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Overfitting analysis
    max_overfit = [convergence_analysis[m]['max_overfitting'] for m in models]
    final_overfit = [convergence_analysis[m]['final_overfitting'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, max_overfit, width, label='Max Overfitting', alpha=0.7)
    axes[0, 1].bar(x + width/2, final_overfit, width, label='Final Overfitting', alpha=0.7)
    axes[0, 1].set_title('Overfitting Analysis')
    axes[0, 1].set_ylabel('Train Acc - Test Acc')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45)
    axes[0, 1].legend()
    
    # Training stability
    stability = [convergence_analysis[m]['late_stability'] for m in models]
    axes[1, 0].bar(models, stability, color=['blue', 'green', 'red', 'orange'])
    axes[1, 0].set_title('Training Stability (Last 10 Epochs)')
    axes[1, 0].set_ylabel('Std Dev of Test Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Best vs Final accuracy
    best_acc = [convergence_analysis[m]['best_test_acc'] for m in models]
    final_acc = [convergence_analysis[m]['final_test_acc'] for m in models]
    
    axes[1, 1].bar(x - width/2, best_acc, width, label='Best Test Acc', alpha=0.7)
    axes[1, 1].bar(x + width/2, final_acc, width, label='Final Test Acc', alpha=0.7)
    axes[1, 1].set_title('Best vs Final Accuracy')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(models, rotation=45)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/convergence_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_temporal_patterns(temporal_analyses):
    """Plot temporal pattern analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Accuracy by sequence length
    for i, (model_name, length_stats) in enumerate(temporal_analyses.items()):
        lengths = sorted(length_stats.keys())
        accuracies = [length_stats[l]['accuracy'] for l in lengths]
        counts = [length_stats[l]['count'] for l in lengths]
        
        # Only plot lengths with sufficient samples
        valid_lengths = [l for l, c in zip(lengths, counts) if c >= 10]
        valid_accuracies = [length_stats[l]['accuracy'] for l in valid_lengths]
        
        axes[0, 0].plot(valid_lengths, valid_accuracies, 
                       marker='o', label=model_name, color=colors[i])
    
    axes[0, 0].set_title('Accuracy by Sequence Length')
    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sample count distribution
    for i, (model_name, length_stats) in enumerate(temporal_analyses.items()):
        if i == 0:  # Only show distribution once (same for all models)
            lengths = sorted(length_stats.keys())
            counts = [length_stats[l]['count'] for l in lengths]
            axes[0, 1].bar(lengths, counts, alpha=0.7)
            break
    
    axes[0, 1].set_title('Sample Count by Sequence Length')
    axes[0, 1].set_xlabel('Sequence Length')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy correlation with length
    for i, (model_name, length_stats) in enumerate(temporal_analyses.items()):
        lengths = sorted(length_stats.keys())
        accuracies = [length_stats[l]['accuracy'] for l in lengths]
        counts = [length_stats[l]['count'] for l in lengths]
        
        # Filter for reliable estimates
        valid_data = [(l, a) for l, a, c in zip(lengths, accuracies, counts) if c >= 10]
        if valid_data:
            valid_lengths, valid_accuracies = zip(*valid_data)
            correlation = np.corrcoef(valid_lengths, valid_accuracies)[0, 1]
            axes[1, 0].bar(i, correlation, color=colors[i], alpha=0.7)
    
    axes[1, 0].set_title('Length-Accuracy Correlation')
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('Correlation')
    axes[1, 0].set_xticks(range(len(temporal_analyses)))
    axes[1, 0].set_xticklabels(list(temporal_analyses.keys()), rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance summary
    model_names = list(temporal_analyses.keys())
    overall_scores = []
    
    for model_name in model_names:
        # Calculate weighted average accuracy
        length_stats = temporal_analyses[model_name]
        total_correct = sum(stats['accuracy'] * stats['count'] for stats in length_stats.values())
        total_samples = sum(stats['count'] for stats in length_stats.values())
        overall_acc = total_correct / total_samples
        overall_scores.append(overall_acc)
    
    axes[1, 1].bar(model_names, overall_scores, color=colors[:len(model_names)], alpha=0.7)
    axes[1, 1].set_title('Overall Accuracy (Weighted)')
    axes[1, 1].set_xlabel('Model')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/temporal_patterns.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrices(error_analysis):
    """Plot confusion matrices for all models."""
    n_models = len(error_analysis)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (model_name, analysis) in enumerate(error_analysis.items()):
        conf_matrix = analysis['confusion_matrix']
        
        # Normalize confusion matrix
        conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        
        # Plot
        sns.heatmap(conf_matrix_norm, annot=True, fmt='.2f', cmap='Blues', 
                   ax=axes[i], cbar=True)
        axes[i].set_title(f'{model_name} Confusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_statistical_significance(p_values, model_names):
    """Plot statistical significance matrix."""
    plt.figure(figsize=(10, 8))
    
    # Create mask for significance (p < 0.05)
    significance_mask = p_values < 0.05
    
    # Plot heatmap
    sns.heatmap(p_values, annot=True, fmt='.3f', cmap='RdYlBu_r',
                xticklabels=model_names, yticklabels=model_names,
                mask=~significance_mask, cbar_kws={'label': 'p-value'})
    
    plt.title('Statistical Significance Test (p-values)\nDark cells: p < 0.05 (significant)')
    plt.xlabel('Model')
    plt.ylabel('Model')
    plt.tight_layout()
    plt.savefig(f"{RESULTS_DIR}/statistical_significance.png", dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 🎯 MAIN ANALYSIS EXECUTION
# ============================================================================

def run_comprehensive_analysis():
    """Run comprehensive analysis of the LSTM time embedding comparison."""
    print("📊 Loading previous results...")
    
    # Load results from previous run
    if not os.path.exists(f"{RESULTS_DIR}/training_histories.json"):
        print("❌ No previous results found. Please run lstm_embedding_comparison.py first.")
        return
    
    with open(f"{RESULTS_DIR}/training_histories.json", 'r') as f:
        results = json.load(f)
    
    print("✅ Results loaded successfully")
    
    # Create device and load models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Recreate models for detailed analysis
    models = {
        'Baseline_LSTM': BaselineLSTM(),
        'SinCos_LSTM': SinCosLSTM(),
        'LETE_LSTM': LETE_LSTM(),
        'KAN_MAMMOTE_LSTM': KAN_MAMMOTE_LSTM()
    }
    
    # Load trained models
    for model_name, model in models.items():
        model_path = f"{RESULTS_DIR}/{model_name}_best.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            print(f"✅ Loaded {model_name}")
        else:
            print(f"⚠️ Could not find saved model for {model_name}")
    
    # Create test dataset
    test_dataset = EventBasedMNIST(root='./data', train=False, threshold=THRESHOLD)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    print("\n🔍 Running convergence analysis...")
    convergence_analysis = analyze_convergence(results)
    
    print("\n🔍 Running temporal pattern analysis...")
    temporal_analyses = {}
    for model_name, model in models.items():
        if os.path.exists(f"{RESULTS_DIR}/{model_name}_best.pth"):
            temporal_analyses[model_name] = analyze_temporal_patterns(model, test_loader, model_name)
    
    print("\n🔍 Running statistical significance testing...")
    p_values, model_names = statistical_significance_test(results)
    
    print("\n🔍 Running detailed error analysis...")
    error_analysis = detailed_error_analysis(models, test_loader)
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    plot_convergence_analysis(convergence_analysis)
    plot_temporal_patterns(temporal_analyses)
    plot_confusion_matrices(error_analysis)
    plot_statistical_significance(p_values, model_names)
    
    # Save detailed analysis results
    print("\n💾 Saving detailed analysis results...")
    
    detailed_results = {
        'convergence_analysis': convergence_analysis,
        'temporal_analyses': temporal_analyses,
        'statistical_significance': {
            'p_values': p_values.tolist(),
            'model_names': model_names
        },
        'error_analysis': {
            name: {
                'classification_report': analysis['classification_report'],
                'overall_accuracy': analysis['overall_accuracy'],
                'length_accuracy': analysis['length_accuracy']
            }
            for name, analysis in error_analysis.items()
        }
    }
    
    with open(f"{RESULTS_DIR}/detailed_analysis.json", 'w') as f:
        json.dump(detailed_results, f, indent=2)
    
    # Create summary report
    print("\n📋 Creating summary report...")
    create_summary_report(results, convergence_analysis, temporal_analyses, p_values, model_names)
    
    print("\n🎉 Comprehensive analysis complete!")
    print(f"📁 All results saved to: {RESULTS_DIR}")

def create_summary_report(results, convergence_analysis, temporal_analyses, p_values, model_names):
    """Create a comprehensive summary report."""
    report = []
    report.append("# LSTM Time Embedding Comparison - Detailed Analysis Report")
    report.append("=" * 70)
    report.append("")
    
    # Configuration summary
    report.append("## Configuration")
    report.append(f"- Batch Size: {BATCH_SIZE}")
    report.append(f"- Learning Rate: {LEARNING_RATE}")
    report.append(f"- Epochs: {NUM_EPOCHS}")
    report.append(f"- LSTM Hidden Dim: {LSTM_HIDDEN_DIM}")
    report.append(f"- Time Embedding Dim: {TIME_EMBEDDING_DIM}")
    report.append("")
    
    # Performance summary
    report.append("## Performance Summary")
    report.append("| Model | Best Acc | Final Acc | Convergence Epoch | Stability |")
    report.append("|-------|----------|-----------|-------------------|-----------|")
    
    for model_name in model_names:
        best_acc = convergence_analysis[model_name]['best_test_acc']
        final_acc = convergence_analysis[model_name]['final_test_acc']
        conv_epoch = convergence_analysis[model_name]['convergence_epoch']
        stability = convergence_analysis[model_name]['late_stability']
        
        report.append(f"| {model_name} | {best_acc:.4f} | {final_acc:.4f} | {conv_epoch} | {stability:.4f} |")
    
    report.append("")
    
    # Statistical significance
    report.append("## Statistical Significance")
    report.append("Models with significant differences (p < 0.05):")
    
    for i, name1 in enumerate(model_names):
        for j, name2 in enumerate(model_names):
            if i < j and p_values[i, j] < 0.05:
                report.append(f"- {name1} vs {name2}: p = {p_values[i, j]:.4f}")
    
    report.append("")
    
    # Recommendations
    report.append("## Recommendations")
    
    # Find best model
    best_model = max(convergence_analysis.keys(), 
                    key=lambda x: convergence_analysis[x]['best_test_acc'])
    
    # Find most stable model
    most_stable = min(convergence_analysis.keys(),
                     key=lambda x: convergence_analysis[x]['late_stability'])
    
    # Find fastest converging model
    fastest_converging = min(convergence_analysis.keys(),
                           key=lambda x: convergence_analysis[x]['convergence_epoch'])
    
    report.append(f"1. **Best Overall Performance**: {best_model}")
    report.append(f"   - Achieves highest accuracy: {convergence_analysis[best_model]['best_test_acc']:.4f}")
    report.append("")
    
    report.append(f"2. **Most Stable Training**: {most_stable}")
    report.append(f"   - Lowest training variance: {convergence_analysis[most_stable]['late_stability']:.4f}")
    report.append("")
    
    report.append(f"3. **Fastest Convergence**: {fastest_converging}")
    report.append(f"   - Converges by epoch: {convergence_analysis[fastest_converging]['convergence_epoch']}")
    report.append("")
    
    # Save report
    with open(f"{RESULTS_DIR}/analysis_report.md", 'w') as f:
        f.write('\n'.join(report))
    
    print("📋 Summary report created!")

# ============================================================================
# 🎯 MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    run_comprehensive_analysis()
