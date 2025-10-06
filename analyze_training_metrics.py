"""
Analyze and Visualize Training Metrics

This script loads and visualizes metrics saved during training,
allowing you to compare different time encoders and analyze training dynamics.

Usage:
    python analyze_training_metrics.py --model TGAT --dataset wikipedia --encoder kan_mammote
    python analyze_training_metrics.py --compare_encoders kan_mammote lete original
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_metrics(model_name, dataset_name, encoder_type, run_id=0, phase='val'):
    """Load metrics from CSV file."""
    metrics_dir = Path(f"./saved_metrics/{model_name}/{dataset_name}/{model_name}_{encoder_type}_seed{run_id}")
    
    # Find the most recent metrics file
    pattern = f"{phase}_metrics_*.csv"
    csv_files = list(metrics_dir.glob(pattern))
    
    if not csv_files:
        print(f"⚠️  No metrics found for {model_name}/{dataset_name}/{encoder_type} (phase={phase})")
        return None
    
    # Get most recent file
    csv_file = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"📂 Loading: {csv_file}")
    
    return pd.read_csv(csv_file)


def plot_single_encoder(model_name, dataset_name, encoder_type, run_id=0):
    """Plot training and validation metrics for a single encoder."""
    
    train_df = load_metrics(model_name, dataset_name, encoder_type, run_id, 'train')
    val_df = load_metrics(model_name, dataset_name, encoder_type, run_id, 'val')
    
    if train_df is None or val_df is None:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} - {dataset_name} - {encoder_type}', fontsize=16, fontweight='bold')
    
    # Plot 1: Loss
    axes[0, 0].plot(train_df['epoch'], train_df['loss'], label='Train', marker='o', alpha=0.7)
    axes[0, 0].plot(val_df['epoch'], val_df['loss'], label='Validation', marker='s', alpha=0.7)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Average Precision
    if 'average_precision' in train_df.columns:
        axes[0, 1].plot(train_df['epoch'], train_df['average_precision'], label='Train', marker='o', alpha=0.7)
        axes[0, 1].plot(val_df['epoch'], val_df['average_precision'], label='Validation', marker='s', alpha=0.7)
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Average Precision')
        axes[0, 1].set_title('Average Precision (AP)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Mark best validation epoch
        best_epoch = val_df.loc[val_df['average_precision'].idxmax(), 'epoch']
        best_ap = val_df['average_precision'].max()
        axes[0, 1].axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5, label=f'Best: Epoch {int(best_epoch)}')
        axes[0, 1].legend()
    
    # Plot 3: ROC-AUC
    if 'roc_auc' in train_df.columns:
        axes[1, 0].plot(train_df['epoch'], train_df['roc_auc'], label='Train', marker='o', alpha=0.7)
        axes[1, 0].plot(val_df['epoch'], val_df['roc_auc'], label='Validation', marker='s', alpha=0.7)
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('ROC-AUC')
        axes[1, 0].set_title('ROC-AUC Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Overfitting Analysis (Train-Val Gap)
    if 'average_precision' in train_df.columns:
        gap = train_df['average_precision'] - val_df['average_precision']
        axes[1, 1].plot(train_df['epoch'], gap, marker='o', color='purple', alpha=0.7)
        axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Train AP - Val AP')
        axes[1, 1].set_title('Overfitting Analysis (AP Gap)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].fill_between(train_df['epoch'], 0, gap, where=(gap > 0), alpha=0.3, color='red', label='Overfitting')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("./analysis_figures")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"{model_name}_{dataset_name}_{encoder_type}_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    plt.show()


def compare_encoders(model_name, dataset_name, encoder_types, run_id=0, metric='average_precision'):
    """Compare multiple encoders on the same plot."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Encoder Comparison: {model_name} on {dataset_name}', fontsize=16, fontweight='bold')
    
    colors = sns.color_palette("husl", len(encoder_types))
    
    for i, encoder_type in enumerate(encoder_types):
        val_df = load_metrics(model_name, dataset_name, encoder_type, run_id, 'val')
        
        if val_df is None:
            continue
        
        if metric not in val_df.columns:
            print(f"⚠️  Metric '{metric}' not found for {encoder_type}")
            continue
        
        # Plot validation metric
        axes[0].plot(val_df['epoch'], val_df[metric], 
                    label=encoder_type, marker='o', alpha=0.7, color=colors[i])
        
        # Plot validation loss
        axes[1].plot(val_df['epoch'], val_df['loss'], 
                    label=encoder_type, marker='s', alpha=0.7, color=colors[i])
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel(metric.replace('_', ' ').title())
    axes[0].set_title(f'Validation {metric.replace("_", " ").title()}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path("./analysis_figures")
    output_dir.mkdir(exist_ok=True)
    encoders_str = '_vs_'.join(encoder_types)
    output_path = output_dir / f"{model_name}_{dataset_name}_comparison_{encoders_str}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved: {output_path}")
    plt.show()


def print_summary(model_name, dataset_name, encoder_type, run_id=0):
    """Print summary statistics."""
    
    val_df = load_metrics(model_name, dataset_name, encoder_type, run_id, 'val')
    test_df = load_metrics(model_name, dataset_name, encoder_type, run_id, 'test')
    
    if val_df is None:
        return
    
    print(f"\n{'='*80}")
    print(f"SUMMARY: {model_name} - {dataset_name} - {encoder_type}")
    print(f"{'='*80}")
    
    print(f"\n📊 Validation Metrics (Best Epoch):")
    for col in val_df.columns:
        if col != 'epoch':
            best_idx = val_df[col].idxmax() if col != 'loss' else val_df[col].idxmin()
            best_value = val_df.loc[best_idx, col]
            best_epoch = val_df.loc[best_idx, 'epoch']
            print(f"  {col:20s}: {best_value:.6f} (epoch {int(best_epoch)})")
    
    if test_df is not None and len(test_df) > 0:
        print(f"\n📊 Test Metrics (Final):")
        test_row = test_df.iloc[-1]
        for col in test_df.columns:
            if col != 'epoch':
                print(f"  {col:20s}: {test_row[col]:.6f}")
    
    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze training metrics')
    parser.add_argument('--model', type=str, default='TGAT', help='Model name')
    parser.add_argument('--dataset', type=str, default='wikipedia', help='Dataset name')
    parser.add_argument('--encoder', type=str, help='Single encoder to analyze')
    parser.add_argument('--compare_encoders', nargs='+', help='List of encoders to compare')
    parser.add_argument('--run_id', type=int, default=0, help='Run/seed ID')
    parser.add_argument('--metric', type=str, default='average_precision', 
                       help='Metric to compare (for comparison mode)')
    parser.add_argument('--summary', action='store_true', help='Print summary only')
    parser.add_argument('--phase', type=str, default='val', 
                       choices=['train', 'val', 'new_node_val', 'test', 'new_node_test', 'test_periodic', 'new_node_test_periodic'],
                       help='Phase to analyze')
    
    args = parser.parse_args()
    
    if args.encoder:
        if args.summary:
            print_summary(args.model, args.dataset, args.encoder, args.run_id)
        else:
            print(f"\n🎯 Analyzing {args.encoder} (phase: {args.phase})...")
            plot_single_encoder(args.model, args.dataset, args.encoder, args.run_id)
            print_summary(args.model, args.dataset, args.encoder, args.run_id)
    
    elif args.compare_encoders:
        print(f"\n📊 Comparing encoders: {', '.join(args.compare_encoders)} (phase: {args.phase})...")
        compare_encoders(args.model, args.dataset, args.compare_encoders, args.run_id, args.metric)
    
    else:
        print("⚠️  Please specify --encoder or --compare_encoders")
        parser.print_help()


if __name__ == "__main__":
    main()
