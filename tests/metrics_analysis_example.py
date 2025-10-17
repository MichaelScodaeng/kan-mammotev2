"""
Comprehensive Metrics Analysis Example

This script demonstrates how to analyze all the different metrics 
saved during training, including new node validation metrics and 
periodic test metrics.

Usage:
    python metrics_analysis_example.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

sns.set_style("whitegrid")
sns.set_palette("husl")

def demonstrate_all_metrics():
    """Show how to load and analyze all available metrics types."""
    
    # Example configuration
    model_name = "TGAT"
    dataset_name = "wikipedia"
    encoder_type = "kan_mammote"
    run_id = 0
    
    metrics_dir = Path(f"./saved_metrics/{model_name}/{dataset_name}/{model_name}_{encoder_type}_seed{run_id}")
    
    print(f"🔍 Looking for metrics in: {metrics_dir}")
    
    if not metrics_dir.exists():
        print(f"❌ Metrics directory not found: {metrics_dir}")
        print("Run training first to generate metrics!")
        return
    
    # List all available metric files
    csv_files = list(metrics_dir.glob("*_metrics_*.csv"))
    if not csv_files:
        print("❌ No metrics CSV files found!")
        return
    
    print(f"\n📊 Available metrics files:")
    for csv_file in sorted(csv_files):
        file_size = csv_file.stat().st_size / 1024  # KB
        print(f"   - {csv_file.name} ({file_size:.1f} KB)")
    
    # Load all available metrics
    metrics_data = {}
    
    metric_types = [
        'train', 'val', 'new_node_val', 
        'test', 'new_node_test', 
        'test_periodic', 'new_node_test_periodic'
    ]
    
    for metric_type in metric_types:
        pattern = f"{metric_type}_metrics_*.csv"
        files = list(metrics_dir.glob(pattern))
        if files:
            # Get most recent file
            latest_file = max(files, key=lambda p: p.stat().st_mtime)
            try:
                df = pd.read_csv(latest_file)
                metrics_data[metric_type] = df
                print(f"✅ Loaded {metric_type}: {len(df)} rows")
            except Exception as e:
                print(f"❌ Error loading {metric_type}: {e}")
        else:
            print(f"⚠️  No {metric_type} metrics found")
    
    if not metrics_data:
        print("❌ No metrics could be loaded!")
        return
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # Plot 1: Training Progress (Train vs Val)
    if 'train' in metrics_data and 'val' in metrics_data:
        ax1 = plt.subplot(2, 3, 1)
        train_df = metrics_data['train']
        val_df = metrics_data['val']
        
        if 'average_precision' in train_df.columns:
            ax1.plot(train_df['epoch'], train_df['average_precision'], 
                    label='Train AP', marker='o', alpha=0.7)
        if 'average_precision' in val_df.columns:
            ax1.plot(val_df['epoch'], val_df['average_precision'], 
                    label='Val AP', marker='s', alpha=0.7)
        
        ax1.set_title('Training Progress: Average Precision')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Average Precision')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Plot 2: New Node Validation
    if 'val' in metrics_data and 'new_node_val' in metrics_data:
        ax2 = plt.subplot(2, 3, 2)
        val_df = metrics_data['val']
        new_node_val_df = metrics_data['new_node_val']
        
        if 'average_precision' in val_df.columns:
            ax2.plot(val_df['epoch'], val_df['average_precision'], 
                    label='Standard Val AP', marker='o', alpha=0.7)
        if 'average_precision' in new_node_val_df.columns:
            ax2.plot(new_node_val_df['epoch'], new_node_val_df['average_precision'], 
                    label='New Node Val AP', marker='^', alpha=0.7)
        
        ax2.set_title('Validation: Standard vs New Nodes')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Average Precision')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss Evolution
    ax3 = plt.subplot(2, 3, 3)
    for metric_type, df in metrics_data.items():
        if 'loss' in df.columns and metric_type in ['train', 'val']:
            ax3.plot(df['epoch'], df['loss'], 
                    label=f'{metric_type.title()} Loss', 
                    marker='o' if metric_type == 'train' else 's', alpha=0.7)
    
    ax3.set_title('Loss Evolution')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Test Performance Over Time (if periodic testing)
    if 'test_periodic' in metrics_data:
        ax4 = plt.subplot(2, 3, 4)
        test_periodic_df = metrics_data['test_periodic']
        
        if 'average_precision' in test_periodic_df.columns:
            ax4.plot(test_periodic_df['epoch'], test_periodic_df['average_precision'], 
                    label='Test AP (Periodic)', marker='d', alpha=0.7, color='red')
        
        # Add final test if available
        if 'test' in metrics_data:
            test_df = metrics_data['test']
            if 'average_precision' in test_df.columns:
                ax4.scatter(test_df['epoch'], test_df['average_precision'], 
                           label='Final Test AP', marker='*', s=200, color='darkred', alpha=0.8)
        
        ax4.set_title('Test Performance During Training')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Average Precision')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # Plot 5: ROC-AUC Comparison
    ax5 = plt.subplot(2, 3, 5)
    for metric_type, df in metrics_data.items():
        if 'roc_auc' in df.columns and metric_type in ['train', 'val', 'new_node_val']:
            ax5.plot(df['epoch'], df['roc_auc'], 
                    label=f'{metric_type.replace("_", " ").title()} ROC-AUC', 
                    marker='o', alpha=0.7)
    
    ax5.set_title('ROC-AUC Comparison')
    ax5.set_xlabel('Epoch')
    ax5.set_ylabel('ROC-AUC')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Final Performance Summary (Bar Chart)
    ax6 = plt.subplot(2, 3, 6)
    
    final_metrics = {}
    metric_names = ['average_precision', 'roc_auc']
    
    for metric_type, df in metrics_data.items():
        if len(df) > 0:
            if metric_type in ['test', 'new_node_test']:
                # Use final values
                final_row = df.iloc[-1]
            else:
                # Use best validation values
                if 'average_precision' in df.columns:
                    best_idx = df['average_precision'].idxmax()
                    final_row = df.iloc[best_idx]
                else:
                    final_row = df.iloc[-1]
            
            for metric_name in metric_names:
                if metric_name in final_row:
                    key = f"{metric_type}_{metric_name}"
                    final_metrics[key] = final_row[metric_name]
    
    if final_metrics:
        # Group by metric type for better visualization
        x_pos = np.arange(len(metric_names))
        width = 0.12
        
        metric_types_to_plot = ['val', 'new_node_val', 'test', 'new_node_test']
        colors = ['blue', 'green', 'red', 'orange']
        
        for i, metric_type in enumerate(metric_types_to_plot):
            values = []
            for metric_name in metric_names:
                key = f"{metric_type}_{metric_name}"
                values.append(final_metrics.get(key, 0))
            
            if any(v > 0 for v in values):
                ax6.bar(x_pos + i*width, values, width, 
                       label=metric_type.replace('_', ' ').title(), 
                       color=colors[i], alpha=0.7)
        
        ax6.set_title('Final Performance Summary')
        ax6.set_xlabel('Metrics')
        ax6.set_ylabel('Score')
        ax6.set_xticks(x_pos + width * 1.5)
        ax6.set_xticklabels([m.replace('_', ' ').title() for m in metric_names])
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Comprehensive Metrics Analysis: {encoder_type.upper()} on {dataset_name}', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("./analysis_figures")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / f"comprehensive_metrics_{model_name}_{dataset_name}_{encoder_type}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n💾 Comprehensive analysis saved: {output_path}")
    
    plt.show()
    
    # Print summary statistics
    print(f"\n📊 METRICS SUMMARY:")
    print(f"={'='*80}")
    
    for metric_type, df in metrics_data.items():
        if len(df) == 0:
            continue
            
        print(f"\n{metric_type.upper().replace('_', ' ')} METRICS:")
        print(f"-" * 50)
        
        # Show final or best values
        if metric_type in ['test', 'new_node_test', 'test_periodic', 'new_node_test_periodic']:
            # Use final values for test metrics
            final_row = df.iloc[-1]
            print(f"  Final epoch: {int(final_row['epoch'])}")
        else:
            # Use best validation epoch
            if 'average_precision' in df.columns:
                best_idx = df['average_precision'].idxmax()
                final_row = df.iloc[best_idx]
                print(f"  Best epoch: {int(final_row['epoch'])}")
            else:
                final_row = df.iloc[-1]
                print(f"  Final epoch: {int(final_row['epoch'])}")
        
        for col in df.columns:
            if col != 'epoch':
                print(f"  {col:20s}: {final_row[col]:.6f}")
        
        if metric_type == 'test_periodic':
            print(f"  Total evaluations: {len(df)}")


if __name__ == "__main__":
    demonstrate_all_metrics()