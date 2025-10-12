"""
Training History Visualization Script
====================================

This script generates publication-quality visualizations of training history
from CSV files, similar to the reference plots showing Testing Accuracy and Testing Loss.

Expected CSV format:
epoch,train_loss,train_acc,val_loss,val_acc
1,0.012507101211945216,43.465,1.2281849031448364,58.27
...
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from pathlib import Path

# Set up matplotlib for publication-quality plots
plt.rcParams['figure.figsize'] = (16, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fancybox'] = True
plt.rcParams['legend.shadow'] = True

# Define color palette for different models
COLORS = {
    'lstm_only': '#1f77b4',                    # Blue
    'sm_kernel_only': '#ff7f0e',              # Orange  
    'kmote_abs_only': '#2ca02c',              # Green
    'kmote_rel_only': '#d62728',              # Red
    'dual_stream_baseline': '#9467bd',        # Purple
    'kan_mammote_lite': '#8c564b',            # Brown
    'kan_mammote_full': '#e377c2',            # Pink
    'kan_mammote_dual_kmote': '#e377c2',      # Pink (same as full)
    'lete': '#7f7f7f',                        # Gray
    'mercer': '#bcbd22',                      # Olive
    'bochner': '#17becf'                      # Cyan
}

# Define display names for legend
DISPLAY_NAMES = {
    'lstm_only': 'LSTM',
    'sm_kernel_only': 'LSTM+sm_kernel_only',
    'kmote_abs_only': 'LSTM+kmote_abs_only',
    'kmote_rel_only': 'LSTM+kmote_rel_only',
    'dual_stream_baseline': 'LSTM+dual_stream_baseline',
    'kan_mammote_lite': 'LSTM+kan_mammote_lite',
    'kan_mammote_full': 'LSTM+kan_mammote_full',
    'kan_mammote_dual_kmote': 'LSTM+kan_mammote_dual_kmote',
    'lete': 'LSTM+lete',
    'mercer': 'LSTM+mercer',
    'bochner': 'LSTM+bochner'
}

def load_training_data(data_folder):
    """Load all CSV files from the epoch_history folder"""
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    
    training_data = {}
    
    for file_path in csv_files:
        # Extract model name from filename
        filename = os.path.basename(file_path)
        model_name = filename.replace('_history.csv', '')
        
        try:
            # Load CSV data
            df = pd.read_csv(file_path)
            
            # Validate required columns
            required_cols = ['epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
            if not all(col in df.columns for col in required_cols):
                print(f"Warning: {filename} missing required columns. Skipping...")
                continue
            
            training_data[model_name] = df
            print(f"✅ Loaded {model_name}: {len(df)} epochs")
            
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
    
    return training_data

def smooth_curve(y, window_size=5):
    """Apply moving average smoothing to reduce noise"""
    if len(y) < window_size:
        return y
    
    smoothed = np.convolve(y, np.ones(window_size)/window_size, mode='valid')
    # Pad the beginning to maintain same length
    padding = np.full(window_size-1, smoothed[0])
    return np.concatenate([padding, smoothed])

def create_training_plots(training_data, output_dir="training_plots", smooth=True):
    """Create publication-quality training plots"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the main comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Testing (Validation) Accuracy
    ax1.set_title('(a) Testing Accuracy', fontsize=14, fontweight='bold', pad=20)
    
    # Plot 2: Testing (Validation) Loss  
    ax2.set_title('(b) Testing Loss', fontsize=14, fontweight='bold', pad=20)
    
    # Keep track of models for legend
    legend_entries = []
    
    # Plot each model
    for model_name, df in training_data.items():
        color = COLORS.get(model_name, '#000000')  # Default to black if not found
        display_name = DISPLAY_NAMES.get(model_name, model_name)
        
        epochs = df['epoch'].values
        val_acc = df['val_acc'].values
        val_loss = df['val_loss'].values
        
        # Apply smoothing if requested
        if smooth and len(val_acc) > 5:
            val_acc_smooth = smooth_curve(val_acc, window_size=5)
            val_loss_smooth = smooth_curve(val_loss, window_size=5)
        else:
            val_acc_smooth = val_acc
            val_loss_smooth = val_loss
        
        # Convert accuracy to percentage if it's in decimal format
        if val_acc_smooth.max() <= 1.0:
            val_acc_smooth = val_acc_smooth * 100
        
        # Plot accuracy
        line1 = ax1.plot(epochs, val_acc_smooth, color=color, linewidth=2, 
                        label=display_name, alpha=0.8)
        
        # Plot loss
        line2 = ax2.plot(epochs, val_loss_smooth, color=color, linewidth=2, 
                        label=display_name, alpha=0.8)
        
        legend_entries.append((display_name, color))
    
    # Configure accuracy plot
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(50, 100)  # Typical accuracy range
    
    # Configure loss plot
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.6)  # Limit loss range for better visibility
    
    # Add legend to the loss plot (right side)
    legend_lines = []
    legend_labels = []
    
    for name, color in legend_entries:
        legend_lines.append(plt.Line2D([0], [0], color=color, linewidth=2))
        legend_labels.append(name)
    
    ax2.legend(legend_lines, legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5),
              fontsize=10, frameon=True, fancybox=True, shadow=True)
    
    plt.tight_layout()
    
    # Save the main plot
    output_path = os.path.join(output_dir, 'training_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Main plot saved to: {output_path}")
    
    plt.show()
    
    # Create individual detailed plots
    create_individual_plots(training_data, output_dir, smooth)
    
    # Create summary statistics
    create_summary_statistics(training_data, output_dir)

def create_individual_plots(training_data, output_dir, smooth=True):
    """Create individual detailed plots for each model"""
    
    individual_dir = os.path.join(output_dir, 'individual_plots')
    os.makedirs(individual_dir, exist_ok=True)
    
    for model_name, df in training_data.items():
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training History: {DISPLAY_NAMES.get(model_name, model_name)}', 
                     fontsize=16, fontweight='bold')
        
        epochs = df['epoch'].values
        train_acc = df['train_acc'].values
        val_acc = df['val_acc'].values
        train_loss = df['train_loss'].values
        val_loss = df['val_loss'].values
        
        # Convert accuracy to percentage if needed
        if train_acc.max() <= 1.0:
            train_acc = train_acc * 100
        if val_acc.max() <= 1.0:
            val_acc = val_acc * 100
        
        # Apply smoothing if requested
        if smooth and len(epochs) > 5:
            train_acc = smooth_curve(train_acc)
            val_acc = smooth_curve(val_acc)
            train_loss = smooth_curve(train_loss)
            val_loss = smooth_curve(val_loss)
        
        color = COLORS.get(model_name, '#1f77b4')
        
        # Plot 1: Training vs Validation Accuracy
        ax1.plot(epochs, train_acc, color='blue', linewidth=2, label='Training Accuracy', alpha=0.8)
        ax1.plot(epochs, val_acc, color='red', linewidth=2, label='Validation Accuracy', alpha=0.8)
        ax1.set_title('Accuracy Comparison')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Training vs Validation Loss
        ax2.plot(epochs, train_loss, color='blue', linewidth=2, label='Training Loss', alpha=0.8)
        ax2.plot(epochs, val_loss, color='red', linewidth=2, label='Validation Loss', alpha=0.8)
        ax2.set_title('Loss Comparison')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Validation Accuracy Only (matching main plot)
        ax3.plot(epochs, val_acc, color=color, linewidth=2, alpha=0.8)
        ax3.set_title('Testing Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy (%)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Validation Loss Only (matching main plot)
        ax4.plot(epochs, val_loss, color=color, linewidth=2, alpha=0.8)
        ax4.set_title('Testing Loss')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save individual plot
        safe_name = model_name.replace('/', '_').replace(' ', '_')
        output_path = os.path.join(individual_dir, f'{safe_name}_detailed.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    
    print(f"✅ Individual plots saved to: {individual_dir}")

def create_summary_statistics(training_data, output_dir):
    """Create summary statistics and final performance comparison"""
    
    summary_data = []
    
    for model_name, df in training_data.items():
        # Get final epoch metrics
        final_metrics = df.iloc[-1]
        
        # Get best validation accuracy and its epoch
        best_val_acc_idx = df['val_acc'].idxmax()
        best_val_acc = df.loc[best_val_acc_idx, 'val_acc']
        best_val_acc_epoch = df.loc[best_val_acc_idx, 'epoch']
        
        # Get minimum validation loss and its epoch
        min_val_loss_idx = df['val_loss'].idxmin()
        min_val_loss = df.loc[min_val_loss_idx, 'val_loss']
        min_val_loss_epoch = df.loc[min_val_loss_idx, 'epoch']
        
        # Convert accuracy to percentage if needed
        final_val_acc = final_metrics['val_acc']
        if final_val_acc <= 1.0:
            final_val_acc *= 100
            best_val_acc *= 100
        
        summary_data.append({
            'Model': DISPLAY_NAMES.get(model_name, model_name),
            'Final_Val_Acc': final_val_acc,
            'Final_Val_Loss': final_metrics['val_loss'],
            'Best_Val_Acc': best_val_acc,
            'Best_Val_Acc_Epoch': best_val_acc_epoch,
            'Min_Val_Loss': min_val_loss,
            'Min_Val_Loss_Epoch': min_val_loss_epoch,
            'Total_Epochs': len(df)
        })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Best_Val_Acc', ascending=False)
    
    # Save summary to CSV
    summary_path = os.path.join(output_dir, 'training_summary.csv')
    summary_df.to_csv(summary_path, index=False, float_format='%.4f')
    
    # Create performance comparison bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    models = summary_df['Model'].values
    best_acc = summary_df['Best_Val_Acc'].values
    min_loss = summary_df['Min_Val_Loss'].values
    
    # Best accuracy comparison
    bars1 = ax1.barh(models, best_acc, alpha=0.7)
    ax1.set_xlabel('Best Validation Accuracy (%)')
    ax1.set_title('Model Performance Comparison - Accuracy')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, acc in zip(bars1, best_acc):
        width = bar.get_width()
        ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
                f'{acc:.2f}%', ha='left', va='center', fontsize=9)
    
    # Minimum loss comparison
    bars2 = ax2.barh(models, min_loss, alpha=0.7, color='red')
    ax2.set_xlabel('Minimum Validation Loss')
    ax2.set_title('Model Performance Comparison - Loss')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, loss in zip(bars2, min_loss):
        width = bar.get_width()
        ax2.text(width + 0.005, bar.get_y() + bar.get_height()/2, 
                f'{loss:.4f}', ha='left', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save performance comparison
    perf_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(perf_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    print(f"✅ Summary statistics saved to: {summary_path}")
    print(f"✅ Performance comparison saved to: {perf_path}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("📊 TRAINING SUMMARY STATISTICS")
    print("="*80)
    print(summary_df.to_string(index=False, float_format='%.4f'))

def main():
    """Main function to run the visualization"""
    print("🚀 Training History Visualization Script")
    print("=" * 50)
    
    # Define data folder path
    data_folder = "/home/s2516027/kan-mammotev2/mnist_experiments/run_20251006_221728/epoch_history"
    
    # Check if folder exists
    if not os.path.exists(data_folder):
        print(f"❌ Data folder not found: {data_folder}")
        print("Please update the path to your epoch_history folder.")
        return
    
    # Load training data
    print(f"📂 Loading data from: {data_folder}")
    training_data = load_training_data(data_folder)
    
    if not training_data:
        print("❌ No valid training data found!")
        return
    
    print(f"✅ Successfully loaded {len(training_data)} models")
    
    # Create visualizations
    output_dir = "training_visualizations"
    print(f"🎨 Creating visualizations...")
    create_training_plots(training_data, output_dir, smooth=True)
    
    print(f"\n🎉 Visualization complete!")
    print(f"📁 All plots saved in: {output_dir}/")
    print(f"📋 Main comparison plot: {output_dir}/training_comparison.png")

if __name__ == "__main__":
    main()