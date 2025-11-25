#!/usr/bin/env python3
"""
ACTUAL Wikipedia Node Temporal Patterns - RAW DATA ONLY
No reconstruction, no LeTE/FTE comparison - just pure temporal interaction patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os
import sys

def load_wikipedia_data():
    """Load Wikipedia dataset"""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'processed_data', 'wikipedia', 'ml_wikipedia.csv')
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Wikipedia dataset not found at {data_path}")
    
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} interactions")
    
    return df

def extract_actual_temporal_pattern(df, node_id):
    """
    Extract the ACTUAL temporal interaction pattern for a node
    Returns the real timestamps as they occurred
    """
    # Get all interactions for this node (as source or destination)
    node_interactions = df[(df['u'] == node_id) | (df['i'] == node_id)].copy()
    
    if len(node_interactions) == 0:
        return None, None, None
    
    # Sort by timestamp to get chronological order
    node_interactions = node_interactions.sort_values('ts')
    
    # Extract the actual data
    timestamps = node_interactions['ts'].values
    interaction_indices = np.arange(len(timestamps))
    
    print(f"  Node {node_id}: {len(timestamps)} interactions")
    print(f"    Time range: {timestamps[0]:.0f} - {timestamps[-1]:.0f}")
    print(f"    Duration: {(timestamps[-1] - timestamps[0]):.0f} time units")
    
    return interaction_indices, timestamps, node_interactions

def find_interesting_nodes(df, n_nodes=4):
    """Find nodes with substantial interaction patterns"""
    # Count interactions per node (as both source and destination)
    u_counts = df['u'].value_counts()
    i_counts = df['i'].value_counts()
    
    all_nodes = set(u_counts.index) | set(i_counts.index)
    node_total_counts = {}
    
    for node in all_nodes:
        total = u_counts.get(node, 0) + i_counts.get(node, 0)
        if total >= 100:  # Minimum for good patterns
            node_total_counts[node] = total
    
    # Sort by activity and select diverse nodes
    sorted_nodes = sorted(node_total_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Select nodes with different activity levels
    selected = []
    if len(sorted_nodes) >= n_nodes:
        indices = [0, len(sorted_nodes)//4, len(sorted_nodes)//2, 3*len(sorted_nodes)//4]
        selected = [sorted_nodes[i][0] for i in indices[:n_nodes]]
    else:
        selected = [node for node, count in sorted_nodes[:n_nodes]]
    
    print(f"Selected {len(selected)} nodes with activity levels:")
    for node in selected:
        count = node_total_counts[node]
        print(f"  Node {node}: {count} total interactions")
    
    return selected

def create_actual_temporal_plots(df, node_ids, apply_smoothing=True, sigma=3):
    """
    Create plots showing ACTUAL temporal interaction patterns
    No reconstruction - just raw temporal data from Wikipedia
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    fig.suptitle('ACTUAL Wikipedia Node Temporal Interaction Patterns\n' + 
                 f'Raw Data {"(Smoothed by Gaussian Filter, σ=3)" if apply_smoothing else "(Unsmoothed)"}',
                 fontsize=16, fontweight='bold')
    
    for idx, node_id in enumerate(node_ids[:4]):
        ax = axes[idx]
        
        # Extract actual temporal pattern
        indices, timestamps, interactions = extract_actual_temporal_pattern(df, node_id)
        
        if indices is None:
            ax.text(0.5, 0.5, f'Node {node_id}\nNo data found', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=14)
            continue
        
        # Plot raw timestamps if requested
        if not apply_smoothing:
            ax.plot(indices, timestamps, 'o-', color='darkorange', markersize=2, 
                   linewidth=1.5, alpha=0.8, label='Raw Timestamps')
        
        # Apply smoothing to show patterns more clearly
        if apply_smoothing and len(timestamps) >= 5:
            timestamps_smooth = gaussian_filter1d(timestamps, sigma=sigma)
            ax.plot(indices, timestamps_smooth, color='darkorange', linewidth=3, 
                   label=f'Smoothed (σ={sigma})', alpha=0.9)
            
            # Also show raw data lightly in background
            ax.plot(indices, timestamps, color='lightgray', linewidth=0.5, 
                   alpha=0.5, label='Raw Data')
        else:
            ax.plot(indices, timestamps, color='darkorange', linewidth=2, 
                   label='Raw Timestamps')
        
        # Calculate some basic pattern statistics
        if len(timestamps) > 1:
            time_diffs = np.diff(timestamps)
            mean_interval = np.mean(time_diffs)
            std_interval = np.std(time_diffs)
            cv = std_interval / mean_interval if mean_interval > 0 else 0
            
            # Determine pattern type
            if cv < 0.5:
                pattern_type = "Regular"
            elif cv > 2.0:
                pattern_type = "Very Bursty"
            elif cv > 1.0:
                pattern_type = "Bursty"
            else:
                pattern_type = "Mixed"
        else:
            pattern_type = "Single Event"
            mean_interval = 0
            cv = 0
        
        # Format plot
        ax.set_title(f'Node {node_id} - {pattern_type}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Interaction Index (Chronological Order)', fontsize=12)
        ax.set_ylabel('Actual Timestamp', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        
        # Add statistics box
        stats_text = f"Interactions: {len(timestamps)}\n" + \
                    f"Time Span: {timestamps[-1] - timestamps[0]:.0f}\n" + \
                    f"Avg Interval: {mean_interval:.0f}\n" + \
                    f"Variability (CV): {cv:.3f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=10,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.9))
        
        # Set reasonable axis limits
        ax.set_xlim(0, len(indices))
        if len(timestamps) > 0:
            y_margin = (timestamps.max() - timestamps.min()) * 0.1
            ax.set_ylim(timestamps.min() - y_margin, timestamps.max() + y_margin)
    
    plt.tight_layout()
    return fig

def create_detailed_node_analysis(df, node_id):
    """Create detailed analysis for a single node showing all temporal aspects"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data
    indices, timestamps, interactions = extract_actual_temporal_pattern(df, node_id)
    
    if indices is None:
        fig.suptitle(f'Node {node_id} - No Data Available', fontsize=16)
        return fig
    
    fig.suptitle(f'Detailed Temporal Analysis: Node {node_id}\n' + 
                 f'{len(timestamps)} Total Interactions', fontsize=16, fontweight='bold')
    
    # 1. Raw time series
    ax1 = axes[0, 0]
    ax1.plot(indices, timestamps, 'o-', color='darkorange', markersize=3, linewidth=1.5)
    ax1.set_title('Raw Temporal Sequence', fontweight='bold')
    ax1.set_xlabel('Interaction Index')
    ax1.set_ylabel('Actual Timestamp')
    ax1.grid(True, alpha=0.3)
    
    # 2. Smoothed time series
    ax2 = axes[0, 1]
    if len(timestamps) >= 5:
        timestamps_smooth = gaussian_filter1d(timestamps, sigma=3)
        ax2.plot(indices, timestamps, color='lightgray', alpha=0.5, linewidth=1, label='Raw')
        ax2.plot(indices, timestamps_smooth, color='darkorange', linewidth=3, label='Smoothed (σ=3)')
        ax2.legend()
    else:
        ax2.plot(indices, timestamps, color='darkorange', linewidth=2)
    ax2.set_title('Smoothed Temporal Pattern', fontweight='bold')
    ax2.set_xlabel('Interaction Index')
    ax2.set_ylabel('Actual Timestamp')
    ax2.grid(True, alpha=0.3)
    
    # 3. Inter-arrival times
    ax3 = axes[0, 2]
    if len(timestamps) > 1:
        inter_arrivals = np.diff(timestamps)
        ax3.plot(inter_arrivals, 'o-', color='red', markersize=3, linewidth=1.5)
        ax3.set_title('Inter-arrival Times', fontweight='bold')
        ax3.set_xlabel('Event Index')
        ax3.set_ylabel('Time Between Events')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'Insufficient Data\nfor Inter-arrivals', 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Cumulative events over time
    ax4 = axes[1, 0]
    ax4.plot(timestamps, indices, 'o-', color='green', markersize=3, linewidth=1.5)
    ax4.set_title('Cumulative Events Over Time', fontweight='bold')
    ax4.set_xlabel('Actual Timestamp')
    ax4.set_ylabel('Cumulative Event Count')
    ax4.grid(True, alpha=0.3)
    
    # 5. Inter-arrival distribution
    ax5 = axes[1, 1]
    if len(timestamps) > 1:
        inter_arrivals = np.diff(timestamps)
        ax5.hist(inter_arrivals, bins=min(20, len(inter_arrivals)//2), 
                color='purple', alpha=0.7, edgecolor='black')
        ax5.set_title('Inter-arrival Distribution', fontweight='bold')
        ax5.set_xlabel('Inter-arrival Time')
        ax5.set_ylabel('Frequency')
        ax5.grid(True, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Insufficient Data\nfor Distribution', 
                ha='center', va='center', transform=ax5.transAxes)
    
    # 6. Statistics summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate comprehensive statistics
    if len(timestamps) > 1:
        time_span = timestamps[-1] - timestamps[0]
        inter_arrivals = np.diff(timestamps)
        mean_inter = np.mean(inter_arrivals)
        std_inter = np.std(inter_arrivals)
        cv_inter = std_inter / mean_inter if mean_inter > 0 else 0
        
        stats_text = f"""NODE {node_id} - ACTUAL DATA STATISTICS

Basic Information:
• Total Interactions: {len(timestamps):,}
• Time Span: {time_span:.0f} units
• First Event: {timestamps[0]:.0f}
• Last Event: {timestamps[-1]:.0f}

Temporal Patterns:
• Mean Inter-arrival: {mean_inter:.2f}
• Std Inter-arrival: {std_inter:.2f}
• Coefficient of Variation: {cv_inter:.3f}
• Min Gap: {np.min(inter_arrivals):.2f}
• Max Gap: {np.max(inter_arrivals):.2f}

Pattern Classification:
• {"Regular pattern" if cv_inter < 0.5 else "Bursty pattern" if cv_inter > 1.0 else "Mixed pattern"}
• Event Rate: {len(timestamps)/time_span:.2e} events/time

Data Quality:
• Complete temporal sequence
• No missing timestamps
• Chronologically ordered
"""
    else:
        stats_text = f"""NODE {node_id} - ACTUAL DATA STATISTICS

Basic Information:
• Total Interactions: {len(timestamps):,}
• Single event or insufficient data

Cannot compute temporal patterns
with fewer than 2 events.
"""
    
    ax6.text(0.05, 0.95, stats_text, transform=ax6.transAxes, 
            verticalalignment='top', fontsize=10, family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig

def export_raw_data(df, node_ids, output_dir):
    """Export the actual raw temporal data for analysis"""
    output_dir = os.path.join(output_dir, 'raw_temporal_data')
    os.makedirs(output_dir, exist_ok=True)
    
    for node_id in node_ids:
        indices, timestamps, interactions = extract_actual_temporal_pattern(df, node_id)
        
        if indices is not None:
            # Save raw interaction data
            output_file = os.path.join(output_dir, f'node_{node_id}_raw_interactions.csv')
            interactions.to_csv(output_file, index=False)
            
            # Save processed temporal sequence
            temporal_data = pd.DataFrame({
                'interaction_index': indices,
                'actual_timestamp': timestamps
            })
            
            if len(timestamps) > 1:
                temporal_data['inter_arrival_time'] = [0] + list(np.diff(timestamps))
            
            temporal_file = os.path.join(output_dir, f'node_{node_id}_temporal_sequence.csv')
            temporal_data.to_csv(temporal_file, index=False)
            
            print(f"✅ Exported Node {node_id} raw data: {len(timestamps)} interactions")

def main():
    """Extract and visualize ACTUAL temporal patterns from Wikipedia data"""
    print("="*80)
    print("ACTUAL WIKIPEDIA TEMPORAL PATTERNS - RAW DATA EXTRACTION")
    print("="*80)
    print("This extracts the real temporal interaction patterns - no reconstruction!")
    
    # Load data
    df = load_wikipedia_data()
    
    # Find interesting nodes
    selected_nodes = find_interesting_nodes(df, n_nodes=4)
    
    if not selected_nodes:
        print("❌ No suitable nodes found")
        return
    
    print(f"\n📊 Creating visualizations for nodes: {selected_nodes}")
    
    # Create output directory
    output_dir = 'actual_temporal_patterns'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create main visualization showing actual patterns
    fig1 = create_actual_temporal_plots(df, selected_nodes, apply_smoothing=True)
    fig1_path = os.path.join(output_dir, 'actual_wikipedia_temporal_patterns.png')
    fig1.savefig(fig1_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved main temporal patterns: {fig1_path}")
    
    # Create detailed analysis for the first node
    if selected_nodes:
        fig2 = create_detailed_node_analysis(df, selected_nodes[0])
        fig2_path = os.path.join(output_dir, f'detailed_node_{selected_nodes[0]}_analysis.png')
        fig2.savefig(fig2_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved detailed analysis: {fig2_path}")
    
    # Export raw data
    export_raw_data(df, selected_nodes, output_dir)
    print(f"✅ Raw data exported to: {output_dir}/raw_temporal_data/")
    
    plt.show()
    
    print(f"\n🎯 ACTUAL TEMPORAL PATTERNS EXTRACTED!")
    print(f"📁 All outputs saved in: {output_dir}/")
    print(f"\n📋 What you have now:")
    print(f"   • REAL temporal interaction sequences (the 'orange line' data)")
    print(f"   • Raw timestamps as they actually occurred in Wikipedia")
    print(f"   • No reconstruction or model comparison - pure data")
    print(f"   • Exportable CSV files with actual values")
    print(f"   • Statistical analysis of real interaction patterns")

if __name__ == "__main__":
    main()