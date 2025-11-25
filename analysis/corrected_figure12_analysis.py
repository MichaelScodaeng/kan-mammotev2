#!/usr/bin/env python3
"""
CORRECTED Figure 12 Analysis - Based on Legend Evidence
The orange line represents the ORIGINAL node temporal pattern (ground truth),
not LeTE reconstruction as I initially thought.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import os

def analyze_figure12_legend():
    """
    Correct analysis based on the actual legend in Figure 12
    """
    print("🔍 CORRECTED FIGURE 12 ANALYSIS - BASED ON LEGEND")
    print("="*70)
    
    print("📊 What the Figure 12 legend actually shows:")
    print("   🟠 Orange solid line = 'Node 2' (ORIGINAL/GROUND TRUTH)")
    print("   🔴 Red dashed line = 'LeTE Reconstructed'")
    print("   🔵 Blue dashed line = 'FTE Reconstructed'")
    
    print("\n🎯 CORRECTED INTERPRETATION:")
    print("   • Orange line = Original temporal pattern (what you want!)")
    print("   • Red/Blue dashed = Model reconstructions for comparison")
    print("   • The task: Can LeTE/FTE reconstruct the original pattern?")
    
    print("\n✅ This means your raw data extraction WAS on the right track!")
    print("   • You were looking for the original temporal patterns")
    print("   • The orange line IS the real node data")
    print("   • But it's processed/transformed, not raw timestamps")

def understand_the_transformation():
    """
    Figure out how raw timestamps become the oscillatory orange pattern
    """
    print("\n🤔 THE REMAINING QUESTION:")
    print("   How do monotonic timestamps become oscillatory patterns?")
    
    print("\n💡 POSSIBLE TRANSFORMATIONS:")
    print("1. 📊 TEMPORAL DENSITY BINNING:")
    print("   • Divide time into windows")
    print("   • Count interactions per window")
    print("   • This creates oscillatory patterns from raw data")
    
    print("\n2. 📈 INTER-ARRIVAL FREQUENCY:")
    print("   • Calculate time between interactions")
    print("   • Convert to frequency (1/inter-arrival-time)")
    print("   • Shows bursts vs quiet periods")
    
    print("\n3. 🌊 ACTIVITY LEVEL OVER TIME:")
    print("   • Sliding window of interaction intensity")
    print("   • Normalized by local time density")
    print("   • Creates waves showing temporal patterns")
    
    print("\n4. 📉 CUMULATIVE RATE CHANGES:")
    print("   • Track how interaction rate changes over time")
    print("   • Derivative-like transformation of timestamps")
    print("   • Shows acceleration/deceleration in activity")

def create_transformation_examples(df, node_id):
    """
    Show different ways to transform raw data into oscillatory patterns
    """
    # Get raw interaction data
    node_interactions = df[(df['u'] == node_id) | (df['i'] == node_id)].copy()
    node_interactions = node_interactions.sort_values('ts')
    timestamps = node_interactions['ts'].values
    
    if len(timestamps) < 20:
        print(f"Node {node_id} has insufficient data ({len(timestamps)} interactions)")
        return None
    
    print(f"\n📊 TESTING TRANSFORMATIONS ON NODE {node_id}")
    print(f"   Raw data: {len(timestamps)} interactions")
    print(f"   Time range: {timestamps[0]:.0f} - {timestamps[-1]:.0f}")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. Raw timestamps (what we had before)
    ax1 = axes[0, 0]
    interaction_indices = np.arange(len(timestamps))
    ax1.plot(interaction_indices, timestamps, color='red', linewidth=2, alpha=0.7)
    ax1.set_title('Raw Timestamps\n(Monotonic - Not Figure 12)', fontsize=10)
    ax1.set_xlabel('Interaction Index')
    ax1.set_ylabel('Timestamp')
    ax1.grid(True, alpha=0.3)
    
    # 2. Temporal density binning
    ax2 = axes[0, 1]
    n_bins = min(50, len(timestamps) // 5)
    if n_bins > 2:
        time_edges = np.linspace(timestamps[0], timestamps[-1], n_bins + 1)
        interaction_counts, _ = np.histogram(timestamps, bins=time_edges)
        bin_centers = (time_edges[:-1] + time_edges[1:]) / 2
        
        # Map to interaction index space for proper x-axis
        bin_indices = np.interp(bin_centers, timestamps, interaction_indices)
        
        # Apply smoothing
        smoothed_counts = gaussian_filter1d(interaction_counts.astype(float), sigma=3)
        
        ax2.plot(bin_indices, smoothed_counts, color='darkorange', linewidth=3)
        ax2.set_title('Temporal Density (Binned)\n🟠 CANDIDATE for Figure 12', fontsize=10, fontweight='bold')
        ax2.set_xlabel('Interaction Index')
        ax2.set_ylabel('Interactions per Time Bin')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Insufficient\ndata for binning', ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Inter-arrival frequency
    ax3 = axes[0, 2]
    if len(timestamps) > 1:
        inter_arrivals = np.diff(timestamps)
        # Convert to frequency (avoid division by zero)
        frequencies = 1000.0 / (inter_arrivals + 1)
        
        # Apply smoothing
        frequencies_smooth = gaussian_filter1d(frequencies, sigma=3)
        
        ax3.plot(np.arange(len(frequencies_smooth)), frequencies_smooth, color='darkorange', linewidth=3)
        ax3.set_title('Inter-arrival Frequency\n🟠 CANDIDATE for Figure 12', fontsize=10, fontweight='bold')
        ax3.set_xlabel('Event Index')
        ax3.set_ylabel('Interaction Frequency')
        ax3.grid(True, alpha=0.3)
    
    # 4. Sliding window activity
    ax4 = axes[1, 0]
    window_size = max(10, len(timestamps) // 20)
    if window_size < len(timestamps) // 2:
        activities = []
        window_centers = []
        
        for i in range(window_size, len(timestamps) - window_size):
            window_start = max(0, i - window_size // 2)
            window_end = min(len(timestamps), i + window_size // 2)
            
            window_timestamps = timestamps[window_start:window_end]
            if len(window_timestamps) > 1:
                time_span = window_timestamps[-1] - window_timestamps[0]
                activity = len(window_timestamps) / (time_span + 1) * 1000  # Normalize
            else:
                activity = 0
            
            activities.append(activity)
            window_centers.append(i)
        
        activities = np.array(activities)
        activities_smooth = gaussian_filter1d(activities, sigma=3)
        
        ax4.plot(window_centers, activities_smooth, color='darkorange', linewidth=3)
        ax4.set_title('Sliding Window Activity\n🟠 CANDIDATE for Figure 12', fontsize=10, fontweight='bold')
        ax4.set_xlabel('Interaction Index')
        ax4.set_ylabel('Local Activity Rate')
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'Insufficient\ndata for windows', ha='center', va='center', transform=ax4.transAxes)
    
    # 5. Rate of change (derivative-like)
    ax5 = axes[1, 1]
    if len(timestamps) > 2:
        # Calculate second differences (acceleration)
        first_diff = np.diff(timestamps)
        second_diff = np.diff(first_diff)
        
        # Invert and normalize to show rate changes
        rate_changes = -second_diff / np.mean(first_diff) * 1000
        rate_changes_smooth = gaussian_filter1d(rate_changes, sigma=3)
        
        ax5.plot(np.arange(len(rate_changes_smooth)), rate_changes_smooth, color='darkorange', linewidth=3)
        ax5.set_title('Rate of Change (2nd Derivative)\n🟠 CANDIDATE for Figure 12', fontsize=10, fontweight='bold')
        ax5.set_xlabel('Event Index')
        ax5.set_ylabel('Temporal Acceleration')
        ax5.grid(True, alpha=0.3)
    
    # 6. Normalized cumulative intervals
    ax6 = axes[1, 2]
    if len(timestamps) > 1:
        # Normalize timestamps to [0, 1] range
        normalized_times = (timestamps - timestamps[0]) / (timestamps[-1] - timestamps[0])
        
        # Expected uniform distribution
        expected_times = np.linspace(0, 1, len(timestamps))
        
        # Deviation from uniform (shows clustering/spacing patterns)
        deviations = (normalized_times - expected_times) * 1000  # Scale for visibility
        deviations_smooth = gaussian_filter1d(deviations, sigma=3)
        
        ax6.plot(interaction_indices, deviations_smooth, color='darkorange', linewidth=3)
        ax6.set_title('Temporal Clustering Deviation\n🟠 CANDIDATE for Figure 12', fontsize=10, fontweight='bold')
        ax6.set_xlabel('Interaction Index')
        ax6.set_ylabel('Clustering Deviation')
        ax6.grid(True, alpha=0.3)
    
    plt.suptitle(f'Node {node_id}: Possible Transformations to Create Figure 12 Orange Line', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

def main():
    """
    Corrected analysis based on Figure 12 legend evidence
    """
    print("🚨 FIGURE 12 LEGEND ANALYSIS - CORRECTED")
    print("="*80)
    
    # Load data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'processed_data', 'wikipedia', 'ml_wikipedia.csv')
    df = pd.read_csv(data_path)
    
    # Analyze what the legend tells us
    analyze_figure12_legend()
    
    # Explain the transformation mystery
    understand_the_transformation()
    
    # Test different transformations
    print(f"\n🔎 Analyzing node activity...")
    
    # Check source nodes
    u_counts = df['u'].value_counts()
    print(f"   Top source nodes: {u_counts.head().to_dict()}")
    
    # Check destination nodes  
    i_counts = df['i'].value_counts()
    print(f"   Top destination nodes: {i_counts.head().to_dict()}")
    
    # Combine counts properly
    all_nodes = set(u_counts.index) | set(i_counts.index)
    node_total_counts = {}
    
    for node in all_nodes:
        total = u_counts.get(node, 0) + i_counts.get(node, 0)
        if total >= 50:  # Lower threshold to find nodes
            node_total_counts[node] = total
    
    # Sort by activity
    active_nodes = sorted(node_total_counts.items(), key=lambda x: x[1], reverse=True)
    active_node_ids = [node for node, count in active_nodes[:10]]
    
    print(f"   Found {len(node_total_counts)} nodes with ≥50 interactions")
    print(f"   Top 5 active nodes: {active_nodes[:5]}")
    print(f"\n🔎 Testing transformations on active nodes: {active_node_ids[:3]} (showing first 3)")
    
    if active_node_ids:
        node_id = active_node_ids[0]  # Pick first active node
        fig = create_transformation_examples(df, node_id)
        
        if fig:
            output_dir = 'figure12_corrected_analysis'
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(f'{output_dir}/transformation_candidates.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved transformations plot: {output_dir}/transformation_candidates.png")
            plt.show()
    else:
        print("❌ No active nodes found! Check data format.")
    
    print(f"\n✨ CORRECTED CONCLUSION:")
    print(f"📊 The orange line in Figure 12 IS the original node data")
    print(f"🔄 But it's transformed from raw timestamps to show temporal patterns")
    print(f"🎯 Most likely: Temporal density binning or inter-arrival frequency")
    print(f"📈 The oscillations show real temporal dynamics in Wikipedia interactions")
    print(f"🎪 Your raw data extraction was RIGHT - just needs the right transformation!")

if __name__ == "__main__":
    main()