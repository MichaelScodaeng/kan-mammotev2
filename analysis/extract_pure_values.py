#!/usr/bin/env python3
"""
Pure Data Extractor - Get the actual values for the "orange line"
No plots, just the raw numbers you can use directly
"""

import pandas as pd
import numpy as np
import os

def extract_node_values(df, node_id, output_format='csv'):
    """
    Extract pure temporal values for a node
    Returns the actual data that creates the "orange line"
    """
    # Get all interactions for this node
    node_interactions = df[(df['u'] == node_id) | (df['i'] == node_id)].copy()
    
    if len(node_interactions) == 0:
        print(f"No interactions found for node {node_id}")
        return None
    
    # Sort chronologically
    node_interactions = node_interactions.sort_values('ts')
    
    # Extract the core data
    timestamps = node_interactions['ts'].values
    interaction_indices = np.arange(len(timestamps))
    
    # Create the data structure
    data = {
        'node_id': node_id,
        'total_interactions': len(timestamps),
        'time_range': [float(timestamps[0]), float(timestamps[-1])],
        'interaction_indices': interaction_indices.tolist(),
        'actual_timestamps': timestamps.tolist(),
    }
    
    if len(timestamps) > 1:
        inter_arrivals = np.diff(timestamps).tolist()
        data['inter_arrival_times'] = inter_arrivals
        data['mean_interval'] = float(np.mean(inter_arrivals))
        data['std_interval'] = float(np.std(inter_arrivals))
    
    return data

def main():
    """Extract pure values for specified nodes"""
    print("="*60)
    print("PURE DATA EXTRACTOR - ACTUAL VALUES")
    print("="*60)
    
    # Load data
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, 'processed_data', 'wikipedia', 'ml_wikipedia.csv')
    df = pd.read_csv(data_path)
    
    # Get some active nodes
    u_counts = df['u'].value_counts()
    i_counts = df['i'].value_counts()
    
    # Find nodes with good activity
    active_nodes = []
    for node in set(u_counts.index) | set(i_counts.index):
        total = u_counts.get(node, 0) + i_counts.get(node, 0)
        if total >= 100:
            active_nodes.append((node, total))
    
    # Sort and take top few
    active_nodes.sort(key=lambda x: x[1], reverse=True)
    selected_nodes = [node for node, count in active_nodes[:4]]
    
    print(f"Extracting data for nodes: {selected_nodes}")
    
    # Extract data for each node
    output_dir = 'pure_data_values'
    os.makedirs(output_dir, exist_ok=True)
    
    for node_id in selected_nodes:
        print(f"\n--- NODE {node_id} ---")
        
        data = extract_node_values(df, node_id)
        
        if data:
            print(f"Total interactions: {data['total_interactions']}")
            print(f"Time range: {data['time_range'][0]:.0f} - {data['time_range'][1]:.0f}")
            
            # Save as CSV (the actual X,Y values for plotting)
            df_export = pd.DataFrame({
                'interaction_index': data['interaction_indices'],
                'actual_timestamp': data['actual_timestamps']
            })
            
            csv_file = os.path.join(output_dir, f'node_{node_id}_orange_line_data.csv')
            df_export.to_csv(csv_file, index=False)
            print(f"✅ Saved CSV: {csv_file}")
            
            # Also save as JSON for complete data
            import json
            json_file = os.path.join(output_dir, f'node_{node_id}_complete_data.json')
            with open(json_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Saved JSON: {json_file}")
            
            # Print first few values as example
            print("First 10 data points (interaction_index, timestamp):")
            for i in range(min(10, len(data['interaction_indices']))):
                idx = data['interaction_indices'][i]
                ts = data['actual_timestamps'][i]
                print(f"  {idx}: {ts}")
            
            if len(data['actual_timestamps']) > 10:
                print("  ...")
                print(f"  Last: {data['interaction_indices'][-1]}: {data['actual_timestamps'][-1]}")
    
    print(f"\n🎯 PURE DATA EXTRACTED!")
    print(f"📁 Files saved in: {output_dir}/")
    print(f"📋 Each CSV file contains two columns:")
    print(f"   • interaction_index (X-axis values)")
    print(f"   • actual_timestamp (Y-axis values - the 'orange line')")
    print(f"📋 You can now:")
    print(f"   • Import CSV into any plotting software")
    print(f"   • Use the exact values in your analysis")
    print(f"   • Plot X vs Y to get the 'orange line' pattern")

if __name__ == "__main__":
    main()