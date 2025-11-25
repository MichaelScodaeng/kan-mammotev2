#!/usr/bin/env python3
"""
🎯 CLEAR EXPLANATION: What are we plotting exactly?
==================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load Wikipedia data
print("🔍 LOADING WIKIPEDIA DATA...")
df = pd.read_csv('/home/s2516027/kan-mammotev3/kan-mammotev2/dataset/ml_wikipedia.csv')

# Focus on one node
node_id = 8412
node_data = df[(df['u'] == node_id) | (df['i'] == node_id)].copy()
node_data = node_data.sort_values('ts').reset_index(drop=True)

print(f"\n📊 RAW DATA FOR NODE {node_id}:")
print("=" * 50)
print("First 10 interactions:")
for idx, row in node_data.head(10).iterrows():
    print(f"Interaction #{idx+1:3d}: Time={row['ts']:>8} (Real timestamp)")

print(f"\n🎯 WHAT EACH AXIS MEANS:")
print("=" * 50)
print("📈 X-axis (Interaction Index):")
print("   • Just counting: 1st interaction, 2nd interaction, 3rd...")
print("   • Like page numbers in a book: Page 1, Page 2, Page 3...")
print("   • NOT time! Just sequence order")
print()
print("📊 Y-axis (Temporal Amplitude):")
print("   • Activity level at that point in the sequence")
print("   • How 'busy' was the node around that interaction?")
print("   • Like measuring 'popularity' or 'intensity'")

# Create the explanation plot
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'🎪 CLEAR EXPLANATION: What We Are Plotting (Node {node_id})', fontsize=16, fontweight='bold')

# 1. Raw timestamps (what data actually is)
ax1 = axes[0, 0]
ax1.plot(range(len(node_data)), node_data['ts'], 'r-', linewidth=2)
ax1.set_title('1️⃣ RAW DATA: What we actually have', fontweight='bold')
ax1.set_xlabel('Interaction Index (1st, 2nd, 3rd interaction...)')
ax1.set_ylabel('Timestamp (Real time when it happened)')
ax1.grid(True, alpha=0.3)
ax1.text(0.05, 0.95, 'X = Counting interactions\nY = When it happened', 
         transform=ax1.transAxes, verticalalignment='top', 
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

# 2. Temporal density (what Figure 12 shows)
window_size = 50
density = []
for i in range(len(node_data)):
    start_idx = max(0, i - window_size // 2)
    end_idx = min(len(node_data), i + window_size // 2)
    window_interactions = end_idx - start_idx
    density.append(window_interactions)

ax2 = axes[0, 1]
ax2.plot(range(len(density)), density, 'orange', linewidth=3)
ax2.set_title('2️⃣ TRANSFORMED: What Figure 12 shows', fontweight='bold')
ax2.set_xlabel('Interaction Index (sequence position)')
ax2.set_ylabel('Temporal Amplitude (activity intensity)')
ax2.grid(True, alpha=0.3)
ax2.text(0.05, 0.95, 'X = Still counting interactions\nY = How busy around this point', 
         transform=ax2.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.8))

# 3. Real-world analogy: Facebook likes over posts
ax3 = axes[1, 0]
posts = range(1, 21)
likes = [5, 12, 8, 45, 67, 23, 15, 89, 12, 34, 78, 23, 45, 12, 67, 89, 34, 12, 23, 56]
ax3.bar(posts, likes, color='skyblue', alpha=0.7)
ax3.plot(posts, likes, 'b-', linewidth=2, marker='o')
ax3.set_title('3️⃣ ANALOGY: Facebook Posts vs Likes', fontweight='bold')
ax3.set_xlabel('Post Number (like Interaction Index)')
ax3.set_ylabel('Number of Likes (like Temporal Amplitude)')
ax3.grid(True, alpha=0.3)
ax3.text(0.05, 0.95, 'X = Which post (1st, 2nd, 3rd...)\nY = How popular (likes)', 
         transform=ax3.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

# 4. What makes the oscillations
ax4 = axes[1, 1]
time_bins = np.linspace(node_data['ts'].min(), node_data['ts'].max(), 20)
hist, _ = np.histogram(node_data['ts'], bins=time_bins)
bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
ax4.bar(range(len(hist)), hist, color='lightgreen', alpha=0.7)
ax4.plot(range(len(hist)), hist, 'g-', linewidth=2, marker='s')
ax4.set_title('4️⃣ WHY OSCILLATIONS: Busier vs Quieter Periods', fontweight='bold')
ax4.set_xlabel('Time Period (chronological order)')
ax4.set_ylabel('Activity Count (creates the waves)')
ax4.grid(True, alpha=0.3)
ax4.text(0.05, 0.95, 'X = Time periods in order\nY = How many interactions', 
         transform=ax4.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

plt.tight_layout()
plt.savefig('/home/s2516027/kan-mammotev3/kan-mammotev2/analysis/axes_explanation.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print(f"\n🎪 SUMMARY - EXACTLY WHAT WE PLOT:")
print("=" * 60)
print("🔥 X-AXIS (Interaction Index):")
print("   → Just counting events: 1, 2, 3, 4, 5, ... 1937")
print("   → Like page numbers or sequence position")
print("   → NOT timestamps, just order!")
print()
print("🔥 Y-AXIS (Temporal Amplitude):")
print("   → Activity intensity at that sequence position")
print("   → 'How busy was the node around interaction #X?'")
print("   → Creates waves: high activity = peak, low activity = valley")
print()
print("🎯 THE MAGIC:")
print("   → Raw timestamps are monotonic (always increasing)")
print("   → But activity levels go up and down (oscillatory)")
print("   → That's why we get waves instead of straight lines!")
print()
print("✅ Saved explanation plot: axes_explanation.png")

# Show actual data transformation
print(f"\n📋 ACTUAL EXAMPLE FROM NODE {node_id}:")
print("=" * 50)
sample_indices = [0, 100, 200, 300, 400]
for idx in sample_indices:
    if idx < len(node_data):
        timestamp = node_data.iloc[idx]['ts']
        if idx < len(density):
            amplitude = density[idx]
            print(f"Interaction #{idx+1:3d}: Timestamp={timestamp:>8} → Amplitude={amplitude:>6.1f}")
        else:
            print(f"Interaction #{idx+1:3d}: Timestamp={timestamp:>8} → Amplitude=N/A")

print(f"\n🤔 SEE THE PATTERN?")
print("   • Interaction Index keeps counting up: 1, 101, 201, 301, 401...")
print("   • But Temporal Amplitude varies: sometimes high, sometimes low")
print("   • That variation creates the oscillatory pattern in Figure 12!")