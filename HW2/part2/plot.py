#In your write-up, plot a graph of speedup (compared to the reference sequential implementation) as a function of the number of threads used for VIEW 1. Is the speedup linear to the number of threads used? Hypothesize why this is (or is not) the case. (You may want to plot a graph for VIEW 2 for further insights.

import matplotlib.pyplot as plt
import numpy as np

# Data from result.txt
threads = np.array([1, 2, 3, 4, 5, 6])

# View 1 speedup data
view1_speedup = np.array([1.00, 1.96, 2.93, 3.82, 4.76, 5.58])

# View 2 speedup data
view2_speedup = np.array([1.00, 1.96, 2.92, 3.80, 4.73, 5.54])

# Ideal linear speedup
ideal_speedup = threads

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot View 1
ax1.plot(threads, view1_speedup, 'bo-', linewidth=2, markersize=8, label='Actual Speedup')
ax1.plot(threads, ideal_speedup, 'r--', linewidth=2, label='Ideal Linear Speedup')
ax1.set_xlabel('Number of Threads', fontsize=12)
ax1.set_ylabel('Speedup', fontsize=12)
ax1.set_title('Mandelbrot Speedup - View 1 (Full View)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xticks(threads)
ax1.set_ylim([0, 7])

# Add efficiency annotations for View 1
for i, (t, s) in enumerate(zip(threads, view1_speedup)):
    efficiency = (s / t) * 100
    ax1.annotate(f'{efficiency:.1f}%', 
                xy=(t, s), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8,
                color='blue')

# Plot View 2
ax2.plot(threads, view2_speedup, 'go-', linewidth=2, markersize=8, label='Actual Speedup')
ax2.plot(threads, ideal_speedup, 'r--', linewidth=2, label='Ideal Linear Speedup')
ax2.set_xlabel('Number of Threads', fontsize=12)
ax2.set_ylabel('Speedup', fontsize=12)
ax2.set_title('Mandelbrot Speedup - View 2 (Zoomed View)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.set_xticks(threads)
ax2.set_ylim([0, 7])

# Add efficiency annotations for View 2
for i, (t, s) in enumerate(zip(threads, view2_speedup)):
    efficiency = (s / t) * 100
    ax2.annotate(f'{efficiency:.1f}%', 
                xy=(t, s), 
                xytext=(5, 5), 
                textcoords='offset points',
                fontsize=8,
                color='green')

plt.tight_layout()
plt.savefig('speedup_plot.png', dpi=300, bbox_inches='tight')
print("Speedup plot saved as 'speedup_plot.png'")

# Create comparison plot
fig2, ax3 = plt.subplots(figsize=(10, 6))
ax3.plot(threads, view1_speedup, 'bo-', linewidth=2, markersize=8, label='View 1 (Full)')
ax3.plot(threads, view2_speedup, 'go-', linewidth=2, markersize=8, label='View 2 (Zoomed)')
ax3.plot(threads, ideal_speedup, 'r--', linewidth=2, label='Ideal Linear')
ax3.set_xlabel('Number of Threads', fontsize=12)
ax3.set_ylabel('Speedup', fontsize=12)
ax3.set_title('Speedup Comparison: View 1 vs View 2', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)
ax3.set_xticks(threads)
ax3.set_ylim([0, 7])

plt.tight_layout()
plt.savefig('speedup_comparison.png', dpi=300, bbox_inches='tight')
print("Comparison plot saved as 'speedup_comparison.png'")

# Print efficiency analysis
print("\n" + "="*60)
print("EFFICIENCY ANALYSIS")
print("="*60)
print(f"{'Threads':<10} {'View 1':<15} {'View 2':<15} {'Efficiency V1':<15} {'Efficiency V2':<15}")
print("-"*60)
for t, s1, s2 in zip(threads, view1_speedup, view2_speedup):
    eff1 = (s1 / t) * 100
    eff2 = (s2 / t) * 100
    print(f"{t:<10} {s1:<15.2f} {s2:<15.2f} {eff1:<15.1f}% {eff2:<15.1f}%")

print("\n" + "="*60)
print("KEY OBSERVATIONS")
print("="*60)
print(f"1. Maximum speedup achieved:")
print(f"   - View 1: {view1_speedup[-1]:.2f}x with 6 threads ({(view1_speedup[-1]/6)*100:.1f}% efficiency)")
print(f"   - View 2: {view2_speedup[-1]:.2f}x with 6 threads ({(view2_speedup[-1]/6)*100:.1f}% efficiency)")

print(f"\n2. Efficiency with 2 threads:")
print(f"   - View 1: {(view1_speedup[1]/2)*100:.1f}%")
print(f"   - View 2: {(view2_speedup[1]/2)*100:.1f}%")

print(f"\n3. Speedup is {'NEAR-LINEAR' if view1_speedup[-1] > 5.0 else 'SUB-LINEAR'}")
print(f"   - 6 threads achieve {(view1_speedup[-1]/6)*100:.1f}% of ideal performance")

plt.show()