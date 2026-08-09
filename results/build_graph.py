import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

lightrag_runs = [7940.52, 8275.44, 7420.50]

labels = ['LightRAG (mean)', 'RLDA 1', 'RLDA 2', 'RLDA 3', 'RLDA 4']

times = [sum(lightrag_runs) / len(lightrag_runs), 386.277, 1162.556, 1755.639, 2687.015]

colors = ['#4C72B0', '#DD8452', '#E8A070', '#C45E30', '#8B2500']

fig, ax = plt.subplots(figsize=(14, 4))

# draw bars tallest → shortest
sorted_indices = sorted(range(len(labels)), key=lambda i: times[i], reverse=True)
for idx in sorted_indices:
    ax.barh(0, times[idx], color=colors[idx], height=0.5,
            edgecolor='white', linewidth=0.8, label=labels[idx])

lightrag = sorted([i for i in range(len(labels)) if times[i] > 3000], key=lambda i: times[i])
lda      = sorted([i for i in range(len(labels)) if times[i] <= 3000], key=lambda i: times[i])

above_y = np.linspace(0.32, 0.52, len(lightrag))
below_y = np.linspace(-0.32, -0.62, len(lda))

for y_lbl, idx in zip(above_y, lightrag):
    ax.plot([times[idx], times[idx]], [0.25, y_lbl],
            color=colors[idx], lw=0.8, alpha=0.7)
    ax.text(times[idx], y_lbl + 0.02, f'{times[idx]:,.0f}s',
            ha='center', va='bottom', fontsize=13,
            color=colors[idx], fontweight='bold')

for y_lbl, idx in zip(below_y, lda):
    ax.plot([times[idx], times[idx]], [-0.25, y_lbl],
            color=colors[idx], lw=0.8, alpha=0.7)
    ax.text(times[idx], y_lbl - 0.02, f'{times[idx]:,.0f}s',
            ha='center', va='top', fontsize=13,
            color=colors[idx], fontweight='bold')

ax.set_yticks([])
ax.set_xlabel('Build Time (seconds)', fontsize=16)
ax.set_title('Graph Build Time Comparison', fontsize=22, fontweight='bold')
ax.set_xlim(0, max(times) * 1.1)
ax.set_ylim(-1.0, 1.0)
ax.xaxis.grid(True, linestyle='--', alpha=0.4, zorder=0)
ax.plot([0, 0], [-1.0, 0.25], color='black', linewidth=1.0, zorder=5)
ax.tick_params(axis='x', labelsize=14)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)

handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors]
ax.legend(handles, labels, fontsize=13, frameon=True, framealpha=0.9,
          edgecolor='gray', loc='upper center',
          bbox_to_anchor=(0.5, 0.98), ncol=7)

plt.tight_layout()
plt.savefig('results/build_time_comparison.pdf', dpi=150, bbox_inches='tight')
plt.savefig('results/build_time_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: build_time_comparison.pdf/.png")
