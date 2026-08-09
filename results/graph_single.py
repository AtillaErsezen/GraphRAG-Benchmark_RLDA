import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────

generation = {
    "LightRAG": [
        [0.59, 0.38, 0.49, 0.16, 0.53, 0.31, 0.20, 0.01, 0.04],
        [0.59, 0.38, 0.49, 0.16, 0.53, 0.31, 0.20, 0.01, 0.04],
        [0.60, 0.38, 0.49, 0.16, 0.53, 0.31, 0.20, 0.01, 0.03],
        [0.59, 0.38, 0.49, 0.16, 0.56, 0.34, 0.21, 0.01, 0.04],
        [0.60, 0.38, 0.49, 0.16, 0.56, 0.35, 0.21, 0.02, 0.04],
        [0.59, 0.38, 0.49, 0.16, 0.56, 0.35, 0.21, 0.01, 0.04],
        [0.59, 0.38, 0.50, 0.17, 0.56, 0.33, 0.19, 0.02, 0.03],
        [0.60, 0.38, 0.50, 0.17, 0.56, 0.34, 0.20, 0.01, 0.04],
        [0.59, 0.38, 0.49, 0.17, 0.55, 0.33, 0.20, 0.02, 0.03],
    ],
    "LightRAG LDA": [
        [0.42, 0.23, 0.28, 0.06, 0.31, 0.14, 0.18, 0.00, 0.01],
        [0.41, 0.23, 0.28, 0.06, 0.31, 0.14, 0.18, 0.00, 0.01],
        [0.42, 0.23, 0.28, 0.06, 0.32, 0.14, 0.18, 0.00, 0.01],
    ],
    "LDA Test 2": [[0.49, 0.29, 0.34, 0.09, 0.39, 0.20, 0.17, 0.01, 0.01]],
    "LDA Test 3": [[0.52, 0.33, 0.36, 0.11, 0.43, 0.23, 0.18, 0.02, 0.00]],
    "LDA Test 4": [[0.55, 0.34, 0.39, 0.12, 0.45, 0.25, 0.18, 0.02, 0.02]],
}

retrieval = {
    "LightRAG": [
        [0.58, 0.88, 0.45, 0.91, 0.60, 0.92, 0.26, 0.94],
        [0.58, 0.88, 0.45, 0.91, 0.59, 0.91, 0.26, 0.95],
        [0.58, 0.88, 0.45, 0.91, 0.60, 0.92, 0.25, 0.94],
        [0.63, 0.89, 0.47, 0.89, 0.62, 0.92, 0.29, 0.94],
        [0.63, 0.89, 0.46, 0.89, 0.63, 0.92, 0.30, 0.94],
        [0.63, 0.88, 0.46, 0.89, 0.62, 0.92, 0.29, 0.95],
        [0.58, 0.87, 0.43, 0.89, 0.61, 0.92, 0.27, 0.96],
        [0.58, 0.88, 0.43, 0.89, 0.61, 0.92, 0.25, 0.95],
        [0.58, 0.87, 0.44, 0.89, 0.61, 0.92, 0.26, 0.94],
    ],
    "LightRAG LDA": [
        [0.24, 0.86, 0.15, 0.90, 0.26, 0.89, 0.13, 0.96],
        [0.23, 0.85, 0.15, 0.88, 0.26, 0.88, 0.12, 0.95],
        [0.24, 0.84, 0.16, 0.88, 0.26, 0.88, 0.14, 0.96],
    ],
    "LDA Test 2": [[0.36, 0.87, 0.28, 0.92, 0.40, 0.93, 0.22, 0.96]],
    "LDA Test 3": [[0.42, 0.92, 0.30, 0.95, 0.42, 0.96, 0.27, 0.97]],
    "LDA Test 4": [[0.34, 0.65, 0.36, 0.96, 0.47, 0.97, 0.22, 0.96]],
}

# ─────────────────────────────────────────────
# TASK DEFINITIONS
# ─────────────────────────────────────────────

tasks = [
    ("Fact Retrieval",       [0, 1], ["ACC", "ROUGE-L"], [0, 1], ["CR", "ER"]),
    ("Complex Reasoning",    [2, 3], ["ACC", "ROUGE-L"], [2, 3], ["CR", "ER"]),
    ("Contextual Summariz.", [4, 5], ["ACC", "Cov"],     [4, 5], ["CR", "ER"]),
    ("Creative Generation",  [6, 7, 8], ["ACC", "FS", "Cov"], [6, 7], ["CR", "ER"]),
]

# ─────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────

MODEL_COLORS = {
    "LightRAG":     "#4C72B0",
    "LightRAG LDA": "#DD8452",
    "LDA Test 2":   "#55A868",
    "LDA Test 3":   "#C44E52",
    "LDA Test 4":   "#8172B2",
}

# outer task zone colors (light)
TASK_COLORS = ["#D6E8FF", "#D6F5D6", "#EDD6FF", "#FFD6D6"]
# inner gen/ret subzone overlays (slightly darker tint)
GEN_OVERLAY  = "#B0CCEE"
RET_OVERLAY  = "#F5C89A"

TICK_SPACING = 1.8   # distance between metric ticks within a subzone
SUBZONE_GAP  = 2.0   # gap between gen and ret subzones inside a task
TASK_GAP     = 3.5   # gap between tasks
PAD          = 0.7   # padding inside each subzone background

# ─────────────────────────────────────────────
# BUILD X POSITIONS
# ─────────────────────────────────────────────

models     = list(generation.keys())
n_models   = len(models)
bar_width  = 0.7 / n_models

cursor = 0.0
task_zones   = []   # (x0, x1, label, color)
gen_zones    = []   # (x0, x1) per task
ret_zones    = []   # (x0, x1) per task
gen_ticks    = []   # list of (x_pos, label) per task
ret_ticks    = []   # list of (x_pos, label) per task

for task_label, gen_cols, gen_labels, ret_cols, ret_labels, task_color in zip(
        *zip(*tasks), TASK_COLORS):

    task_x0 = cursor

    # gen positions
    gx = [cursor + i * TICK_SPACING for i in range(len(gen_labels))]
    cursor = gx[-1] + SUBZONE_GAP

    # ret positions
    rx = [cursor + i * TICK_SPACING for i in range(len(ret_labels))]
    cursor = rx[-1] + TASK_GAP

    task_x1 = rx[-1]

    task_zones.append((task_x0 - PAD, task_x1 + PAD, task_label, task_color))
    gen_zones.append((gx[0] - PAD * 0.7, gx[-1] + PAD * 0.7))
    ret_zones.append((rx[0] - PAD * 0.7, rx[-1] + PAD * 0.7))
    gen_ticks.append(list(zip(gx, gen_labels)))
    ret_ticks.append(list(zip(rx, ret_labels)))

# ─────────────────────────────────────────────
# DRAW
# ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(42, 9))
fig.suptitle("LightRAG vs LDA Variants — Full Evaluation",
             fontsize=20, fontweight='bold', y=0.98)

Y_MAX = 120

# ── background zones ──
for (x0, x1, label, color), (gx0, gx1), (rx0, rx1) in zip(task_zones, gen_zones, ret_zones):
    # task zone
    ax.axvspan(x0, x1, color=color, alpha=0.5, zorder=0)
    # gen subzone overlay
    ax.axvspan(gx0, gx1, color=GEN_OVERLAY, alpha=0.35, zorder=1)
    # ret subzone overlay
    ax.axvspan(rx0, rx1, color=RET_OVERLAY, alpha=0.35, zorder=1)
    # task label at top
    ax.text((x0 + x1) / 2, Y_MAX * 0.99, label,
            ha='center', va='top', fontsize=15, fontweight='bold', color='#333333')
    # gen/ret subzone labels
    ax.text((gx0 + gx1) / 2, Y_MAX * 0.91, "Generation",
            ha='center', va='top', fontsize=11, color='#2255AA', style='italic')
    ax.text((rx0 + rx1) / 2, Y_MAX * 0.91, "Retrieval",
            ha='center', va='top', fontsize=11, color='#AA5522', style='italic')

# ── bars ──
tick_positions = []
tick_labels    = []

for task_idx, (_, gen_cols, _, ret_cols, _, _) in enumerate(zip(*zip(*tasks), TASK_COLORS)):
    for model_idx, model in enumerate(models):
        offset = (model_idx - (n_models - 1) / 2) * bar_width
        color  = MODEL_COLORS[model]
        gen_rows = generation[model]
        ret_rows = retrieval[model]
        gen_mean = np.mean(np.array(gen_rows), axis=0)[gen_cols] * 100
        ret_mean = np.mean(np.array(ret_rows), axis=0)[ret_cols] * 100

        for (x_pos, _), val in zip(gen_ticks[task_idx], gen_mean):
            ax.bar(x_pos + offset, val, bar_width, color=color,
                   edgecolor='white', linewidth=0.5,
                   label=model if task_idx == 0 and model_idx == 0 else "_nolegend_",
                   zorder=3)
            ax.text(x_pos + offset, val + 0.5, f"{val:.0f}%",
                    ha='center', va='bottom', fontsize=9, rotation=45, zorder=4)

        for (x_pos, _), val in zip(ret_ticks[task_idx], ret_mean):
            ax.bar(x_pos + offset, val, bar_width, color=color,
                   edgecolor='white', linewidth=0.5, zorder=3)
            ax.text(x_pos + offset, val + 0.5, f"{val:.0f}%",
                    ha='center', va='bottom', fontsize=9, rotation=45, zorder=4)

    # collect tick positions/labels
    for x_pos, lbl in gen_ticks[task_idx] + ret_ticks[task_idx]:
        tick_positions.append(x_pos)
        tick_labels.append(lbl)

# ── axes styling ──
ax.set_xticks(tick_positions)
ax.set_xticklabels(tick_labels, fontsize=12)
ax.set_ylim(0, Y_MAX)
ax.set_ylabel("Score (%)", fontsize=13)
ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=2)
ax.set_axisbelow(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(task_zones[0][0] - 0.3, task_zones[-1][1] + 0.3)

# ── legend ──
legend_handles = [mpatches.Patch(color=MODEL_COLORS[m], label=m) for m in models]
fig.legend(handles=legend_handles, loc='lower center', ncol=len(models),
           fontsize=13, frameon=True, framealpha=0.9,
           edgecolor='gray', facecolor='white', bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.08, 1, 0.95])
plt.savefig('results/single_comparison.pdf', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: results/single_comparison.pdf")
