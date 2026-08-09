import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# GENERATION DATA
# LightRAG: 9 rows → plotted as mean
# LDA variants: 1 row  → plotted as-is
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
    "LightRAG LDA": [       # 3 original runs → averaged
        [0.42, 0.23, 0.28, 0.06, 0.31, 0.14, 0.18, 0.00, 0.01],
        [0.41, 0.23, 0.28, 0.06, 0.31, 0.14, 0.18, 0.00, 0.01],
        [0.42, 0.23, 0.28, 0.06, 0.32, 0.14, 0.18, 0.00, 0.01]
    ],
    "LDA Test 2": [[0.49, 0.29, 0.34, 0.09, 0.39, 0.20, 0.17, 0.01, 0.01]],
    "LDA Test 3": [[0.52, 0.33, 0.36, 0.11, 0.43, 0.23, 0.18, 0.02, 0.00]],
    "LDA Test 4": [[0.55, 0.34, 0.39, 0.12, 0.45, 0.25, 0.18, 0.02, 0.02]],
}

# ─────────────────────────────────────────────
# RETRIEVAL DATA
# ─────────────────────────────────────────────

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
    "LightRAG LDA": [       # 3 original runs → averaged
          [0.24, 0.86, 0.15, 0.90, 0.26, 0.89, 0.13, 0.96],
    [0.23, 0.85, 0.15, 0.88, 0.26, 0.88, 0.12, 0.95],
    [0.24, 0.84, 0.16, 0.88, 0.26, 0.88, 0.14, 0.96]
    ],
    "LDA Test 2": [[0.36, 0.87, 0.28, 0.92, 0.40, 0.93, 0.22, 0.96]],
    "LDA Test 3": [[0.42, 0.92, 0.30, 0.95, 0.42, 0.96, 0.27, 0.97]],
    "LDA Test 4": [[0.34, 0.65, 0.36, 0.96, 0.47, 0.97, 0.22, 0.96]],
}

# ─────────────────────────────────────────────
# PLOT FUNCTION
# ─────────────────────────────────────────────

COLORS = {
    "LightRAG":      "#4C72B0",
    "LightRAG LDA":  "#DD8452",
    "LDA Test 2":        "#55A868",
    "LDA Test 3":        "#C44E52",
    "LDA Test 4":        "#8172B2",
}

def plot_comparison(data, task_groups, title, filename):
    fig, axes = plt.subplots(1, len(task_groups), figsize=(20, 5), sharey=False)
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

    n = len(data)
    bar_width = 0.9 / n

    for ax, (task_label, col_indices, sub_labels) in zip(axes, task_groups):
        x = np.arange(len(sub_labels))
        legend_handles = []

        for i, (model, rows) in enumerate(data.items()):
            vals  = np.mean(np.array(rows), axis=0)[col_indices]
            offset = (i - (n - 1) / 2) * bar_width
            color  = COLORS.get(model, f"C{i}")
            bars   = ax.bar(x + offset, vals, bar_width,
                            label=model, color=color,
                            edgecolor='white', linewidth=0.6)
            legend_handles.append(bars[0])
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val*100:.0f}%",
                        ha='center', va='bottom', fontsize=11, rotation=45)

        ax.set_title(task_label, fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sub_labels, fontsize=13)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Score", fontsize=13)
        ax.yaxis.grid(True, linestyle='--', alpha=0.5)
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               ncol=len(handles),
               fontsize=13,
               frameon=True,
               framealpha=0.9,
               edgecolor='gray',
               facecolor='white',
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.12, 1, 0.95])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved: {filename}")

# ─────────────────────────────────────────────
# GENERATION FIGURE
# ─────────────────────────────────────────────

gen_task_groups = [
    ("Fact Retrieval",       [0, 1], ["ACC", "ROUGE-L"]),
    ("Complex Reasoning",    [2, 3], ["ACC", "ROUGE-L"]),
    ("Contextual Summariz.", [4, 5], ["ACC", "Cov"]),
    ("Creative Generation",  [6, 7, 8], ["ACC", "FS", "Cov"]),
]

plot_comparison(
    generation,
    gen_task_groups,
    title="Generation Evaluation — LightRAG (mean of 9) vs LDA Variants",
    filename="generation_comparison.pdf"
)

# ─────────────────────────────────────────────
# RETRIEVAL FIGURE
# ─────────────────────────────────────────────

ret_task_groups = [
    ("Fact Retrieval",       [0, 1], ["CR", "ER"]),
    ("Complex Reasoning",    [2, 3], ["CR", "ER"]),
    ("Contextual Summariz.", [4, 5], ["CR", "ER"]),
    ("Creative Generation",  [6, 7], ["CR", "ER"]),
]

plot_comparison(
    retrieval,
    ret_task_groups,
    title="Retrieval Evaluation — LightRAG (mean of 9) vs LDA Variants",
    filename="retrieval_comparison.pdf"
)
