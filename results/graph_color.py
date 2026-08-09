import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# GENERATION DATA
# LightRAG: 9 rows → plotted as mean
# RLDA variants: 1 row  → plotted as-is
# ─────────────────────────────────────────────

generation = {
    "LightRAG": [   # lightrag_1 evals 1-3, lightrag_2 evals 1-3, lightrag_3 evals 1-3
        [0.5918, 0.3898, 0.4965, 0.1681, 0.5341, 0.3123, 0.2052, 0.0131, 0.0473],
        [0.5970, 0.3898, 0.4986, 0.1681, 0.5373, 0.3127, 0.2055, 0.0123, 0.0412],
        [0.6008, 0.3898, 0.4959, 0.1681, 0.5371, 0.3150, 0.2084, 0.0141, 0.0397],
        [0.5948, 0.3840, 0.4990, 0.1692, 0.5653, 0.3475, 0.2144, 0.0108, 0.0492],
        [0.6003, 0.3840, 0.4973, 0.1692, 0.5641, 0.3554, 0.2160, 0.0204, 0.0498],
        [0.5962, 0.3840, 0.4946, 0.1692, 0.5615, 0.3535, 0.2153, 0.0128, 0.0468],
        [0.5939, 0.3812, 0.5037, 0.1715, 0.5674, 0.3398, 0.1986, 0.0215, 0.0349],
        [0.6053, 0.3812, 0.5001, 0.1715, 0.5676, 0.3439, 0.2014, 0.0190, 0.0436],
        [0.5987, 0.3812, 0.4999, 0.1715, 0.5598, 0.3400, 0.2045, 0.0271, 0.0368],
    ],
    "RLDA 1": [     # lightrag_lda_1 evals 1-3, lightrag_lda_2 evals 1-3, lightrag_lda_3 evals 1-3
        [0.4211, 0.2425, 0.2871, 0.0659, 0.3214, 0.1387, 0.1848, 0.0167, 0.0187],
        [0.4248, 0.2425, 0.2862, 0.0659, 0.3220, 0.1406, 0.1835, 0.0090, 0.0168],
        [0.4230, 0.2425, 0.2897, 0.0659, 0.3156, 0.1379, 0.1823, 0.0061, 0.0175],
        [0.4205, 0.2303, 0.2891, 0.0668, 0.3154, 0.1402, 0.1800, 0.0082, 0.0181],
        [0.4165, 0.2303, 0.2893, 0.0668, 0.3187, 0.1422, 0.1810, 0.0000, 0.0169],
        [0.4200, 0.2303, 0.2888, 0.0668, 0.3203, 0.1453, 0.1810, 0.0079, 0.0197],
        [0.4044, 0.2294, 0.2956, 0.0697, 0.3180, 0.1404, 0.1781, 0.0000, 0.0170],
        [0.4060, 0.2294, 0.2983, 0.0697, 0.3178, 0.1424, 0.1776, 0.0019, 0.0137],
        [0.4076, 0.2294, 0.2961, 0.0697, 0.3185, 0.1449, 0.1835, 0.0000, 0.0224],
    ],
    "RLDA 2": [[0.4978, 0.2904, 0.3413, 0.0918, 0.3944, 0.2042, 0.1779, 0.0154, 0.0166]],
    "RLDA 3": [[0.5293, 0.3308, 0.3630, 0.1105, 0.4318, 0.2318, 0.1827, 0.0093, 0.0284]],
    "RLDA 4": [[0.5524, 0.3485, 0.3923, 0.1217, 0.4520, 0.2566, 0.1895, 0.0236, 0.0290]],
}

# ─────────────────────────────────────────────
# RETRIEVAL DATA
# ─────────────────────────────────────────────

retrieval = {
    "LightRAG": [   # lightrag_1 evals 1-3, lightrag_2 evals 1-3, lightrag_3 evals 1-3
        [0.5858, 0.8881, 0.4528, 0.9156, 0.6003, 0.9208, 0.2620, 0.9481],
        [0.5883, 0.8850, 0.4519, 0.9127, 0.5978, 0.9116, 0.2620, 0.9544],
        [0.5838, 0.8871, 0.4514, 0.9136, 0.6012, 0.9221, 0.2575, 0.9480],
        [0.6384, 0.8914, 0.4720, 0.8907, 0.6280, 0.9291, 0.2922, 0.9445],
        [0.6357, 0.8930, 0.4661, 0.8981, 0.6332, 0.9263, 0.3042, 0.9496],
        [0.6368, 0.8891, 0.4681, 0.8929, 0.6263, 0.9290, 0.2937, 0.9507],
        [0.5849, 0.8766, 0.4391, 0.8920, 0.6159, 0.9220, 0.2741, 0.9620],
        [0.5849, 0.8803, 0.4391, 0.8975, 0.6151, 0.9257, 0.2545, 0.9525],
        [0.5817, 0.8745, 0.4445, 0.8979, 0.6116, 0.9219, 0.2605, 0.9419],
    ],
    "RLDA 1": [     # lightrag_lda_1 evals 1-3, lightrag_lda_2 evals 1-3, lightrag_lda_3 evals 1-3
        [0.2404, 0.8622, 0.1582, 0.9081, 0.2647, 0.8942, 0.1386, 0.9610],
        [0.2361, 0.8553, 0.1572, 0.8887, 0.2647, 0.8856, 0.1235, 0.9579],
        [0.2402, 0.8496, 0.1626, 0.8869, 0.2604, 0.8803, 0.1431, 0.9622],
        [0.2434, 0.8463, 0.1611, 0.9149, 0.2690, 0.8927, 0.1190, 0.9569],
        [0.2400, 0.8507, 0.1596, 0.9047, 0.2699, 0.9090, 0.1114, 0.9448],
        [0.2441, 0.8426, 0.1645, 0.8915, 0.2716, 0.8827, 0.1130, 0.9582],
        [0.2384, 0.8376, 0.1768, 0.8990, 0.2500, 0.8987, 0.1190, 0.9567],
        [0.2391, 0.8322, 0.1807, 0.9046, 0.2561, 0.8886, 0.1265, 0.9570],
        [0.2370, 0.8416, 0.1783, 0.8988, 0.2526, 0.9178, 0.1310, 0.9683],
    ],
    "RLDA 2": [[0.3636, 0.8780, 0.2819, 0.9298, 0.4074, 0.9303, 0.2229, 0.9630]],
    "RLDA 3": [[0.4240, 0.9276, 0.3094, 0.9536, 0.4204, 0.9681, 0.2756, 0.9705]],
    "RLDA 4": [[0.3486, 0.6527, 0.3630, 0.9667, 0.4732, 0.9776, 0.2289, 0.9694]],
}

# ─────────────────────────────────────────────
# TASK GROUPS  (gen_cols, gen_labels, ret_cols, ret_labels)
# ─────────────────────────────────────────────

task_groups = [
    ("Fact Retrieval",
     [0, 1], ["ACC", "ROUGE-L"],
     [0, 1], ["CR", "ER"]),
    ("Complex Reasoning",
     [2, 3], ["ACC", "ROUGE-L"],
     [2, 3], ["CR", "ER"]),
    ("Contextual Summariz.",
     [4, 5], ["ACC", "Cov"],
     [4, 5], ["CR", "ER"]),
    ("Creative Generation",
     [6, 7, 8], ["ACC", "FS", "Cov"],
     [6, 7],   ["CR", "ER"]),
]

# ─────────────────────────────────────────────
# COLORS
# ─────────────────────────────────────────────

COLORS = {
    "LightRAG":     "#4C72B0",
    "RLDA 1": "#DD8452",
    "RLDA 2":   "#55A868",
    "RLDA 3":   "#C44E52",
    "RLDA 4":   "#8172B2",
}

GEN_ZONE_COLOR = "#D0E8FF"   # light blue
RET_ZONE_COLOR = "#FFE8D0"   # light orange

GAP = 1   # space between generation and retrieval zones

# ─────────────────────────────────────────────
# COMBINED PLOT FUNCTION
# ─────────────────────────────────────────────

TICK_SPACING = 1.8  # distance between metric ticks (increase to spread bars apart)

def plot_combined(gen_data, ret_data, task_groups, title, filename):
    n_models = len(gen_data)
    bar_width = 0.9 / n_models

    fig, axes = plt.subplots(2, 2, figsize=(32, 20), sharey=False)
    axes = axes.flatten()
    fig.suptitle(title, fontsize=45, fontweight='bold', y=0.98)

    for ax, (task_label, gen_cols, gen_labels, ret_cols, ret_labels) in zip(axes, task_groups):
        n_gen = len(gen_labels)
        n_ret = len(ret_labels)

        gen_x = np.arange(n_gen, dtype=float) * TICK_SPACING
        ret_x = np.arange(n_ret, dtype=float) * TICK_SPACING + n_gen * TICK_SPACING + GAP

        # ── colored background zones ──
        gen_pad = 0.8
        ret_pad = 0.8
        ax.axvspan(gen_x[0] - gen_pad, gen_x[-1] + gen_pad,
                   color=GEN_ZONE_COLOR, zorder=0)
        ax.axvspan(ret_x[0] - ret_pad, ret_x[-1] + ret_pad,
                   color=RET_ZONE_COLOR, zorder=0)

        # ── zone labels ──
        ax.text((gen_x[0] + gen_x[-1]) / 2, 158, "Generation",
                ha='center', va='top', fontsize=40, color='#2255AA',
                fontweight='bold')
        ax.text((ret_x[0] + ret_x[-1]) / 2, 158, "Retrieval",
                ha='center', va='top', fontsize=40, color='#AA5522',
                fontweight='bold')

        # ── bars: first pass (draw only) ──
        label_queue = []  # (tick_x, bar_x, val, color)
        for i, (model, rows) in enumerate(gen_data.items()):
            offset   = (i - (n_models - 1) / 2) * bar_width
            color    = COLORS.get(model, f"C{i}")
            gen_vals = np.mean(np.array(rows), axis=0)[gen_cols] * 100
            ret_vals = np.mean(np.array(list(ret_data.values())[i]), axis=0)[ret_cols] * 100

            for x_pos, val in zip(gen_x + offset, gen_vals):
                ax.bar(x_pos, val, bar_width, color=color,
                       edgecolor='white', linewidth=0.6,
                       label=model if x_pos == gen_x[0] + offset else "_nolegend_",
                       zorder=2)
                tick = gen_x[int(np.argmin(np.abs(gen_x - x_pos)))]
                label_queue.append((round(float(tick), 5), x_pos, float(val), color))

            for x_pos, val in zip(ret_x + offset, ret_vals):
                ax.bar(x_pos, val, bar_width, color=color,
                       edgecolor='white', linewidth=0.6, zorder=2)
                tick = ret_x[int(np.argmin(np.abs(ret_x - x_pos)))]
                label_queue.append((round(float(tick), 5), x_pos, float(val), color))

        # ── labels: second pass (sort by value per group, enforce min spacing) ──
        groups = {}
        for tick, xp, val, color in label_queue:
            groups.setdefault(tick, []).append((xp, val, color))

        LABEL_CLEARANCE = 3   # gap between bar top and its label
        LABEL_SPACING   = 10  # min vertical distance between adjacent labels
        for items in groups.values():
            cur_y = None
            for xp, val, color in sorted(items, key=lambda t: t[1]):
                y = val + LABEL_CLEARANCE
                if cur_y is not None:
                    y = max(y, cur_y + LABEL_SPACING)
                cur_y = y
                ax.text(xp, y, f"{round(val):.0f}%",
                        ha='center', va='bottom', fontsize=32,
                        color=color, fontweight='bold')

        # ── x ticks: combine gen and ret labels ──
        all_x      = np.concatenate([gen_x, ret_x])
        all_labels = gen_labels + ret_labels
        ax.set_xticks(all_x)
        ax.set_xticklabels(all_labels, fontsize=30)

        ax.set_title(task_label, fontsize=40, fontweight='bold')
        ax.set_xlim(gen_x[0] - gen_pad - 0.2, ret_x[-1] + ret_pad + 0.2)
        ax.set_ylim(0, 160)
        ax.set_ylabel("Score", fontsize=15)
        ax.yaxis.grid(True, linestyle='--', alpha=0.4, zorder=1)
        ax.set_axisbelow(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # ── legend (one entry per model, taken from first subplot) ──
    handles, labels = axes[0].get_legend_handles_labels()  # axes is already flattened
    # deduplicate while preserving order
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen and not l.startswith("_"):
            seen[l] = h
    fig.legend(seen.values(), seen.keys(),
               loc='lower center',
               ncol=len(seen),
               fontsize=35,
               frameon=True,
               framealpha=0.9,
               edgecolor='gray',
               facecolor='white',
               bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    png_filename = filename.rsplit('.', 1)[0] + '.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.savefig(png_filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename} and {png_filename}")

# ─────────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────────

plot_combined(
    generation,
    retrieval,
    task_groups,
    title="LightRAG vs RLDA Variants — Generation & Retrieval Evaluation",
    filename="comparison.pdf"
)
