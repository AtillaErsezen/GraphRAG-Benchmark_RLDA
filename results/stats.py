import numpy as np

# ─────────────────────────────────────────────
# GENERATION DATA
# ─────────────────────────────────────────────

generation = {
    "LightRAG Run 1":     [
        [0.59, 0.38, 0.49, 0.16, 0.53, 0.31, 0.20, 0.01, 0.04],
        [0.59, 0.38, 0.49, 0.16, 0.53, 0.31, 0.20, 0.01, 0.04],
        [0.60, 0.38, 0.49, 0.16, 0.53, 0.31, 0.20, 0.01, 0.03],
    ],
    "LightRAG Run 2":     [
        [0.59, 0.38, 0.49, 0.16, 0.56, 0.34, 0.21, 0.01, 0.04],
        [0.60, 0.38, 0.49, 0.16, 0.56, 0.35, 0.21, 0.02, 0.04],
        [0.59, 0.38, 0.49, 0.16, 0.56, 0.35, 0.21, 0.01, 0.04],
    ],
    "LightRAG Run 3":     [
        [0.59, 0.38, 0.50, 0.17, 0.56, 0.33, 0.19, 0.02, 0.03],
        [0.60, 0.38, 0.50, 0.17, 0.56, 0.34, 0.20, 0.01, 0.04],
        [0.59, 0.38, 0.49, 0.17, 0.55, 0.33, 0.20, 0.02, 0.03],
    ],
    "LightRAG LDA Run 1": [
        [0.42, 0.23, 0.28, 0.06, 0.31, 0.14, 0.18, 0.00, 0.01],
        [0.41, 0.23, 0.28, 0.06, 0.31, 0.14, 0.18, 0.00, 0.01],
        [0.42, 0.23, 0.28, 0.06, 0.32, 0.14, 0.18, 0.00, 0.01],
    ],
    "LightRAG LDA Run 2": [
        [0.42, 0.23, 0.28, 0.06, 0.31, 0.14, 0.18, 0.00, 0.01],
        [0.41, 0.23, 0.28, 0.06, 0.31, 0.14, 0.18, 0.00, 0.01],
        [0.42, 0.23, 0.28, 0.06, 0.32, 0.14, 0.18, 0.00, 0.01],
    ],
    "LightRAG LDA Run 3": [
        [0.40, 0.22, 0.29, 0.06, 0.31, 0.14, 0.17, 0.00, 0.01],
        [0.40, 0.22, 0.29, 0.06, 0.31, 0.14, 0.17, 0.00, 0.01],
        [0.40, 0.22, 0.29, 0.06, 0.31, 0.14, 0.18, 0.00, 0.02],
    ],
}

# ─────────────────────────────────────────────
# RETRIEVAL DATA
# ─────────────────────────────────────────────

retrieval = {
    "LightRAG Run 1":     [
        [0.58, 0.88, 0.45, 0.91, 0.60, 0.92, 0.26, 0.94],
        [0.58, 0.88, 0.45, 0.91, 0.59, 0.91, 0.26, 0.95],
        [0.58, 0.88, 0.45, 0.91, 0.60, 0.92, 0.25, 0.94],
    ],
    "LightRAG Run 2":     [
        [0.63, 0.89, 0.47, 0.89, 0.62, 0.92, 0.29, 0.94],
        [0.63, 0.89, 0.46, 0.89, 0.63, 0.92, 0.30, 0.94],
        [0.63, 0.88, 0.46, 0.89, 0.62, 0.92, 0.29, 0.95],
    ],
    "LightRAG Run 3":     [
        [0.58, 0.87, 0.43, 0.89, 0.61, 0.92, 0.27, 0.96],
        [0.58, 0.88, 0.43, 0.89, 0.61, 0.92, 0.25, 0.95],
        [0.58, 0.87, 0.44, 0.89, 0.61, 0.92, 0.26, 0.94],
    ],
    "LightRAG LDA Run 1": [
        [0.24, 0.86, 0.15, 0.90, 0.26, 0.89, 0.13, 0.96],
        [0.23, 0.85, 0.15, 0.88, 0.26, 0.88, 0.12, 0.95],
        [0.24, 0.84, 0.16, 0.88, 0.26, 0.88, 0.14, 0.96],
    ],
    "LightRAG LDA Run 2": [
        [0.24, 0.84, 0.16, 0.91, 0.26, 0.89, 0.11, 0.95],
        [0.23, 0.85, 0.15, 0.90, 0.26, 0.90, 0.11, 0.94],
        [0.24, 0.84, 0.16, 0.89, 0.27, 0.88, 0.11, 0.95],
    ],
    "LightRAG LDA Run 3": [
        [0.23, 0.83, 0.17, 0.89, 0.25, 0.89, 0.11, 0.95],
        [0.23, 0.83, 0.18, 0.90, 0.25, 0.88, 0.12, 0.95],
        [0.23, 0.84, 0.17, 0.89, 0.25, 0.91, 0.13, 0.96],
    ],
}

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

# Add \usepackage{colortbl} and \usepackage{xcolor} to your preamble,
# then define: \definecolor{statgray}{gray}{0.88}

def stat_rows(arr, label):
    """Print Mean, Var and Std as three gray-shaded rows."""
    mean = np.mean(arr, axis=0)
    var  = np.var( arr, axis=0, ddof=1)
    std  = np.std( arr, axis=0, ddof=1)

    mean_cells = " & ".join(f"{v:.4f}" for v in mean)
    var_cells  = " & ".join(f"{v:.6f}" for v in var)
    std_cells  = " & ".join(f"{v:.4f}" for v in std)

    print(f"% {label} -- stats")
    print(f"\\rowcolor{{statgray}} \\textit{{Mean}} & {mean_cells} \\\\")
    print(f"\\rowcolor{{statgray}} \\textit{{Var}}  & {var_cells}  \\\\")
    print(f"\\rowcolor{{statgray}} \\textit{{Std}}  & {std_cells}  \\\\ \\midrule")
    print()

def collect_rows(data, keys):
    return np.vstack([data[k] for k in keys])

# ─────────────────────────────────────────────
# PRINT STAT ROWS
# ─────────────────────────────────────────────

def print_all_stats(data, table_name, n_cols):
    print(f"% ════════════════════════════════════════")
    print(f"% {table_name}")
    print(f"% ════════════════════════════════════════\n")

    # ── Per-run stats ──────────────────────────
    for run_name, rows in data.items():
        stat_rows(np.array(rows), run_name)

    # ── Overall: LightRAG (all 9 rows) ─────────
    lr_keys  = [k for k in data if "LDA" not in k]
    lda_keys = [k for k in data if "LDA"     in k]

    stat_rows(collect_rows(data, lr_keys),  "LightRAG Overall (across all runs)")
    stat_rows(collect_rows(data, lda_keys), "LightRAG LDA Overall (across all runs)")


print_all_stats(generation, "GENERATION", n_cols=9)
print_all_stats(retrieval,  "RETRIEVAL",  n_cols=8)