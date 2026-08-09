const { readFileSync, writeFileSync } = require('fs');

const folders = [
    ...['1', '2'].map(i => `lightrag_lda_changed_${i}`),
    ...[''].map(i => `lightrag_lda${i}`),
];

const rc = (v) => {
    if (v == null) return 'N/A';
    const s = v.trim();
    if (s === 'inf') return '$\\infty$';
    const n = parseFloat(s);
    if (isNaN(n)) return s;
    return Number.isInteger(n) ? String(n) : n.toFixed(4).replace(/\.?0+$/, '');
};

function parseFile(path) {
    const lines = readFileSync(path, 'utf8').trim().split(/\r?\n|\r/);
    const result = {};
    for (const line of lines) {
        const m = line.match(/^\s+(\w+):\s+(.+)$/);
        if (m) result[m[1]] = m[2].trim();
    }
    return result;
}

function makeTable(folder, d) {
    const label = folder.replace(/_/g, '-');
    const caption = `Graph Structure Metrics for ${folder}.`;
    return `% --- ${folder} ---
\\begin{table}[h!]
\\centering
\\caption{${caption}}
\\label{tab:${label}-metrics}
\\small
\\setlength{\\tabcolsep}{6pt}
\\begin{tabular}{lr}
\\toprule
\\textbf{Metric} & \\textbf{Value} \\\\ \\midrule
\\multicolumn{2}{l}{\\textit{Basic Structure}} \\\\ \\midrule
Number of Nodes                     & ${rc(d.num_nodes)}   \\\\
Number of Edges                     & ${rc(d.num_edges)}   \\\\
Average Degree                      & ${rc(d.average_degree)}   \\\\
Density                             & ${rc(d.density)} \\\\ \\midrule
\\multicolumn{2}{l}{\\textit{Component Analysis}} \\\\ \\midrule
Number of Components                & ${rc(d.num_components)}    \\\\
Largest Component Size              & ${rc(d.largest_component_size)}   \\\\
Average Component Size              & ${rc(d.average_component_size)}  \\\\
Median Component Size               & ${rc(d.median_component_size)}   \\\\
Trimmed Mean Component Size         & ${rc(d.trimmed_mean_component_size)}   \\\\
Geometric Mean Component Size       & ${rc(d.geometric_mean_component_size)}   \\\\
Harmonic Mean Component Size        & ${rc(d.harmonic_mean_component_size)}   \\\\
Components Excl. Isolated           & ${rc(d.num_components_excluding_isolated)}     \\\\
Components Above Average Size       & ${rc(d.num_components_above_average)}      \\\\ \\midrule
\\multicolumn{2}{l}{\\textit{Node Degree Distribution}} \\\\ \\midrule
Nodes Excl. Isolated                & ${rc(d.num_nodes_excluding_isolated)}   \\\\
Isolated Nodes                      & ${rc(d.num_isolated_nodes)}    \\\\
Nodes with Degree $> 1$             & ${rc(d.num_nodes_degree_above_1)}   \\\\
Nodes with Degree $> 2$             & ${rc(d.num_nodes_degree_above_2)}   \\\\
Nodes with Degree $> 3$             & ${rc(d.num_nodes_degree_above_3)}   \\\\ \\midrule
\\multicolumn{2}{l}{\\textit{Clustering \\& Connectivity}} \\\\ \\midrule
Average Clustering Coefficient      & ${rc(d.average_clustering_coefficient)}   \\\\
Diameter                            & ${rc(d.diameter)} \\\\
\\bottomrule
\\end{tabular}
\\end{table}`;
}

const out = [];
for (const folder of folders) {
    const path = require('path').join('results', folder, 'Medical', 'indexing_results.json');
    try {
        const d = parseFile(path);
        out.push(makeTable(folder, d));
    } catch (e) {
        out.push(`% MISSING: ${path} — ${e.message}`);
    }
}

writeFileSync('results/tables.txt', out.join('\n\n') + '\n', 'utf8');
console.log('Written to results/tables.txt');
