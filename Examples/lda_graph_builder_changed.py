import json
import os
import sys
import time
import hashlib
from collections import defaultdict
from typing import List, Dict, Any, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'LightRAG'))

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation
from lightrag.operate import chunking_by_token_size
from lightrag.prompt import GRAPH_FIELD_SEP

# ─── Constants ────────────────────────────────────────────────────────────────
GLOBAL_N_TOPICS       = 10    # depth-0: corpus-level topic count
SUB_N_TOPICS          = 3     # depth-1: sub-chunk topic count (capped lower)
CROSS_TOPIC_THRESHOLD = 0.25  # min soft-assignment probability for cross-topic edge
SUB_CHUNK_TOKEN_SIZE  = 200   # max tokens per depth-1 sub-chunk
SUB_CHUNK_OVERLAP     = 20    # overlap tokens between depth-1 sub-chunks

MEDICAL_STOPWORDS = frozenset({
    "patient", "patients", "study", "studies", "clinical", "clinically",
    "analysis", "analyzed", "results", "result", "data",
    "associated", "significant", "significantly", "method", "methods",
    "showed", "shown", "observed", "reported", "group", "groups",
    "including", "based", "used", "using", "compared",
    "however", "therefore", "whereas", "within", "between", "among",
    "without", "after", "before", "during", "following",
})
COMBINED_STOPWORDS = list(ENGLISH_STOP_WORDS | MEDICAL_STOPWORDS)


# ─── LDA tree construction ────────────────────────────────────────────────────

def perform_lda_on_layer(docs: List[str], depth: int) -> Optional[Dict]:
    """
    Fit LDA on `docs` at the given recursion depth.

    depth=0: corpus chunks (1200 tok each) → up to GLOBAL_N_TOPICS topics.
             Each chunk is sub-chunked (200 tok) and recursed to depth=1.
    depth≥1: sub-chunks → up to SUB_N_TOPICS topics, then leaf.

    Returns a tree dict with keys: depth, topics, results.
    Each result carries `all_probs` (full topic distribution) for soft edges.
    """
    print(f"[LDA] depth={depth} | {len(docs)} documents")

    if not docs:
        return None

    # Single very-short document → leaf immediately
    if len(docs) == 1 and len(docs[0].split()) <= 50:
        return {"status": "leaf", "content": docs[0]}

    vectorizer = CountVectorizer(
        stop_words=COMBINED_STOPWORDS,
        token_pattern=r"(?u)\b\w+\b",
        min_df=1,
    )
    try:
        dtm = vectorizer.fit_transform(docs)
        n_topics_cap = GLOBAL_N_TOPICS if depth == 0 else SUB_N_TOPICS
        n_topics = min(n_topics_cap, dtm.shape[0], dtm.shape[1])
        if n_topics < 1:
            n_topics = 1
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
        lda.fit(dtm)
        doc_topic_probs = lda.transform(dtm)
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        print(f"[LDA] depth={depth} | LDA failed ({e}), returning leaf")
        return {"status": "leaf", "content": docs}

    topics = []
    for idx, weights in enumerate(lda.components_):
        top_indices = weights.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_indices if i < len(feature_names)]
        topics.append({"id": idx + 1, "words": top_words})
        print(f"[LDA] depth={depth} | topic {idx + 1}: {' '.join(top_words[:3])}")

    topic_counts: Dict[int, int] = defaultdict(int)
    processed_docs = []

    for i, content in enumerate(docs):
        dom_topic = int(doc_topic_probs[i].argmax() + 1)
        all_probs = doc_topic_probs[i].tolist()
        topic_counts[dom_topic] += 1

        # Leaf: already at depth≥1, or content too short to sub-chunk (~50 words ≈ 200 tokens)
        word_count = len(content.split())
        if depth >= 1 or word_count <= 50:
            sub_tree = {"status": "leaf", "content": content}
        else:
            raw = chunking_by_token_size(
                content,
                overlap_token_size=SUB_CHUNK_OVERLAP,
                max_token_size=SUB_CHUNK_TOKEN_SIZE,
            )
            sub_texts = [c["content"] for c in raw]
            if len(sub_texts) <= 1:
                sub_tree = {"status": "leaf", "content": content}
            else:
                print(f"[LDA] depth={depth} | chunk[{i}] → {len(sub_texts)} sub-chunks")
                sub_tree = perform_lda_on_layer(sub_texts, depth + 1)

        processed_docs.append({
            "topic":     dom_topic,
            "content":   content,
            "children":  sub_tree,
            "prob":      float(doc_topic_probs[i].max()),
            "all_probs": all_probs,   # full soft-assignment vector for cross-topic edges
        })

    print(f"[LDA] depth={depth} | topic distribution: "
          f"{ {t: topic_counts[t] for t in sorted(topic_counts)} }")

    return {"depth": depth, "topics": topics, "results": processed_docs}


# ─── KG construction helpers ─────────────────────────────────────────────────

def _make_node_name(topic_words: List[str], depth: int, topic_id: int) -> str:
    """Return a node name from the first 6 topic words. Duplicates are resolved later by merging."""
    return "_".join(topic_words[:6]) if topic_words else "topic"


def _extract_subtree(
    tree: Dict,
    chunk_source_id: str,
    parent_name: str,
    parent_words: List[str],
    entities: List[Dict],
    relationships: List[Dict],
    depth: int,
) -> None:
    """Recursively extract sub-topic entities and relationships from a depth-1+ LDA subtree."""
    if not tree or tree.get("status") == "leaf":
        return

    topics  = tree.get("topics", [])
    results = tree.get("results", [])

    topic_to_results: Dict[int, List[Dict]] = defaultdict(list)
    for r in results:
        topic_to_results[r["topic"]].append(r)

    parent_word_set = set(parent_words[:10])

    for topic in topics:
        tid    = topic["id"]
        twords = topic["words"]
        name   = _make_node_name(twords, depth, tid)

        entities.append({
            "entity_name": name,
            "entity_type": "TOPIC",
            "description": " ".join(twords),
            "source_id":   chunk_source_id,
        })

        child_word_set = set(twords[:10])
        overlap  = parent_word_set & child_word_set
        edge_kw  = " ".join(overlap) if overlap else " ".join(twords[:3])

        assigned = topic_to_results[tid]
        avg_prob = (
            sum(r.get("prob", 1.0) for r in assigned) / len(assigned)
            if assigned else 1.0
        )

        relationships.append({
            "src_id":      parent_name,
            "tgt_id":      name,
            "description": f"Topic '{parent_name}' contains subtopic '{name}' at depth {depth}",
            "keywords":    edge_kw,
            "weight":      avg_prob,
            "source_id":   chunk_source_id,
        })

        for r in assigned:
            child_tree = r.get("children")
            if child_tree and child_tree.get("status") != "leaf":
                _extract_subtree(
                    child_tree, chunk_source_id, name, twords,
                    entities, relationships, depth + 1,
                )


def _merge_duplicate_nodes(
    entities: List[Dict],
    relationships: List[Dict],
) -> Tuple[List[Dict], List[Dict]]:
    """
    Post-processing pass:
    - Entities with the same name → collapsed into one (primary source_id kept).
    - Relationship pairs with the same (src, tgt) → collapsed, weights averaged.
    """
    # ── Entity merge ─────────────────────────────────────────────
    name_to_ents: Dict[str, List[Dict]] = defaultdict(list)
    for e in entities:
        name_to_ents[e["entity_name"]].append(e)

    merged_entities: List[Dict] = []
    merged_count = 0
    for name, ents in name_to_ents.items():
        if len(ents) == 1:
            merged_entities.append(ents[0])
        else:
            merged_count += len(ents) - 1
            all_src: List[str] = []
            for e in ents:
                all_src.extend(e["source_id"].split(GRAPH_FIELD_SEP))
            merged_entities.append({
                "entity_name": name,
                "entity_type": ents[0]["entity_type"],
                "description": ents[0]["description"],
                # union all members' chunk sources (dedup, keep order) so a recurring
                # name keeps reach to every parent's chunks
                "source_id":   GRAPH_FIELD_SEP.join(dict.fromkeys(all_src)),
            })

    if merged_count:
        print(f"[KG] Merged {merged_count} duplicate nodes → {len(merged_entities)} unique entities")

    # ── Relationship dedup ────────────────────────────────────────
    seen: Dict[Tuple[str, str], int] = {}
    merged_rels: List[Dict] = []
    dup_count = 0
    for rel in relationships:
        key = (rel["src_id"], rel["tgt_id"])
        if key in seen:
            idx = seen[key]
            merged_rels[idx]["weight"] = (merged_rels[idx]["weight"] + rel["weight"]) / 2
            # union source_ids so the merged edge reaches all contributing chunks
            existing = merged_rels[idx]["source_id"].split(GRAPH_FIELD_SEP)
            incoming = rel["source_id"].split(GRAPH_FIELD_SEP)
            merged_rels[idx]["source_id"] = GRAPH_FIELD_SEP.join(
                dict.fromkeys(existing + incoming)
            )
            dup_count += 1
        else:
            seen[key] = len(merged_rels)
            merged_rels.append(dict(rel))

    if dup_count:
        print(f"[KG] Collapsed {dup_count} duplicate relationships → {len(merged_rels)} unique")

    return merged_entities, merged_rels


# ─── Visualization ───────────────────────────────────────────────────────────

def visualize_kg(kg: Dict, output_path: str) -> None:
    """Generate a self-contained interactive HTML visualization of the LDA knowledge graph."""
    import json as _json

    entities      = kg["entities"]
    relationships = kg["relationships"]
    chunks        = kg["chunks"]

    # Chunk text previews keyed by source_id
    chunk_preview = {
        c["source_id"]: c["content"][:300]
        for c in chunks
    }

    # Identify sub-topics from "contains subtopic" relationships
    sub_topic_names: set = set()
    parent_of: Dict[str, str] = {}
    for rel in relationships:
        if "contains subtopic" in rel.get("description", ""):
            sub_topic_names.add(rel["tgt_id"])
            parent_of[rel["tgt_id"]] = rel["src_id"]

    top_level_entities = [e for e in entities if e["entity_name"] not in sub_topic_names]

    PALETTE = [
        "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
        "#1abc9c", "#e67e22", "#607d8b", "#e91e63", "#00bcd4",
    ]
    topic_color: Dict[str, str] = {
        e["entity_name"]: PALETTE[i % len(PALETTE)]
        for i, e in enumerate(top_level_entities)
    }

    def get_color(name: str) -> str:
        if name in topic_color:
            return topic_color[name]
        parent = parent_of.get(name)
        return topic_color.get(parent, "#666666") if parent else "#666666"

    # vis.js node records
    vis_nodes = []
    for e in entities:
        name    = e["entity_name"]
        is_top  = name not in sub_topic_names
        color   = get_color(name)
        # source_id may be a GRAPH_FIELD_SEP-joined list of chunks; preview the first
        src_ids_list = e["source_id"].split(GRAPH_FIELD_SEP) if e["source_id"] else []
        preview = chunk_preview.get(src_ids_list[0], "") if src_ids_list else ""
        vis_nodes.append({
            "id":            name,
            "label":         name[:22],
            "description":   e["description"],
            "source_id":     e["source_id"],
            "chunk_preview": preview,
            "is_top":        is_top,
            "color_hex":     color,
        })

    # vis.js edge records
    vis_edges = []
    for i, rel in enumerate(relationships):
        is_cross = "co-occur" in rel.get("description", "")
        vis_edges.append({
            "id":          i,
            "from":        rel["src_id"],
            "to":          rel["tgt_id"],
            "weight":      rel["weight"],
            "description": rel["description"],
            "keywords":    rel.get("keywords", ""),
            "is_cross":    is_cross,
        })

    # Topic summary for sidebar
    topic_summary = []
    for e in top_level_entities:
        name      = e["entity_name"]
        sub_topics = [n for n, p in parent_of.items() if p == name]
        topic_summary.append({
            "name":            name,
            "color":           topic_color[name],
            "words":           e["description"],
            "sub_topic_count": len(sub_topics),
            "sub_topics":      sub_topics[:10],
        })

    stats = {
        "top_topics":    len(top_level_entities),
        "sub_topics":    len(entities) - len(top_level_entities),
        "relationships": len(relationships),
        "cross_edges":   sum(1 for r in relationships if "co-occur" in r.get("description", "")),
        "chunks":        len(chunks),
    }

    nodes_json  = _json.dumps(vis_nodes,     ensure_ascii=False)
    edges_json  = _json.dumps(vis_edges,     ensure_ascii=False)
    topics_json = _json.dumps(topic_summary, ensure_ascii=False)
    stats_json  = _json.dumps(stats,         ensure_ascii=False)

    # Use placeholder replacement to avoid Python f-string / JS brace conflicts
    html = _HTML_TEMPLATE
    html = html.replace("%%NODES%%",  nodes_json)
    html = html.replace("%%EDGES%%",  edges_json)
    html = html.replace("%%TOPICS%%", topics_json)
    html = html.replace("%%STATS%%",  stats_json)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[LDA] Visualization saved → {output_path}")


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>LDA Knowledge Graph</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/vis-network.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #c9d1d9; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
#header { background: #161b22; border-bottom: 1px solid #30363d; padding: 0.6rem 1rem; display: flex; align-items: center; gap: 0.6rem; flex-shrink: 0; flex-wrap: wrap; }
#header h1 { font-size: 0.95rem; color: #f0f6fc; font-weight: 600; margin-right: 0.4rem; }
.chip { background: #21262d; border: 1px solid #30363d; border-radius: 20px; padding: 0.18rem 0.55rem; font-size: 0.72rem; color: #8b949e; white-space: nowrap; }
.chip b { color: #58a6ff; }
#main { display: flex; flex: 1; overflow: hidden; }
#graph-wrap { flex: 1; position: relative; }
#mynetwork { width: 100%; height: 100%; }
#sidebar { width: 310px; min-width: 310px; background: #161b22; border-left: 1px solid #30363d; display: flex; flex-direction: column; overflow: hidden; }
#ctrl { padding: 0.5rem 0.7rem; border-bottom: 1px solid #21262d; display: flex; gap: 0.35rem; flex-wrap: wrap; }
button { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; border-radius: 5px; padding: 0.28rem 0.55rem; font-size: 0.72rem; cursor: pointer; transition: background 0.15s; }
button:hover { background: #30363d; }
button.on { background: #1f3a5f; border-color: #58a6ff; color: #79c0ff; }
#scroll { flex: 1; overflow-y: auto; padding: 0.7rem; }
.sec { font-size: 0.67rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.08em; color: #6e7681; margin: 0.9rem 0 0.4rem; }
.sec:first-child { margin-top: 0; }
/* node info */
#ninfo { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 0.7rem; min-height: 60px; }
#ninfo h3 { font-size: 0.85rem; color: #e6edf3; margin-bottom: 0.35rem; word-break: break-all; }
.irow { font-size: 0.75rem; color: #8b949e; margin-bottom: 0.2rem; line-height: 1.45; }
.irow b { color: #c9d1d9; }
#chunk-box { margin-top: 0.45rem; background: #161b22; border: 1px solid #30363d; border-radius: 4px; padding: 0.45rem; font-size: 0.7rem; color: #6e7681; line-height: 1.55; max-height: 110px; overflow-y: auto; white-space: pre-wrap; word-break: break-word; }
/* legend */
.leg { display: flex; align-items: center; gap: 0.45rem; font-size: 0.73rem; color: #8b949e; margin-bottom: 0.22rem; }
.ldot { width: 11px; height: 11px; border-radius: 50%; flex-shrink: 0; }
.lline { width: 22px; height: 2px; flex-shrink: 0; background: #444c56; }
.ldash { width: 22px; height: 0; border-top: 2px dashed #f39c12; flex-shrink: 0; }
/* topic list */
.titem { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 0.55rem; margin-bottom: 0.35rem; cursor: pointer; transition: border-color 0.15s; }
.titem:hover { border-color: #30363d; }
.titem.sel { border-color: #58a6ff; }
.thead { display: flex; align-items: center; gap: 0.45rem; margin-bottom: 0.25rem; }
.tdot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }
.tname { font-size: 0.8rem; color: #e6edf3; font-weight: 600; word-break: break-all; }
.twords { font-size: 0.7rem; color: #6e7681; line-height: 1.4; }
.tmeta { font-size: 0.68rem; color: #58a6ff; margin-top: 0.18rem; }
.subs { margin-top: 0.28rem; display: flex; flex-wrap: wrap; gap: 0.25rem; }
.schip { background: #21262d; border-radius: 3px; padding: 0.08rem 0.32rem; font-size: 0.65rem; color: #8b949e; word-break: break-all; }
/* scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 2px; }
</style>
</head>
<body>
<div id="header">
  <h1>LDA Knowledge Graph</h1>
  <div class="chip">Top topics <b id="s1">–</b></div>
  <div class="chip">Sub-topics <b id="s2">–</b></div>
  <div class="chip">Relationships <b id="s3">–</b></div>
  <div class="chip">Cross-edges <b id="s4">–</b></div>
  <div class="chip">Chunks <b id="s5">–</b></div>
</div>
<div id="main">
  <div id="graph-wrap"><div id="mynetwork"></div></div>
  <div id="sidebar">
    <div id="ctrl">
      <button id="bphy" class="on" onclick="togPhys()">Physics ON</button>
      <button id="bhier" onclick="togHier()">Hierarchical</button>
      <button onclick="network.fit()">Fit view</button>
      <button onclick="showCross()">Cross only</button>
      <button onclick="resetEdges()">Reset</button>
    </div>
    <div id="scroll">
      <div class="sec">Selected node</div>
      <div id="ninfo"><span style="color:#6e7681;font-size:0.78rem">Click a node to inspect it</span></div>

      <div class="sec">Legend</div>
      <div class="leg"><div class="ldot" style="border:2px solid #ccc;background:#444"></div> Top-level topic (large)</div>
      <div class="leg"><div class="ldot" style="opacity:0.45;background:#888"></div> Sub-topic (small)</div>
      <div class="leg"><div class="lline"></div> Parent → child edge</div>
      <div class="leg"><div class="ldash"></div> Cross-topic (soft assign)</div>

      <div class="sec">Topics</div>
      <div id="tlist"></div>
    </div>
  </div>
</div>
<script>
const RAW_NODES = %%NODES%%;
const RAW_EDGES = %%EDGES%%;
const TOPICS    = %%TOPICS%%;
const STATS     = %%STATS%%;

document.getElementById('s1').textContent = STATS.top_topics;
document.getElementById('s2').textContent = STATS.sub_topics;
document.getElementById('s3').textContent = STATS.relationships;
document.getElementById('s4').textContent = STATS.cross_edges;
document.getElementById('s5').textContent = STATS.chunks;

function esc(s) { return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }

const visNodes = new vis.DataSet(RAW_NODES.map(n => ({
  id:    n.id,
  label: n.label,
  title: '<b>' + esc(n.id) + '</b><br><i>' + esc(n.description) + '</i>',
  size:  n.is_top ? 26 : 12,
  shape: 'dot',
  color: {
    background: n.is_top ? n.color_hex : n.color_hex + '60',
    border:     n.color_hex,
    highlight:  { background: n.color_hex, border: '#f0f6fc' },
    hover:      { background: n.color_hex, border: '#f0f6fc' },
  },
  font: {
    size:  n.is_top ? 11 : 8,
    color: n.is_top ? '#f0f6fc' : '#8b949e',
  },
  // extras stored for click panel
  _desc:    n.description,
  _src:     n.source_id,
  _chunk:   n.chunk_preview,
  _is_top:  n.is_top,
  _color:   n.color_hex,
})));

const visEdges = new vis.DataSet(RAW_EDGES.map(e => ({
  id:     e.id,
  from:   e.from,
  to:     e.to,
  label:  e.weight.toFixed(2),
  title:  esc(e.description) + (e.keywords ? '<br><i>kw: ' + esc(e.keywords) + '</i>' : ''),
  dashes: e.is_cross,
  color:  { color: e.is_cross ? '#f39c12' : '#444c56', opacity: e.is_cross ? 0.85 : 0.55,
            highlight: '#58a6ff', hover: '#79c0ff' },
  width:  Math.max(1, e.weight * 2.5),
  arrows: { to: { enabled: true, scaleFactor: 0.45 } },
  font:   { size: 8, color: '#6e7681', align: 'middle' },
  smooth: e.is_cross ? { type: 'curvedCW', roundness: 0.35 } : { type: 'continuous' },
})));

const network = new vis.Network(
  document.getElementById('mynetwork'),
  { nodes: visNodes, edges: visEdges },
  {
    physics: {
      enabled: true,
      forceAtlas2Based: {
        gravitationalConstant: -55,
        centralGravity: 0.003,
        springLength: 130,
        springConstant: 0.05,
        damping: 0.5,
      },
      solver: 'forceAtlas2Based',
      stabilization: { iterations: 250 },
    },
    interaction: { hover: true, tooltipDelay: 80, navigationButtons: true, keyboard: true },
  }
);

// Click → node info panel
network.on('click', function(p) {
  if (!p.nodes.length) { document.getElementById('ninfo').innerHTML = '<span style="color:#6e7681;font-size:0.78rem">Click a node to inspect it</span>'; return; }
  const id   = p.nodes[0];
  const node = visNodes.get(id);
  const conn = network.getConnectedEdges(id).length;
  const nbrs = network.getConnectedNodes(id).length;
  document.getElementById('ninfo').innerHTML =
    '<h3>' + (node._is_top ? '⬤ ' : '◦ ') + esc(id) + '</h3>' +
    '<div class="irow"><b>Type:</b> ' + (node._is_top ? 'Top-level topic' : 'Sub-topic') + '</div>' +
    '<div class="irow"><b>Words:</b> ' + esc(node._desc) + '</div>' +
    '<div class="irow"><b>Edges:</b> ' + conn + ' &nbsp;|&nbsp; <b>Neighbors:</b> ' + nbrs + '</div>' +
    '<div class="irow"><b>Source chunk:</b> <code style="font-size:0.67rem;color:#58a6ff">' + esc(node._src.substring(0,12)) + '…</code></div>' +
    (node._chunk ? '<div id="chunk-box">' + esc(node._chunk) + '…</div>' : '');
});

// Controls
let physOn = true, hierOn = false;
function togPhys() {
  physOn = !physOn;
  network.setOptions({ physics: { enabled: physOn } });
  const b = document.getElementById('bphy');
  b.textContent = 'Physics ' + (physOn ? 'ON' : 'OFF');
  b.classList.toggle('on', physOn);
}
function togHier() {
  hierOn = !hierOn;
  document.getElementById('bhier').classList.toggle('on', hierOn);
  if (hierOn) {
    network.setOptions({ layout: { hierarchical: { direction: 'UD', sortMethod: 'directed', levelSeparation: 130, nodeSpacing: 90 } }, physics: { enabled: false } });
    physOn = false;
    document.getElementById('bphy').textContent = 'Physics OFF';
    document.getElementById('bphy').classList.remove('on');
  } else {
    network.setOptions({ layout: { hierarchical: false }, physics: { enabled: true } });
    physOn = true;
    document.getElementById('bphy').textContent = 'Physics ON';
    document.getElementById('bphy').classList.add('on');
  }
}
function showCross() {
  const crossIds = new Set(RAW_EDGES.filter(e => e.is_cross).map(e => e.id));
  visEdges.forEach(e => visEdges.update({ id: e.id, hidden: !crossIds.has(e.id) }));
}
function resetEdges() {
  visEdges.forEach(e => visEdges.update({ id: e.id, hidden: false }));
  network.fit();
}

// Topic list
const tlist = document.getElementById('tlist');
TOPICS.forEach(t => {
  const d = document.createElement('div');
  d.className = 'titem';
  d.innerHTML =
    '<div class="thead"><div class="tdot" style="background:' + t.color + '"></div><div class="tname">' + esc(t.name) + '</div></div>' +
    '<div class="twords">' + esc(t.words) + '</div>' +
    '<div class="tmeta">' + t.sub_topic_count + ' sub-topics</div>' +
    (t.sub_topics.length ? '<div class="subs">' + t.sub_topics.map(s => '<span class="schip">' + esc(s) + '</span>').join('') + '</div>' : '');
  d.onclick = () => {
    document.querySelectorAll('.titem').forEach(el => el.classList.remove('sel'));
    d.classList.add('sel');
    const toSelect = [t.name, ...t.sub_topics].filter(n => visNodes.get(n));
    network.selectNodes(toSelect);
    if (visNodes.get(t.name)) network.focus(t.name, { scale: 0.9, animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
  };
  tlist.appendChild(d);
});
</script>
</body>
</html>"""


# ─── Main KG builder ─────────────────────────────────────────────────────────

def tree_to_kg(lda_tree: Dict, source_chunks: List[Dict]) -> Dict:
    """
    Convert the LDA topic tree into a LightRAG-compatible knowledge graph dict.

    Four phases:
      1. Top-level topic entities
      2. Cross-topic edges from soft LDA assignments
      3. Sub-topic entity/relationship extraction
      4. Node merging and relationship deduplication
    """
    entities:      List[Dict] = []
    relationships: List[Dict] = []

    top_topics  = lda_tree.get("topics", [])
    top_results = lda_tree.get("results", [])
    print(f"\n[KG] Building graph: {len(top_topics)} top topics, {len(top_results)} chunks")

    topic_to_chunk_sources:    Dict[int, List[str]]   = defaultdict(list)
    topic_to_results_with_src: Dict[int, List[tuple]] = defaultdict(list)
    for i, result in enumerate(top_results):
        tid    = result["topic"]
        src_id = source_chunks[i]["source_id"]
        topic_to_chunk_sources[tid].append(src_id)
        topic_to_results_with_src[tid].append((result, src_id))

    # ── Phase 1: top-level topic entities ────────────────────────
    top_topic_names: Dict[int, str] = {}
    for topic in top_topics:
        tid    = topic["id"]
        twords = topic["words"]
        name   = _make_node_name(twords, 0, tid)
        top_topic_names[tid] = name

        src_ids    = topic_to_chunk_sources[tid]
        joined_src = GRAPH_FIELD_SEP.join(src_ids) if src_ids else source_chunks[0]["source_id"]

        entities.append({
            "entity_name": name,
            "entity_type": "TOPIC",
            "description": " ".join(twords),
            "source_id":   joined_src,
        })
        print(f"[KG] Topic {tid}: '{name}' | {len(src_ids)} chunks "
              f"| words: {' '.join(twords[:6])}")

    # ── Phase 2: cross-topic edges (soft assignment) ──────────────
    edge_probs:   Dict[tuple, List[float]] = defaultdict(list)
    edge_sources: Dict[tuple, List[str]]   = defaultdict(list)

    for i, result in enumerate(top_results):
        all_probs   = result.get("all_probs", [])
        primary_tid = result["topic"]
        for j, prob in enumerate(all_probs):
            secondary_tid = j + 1
            if secondary_tid != primary_tid and prob > CROSS_TOPIC_THRESHOLD:
                key = tuple(sorted([primary_tid, secondary_tid]))
                edge_probs[key].append(prob)
                edge_sources[key].append(source_chunks[i]["source_id"])

    print(f"[KG] Cross-topic edges: {len(edge_probs)} unique pairs "
          f"(threshold={CROSS_TOPIC_THRESHOLD})")

    for (tid_a, tid_b), probs in edge_probs.items():
        avg_p = sum(probs) / len(probs)
        relationships.append({
            "src_id":      top_topic_names[tid_a],
            "tgt_id":      top_topic_names[tid_b],
            "description": (f"Topics co-occur in {len(probs)} chunk(s) "
                            f"(avg p={avg_p:.2f})"),
            "keywords":    "related",
            "weight":      float(avg_p),
            "source_id":   GRAPH_FIELD_SEP.join(dict.fromkeys(edge_sources[(tid_a, tid_b)])),
        })
        print(f"[KG]   '{top_topic_names[tid_a]}' ↔ '{top_topic_names[tid_b]}' "
              f"| n={len(probs)}, avg_p={avg_p:.2f}")

    # ── Phase 3: sub-topic extraction ────────────────────────────
    for topic in top_topics:
        tid = topic["id"]
        for result, chunk_src_id in topic_to_results_with_src[tid]:
            child_tree = result.get("children")
            if child_tree and child_tree.get("status") != "leaf":
                _extract_subtree(
                    child_tree, chunk_src_id, top_topic_names[tid], topic["words"],
                    entities, relationships, depth=1,
                )

    print(f"[KG] Before merge: {len(entities)} entities, {len(relationships)} relationships")

    # ── Phase 4: node merging ─────────────────────────────────────
    entities, relationships = _merge_duplicate_nodes(entities, relationships)

    chunks = [
        {
            "content":           c["content"],
            "source_id":         c["source_id"],
            "chunk_order_index": c["chunk_order_index"],
        }
        for c in source_chunks
    ]

    print(f"[KG] Final: {len(chunks)} chunks | "
          f"{len(entities)} entities | {len(relationships)} relationships\n")
    return {"chunks": chunks, "entities": entities, "relationships": relationships}


# ─── Entry point ─────────────────────────────────────────────────────────────

def build_lda_graph(
    corpus_text: str,
    cache_path: str,
    chunk_token_size: int = 500,
    chunk_overlap: int = 50,
) -> Dict:
    if os.path.exists(cache_path):
        print(f"[LDA] Loading cached LDA graph from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    build_start = time.perf_counter()

    print(f"[LDA] Corpus: {len(corpus_text):,} chars, ~{len(corpus_text.split()):,} words")
    print(f"[LDA] Config: global_topics={GLOBAL_N_TOPICS}, sub_topics={SUB_N_TOPICS}, "
          f"leaf_threshold=~200tok, cross_topic_threshold={CROSS_TOPIC_THRESHOLD}")

    print("[LDA] Chunking corpus...")
    raw_chunks = chunking_by_token_size(
        corpus_text,
        overlap_token_size=chunk_overlap,
        max_token_size=chunk_token_size,
    )

    source_chunks = []
    for chunk in raw_chunks:
        content   = chunk["content"]
        source_id = hashlib.md5(content.encode("utf-8")).hexdigest()
        source_chunks.append({
            "content":           content,
            "source_id":         source_id,
            "chunk_order_index": chunk["chunk_order_index"],
            "tokens":            chunk["tokens"],
        })

    print(f"[LDA] {len(source_chunks)} chunks created "
          f"(chunk_size={chunk_token_size}, overlap={chunk_overlap})")
    print(f"[LDA] Running recursive LDA...")

    chunk_texts = [c["content"] for c in source_chunks]
    lda_tree = perform_lda_on_layer(chunk_texts, depth=0)

    print("[LDA] Converting tree to knowledge graph...")
    kg = tree_to_kg(lda_tree, source_chunks)

    build_elapsed = time.perf_counter() - build_start
    print(f"[LDA] Graph build time: {build_elapsed:.2f}s ({build_elapsed / 60:.2f} min)")

    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(kg, f, ensure_ascii=False)

    print(f"[LDA] Graph cached to {cache_path}")
    print(f"[LDA] Stats: {len(kg['chunks'])} chunks, "
          f"{len(kg['entities'])} entities, {len(kg['relationships'])} relationships")

    viz_path = cache_path.replace(".json", "_viz.html")
    visualize_kg(kg, viz_path)

    return kg
