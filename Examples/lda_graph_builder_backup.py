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
        token_pattern=r"(?u)\b\w+\b",  # allow single-char tokens; default pattern requires ≥2 chars
        min_df=1,  # keep every token — deeper recursion levels may have only 2-3 docs, min_df>1 would silently discard most of the vocabulary
    )
    try:
        dtm = vectorizer.fit_transform(docs)
        n_topics_cap = GLOBAL_N_TOPICS if depth == 0 else SUB_N_TOPICS
        # LDA requires n_topics ≤ min(n_documents, n_vocabulary); clamp to avoid sklearn errors
        n_topics = min(n_topics_cap, dtm.shape[0], dtm.shape[1])
        # guard against an empty vocabulary after stopword filtering (dtm.shape[1] == 0)
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
        top_indices = weights.argsort()[:-11:-1]  # indices of top-10 words by weight, descending
        top_words = [feature_names[i] for i in top_indices if i < len(feature_names)]  # bounds guard: skips any index that exceeds the vocabulary size instead of raising IndexError
        topics.append({"id": idx + 1, "words": top_words})  # +1: LDA is 0-indexed, topic IDs are 1-indexed throughout
        print(f"[LDA] depth={depth} | topic {idx + 1}: {' '.join(top_words[:3])}")

    topic_counts: Dict[int, int] = defaultdict(int)
    processed_docs = []

    for i, content in enumerate(docs):
        dom_topic = int(doc_topic_probs[i].argmax() + 1)  # +1: convert 0-based argmax to 1-based topic ID
        all_probs = doc_topic_probs[i].tolist()  # full soft-assignment vector; used later for cross-topic edges
        topic_counts[dom_topic] += 1

        # Recursion stops at depth≥1 or for short content — sub-chunking a single tiny passage produces no signal
        word_count = len(content.split())
        if depth >= 1 or word_count <= 50:
            sub_tree = {"status": "leaf", "content": content}
        else:
            # Break this chunk into smaller sub-chunks and recurse to find finer-grained topics
            raw = chunking_by_token_size(
                content,
                overlap_token_size=SUB_CHUNK_OVERLAP,
                max_token_size=SUB_CHUNK_TOKEN_SIZE,
            )
            sub_texts = [c["content"] for c in raw]
            if len(sub_texts) <= 1:
                # chunker returned a single piece — nothing to split, treat as leaf
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
    """Return a node name from the first 3 topic words. Duplicates are resolved later by merging."""
    return "_".join(topic_words[:3]) if topic_words else "topic"


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
    results = tree.get("results", []) #documents with dom topic and all probs

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
            secondary_tid = j + 1  # convert 0-based index to 1-based topic ID
            # a chunk "belongs" to multiple topics when its secondary probability exceeds the threshold
            if secondary_tid != primary_tid and prob > CROSS_TOPIC_THRESHOLD:
                key = tuple(sorted([primary_tid, secondary_tid]))  # sort so (A,B) and (B,A) map to the same edge
                edge_probs[key].append(prob)
                edge_sources[key].append(source_chunks[i]["source_id"])

    print(f"[KG] Cross-topic edges: {len(edge_probs)} unique pairs "
          f"(threshold={CROSS_TOPIC_THRESHOLD})")

    for (tid_a, tid_b), probs in edge_probs.items():
        # probs holds one probability value per chunk where both topics co-occurred; average them for edge weight
        avg_p = sum(probs) / len(probs)
        relationships.append({
            "src_id":      top_topic_names[tid_a],
            "tgt_id":      top_topic_names[tid_b],
            "description": (f"Topics co-occur in {len(probs)} chunk(s) "
                            f"(avg p={avg_p:.2f})"),
            "keywords":    "related",
            "weight":      float(avg_p),  # higher weight = stronger thematic overlap between the two topics
            # union of all chunk source_ids that contributed to this edge, deduplicated while preserving order
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
    chunk_token_size: int = 1200,
    chunk_overlap: int = 100,
) -> Dict:
    # LDA is non-deterministic and slow; cache the result so re-runs skip the build entirely
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
        source_id = hashlib.md5(content.encode("utf-8")).hexdigest()  # stable ID so the same chunk always gets the same source_id across runs
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

    return kg
