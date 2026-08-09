# lightrag_example.py
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import requests
_orig_request = requests.Session.request
def _no_verify_request(self, method, url, **kwargs):
    kwargs['verify'] = False
    return _orig_request(self, method, url, **kwargs)
requests.Session.request = _no_verify_request

import httpx
_orig_async_init = httpx.AsyncClient.__init__
def _patched_async_init(self, *args, **kwargs):
    kwargs.setdefault('verify', False)
    _orig_async_init(self, *args, **kwargs)
httpx.AsyncClient.__init__ = _patched_async_init

_orig_sync_init = httpx.Client.__init__
def _patched_sync_init(self, *args, **kwargs):
    kwargs.setdefault('verify', False)
    _orig_sync_init(self, *args, **kwargs)
httpx.Client.__init__ = _patched_sync_init

import asyncio
import os
import time
import logging
import nest_asyncio
import argparse
import json
from typing import Dict, List
from datasets import load_dataset
from lda_graph_builder import build_lda_graph

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.hf import hf_embed
from lightrag.utils import EmbeddingFunc
from lightrag.kg.shared_storage import initialize_pipeline_status
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from lightrag.llm.ollama import ollama_model_complete, ollama_embed

# Apply nest_asyncio for Jupyter environments
nest_asyncio.apply()
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)


def group_questions_by_source(question_list):
    grouped_questions = {}

    for question in question_list:
        source = question.get("source")

        if source not in grouped_questions:
            grouped_questions[source] = []

        grouped_questions[source].append(question)

    return grouped_questions


SYSTEM_PROMPT = """
---Role---
You are a helpful assistant responding to user queries.

---Goal---
Generate direct and concise answers based strictly on the provided Knowledge Base.
Respond in plain text without explanations or formatting.
Maintain conversation continuity and use the same language as the query.
If the answer is unknown, respond with "I don't know".

---Conversation History---
{history}

---Knowledge Base---
{context_data}
"""

async def llm_model_func(
    prompt: str,
    system_prompt: str = None,
    history_messages: list = [],
    keyword_extraction: bool = False,
    **kwargs
) -> str:
    """LLM interface function using OpenAI-compatible API"""
    # Get API configuration from kwargs
    model_name = kwargs.pop("model_name", "qwen2.5-14b-instruct")
    base_url = kwargs.pop("base_url", "")
    api_key = kwargs.pop("api_key", "")
    
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=base_url,
        api_key=api_key,
        **kwargs
    )

async def initialize_rag(
    base_dir: str,
    source: str,
    mode:str,
    model_name: str,
    embed_model_name: str,
    llm_base_url: str,
    llm_api_key: str,
    llm_max_async: int = 2
) -> LightRAG:
    """Initialize LightRAG instance for a specific corpus"""
    working_dir = os.path.join(base_dir, source)
    
    # Create directory for this corpus
    os.makedirs(working_dir, exist_ok=True)
    
    if mode == "API":
        tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        embed_model = AutoModel.from_pretrained(embed_model_name)
        # Initialize embedding function
        embedding_func = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: hf_embed(texts, tokenizer, embed_model),
        )
        
        # Create LLM configuration
        llm_kwargs = {
            "model_name": model_name,
            "base_url": llm_base_url,
            "api_key": llm_api_key
        }

        llm_model_func_input = llm_model_func
    elif mode == "ollama":
        embedding_func = EmbeddingFunc(
            embedding_dim=1024,
            max_token_size=8192,
            func=lambda texts: ollama_embed(
                texts, embed_model=embed_model_name, host=llm_base_url
            ),
        )

        llm_kwargs = {
            "host": llm_base_url,
            "options": {"num_ctx": 32768},
        }

        llm_model_func_input = ollama_model_complete
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'API' or 'ollama'.")
    
    # Create RAG instance
    rag = LightRAG(
        working_dir=working_dir,
        llm_model_func=llm_model_func_input,
        llm_model_name=model_name,
        llm_model_max_async=llm_max_async,
        llm_model_max_token_size=32768,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        embedding_func=embedding_func,
        llm_model_kwargs=llm_kwargs
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    return rag

async def process_corpus(
    corpus_name: str,
    context: str,
    base_dir: str,
    mode: str,
    model_name: str,
    embed_model_name: str,
    llm_base_url: str,
    llm_api_key: str,
    questions: List[dict],
    sample: int,
    retrieve_topk: int,
    llm_max_async: int = 2,
    use_lda: bool = False,
):
    """Process a single corpus: index it and answer its questions"""
    logging.info(f"📚 Processing corpus: {corpus_name}")
    
    # Initialize RAG for this corpus
    rag = await initialize_rag(
        base_dir=base_dir,
        source=corpus_name,
        mode=mode,
        model_name=model_name,
        embed_model_name=embed_model_name,
        llm_base_url=llm_base_url,
        llm_api_key=llm_api_key,
        llm_max_async=llm_max_async
    )
    
    # Index the corpus content. Total (build + insertion) is timed for both paths
    # so LDA and default LightRAG are an apples-to-apples comparison. For LDA the
    # build and insertion are separate calls, so they are also reported individually;
    # default LightRAG fuses both inside ainsert(), so only the combined total exists.
    t_start = time.perf_counter()
    if use_lda:
        cache_path = os.path.join(base_dir, corpus_name, "lda_graph_cache.json")
        lda_kg = build_lda_graph(context, cache_path)
        t_built = time.perf_counter()
        await rag.ainsert_custom_kg(lda_kg)
        t_end = time.perf_counter()
        build_t, insert_t, total_t = t_built - t_start, t_end - t_built, t_end - t_start
        logging.info(
            f"⏱️ Indexing time ({corpus_name}) — build: {build_t:.2f}s | "
            f"insert: {insert_t:.2f}s | total: {total_t:.2f}s ({total_t / 60:.2f} min)"
        )
        timing = {"build_seconds": round(build_t, 3), "insert_seconds": round(insert_t, 3)}
    else:
        await rag.ainsert(context)
        total_t = time.perf_counter() - t_start
        logging.info(
            f"⏱️ Indexing time ({corpus_name}) — build+insert (fused in ainsert): "
            f"{total_t:.2f}s ({total_t / 60:.2f} min)"
        )
        timing = {"build_seconds": None, "insert_seconds": None}
    logging.info(f"✅ Indexed corpus: {corpus_name} ({len(context.split())} words)")

    # Persist indexing timing to JSON for the apples-to-apples comparison
    results_tag = "lightrag_lda" if use_lda else "lightrag"
    timing.update({
        "corpus": corpus_name,
        "method": results_tag,
        "total_seconds": round(total_t, 3),
        "total_minutes": round(total_t / 60, 3),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    timing_dir = os.path.join("./results", results_tag, corpus_name)
    os.makedirs(timing_dir, exist_ok=True)
    timing_path = os.path.join(timing_dir, "indexing_time.json")
    with open(timing_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)
    logging.info(f"💾 Saved indexing time → {timing_path}")
    
    corpus_questions = questions.get(corpus_name, [])
    
    if not corpus_questions:
        logging.warning(f"No questions found for corpus: {corpus_name}")
        return
    
    # Sample questions if requested
    if sample and sample < len(corpus_questions):
        corpus_questions = corpus_questions[:sample]
    
    logging.info(f"🔍 Found {len(corpus_questions)} questions for {corpus_name}")
    
    # Prepare output path
    results_tag = "lightrag_lda" if use_lda else "lightrag"
    output_dir = f"./results/{results_tag}/{corpus_name}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"predictions_{corpus_name}.json")
    
    # Process questions
    results = []
    query_type = 'hybrid'
    max_retries = 7
    save_every = 50

    for q_idx, q in enumerate(tqdm(corpus_questions, desc=f"Answering questions for {corpus_name}")):
        predicted_answer, context = "", None

        for attempt in range(1, max_retries + 1):
            # Fresh params each attempt — kg_query mutates query_param.mode on keyword fallback
            query_param = QueryParam(
                mode=query_type,
                top_k=retrieve_topk,
                max_token_for_text_unit=4000,
                max_token_for_global_context=4000,
                max_token_for_local_context=4000
            )
            try:
                # Execute query
                response = rag.query(
                    q["question"],
                    param=query_param,
                    system_prompt=SYSTEM_PROMPT
                )

                # Handle both async and sync responses
                if asyncio.iscoroutine(response):
                    response = await response
                predicted_answer, context = response
                predicted_answer = str(predicted_answer)
                break  # success
            except Exception as e:
                logging.warning(f"⚠️ Question {q['id']} failed (attempt {attempt}/{max_retries}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(2 * attempt)  # backoff before retry
                else:
                    logging.error(f"❌ Question {q['id']} failed after {max_retries} attempts; recording empty result")
                    predicted_answer, context = "", None

        # Collect results
        results.append({
            "id": q["id"],
            "question": q["question"],
            "source": corpus_name,
            "context": [context] if context else [],
            "evidence": q["evidence"],
            "question_type": q["question_type"],
            "generated_answer": predicted_answer,
            "ground_truth": q.get("answer"),

        })

        # Incremental checkpoint so a later crash never wipes all progress
        if (q_idx + 1) % save_every == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logging.info(f"💾 Checkpoint: {len(results)}/{len(corpus_questions)} saved to: {output_path}")

    # Save results
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logging.info(f"💾 Saved {len(results)} predictions to: {output_path}")

def main():
    # Define subset paths
    SUBSET_PATHS = {
        "medical": {
            "corpus": "./Datasets/Corpus/medical.parquet",
            "questions": "./Datasets/Questions/medical_questions.parquet"
        },
        "novel": {
            "corpus": "./Datasets/Corpus/novel.parquet",
            "questions": "./Datasets/Questions/novel_questions.parquet"
        }
    }
    
    parser = argparse.ArgumentParser(description="LightRAG: Process Corpora and Answer Questions")
    
    # Core arguments
    parser.add_argument("--subset", required=True, choices=["medical", "novel"], 
                        help="Subset to process (medical or novel)")
    parser.add_argument("--base_dir", default="./lightrag_workspace", help="Base working directory")
    
    # Model configuration
    parser.add_argument("--mode", required=True, choices=["API", "ollama"], help="Use API or ollama for LLM")
    parser.add_argument("--model_name", default="qwen2.5-14b-instruct", help="LLM model identifier")
    parser.add_argument("--embed_model", default="bge-base-en", help="Embedding model name")
    parser.add_argument("--retrieve_topk", type=int, default=5, help="Number of top documents to retrieve")
    parser.add_argument("--llm_max_async", type=int, default=2, help="Max concurrent LLM requests per corpus (lower to avoid rate limits)")
    parser.add_argument("--max_concurrent_corpus", type=int, default=2, help="Max number of corpora processed simultaneously")
    parser.add_argument("--sample", type=int, default=None, help="Number of questions to sample per corpus")
    parser.add_argument("--use_lda", action="store_true", help="Use LDA-based KG instead of LLM entity extraction")
    
    # API configuration
    parser.add_argument("--llm_base_url", default="https://api.openai.com/v1", 
                        help="Base URL for LLM API")
    parser.add_argument("--llm_api_key", default="", 
                        help="API key for LLM service (can also use LLM_API_KEY environment variable)")

    args = parser.parse_args()
    
    # Validate subset and mode
    if args.subset not in SUBSET_PATHS:
        logging.error(f"Invalid subset: {args.subset}. Valid options: {list(SUBSET_PATHS.keys())}")
        return
    if args.mode not in ["API", "ollama"]:
        logging.error(f'Invalid mode: {args.subset}. Valid options: {["API", "ollama"]}')
        return
    
    # Get file paths for this subset
    corpus_path = SUBSET_PATHS[args.subset]["corpus"]
    questions_path = SUBSET_PATHS[args.subset]["questions"]
    
    # Handle API key security
    api_key = args.llm_api_key or os.getenv("LLM_API_KEY", "")
    if not api_key:
        logging.warning("No API key provided! Requests may fail.")
    
    # Create workspace directory
    os.makedirs(args.base_dir, exist_ok=True)
    
    # Load corpus data
    try:
        corpus_dataset = load_dataset("parquet", data_files=corpus_path, split="train")
        corpus_data = []
        for item in corpus_dataset:
            corpus_data.append({
                "corpus_name": item["corpus_name"],
                "context": item["context"]
            })
        logging.info(f"Loaded corpus with {len(corpus_data)} documents from {corpus_path}")
    except Exception as e:
        logging.error(f"Failed to load corpus: {e}")
        return
    
    # Sample corpus data if requested
    if args.sample:
        corpus_data = corpus_data[:1]

    # Load question data
    try:
        questions_dataset = load_dataset("parquet", data_files=questions_path, split="train")
        question_data = []
        for item in questions_dataset:
            question_data.append({
                "id": item["id"],
                "source": item["source"],
                "question": item["question"],
                "answer": item["answer"],
                "question_type": item["question_type"],
                "evidence": item["evidence"]
            })
        grouped_questions = group_questions_by_source(question_data)
        logging.info(f"Loaded questions with {len(question_data)} entries from {questions_path}")
    except Exception as e:
        logging.error(f"Failed to load questions: {e}")
        return
    
    # Process each corpus concurrently in a single event loop
    async def _run_all():
        semaphore = asyncio.Semaphore(args.max_concurrent_corpus)

        async def _run_one(item):
            async with semaphore:
                return await process_corpus(
                    corpus_name=item["corpus_name"],
                    context=item["context"],
                    base_dir=args.base_dir,
                    mode=args.mode,
                    model_name=args.model_name,
                    embed_model_name=args.embed_model,
                    llm_base_url=args.llm_base_url,
                    llm_api_key=api_key,
                    questions=grouped_questions,
                    sample=args.sample,
                    retrieve_topk=args.retrieve_topk,
                    llm_max_async=args.llm_max_async,
                    use_lda=args.use_lda
                )

        results = await asyncio.gather(*[_run_one(item) for item in corpus_data], return_exceptions=True)
        for r in results:
            if isinstance(r, Exception):
                logging.exception(f"Task failed: {r}")

    asyncio.run(_run_all())

if __name__ == "__main__":
    main()

