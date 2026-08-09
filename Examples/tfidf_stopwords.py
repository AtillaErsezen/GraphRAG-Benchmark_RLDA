import json
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'LightRAG'))
from lightrag.operate import chunking_by_token_size

CORPUS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Datasets/Corpus/medical.json")
MAX_DF = 0.90


def load_chunks(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    chunks = chunking_by_token_size(
        data["context"],
        overlap_token_size=0,
        max_token_size=512,
        tiktoken_model="gpt-4o",
    )
    return [c["content"] for c in chunks]


def main():
    print("Loading and chunking corpus...")
    docs = load_chunks(CORPUS_PATH)
    print(f"  {len(docs)} chunks")

    common = dict(token_pattern=r"(?u)\b\w+\b", min_df=1, stop_words="english")

    vocab_full = set(TfidfVectorizer(**common).fit(docs).vocabulary_.keys())
    vocab_filtered = set(TfidfVectorizer(**common, max_df=MAX_DF).fit(docs).vocabulary_.keys())

    derived = sorted((vocab_full - vocab_filtered) - set(ENGLISH_STOP_WORDS))

    print(f"\nDerived stopwords via max_df={MAX_DF} ({len(derived)} words):\n")
    print(derived)


if __name__ == "__main__":
    main()
