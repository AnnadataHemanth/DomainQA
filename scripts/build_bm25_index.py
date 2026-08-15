from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.jsonl"
)

INDEX_DIR = PROJECT_ROOT / "data" / "rag"
INDEX_PATH = INDEX_DIR / "bm25.pkl"
METADATA_PATH = INDEX_DIR / "bm25_metadata.json"


def tokenize(text: str) -> list[str]:
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def load_chunks() -> list[dict]:
    chunks = []

    with CHUNK_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    chunks = load_chunks()

    if not chunks:
        raise RuntimeError("No chunks found.")

    print(f"Loaded {len(chunks)} chunks.")
    print("Building BM25 index...")

    tokenized_corpus = [
        tokenize(chunk["text"])
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    with INDEX_PATH.open("wb") as file:
        pickle.dump(bm25, file)

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== BM25 Index ===")
    print(f"Chunks:   {len(chunks)}")
    print(f"Index:    {INDEX_PATH}")
    print(f"Metadata: {METADATA_PATH}")


if __name__ == "__main__":
    main()