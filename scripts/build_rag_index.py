from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.jsonl"
)

INDEX_DIR = PROJECT_ROOT / "data" / "rag"
INDEX_PATH = INDEX_DIR / "dense.index"
METADATA_PATH = INDEX_DIR / "metadata.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


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
    print(f"Loading embedding model: {EMBEDDING_MODEL}")

    encoder = SentenceTransformer(EMBEDDING_MODEL)

    texts = [chunk["text"] for chunk in chunks]

    embeddings = encoder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    dimension = embeddings.shape[1]

    # For normalized embeddings, inner product = cosine similarity.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))

    with METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(
            chunks,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== Dense RAG Index ===")
    print(f"Chunks:    {len(chunks)}")
    print(f"Dimension: {dimension}")
    print(f"Index:     {INDEX_PATH}")
    print(f"Metadata:  {METADATA_PATH}")


if __name__ == "__main__":
    main()