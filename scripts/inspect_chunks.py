from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.jsonl"
)


def main() -> None:
    if not CHUNK_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNK_PATH}"
        )

    chunks = []

    with CHUNK_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    print("\n=== DomainQA Chunk Inspection ===")
    print(f"Total chunks: {len(chunks)}")

    if not chunks:
        return

    word_counts = [
        chunk["word_count"]
        for chunk in chunks
    ]

    print(
        f"Average words/chunk: "
        f"{sum(word_counts) / len(word_counts):.1f}"
    )
    print(f"Shortest chunk:      {min(word_counts)}")
    print(f"Longest chunk:       {max(word_counts)}")

    print("\n=== First 5 Chunks ===\n")

    for chunk in chunks[:5]:
        print(f"Chunk ID: {chunk['chunk_id']}")
        print(f"Paper:    {chunk['paper_id']}")
        print(f"Pages:    {chunk['pages']}")
        print(f"Words:    {chunk['word_count']}")
        print("\nText:")
        print(chunk["text"][:1000])
        print("\n" + "=" * 80)


if __name__ == "__main__":
    main()