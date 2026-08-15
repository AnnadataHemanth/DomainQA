from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final"
    / "chunks"
    / "chunks.jsonl"
)


def main() -> None:
    chunks = []

    with CHUNK_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    print("=== Final Chunk Inspection ===")
    print(f"Total chunks: {len(chunks)}")

    split_counts = Counter(
        chunk["split"]
        for chunk in chunks
    )

    paper_counts = Counter(
        chunk["paper_id"]
        for chunk in chunks
    )

    word_counts = [
        chunk["word_count"]
        for chunk in chunks
    ]

    print("\nBy split:")
    for split, count in split_counts.items():
        print(f"  {split:12}: {count}")

    print("\nBy paper:")
    for paper_id, count in sorted(
        paper_counts.items()
    ):
        print(
            f"  {paper_id:35} {count}"
        )

    print("\nChunk sizes:")
    print(
        f"  Average: {sum(word_counts) / len(word_counts):.1f}"
    )
    print(
        f"  Shortest: {min(word_counts)}"
    )
    print(
        f"  Longest:  {max(word_counts)}"
    )

    print("\nFirst 3 chunks:\n")

    for chunk in chunks[:3]:
        print(
            f"ID:       {chunk['chunk_id']}"
        )
        print(
            f"Paper:    {chunk['paper_id']}"
        )
        print(
            f"Split:    {chunk['split']}"
        )
        print(
            f"Pages:    {chunk['pages']}"
        )
        print(
            f"Words:    {chunk['word_count']}"
        )
        print(
            f"Text:     {chunk['text'][:600]}"
        )
        print("-" * 80)


if __name__ == "__main__":
    main()