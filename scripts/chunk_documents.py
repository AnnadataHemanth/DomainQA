from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR / "chunks"
OUTPUT_PATH = OUTPUT_DIR / "chunks.jsonl"

TARGET_WORDS = 450
MIN_WORDS = 150
OVERLAP_WORDS = 80


def clean_text(text: str) -> str:
    """Normalize whitespace and common PDF extraction artifacts."""
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # Fix words that were split across lines/hyphenated.
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)

    # Collapse repeated whitespace.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Perform simple sentence-aware segmentation."""
    sentences = re.split(
        r"(?<=[.!?])\s+(?=[A-Z0-9])",
        text,
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def build_chunks(
    pages: list[dict],
) -> list[dict]:
    """
    Combine pages into a continuous stream of sentences,
    then build overlapping chunks while tracking page provenance.
    """

    sentence_records = []

    for page in pages:
        page_number = page["page"]
        text = clean_text(page.get("text", ""))

        if not text:
            continue

        sentences = split_sentences(text)

        for sentence in sentences:
            sentence_records.append(
                {
                    "text": sentence,
                    "page": page_number,
                }
            )

    chunks = []

    current_sentences = []
    current_words = 0

    for record in sentence_records:
        sentence = record["text"]
        sentence_words = len(sentence.split())

        # Start a new chunk if adding the sentence exceeds target size.
        if (
            current_sentences
            and current_words + sentence_words > TARGET_WORDS
        ):
            chunk_text = " ".join(
                item["text"]
                for item in current_sentences
            )

            if current_words >= MIN_WORDS:
                chunks.append(
                    {
                        "text": chunk_text,
                        "pages": sorted(
                            {
                                item["page"]
                                for item in current_sentences
                            }
                        ),
                    }
                )

            # Keep semantic overlap using whole sentences.
            overlap_sentences = []
            overlap_count = 0

            for item in reversed(current_sentences):
                item_words = len(item["text"].split())

                if overlap_count + item_words > OVERLAP_WORDS:
                    break

                overlap_sentences.insert(0, item)
                overlap_count += item_words

            current_sentences = overlap_sentences.copy()
            current_words = overlap_count

        current_sentences.append(record)
        current_words += sentence_words

    # Add final chunk.
    if current_sentences:
        chunk_text = " ".join(
            item["text"]
            for item in current_sentences
        )

        if (
            len(chunk_text.split()) >= MIN_WORDS
            or not chunks
        ):
            chunks.append(
                {
                    "text": chunk_text,
                    "pages": sorted(
                        {
                            item["page"]
                            for item in current_sentences
                        }
                    ),
                }
            )

    return chunks


def load_documents() -> list[dict]:
    """Load processed documents."""

    documents = []

    for path in sorted(PROCESSED_DIR.glob("*.json")):
        with path.open("r", encoding="utf-8") as file:
            documents.append(json.load(file))

    return documents


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents()

    if not documents:
        raise RuntimeError(
            "No processed documents found."
        )

    total_chunks = 0

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for document in documents:

            chunks = build_chunks(
                document["pages"]
            )

            for index, chunk in enumerate(
                chunks,
                start=1,
            ):
                record = {
                    "chunk_id": (
                        f"{document['id']}_"
                        f"c{index:04d}"
                    ),
                    "paper_id": document["id"],
                    "title": document["title"],
                    "pages": chunk["pages"],
                    "text": chunk["text"],
                    "word_count": len(
                        chunk["text"].split()
                    ),
                }

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            total_chunks += len(chunks)

            print(
                f"[OK] {document['id']} → "
                f"{len(chunks)} chunks"
            )

    print("\n=== Chunking Summary ===")
    print(f"Papers:       {len(documents)}")
    print(f"Total chunks: {total_chunks}")
    print(f"Output:       {OUTPUT_PATH}")


if __name__ == "__main__":
    main()