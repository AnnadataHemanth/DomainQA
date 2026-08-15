from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "final_paper_split.json"
)

OUTPUT_DIR = PROCESSED_DIR / "chunks"
OUTPUT_PATH = OUTPUT_DIR / "chunks.jsonl"

TARGET_WORDS = 450
MIN_WORDS = 150
OVERLAP_WORDS = 80


def clean_text(text: str) -> str:
    text = text.replace("\r", " ")
    text = text.replace("\n", " ")

    # Rejoin words broken by PDF line wrapping.
    text = re.sub(
        r"(\w)-\s+(\w)",
        r"\1\2",
        text,
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def split_sentences(text: str) -> list[str]:
    raw_sentences = [
        sentence.strip()
        for sentence in re.split(
            r"(?<=[.!?])\s+(?=[A-Z0-9])",
            text,
        )
        if sentence.strip()
    ]

    sentences = []

    MAX_SENTENCE_WORDS = 120

    for sentence in raw_sentences:
        words = sentence.split()

        if len(words) <= MAX_SENTENCE_WORDS:
            sentences.append(sentence)
            continue

        # Split unusually long PDF-extracted "sentences"
        # into smaller word-based segments.
        for start in range(
            0,
            len(words),
            MAX_SENTENCE_WORDS,
        ):
            piece = words[
                start:start + MAX_SENTENCE_WORDS
            ]

            if piece:
                sentences.append(
                    " ".join(piece)
                )

    return sentences


def load_split() -> dict:
    with SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_paper_split(
    paper_id: str,
    split: dict,
) -> str:
    if paper_id in split["train"]:
        return "train"

    if paper_id in split["validation"]:
        return "validation"

    if paper_id in split["benchmark"]:
        return "benchmark"

    raise ValueError(
        f"Paper {paper_id} not found in final split."
    )


def build_chunks(
    pages: list[dict],
) -> list[dict]:

    sentence_records = []

    for page in pages:
        page_number = page["page"]
        text = clean_text(
            page.get("text", "")
        )

        if not text:
            continue

        for sentence in split_sentences(text):
            sentence_records.append(
                {
                    "text": sentence,
                    "page": page_number,
                }
            )

    chunks = []

    current = []
    current_words = 0

    for record in sentence_records:
        sentence_words = len(
            record["text"].split()
        )

        if (
            current
            and current_words + sentence_words
            > TARGET_WORDS
        ):
            chunk_text = " ".join(
                item["text"]
                for item in current
            )

            if current_words >= MIN_WORDS:
                chunks.append(
                    {
                        "text": chunk_text,
                        "pages": sorted(
                            {
                                item["page"]
                                for item in current
                            }
                        ),
                    }
                )

            # Sentence-level overlap.
            overlap = []
            overlap_words_count = 0

            for item in reversed(current):
                item_words = len(
                    item["text"].split()
                )

                if (
                    overlap_words_count
                    + item_words
                    > OVERLAP_WORDS
                ):
                    break

                overlap.insert(0, item)
                overlap_words_count += item_words

            current = overlap
            current_words = overlap_words_count

        current.append(record)
        current_words += sentence_words

    if current:
        chunk_text = " ".join(
            item["text"]
            for item in current
        )

        if (
            len(chunk_text.split())
            >= MIN_WORDS
        ):
            chunks.append(
                {
                    "text": chunk_text,
                    "pages": sorted(
                        {
                            item["page"]
                            for item in current
                        }
                    ),
                }
            )

    return chunks


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    split = load_split()

    paper_files = sorted(
        PROCESSED_DIR.glob("*.json")
    )

    if not paper_files:
        raise RuntimeError(
            "No final processed documents found."
        )

    total_chunks = 0
    split_counts = {
        "train": 0,
        "validation": 0,
        "benchmark": 0,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for path in paper_files:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                document = json.load(file)

            paper_id = document["id"]
            split_name = get_paper_split(
                paper_id,
                split,
            )

            chunks = build_chunks(
                document["pages"]
            )

            for index, chunk in enumerate(
                chunks,
                start=1,
            ):
                record = {
                    "chunk_id": (
                        f"{paper_id}_"
                        f"c{index:04d}"
                    ),
                    "paper_id": paper_id,
                    "split": split_name,
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

                total_chunks += 1
                split_counts[split_name] += 1

            print(
                f"[OK] {paper_id} → "
                f"{len(chunks)} chunks → "
                f"{split_name}"
            )

    print("\n=== Final Chunking Summary ===")
    print(
        f"Papers:       {len(paper_files)}"
    )
    print(
        f"Total chunks: {total_chunks}"
    )
    print(
        f"Train chunks: {split_counts['train']}"
    )
    print(
        f"Validation:   {split_counts['validation']}"
    )
    print(
        f"Benchmark:    {split_counts['benchmark']}"
    )
    print(
        f"Output:       {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()