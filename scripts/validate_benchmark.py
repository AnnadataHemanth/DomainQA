from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "benchmark_dev.jsonl"
)

CHUNK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.jsonl"
)


def load_chunks() -> dict[str, dict]:
    chunks = {}

    with CHUNK_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                chunks[record["chunk_id"]] = record

    return chunks


def main() -> None:
    chunks = load_chunks()

    errors = []
    count = 0

    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            count += 1

            required = [
                "id",
                "question",
                "gold_answer",
                "source_paper",
                "source_pages",
                "source_chunk",
                "category",
                "answerable",
            ]

            for field in required:
                if field not in record:
                    errors.append(
                        f"{record.get('id', line_number)}: "
                        f"missing field '{field}'"
                    )

            chunk_id = record.get("source_chunk")

            if record.get("answerable") and chunk_id:
                if chunk_id not in chunks:
                    errors.append(
                        f"{record['id']}: "
                        f"chunk '{chunk_id}' not found"
                    )
                else:
                    chunk = chunks[chunk_id]

                    if chunk["paper_id"] != record["source_paper"]:
                        errors.append(
                            f"{record['id']}: "
                            "paper/chunk mismatch"
                        )

                    if not set(record["source_pages"]).issubset(
                        set(chunk["pages"])
                    ):
                        errors.append(
                            f"{record['id']}: "
                            "page/chunk mismatch"
                        )

    print("\n=== Benchmark Validation ===")
    print(f"Questions: {count}")

    if errors:
        print(f"Errors: {len(errors)}\n")

        for error in errors:
            print(f"[ERROR] {error}")

        raise SystemExit(1)

    print("✅ Benchmark metadata is valid.")


if __name__ == "__main__":
    main()