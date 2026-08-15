from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT / "data" / "datasets"
SPLIT_PATH = PROJECT_ROOT / "data" / "splits" / "paper_split.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_jsonl(path: Path) -> list[dict]:
    records = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from exc

    return records


def main() -> None:
    split = load_json(SPLIT_PATH)

    train_papers = set(split["train"])
    validation_papers = set(split["validation"])
    benchmark_papers = set(split["benchmark"])

    train_records = load_jsonl(DATASET_DIR / "train.jsonl")
    validation_records = load_jsonl(DATASET_DIR / "validation.jsonl")
    benchmark_records = load_jsonl(DATASET_DIR / "benchmark.jsonl")

    errors = []

    for record in train_records:
        if record["source_paper"] not in train_papers:
            errors.append(
                f"Training leakage: {record['id']} "
                f"references {record['source_paper']}"
            )

    for record in validation_records:
        if record["source_paper"] not in validation_papers:
            errors.append(
                f"Validation leakage: {record['id']} "
                f"references {record['source_paper']}"
            )

    for record in benchmark_records:
        source = record.get("source_paper")

        if source is not None and source not in benchmark_papers:
            errors.append(
                f"Benchmark leakage: {record['id']} "
                f"references {source}"
            )

    print("\n=== DomainQA Dataset Validation ===")
    print(f"Training examples:   {len(train_records)}")
    print(f"Validation examples: {len(validation_records)}")
    print(f"Benchmark examples:  {len(benchmark_records)}")

    if errors:
        print("\n❌ Dataset validation failed:\n")

        for error in errors:
            print(f"- {error}")

        raise SystemExit(1)

    print("\n✅ No split leakage detected.")


if __name__ == "__main__":
    main()