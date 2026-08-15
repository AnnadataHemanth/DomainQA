from __future__ import annotations

import json
import random
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = PROJECT_ROOT / "data" / "raw" / "papers.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "data" / "splits"
OUTPUT_PATH = OUTPUT_DIR / "paper_split.json"

SEED = 42


def load_manifest() -> list[dict]:
    """Load the paper manifest."""
    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    return data["papers"]


def main() -> None:
    papers = load_manifest()

    # Only include papers that were successfully extracted.
    available_papers = []

    for paper in papers:
        processed_file = PROCESSED_DIR / f"{paper['id']}.json"

        if processed_file.exists():
            available_papers.append(paper)
        else:
            print(f"[SKIP] {paper['id']} - processed file not found")

    if len(available_papers) < 3:
        raise RuntimeError(
            "At least 3 processed papers are required for a split."
        )

    # Deterministic shuffle.
    rng = random.Random(SEED)
    shuffled = available_papers.copy()
    rng.shuffle(shuffled)

    total = len(shuffled)

    # For the current small corpus:
    # ~60% train, ~20% validation, ~20% benchmark.
    train_count = max(1, round(total * 0.60))
    validation_count = max(1, round(total * 0.20))

    # Make sure the benchmark always gets at least one paper.
    if train_count + validation_count >= total:
        validation_count = 1
        train_count = total - 2

    train = shuffled[:train_count]
    validation = shuffled[train_count:train_count + validation_count]
    benchmark = shuffled[train_count + validation_count:]

    split = {
        "seed": SEED,
        "total_papers": total,
        "train": [paper["id"] for paper in train],
        "validation": [paper["id"] for paper in validation],
        "benchmark": [paper["id"] for paper in benchmark],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(split, file, indent=2)

    print("\n=== DomainQA Paper Split ===")
    print(f"Total papers: {total}")
    print(f"Training:     {len(train)}")
    print(f"Validation:   {len(validation)}")
    print(f"Benchmark:    {len(benchmark)}")
    print(f"Seed:         {SEED}")

    print("\nTraining papers:")
    for paper in train:
        print(f"  - {paper['id']}")

    print("\nValidation papers:")
    for paper in validation:
        print(f"  - {paper['id']}")

    print("\nBenchmark papers:")
    for paper in benchmark:
        print(f"  - {paper['id']}")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()