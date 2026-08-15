from __future__ import annotations

import json
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_papers.yaml"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "final_paper_split.json"
)

SEED = 42

TRAIN_COUNT = 15
VALIDATION_COUNT = 4
BENCHMARK_COUNT = 5


def load_manifest() -> list[dict]:
    import yaml

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = yaml.safe_load(file)

    return data["papers"]


def main() -> None:
    papers = load_manifest()

    if len(papers) != 24:
        raise ValueError(
            f"Expected 24 papers, found {len(papers)}."
        )

    ids = [paper["id"] for paper in papers]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate paper IDs detected."
        )

    rng = random.Random(SEED)
    shuffled = ids.copy()
    rng.shuffle(shuffled)

    train = shuffled[:TRAIN_COUNT]

    validation_end = (
        TRAIN_COUNT + VALIDATION_COUNT
    )

    validation = shuffled[
        TRAIN_COUNT:validation_end
    ]

    benchmark = shuffled[
        validation_end:
    ]

    result = {
        "seed": SEED,
        "total_papers": len(ids),
        "train": train,
        "validation": validation,
        "benchmark": benchmark,
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print("=== DomainQA Final Paper Split ===")
    print(f"Total papers: {len(ids)}")
    print(f"Training:     {len(train)}")
    print(f"Validation:   {len(validation)}")
    print(f"Benchmark:    {len(benchmark)}")
    print(f"Seed:         {SEED}")

    print("\nTraining papers:")
    for paper_id in train:
        print(f"  - {paper_id}")

    print("\nValidation papers:")
    for paper_id in validation:
        print(f"  - {paper_id}")

    print("\nBenchmark papers:")
    for paper_id in benchmark:
        print(f"  - {paper_id}")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()