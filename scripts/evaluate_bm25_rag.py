from __future__ import annotations

import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "bm25_rag_dev.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "results"
    / "bm25_rag_metrics.json"
)

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "benchmark_dev.jsonl"
)


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def token_set(text: str) -> set[str]:
    return set(normalize(text).split())


def lexical_f1(gold: str, prediction: str) -> float:
    gold_tokens = token_set(gold)
    prediction_tokens = token_set(prediction)

    if not gold_tokens or not prediction_tokens:
        return 0.0

    overlap = gold_tokens & prediction_tokens

    if not overlap:
        return 0.0

    precision = len(overlap) / len(prediction_tokens)
    recall = len(overlap) / len(gold_tokens)

    return 2 * precision * recall / (precision + recall)


def exact_match(gold: str, prediction: str) -> bool:
    return normalize(gold) == normalize(prediction)


def load_benchmark() -> dict[str, dict]:
    records = {}

    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                record = json.loads(line)
                records[record["id"]] = record

    return records


def retrieval_metrics(
    record: dict,
    benchmark: dict[str, dict],
) -> dict:
    benchmark_record = benchmark[record["id"]]
    gold_chunk = benchmark_record["source_chunk"]

    retrieved = record["retrieved_chunks"]
    retrieved_ids = [
        chunk["chunk_id"]
        for chunk in retrieved
    ]

    if gold_chunk in retrieved_ids:
        rank = retrieved_ids.index(gold_chunk) + 1

        return {
            "gold_rank": rank,
            "recall_at_1": 1.0 if rank <= 1 else 0.0,
            "recall_at_3": 1.0 if rank <= 3 else 0.0,
            "recall_at_5": 1.0 if rank <= 5 else 0.0,
            "reciprocal_rank": 1.0 / rank,
        }

    return {
        "gold_rank": None,
        "recall_at_1": 0.0,
        "recall_at_3": 0.0,
        "recall_at_5": 0.0,
        "reciprocal_rank": 0.0,
    }


def main() -> None:
    if not RESULTS_PATH.exists():
        raise FileNotFoundError(
            f"BM25 results not found: {RESULTS_PATH}"
        )

    records = []

    with RESULTS_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise RuntimeError("No BM25 results found.")

    benchmark = load_benchmark()

    total_em = 0
    total_f1 = 0.0
    total_r1 = 0.0
    total_r3 = 0.0
    total_r5 = 0.0
    total_mrr = 0.0

    category_metrics = {}
    detailed = []

    for record in records:
        gold = record["gold_answer"]
        prediction = record["model_answer"]
        category = record["category"]

        em = exact_match(gold, prediction)
        f1 = lexical_f1(gold, prediction)

        retrieval = retrieval_metrics(
            record,
            benchmark,
        )

        total_em += int(em)
        total_f1 += f1
        total_r1 += retrieval["recall_at_1"]
        total_r3 += retrieval["recall_at_3"]
        total_r5 += retrieval["recall_at_5"]
        total_mrr += retrieval["reciprocal_rank"]

        category_metrics.setdefault(
            category,
            {
                "count": 0,
                "exact_match": 0,
                "f1": 0.0,
            },
        )

        category_metrics[category]["count"] += 1
        category_metrics[category]["exact_match"] += int(em)
        category_metrics[category]["f1"] += f1

        detailed.append(
            {
                "id": record["id"],
                "category": category,
                "exact_match": em,
                "lexical_f1": f1,
                **retrieval,
            }
        )

        print(
            f"{record['id']}: "
            f"EM={em} "
            f"F1={f1:.3f} "
            f"GoldRank={retrieval['gold_rank']}"
        )

    count = len(records)

    for metrics in category_metrics.values():
        metrics["exact_match"] /= metrics["count"]
        metrics["f1"] /= metrics["count"]

    results = {
        "model": "Qwen3-4B-Base + BM25 RAG",
        "benchmark": "DomainQA development benchmark",
        "num_questions": count,
        "overall": {
            "exact_match": total_em / count,
            "lexical_f1": total_f1 / count,
            "recall_at_1": total_r1 / count,
            "recall_at_3": total_r3 / count,
            "recall_at_5": total_r5 / count,
            "mrr": total_mrr / count,
        },
        "by_category": category_metrics,
        "detailed": detailed,
    }

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=2)

    print("\n=== BM25 RAG Evaluation ===")
    print(f"Questions:   {count}")
    print(f"Exact Match: {total_em / count:.3f}")
    print(f"Lexical F1:  {total_f1 / count:.3f}")
    print(f"Recall@1:    {total_r1 / count:.3f}")
    print(f"Recall@3:    {total_r3 / count:.3f}")
    print(f"Recall@5:    {total_r5 / count:.3f}")
    print(f"MRR:         {total_mrr / count:.3f}")

    print("\nBy category:")

    for category, metrics in category_metrics.items():
        print(
            f"  {category:12} "
            f"EM={metrics['exact_match']:.3f} "
            f"F1={metrics['f1']:.3f}"
        )

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()