from __future__ import annotations

import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "final_papers.yaml"
)


TOPIC_HINTS = {
    "retrieval": {
        "dpr_2020",
        "colbert_2020",
        "contriever_2021",
        "beir_2021",
        "e5_2022",
        "mteb_2022",
        "sentence_t5_2021",
    },
    "rag": {
        "rag_2020",
        "rag_survey_2024",
        "rag_llm_survey",
        "raft_2024",
        "self_rag_2023",
        "crag_2024",
    },
    "fine_tuning": {
        "lora_2021",
        "qlora_2023",
        "peft_tradeoffs_2024",
        "instructgpt_2022",
    },
    "hallucination": {
        "hallucination_survey",
        "retrieval_hallucination_2021",
    },
    "evaluation": {
        "ragas_2023",
        "ares_2023",
        "rag_eval_survey_2024",
        "ragchecker_2024",
    },
    "embeddings": {
        "llm_embeddings_2024",
    },
}


def topic_for(paper_id: str) -> list[str]:
    return [
        topic
        for topic, ids in TOPIC_HINTS.items()
        if paper_id in ids
    ]


def main() -> None:
    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        papers = yaml.safe_load(file)["papers"]

    print("=== Final Corpus Topics ===\n")

    for paper in papers:
        topics = topic_for(paper["id"])

        print(
            f"{paper['id']:35} "
            f"{', '.join(topics) if topics else 'UNASSIGNED'}"
        )


if __name__ == "__main__":
    main()