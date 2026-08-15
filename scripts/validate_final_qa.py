from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "final"
)

OUTPUT_DIR = INPUT_DIR


VALID_CATEGORIES = {
    "factual",
    "conceptual",
    "comparative",
    "reasoning",
    "unanswerable",
}


GENERIC_PATTERNS = [
    r"what is the main focus",
    r"what is the main purpose",
    r"what does this paper discuss",
    r"what is this paper about",
    r"why do you think",
    r"what do you think",
]


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def validate_record(
    record: dict,
) -> list[str]:

    reasons = []

    question = normalize(
        record.get("question", "")
    )

    answer = record.get("answer")

    category = record.get("category")

    answerable = record.get(
        "answerable"
    )

    if not question:
        reasons.append("empty_question")

    if category not in VALID_CATEGORIES:
        reasons.append("invalid_category")

    if answerable is True and not answer:
        reasons.append(
            "answerable_without_answer"
        )

    if (
        category == "unanswerable"
        and answerable is True
    ):
        reasons.append(
            "unanswerable_category_but_answerable"
        )

    if (
        category == "unanswerable"
        and answer is not None
    ):
        reasons.append(
            "unanswerable_has_answer"
        )

    if len(question.split()) < 5:
        reasons.append("question_too_short")

    if answer is not None:
        if len(
            str(answer).split()
        ) < 4:
            reasons.append("answer_too_short")

    for pattern in GENERIC_PATTERNS:
        if re.search(
            pattern,
            question,
        ):
            reasons.append(
                "generic_or_speculative_question"
            )
            break

    return reasons


def main() -> None:

    files = sorted(
        INPUT_DIR.glob(
            "*_candidates.jsonl"
        )
    )

    if not files:
        raise RuntimeError(
            "No candidate files found."
        )

    for input_path in files:

        output_path = (
            OUTPUT_DIR
            / input_path.name.replace(
                "_candidates",
                "_rule_validated",
            )
        )

        total = 0
        accepted = 0
        rejected = 0

        seen_questions = set()

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as infile, output_path.open(
            "w",
            encoding="utf-8",
        ) as outfile:

            for line in infile:

                if not line.strip():
                    continue

                total += 1

                record = json.loads(line)

                reasons = validate_record(
                    record
                )

                normalized_question = normalize(
                    record["question"]
                )

                if normalized_question in seen_questions:
                    reasons.append(
                        "duplicate_question"
                    )
                else:
                    seen_questions.add(
                        normalized_question
                    )

                if reasons:
                    rejected += 1

                    print(
                        f"[REJECT] "
                        f"{record['id']}: "
                        f"{', '.join(sorted(set(reasons)))}"
                    )

                    continue

                record[
                    "validation_status"
                ] = "rule_accepted"

                outfile.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                accepted += 1

        print(
            f"\n{input_path.name}"
        )
        print(
            f"Total:    {total}"
        )
        print(
            f"Accepted: {accepted}"
        )
        print(
            f"Rejected: {rejected}"
        )
        print(
            f"Output:   {output_path}"
        )


if __name__ == "__main__":
    main()