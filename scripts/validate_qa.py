from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = PROJECT_ROOT / "data" / "datasets" / "qa_candidates.jsonl"
OUTPUT_PATH = PROJECT_ROOT / "data" / "datasets" / "qa_validated.jsonl"

VALID_CATEGORIES = {
    "factual",
    "conceptual",
    "comparative",
    "reasoning",
}


def normalize(text: str) -> str:
    """Normalize text for basic comparison."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str) -> set[str]:
    """Create a simple word set."""
    return set(
        re.findall(r"\b[a-zA-Z0-9]{3,}\b", normalize(text))
    )


def validate_record(record: dict) -> tuple[bool, list[str]]:
    """Run deterministic quality checks."""

    errors = []

    question = record.get("question", "").strip()
    answer = record.get("answer", "").strip()
    passage = record.get("supporting_passage", "").strip()
    category = record.get("category")

    if not question:
        errors.append("missing_question")

    if not answer:
        errors.append("missing_answer")

    if not passage:
        errors.append("missing_passage")

    if category not in VALID_CATEGORIES:
        errors.append("invalid_category")

    if not question.endswith("?"):
        errors.append("question_not_formatted")

    if len(question.split()) < 5:
        errors.append("question_too_short")

    if len(answer.split()) < 5:
        errors.append("answer_too_short")

    if len(question.split()) > 40:
        errors.append("question_too_long")

    # Very rough grounding check:
    # at least some non-trivial answer terms should appear
    # in the source passage.
    passage_tokens = tokenize(passage)
    answer_tokens = tokenize(answer)

    if answer_tokens:
        overlap = len(answer_tokens & passage_tokens) / len(answer_tokens)

        if overlap < 0.20:
            errors.append("low_lexical_grounding")

    # Detect questions that make unsupported global claims.
    risky_phrases = [
        "most accurate",
        "best way",
        "best method",
        "most effective",
        "why is this the best",
    ]

    normalized_question = normalize(question)

    if any(
        phrase in normalized_question
        for phrase in risky_phrases
    ):
        errors.append("potentially_unsupported_comparison")

    return len(errors) == 0, errors


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    accepted = 0
    rejected = 0

    with (
        INPUT_PATH.open("r", encoding="utf-8") as input_file,
        OUTPUT_PATH.open("w", encoding="utf-8") as output_file,
    ):
        for line_number, line in enumerate(
            input_file,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"[REJECT] Line {line_number}: invalid JSON"
                )
                rejected += 1
                continue

            is_valid, errors = validate_record(record)

            if is_valid:
                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                accepted += 1

                print(
                    f"[ACCEPT] {record['id']}"
                )

            else:
                rejected += 1

                print(
                    f"[REJECT] {record['id']}: "
                    f"{', '.join(errors)}"
                )

    print("\n=== QA Validation ===")
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()