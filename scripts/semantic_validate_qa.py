from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "qa_validated.jsonl"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "qa_semantically_validated.jsonl"
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 200


def load_model():
    print(f"Loading validator model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )
    model = model.to("cpu")

    model.eval()

    return tokenizer, model


def generate(
    tokenizer,
    model,
    prompt: str,
) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict academic dataset validator. "
                "Judge only whether the question and answer are "
                "supported by the supplied passage. "
                "Do not use outside knowledge. "
                "Return only valid JSON."
            ),
        },
        {
            "role": "user",
            "content": prompt,
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(model.device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    text = re.sub(r"^```json\s*", "", text)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if not match:
        raise ValueError(
            f"Validator returned no JSON:\n{text}"
        )

    return json.loads(match.group(0))


def build_prompt(record: dict) -> str:
    return f"""
Evaluate this QA pair.

QUESTION:
{record["question"]}

ANSWER:
{record["answer"]}

SOURCE PASSAGE:
{record["supporting_passage"]}

Evaluate:

1. grounded:
Is every important claim in the answer supported by the passage?

2. fully_answered:
Does the answer actually answer the question completely?

3. quality:
Rate the QA pair from 1 to 5.

5 = precise, well-grounded, complete, useful question
4 = good with minor issues
3 = acceptable but somewhat weak
2 = significant issue
1 = unusable

Return ONLY:

{{
  "grounded": true,
  "fully_answered": true,
  "quality": 5
}}
""".strip()


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing input: {INPUT_PATH}"
        )

    tokenizer, model = load_model()

    records = []

    with INPUT_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    accepted = 0
    rejected = 0

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for record in records:
            print(
                f"\nValidating {record['id']}"
            )

            try:
                prompt = build_prompt(record)

                raw_output = generate(
                    tokenizer,
                    model,
                    prompt,
                )

                evaluation = extract_json(
                    raw_output
                )

                record["grounded"] = bool(
                    evaluation["grounded"]
                )

                record["fully_answered"] = bool(
                    evaluation["fully_answered"]
                )

                record["quality_score"] = int(
                    evaluation["quality"]
                )

                # Final automatic acceptance rule.
                is_accepted = (
                    record["grounded"]
                    and record["fully_answered"]
                    and record["quality_score"] >= 4
                )

                record["semantically_valid"] = (
                    is_accepted
                )

                if is_accepted:
                    accepted += 1
                    print(
                        f"[ACCEPT] "
                        f"quality={record['quality_score']}"
                    )
                else:
                    rejected += 1
                    print(
                        f"[REJECT] "
                        f"grounded={record['grounded']} "
                        f"fully_answered="
                        f"{record['fully_answered']} "
                        f"quality="
                        f"{record['quality_score']}"
                    )

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            except Exception as exc:
                rejected += 1
                print(
                    f"[ERROR] {record['id']}: {exc}"
                )

    print("\n=== Semantic QA Validation ===")
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print(f"Output:   {OUTPUT_PATH}")


if __name__ == "__main__":
    main()