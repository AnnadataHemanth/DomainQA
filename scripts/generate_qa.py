from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHUNK_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "chunks"
    / "chunks.jsonl"
)

SPLIT_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "paper_split.json"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "datasets"

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_INPUT_CHARS = 6000
MAX_NEW_TOKENS = 300

# Development run only.
QUESTIONS_PER_CATEGORY = 2

CATEGORIES = [
    "factual",
    "conceptual",
    "comparative",
    "reasoning",
]


def load_model():
    """Load the lightweight development model."""

    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    model = model.to("cpu")
    model.eval()

    print("Model loaded successfully.")

    return tokenizer, model


def generate_text(
    tokenizer,
    model,
    prompt: str,
) -> str:
    """Generate model output."""

    messages = [
        {
            "role": "system",
            "content": (
                "You are an academic dataset generation "
                "assistant. Use only the supplied source "
                "text. Never use outside knowledge. "
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
        key: value.to("cpu")
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0][
        inputs["input_ids"].shape[-1]:
    ]

    return tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()


def extract_json(text: str) -> dict[str, Any]:
    """Extract a JSON object from model output."""

    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
    )

    text = re.sub(
        r"^```\s*",
        "",
        text,
    )

    text = re.sub(
        r"\s*```$",
        "",
        text,
    )

    match = re.search(
        r"\{.*\}",
        text,
        re.DOTALL,
    )

    if not match:
        raise ValueError(
            f"No JSON object found:\n{text}"
        )

    return json.loads(match.group(0))


def load_chunks() -> list[dict[str, Any]]:
    """Load semantic chunks."""

    chunks = []

    with CHUNK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line in file:
            if line.strip():
                chunks.append(
                    json.loads(line)
                )

    return chunks


def load_split() -> dict[str, Any]:
    """Load frozen paper split."""

    with SPLIT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def clean_text(
    text: str,
    max_chars: int = MAX_INPUT_CHARS,
) -> str:
    """Normalize and truncate source text."""

    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text[:max_chars]


def build_prompt(
    chunk: dict[str, Any],
    category: str,
) -> str:
    """Build a category-specific QA generation prompt."""

    instructions = {
        "factual": (
            "Ask for a specific fact, definition, "
            "value, method, or stated result."
        ),
        "conceptual": (
            "Ask the learner to explain a concept, "
            "mechanism, or relationship described "
            "in the source."
        ),
        "comparative": (
            "Compare two methods, concepts, results, "
            "or approaches that are BOTH explicitly "
            "described in the source."
        ),
        "reasoning": (
            "Ask a reasoning question whose answer "
            "can be derived from the source without "
            "requiring outside knowledge."
        ),
    }

    source = clean_text(
        chunk["text"]
    )

    return f"""
Generate ONE {category} question-answer pair.

Question type:
{instructions[category]}

Source paper:
{chunk["title"]}

Source pages:
{chunk["pages"]}

Source text:
--- BEGIN SOURCE ---
{source}
--- END SOURCE ---

Rules:

1. The question must be answerable entirely from the source.
2. The answer must be supported by the source.
3. Do not use outside knowledge.
4. Do not ask about information absent from the source.
5. Do not use phrases such as:
   - "most accurate"
   - "best method"
   - "most effective"
   unless the source explicitly establishes such a comparison.
6. For comparative questions, both compared items must appear
   meaningfully in the source.
7. For reasoning questions, the reasoning must be derivable
   from information in the source.
8. Write a clear, specific academic question.
9. Return ONLY this JSON:

{{
  "question": "...",
  "answer": "...",
  "category": "{category}"
}}
""".strip()


def main() -> None:

    if not CHUNK_PATH.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNK_PATH}"
        )

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Split file not found: {SPLIT_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer, model = load_model()

    chunks = load_chunks()
    split = load_split()

    train_papers = set(
        split["train"]
    )

    validation_papers = set(
        split["validation"]
    )

    benchmark_papers = set(
        split["benchmark"]
    )

    # Development run:
    # generate training examples only.
    candidate_chunks = [
        chunk
        for chunk in chunks
        if chunk["paper_id"] in train_papers
    ]

    if not candidate_chunks:
        raise RuntimeError(
            "No training chunks found."
        )

    # Use deterministic ordering.
    candidate_chunks.sort(
        key=lambda chunk: chunk["chunk_id"]
    )

    # We deliberately use different chunks.
    selected_chunks = candidate_chunks[:]

    category_counts = {
        category: 0
        for category in CATEGORIES
    }

    output_path = (
        OUTPUT_DIR
        / "train_candidates.jsonl"
    )

    generated = 0

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for chunk in selected_chunks:

            # Stop once all categories reach their
            # development target.
            if all(
                count >= QUESTIONS_PER_CATEGORY
                for count in category_counts.values()
            ):
                break

            remaining_categories = [
                category
                for category in CATEGORIES
                if category_counts[category]
                < QUESTIONS_PER_CATEGORY
            ]

            category = remaining_categories[
                generated % len(remaining_categories)
            ]

            print(
                f"\nGenerating {category} QA"
            )
            print(
                f"Chunk: {chunk['chunk_id']}"
            )

            prompt = build_prompt(
                chunk,
                category,
            )

            try:
                raw_output = generate_text(
                    tokenizer,
                    model,
                    prompt,
                )

                qa = extract_json(
                    raw_output
                )

                record = {
                    "id": (
                        f"train_candidate_"
                        f"{generated + 1:04d}"
                    ),
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "source_paper": chunk["paper_id"],
                    "source_pages": chunk["pages"],
                    "source_chunk": chunk["chunk_id"],
                    "category": category,
                    "answerable": True,
                    "supporting_passage": chunk["text"],
                }

                output_file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                output_file.flush()

                category_counts[category] += 1
                generated += 1

                print(
                    f"Question: "
                    f"{record['question']}"
                )

            except Exception as exc:
                print(
                    f"[ERROR] "
                    f"{chunk['chunk_id']}: {exc}"
                )

    print("\n=== QA Generation Summary ===")
    print(f"Generated: {generated}")

    for category, count in category_counts.items():
        print(
            f"{category.capitalize():12}: {count}"
        )

    print(
        f"\nSaved to: {output_path}"
    )


if __name__ == "__main__":
    main()