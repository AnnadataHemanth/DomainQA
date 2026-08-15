from __future__ import annotations

import json
import random
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
    / "final"
    / "chunks"
    / "chunks.jsonl"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "final"
)

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SEED = 42

QUESTIONS_PER_PAPER = {
    "train": 12,
    "validation": 12,
    "benchmark": 12,
}

CATEGORIES = [
    "factual",
    "conceptual",
    "comparative",
    "reasoning",
    "unanswerable",
]

QUESTIONS_PER_CATEGORY = 2

MAX_INPUT_CHARS = 5000
MAX_NEW_TOKENS = 250


def load_model():
    print(f"Loading model: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    model.eval()

    return tokenizer, model


def load_chunks() -> list[dict[str, Any]]:
    chunks = []

    with CHUNK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                chunks.append(json.loads(line))

    return chunks


def normalize_text(text: str) -> str:
    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    text = re.sub(
        r"^```json\s*",
        "",
        text,
        flags=re.IGNORECASE,
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


def generate(
    tokenizer,
    model,
    prompt: str,
) -> dict[str, Any]:

    messages = [
        {
            "role": "system",
            "content": (
                "You are generating an academic QA "
                "dataset. Use only the supplied source. "
                "Never use outside knowledge. "
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

    text = tokenizer.decode(
        generated,
        skip_special_tokens=True,
    )

    return extract_json(text)


def build_prompt(
    chunk: dict[str, Any],
    category: str,
) -> str:

    source = normalize_text(
        chunk["text"]
    )[:MAX_INPUT_CHARS]

    instructions = {
        "factual": (
            "Ask for a specific fact, definition, "
            "method, result, or explicitly stated detail."
        ),
        "conceptual": (
            "Ask the learner to explain a concept, "
            "mechanism, motivation, or relationship."
        ),
        "comparative": (
            "Compare two methods, concepts, "
            "approaches, or results that are both "
            "explicitly present in the source."
        ),
        "reasoning": (
            "Ask a question requiring a short inference "
            "that follows directly from the source."
        ),
        "unanswerable": (
            "Construct a near-miss question whose answer "
            "is NOT supported by the source. It should "
            "sound plausible and be closely related to "
            "the source, but the requested fact must be "
            "absent or contradicted."
        ),
    }

    if category == "unanswerable":
        answer_instructions = """
For an unanswerable question:
- answer must be null
- answerable must be false
- the source must genuinely not support the requested fact
"""
    else:
        answer_instructions = """
For an answerable question:
- answerable must be true
- answer must be fully supported by the source
"""

    return f"""
Generate ONE {category} QA example.

Category instructions:
{instructions[category]}

Source paper:
{chunk["title"]}

Source pages:
{chunk["pages"]}

Source:
--- BEGIN SOURCE ---
{source}
--- END SOURCE ---

Rules:
1. Use ONLY the supplied source.
2. Do not use outside knowledge.
3. Do not invent citations or facts.
4. Do not write vague questions.
5. The question must be academically useful.
6. Avoid yes/no questions unless necessary.
7. Do not repeat generic prompts such as:
   "What is the main focus of this paper?"
8. For comparative questions, both compared items
   must be explicitly discussed.
9. For reasoning questions, the inference must be
   directly supported by the source.
{answer_instructions}

Return ONLY:

{{
  "question": "...",
  "answer": "... or null",
  "category": "{category}",
  "answerable": true
}}

""".strip()


def group_by_split_and_paper(
    chunks: list[dict[str, Any]],
) -> dict[str, dict[str, list[dict[str, Any]]]]:

    grouped: dict[
        str,
        dict[str, list[dict[str, Any]]]
    ] = {
        "train": {},
        "validation": {},
        "benchmark": {},
    }

    for chunk in chunks:
        split = chunk["split"]
        paper = chunk["paper_id"]

        grouped.setdefault(
            split,
            {},
        ).setdefault(
            paper,
            [],
        ).append(chunk)

    return grouped


def sample_chunk(
    chunks: list[dict[str, Any]],
    rng: random.Random,
) -> dict[str, Any]:

    return rng.choice(chunks)


def main() -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = random.Random(SEED)

    tokenizer, model = load_model()

    chunks = load_chunks()

    grouped = group_by_split_and_paper(
        chunks
    )

    print(
        f"Loaded {len(chunks)} final chunks."
    )

    category_sequence = [
        "factual",
        "conceptual",
        "comparative",
        "reasoning",
        "unanswerable",
    ]

    for split in (
        "train",
        "validation",
        "benchmark",
    ):

        output_path = (
            OUTPUT_DIR
            / f"{split}_candidates.jsonl"
        )

        papers = grouped[split]

        print(
            f"\n=== Generating {split.upper()} QA ==="
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:

            generated_count = 0

            for paper_id in sorted(papers):

                target_count = (
                    QUESTIONS_PER_PAPER[split]
                )

                selected_categories = []

                for category in category_sequence:
                    selected_categories.extend(
                        [category]
                        * QUESTIONS_PER_CATEGORY
                    )

                while len(selected_categories) < target_count:
                    selected_categories.append(
                        rng.choice(CATEGORIES)
                    )

                selected_categories = (
                    selected_categories[:target_count]
                )

                for category in selected_categories:

                    chunk = sample_chunk(
                        papers[paper_id],
                        rng,
                    )

                    print(
                        f"[{split}] "
                        f"{paper_id} "
                        f"→ {category}"
                    )

                    prompt = build_prompt(
                        chunk,
                        category,
                    )

                    try:
                        qa = generate(
                            tokenizer,
                            model,
                            prompt,
                        )

                        record = {
                            "id": (
                                f"{split}_candidate_"
                                f"{generated_count + 1:05d}"
                            ),
                            "question": qa["question"],
                            "answer": qa["answer"],
                            "category": category,
                            "answerable": bool(
                                qa["answerable"]
                            ),
                            "source_paper": paper_id,
                            "source_pages": chunk["pages"],
                            "source_chunk": chunk[
                                "chunk_id"
                            ],
                            "split": split,
                        }

                        output_file.write(
                            json.dumps(
                                record,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                        output_file.flush()

                        generated_count += 1

                    except Exception as exc:
                        print(
                            f"[ERROR] "
                            f"{paper_id} "
                            f"{category}: {exc}"
                        )

        print(
            f"Saved: {output_path}"
        )


if __name__ == "__main__":
    main()