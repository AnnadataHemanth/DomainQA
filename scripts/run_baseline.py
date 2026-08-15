from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_PATH = (
    PROJECT_ROOT
    / "data"
    / "datasets"
    / "benchmark_dev.jsonl"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_PATH = OUTPUT_DIR / "baseline_dev.jsonl"

MODEL_NAME = "Qwen/Qwen3-4B-Base"

MAX_NEW_TOKENS = 250


def load_model():
    print(f"Loading: {MODEL_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()

    print("Model loaded.")

    return tokenizer, model


def generate_answer(
    tokenizer,
    model,
    question: str,
) -> str:

    prompt = (
        "Answer the following question as accurately as possible "
        "using your pretrained knowledge.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

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


def main():
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark not found: {BENCHMARK_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    tokenizer, model = load_model()

    with BENCHMARK_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        benchmark = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
    ) as output_file:

        for index, record in enumerate(
            benchmark,
            start=1,
        ):
            print(
                f"\n[{index}/{len(benchmark)}]"
            )
            print(
                f"Question: {record['question']}"
            )

            answer = generate_answer(
                tokenizer,
                model,
                record["question"],
            )

            result = {
                "id": record["id"],
                "question": record["question"],
                "gold_answer": record["gold_answer"],
                "model_answer": answer,
                "category": record["category"],
                "answerable": record["answerable"],
                "source_paper": record["source_paper"],
            }

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            print(
                f"Answer: {answer}"
            )

    print(
        f"\nSaved results to:\n{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()