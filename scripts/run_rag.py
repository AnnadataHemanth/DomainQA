from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_PATH = PROJECT_ROOT / "data" / "datasets" / "benchmark_dev.jsonl"

INDEX_PATH = PROJECT_ROOT / "data" / "rag" / "dense.index"
METADATA_PATH = PROJECT_ROOT / "data" / "rag" / "metadata.json"

OUTPUT_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_PATH = OUTPUT_DIR / "rag_dev.jsonl"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "Qwen/Qwen3-4B-Base"

TOP_K = 3
MAX_NEW_TOKENS = 160
MAX_CONTEXT_CHARS_PER_CHUNK = 2500


def load_retriever():
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    index = faiss.read_index(str(INDEX_PATH))

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return encoder, index, metadata


def load_llm():
    print(f"Loading LLM: {LLM_NAME}")

    tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        LLM_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    model.eval()
    return tokenizer, model


def retrieve(question, encoder, index, metadata):
    query_embedding = encoder.encode(
        [question],
        normalize_embeddings=True,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    scores, indices = index.search(
        query_embedding,
        TOP_K,
    )

    results = []

    for score, index_id in zip(scores[0], indices[0]):
        chunk = metadata[int(index_id)]

        results.append(
            {
                "score": float(score),
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "pages": chunk["pages"],
                "text": chunk["text"],
            }
        )

    return results


def build_prompt(question, retrieved_chunks):
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk["text"][:MAX_CONTEXT_CHARS_PER_CHUNK]

        context_parts.append(
            f"""
SOURCE {i}
Paper: {chunk['title']}
Pages: {chunk['pages']}
Similarity: {chunk['score']:.4f}

{text}
""".strip()
        )

    context = "\n\n".join(context_parts)

    return f"""
Answer the question using ONLY the supplied sources.

If the sources do not provide enough information,
say so explicitly.

Do not use outside knowledge.

Question:
{question}

Sources:
{context}

Answer:
""".strip()


def generate_answer(tokenizer, model, prompt):
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

    generated = outputs[0][inputs["input_ids"].shape[-1]:]

    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
    ).strip()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    encoder, index, metadata = load_retriever()
    tokenizer, model = load_llm()

    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        benchmark = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for number, record in enumerate(benchmark, start=1):
            question = record["question"]

            print(f"\n[{number}/{len(benchmark)}]")
            print(f"Question: {question}")

            retrieved = retrieve(
                question,
                encoder,
                index,
                metadata,
            )

            for result in retrieved:
                print(
                    f"  {result['chunk_id']} "
                    f"score={result['score']:.4f}"
                )

            prompt = build_prompt(
                question,
                retrieved,
            )

            answer = generate_answer(
                tokenizer,
                model,
                prompt,
            )

            result = {
                "id": record["id"],
                "question": question,
                "gold_answer": record["gold_answer"],
                "model_answer": answer,
                "category": record["category"],
                "answerable": record["answerable"],
                "source_paper": record["source_paper"],
                "retrieved_chunks": retrieved,
            }

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )

            output_file.flush()

            print(f"Answer: {answer}")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()