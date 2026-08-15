from __future__ import annotations

import json
import pickle
import re
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

BENCHMARK_PATH = PROJECT_ROOT / "data" / "datasets" / "benchmark_dev.jsonl"

DENSE_INDEX_PATH = PROJECT_ROOT / "data" / "rag" / "dense.index"
DENSE_METADATA_PATH = PROJECT_ROOT / "data" / "rag" / "metadata.json"

BM25_INDEX_PATH = PROJECT_ROOT / "data" / "rag" / "bm25.pkl"
BM25_METADATA_PATH = PROJECT_ROOT / "data" / "rag" / "bm25_metadata.json"

RESULTS_DIR = PROJECT_ROOT / "data" / "results"
OUTPUT_PATH = RESULTS_DIR / "hybrid_rag_dev.jsonl"
CONFIG_PATH = RESULTS_DIR / "hybrid_rag_config.json"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_NAME = "Qwen/Qwen3-4B-Base"

DENSE_TOP_K = 10
BM25_TOP_K = 10
FINAL_TOP_K = 3

RRF_K = 60

MAX_NEW_TOKENS = 160
MAX_CONTEXT_CHARS_PER_CHUNK = 2500


def tokenize(text: str) -> list[str]:
    """Tokenize text for BM25."""
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def load_dense():
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    index = faiss.read_index(str(DENSE_INDEX_PATH))

    with DENSE_METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return encoder, index, metadata


def load_bm25():
    with BM25_INDEX_PATH.open("rb") as file:
        bm25 = pickle.load(file)

    with BM25_METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    return bm25, metadata


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


def dense_retrieve(question, encoder, index, metadata):
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
        DENSE_TOP_K,
    )

    results = []

    for rank, (score, index_id) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        chunk = metadata[int(index_id)]

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "pages": chunk["pages"],
                "text": chunk["text"],
                "dense_score": float(score),
                "dense_rank": rank,
            }
        )

    return results


def bm25_retrieve(question, bm25, metadata):
    tokens = tokenize(question)
    scores = bm25.get_scores(tokens)

    top_indices = scores.argsort()[::-1][:BM25_TOP_K]

    results = []

    for rank, index_id in enumerate(top_indices, start=1):
        chunk = metadata[int(index_id)]

        results.append(
            {
                "chunk_id": chunk["chunk_id"],
                "paper_id": chunk["paper_id"],
                "title": chunk["title"],
                "pages": chunk["pages"],
                "text": chunk["text"],
                "bm25_score": float(scores[index_id]),
                "bm25_rank": rank,
            }
        )

    return results


def reciprocal_rank_fusion(dense_results, bm25_results):
    fused = {}

    for result in dense_results:
        chunk_id = result["chunk_id"]

        fused.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "paper_id": result["paper_id"],
                "title": result["title"],
                "pages": result["pages"],
                "text": result["text"],
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            },
        )

        fused[chunk_id]["dense_rank"] = result["dense_rank"]
        fused[chunk_id]["dense_score"] = result["dense_score"]
        fused[chunk_id]["rrf_score"] += (
            1.0 / (RRF_K + result["dense_rank"])
        )

    for result in bm25_results:
        chunk_id = result["chunk_id"]

        fused.setdefault(
            chunk_id,
            {
                "chunk_id": chunk_id,
                "paper_id": result["paper_id"],
                "title": result["title"],
                "pages": result["pages"],
                "text": result["text"],
                "dense_rank": None,
                "bm25_rank": None,
                "dense_score": None,
                "bm25_score": None,
                "rrf_score": 0.0,
            },
        )

        fused[chunk_id]["bm25_rank"] = result["bm25_rank"]
        fused[chunk_id]["bm25_score"] = result["bm25_score"]
        fused[chunk_id]["rrf_score"] += (
            1.0 / (RRF_K + result["bm25_rank"])
        )

    ranked = sorted(
        fused.values(),
        key=lambda item: item["rrf_score"],
        reverse=True,
    )

    return ranked[:FINAL_TOP_K]


def build_prompt(question, retrieved_chunks):
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks, start=1):
        text = chunk["text"][:MAX_CONTEXT_CHARS_PER_CHUNK]

        context_parts.append(
            f"""
SOURCE {i}
Paper: {chunk['title']}
Pages: {chunk['pages']}
RRF score: {chunk['rrf_score']:.6f}

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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "model": LLM_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "dense_top_k": DENSE_TOP_K,
        "bm25_top_k": BM25_TOP_K,
        "final_top_k": FINAL_TOP_K,
        "rrf_k": RRF_K,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_context_chars_per_chunk": MAX_CONTEXT_CHARS_PER_CHUNK,
        "benchmark": str(BENCHMARK_PATH),
    }

    with CONFIG_PATH.open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    print("Loading dense retriever...")
    encoder, dense_index, dense_metadata = load_dense()

    print("Loading BM25 retriever...")
    bm25, bm25_metadata = load_bm25()

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

            dense_results = dense_retrieve(
                question,
                encoder,
                dense_index,
                dense_metadata,
            )

            bm25_results = bm25_retrieve(
                question,
                bm25,
                bm25_metadata,
            )

            fused_results = reciprocal_rank_fusion(
                dense_results,
                bm25_results,
            )

            for result in fused_results:
                print(
                    f"  {result['chunk_id']} "
                    f"RRF={result['rrf_score']:.6f} "
                    f"dense_rank={result['dense_rank']} "
                    f"bm25_rank={result['bm25_rank']}"
                )

            prompt = build_prompt(
                question,
                fused_results,
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
                "source_chunk": record["source_chunk"],
                "retrieved_chunks": fused_results,
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

    print(f"\nSaved results to: {OUTPUT_PATH}")
    print(f"Saved config to:  {CONFIG_PATH}")


if __name__ == "__main__":
    main()