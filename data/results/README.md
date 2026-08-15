# DomainQA Experiment Results

This directory stores reproducible experiment outputs.

## Development Experiments

### Baseline
Model: Qwen3-4B-Base
Benchmark: 10-question DomainQA development benchmark
Lexical F1: 0.223

### Dense RAG
Model: Qwen3-4B-Base
Retriever: Sentence Transformers + FAISS
Top-K: 3
Lexical F1: 0.255
Recall@1: 0.100
Recall@3: 0.400
Recall@5: 0.400
MRR: 0.233

### BM25 RAG
Model: Qwen3-4B-Base
Retriever: BM25
Top-K: 3
Lexical F1: 0.270
Recall@1: 0.100
Recall@3: 0.400
Recall@5: 0.400
MRR: 0.233