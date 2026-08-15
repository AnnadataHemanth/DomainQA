# DomainQA

## Comparative Analysis of Fine-Tuning and Retrieval-Augmented Generation for Domain-Specific Question Answering

DomainQA is an experimental study comparing four approaches to domain-specific question answering:

1. Base LLM
2. Fine-tuned LLM
3. Retrieval-Augmented Generation (RAG)
4. Fine-tuned LLM + RAG

### Research Question

How do fine-tuning and retrieval-augmented generation compare for domain-specific question answering, and does combining both approaches provide measurable advantages?

### Experimental Objectives

- Establish a base-model benchmark
- Build a domain-specific RAG pipeline
- Fine-tune an open-weight language model using parameter-efficient fine-tuning
- Develop a hybrid fine-tuning + RAG system
- Compare the systems using retrieval, generation, accuracy, faithfulness, hallucination, latency, and compute metrics
-The model was evaluated on documents that were unseen during fine-tuning.

### Planned Model

Qwen3-4B family

### Domain

Scientific literature in Artificial Intelligence and Machine Learning