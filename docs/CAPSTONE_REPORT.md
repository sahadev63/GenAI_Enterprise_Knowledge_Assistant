# GenAI Enterprise Knowledge Assistant — Capstone Report

## 1. Executive Summary

The GenAI Enterprise Knowledge Assistant is an agentic Retrieval-Augmented Generation (RAG) application that allows users to ask natural-language questions about information contained in enterprise documents.

The system combines document ingestion, heading-aware chunking, embeddings, ChromaDB semantic retrieval, relevance filtering, planner/retriever-reasoner/generator/validator agents, and grounded local LLM generation through Ollama.

A key design goal is to avoid unsupported answers. When information is not available in the supplied documents, the assistant reports that it was not found instead of relying on outside knowledge.

## 2. Objectives

- Build an end-to-end enterprise RAG pipeline.
- Convert document content into searchable vector representations.
- Retrieve relevant document chunks for user questions.
- Apply Top-K retrieval and distance-threshold filtering.
- Provide relevant context to the LLM.
- Generate grounded answers.
- Handle unknown and partially answerable questions.
- Evaluate retrieval and RAG behavior.
- Coordinate planning, retrieval reasoning, generation and validation through an agent orchestrator.

## 3. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | User interface |
| Sentence Transformers | Embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Vector storage and semantic retrieval |
| Ollama | Local LLM integration |
| Llama 3.2 3B | Answer generation |
| PyPDF | PDF processing |
| Pandas | CSV/data processing |
| OpenPyXL | XLSX processing |

## 4. Solution Overview

The application follows this flow:

Document → Text Extraction → Chunking → Embedding → ChromaDB

User Question → Embedding → Semantic Search → Top-K → Distance Filtering → Context → Prompt + Context → LLM → Grounded Answer

## 5. Agentic Workflow

The query path is coordinated by `AgentOrchestrator`:

1. **Planner Agent** — creates subtasks and an execution strategy.
2. **Retriever/Reasoner Agent** — retrieves evidence, checks grounding and decides whether the evidence is sufficient.
3. **Generator Agent** — generates an answer only from sufficient evidence.
4. **Validator Agent** — checks generated claims against the evidence.
5. **Correction fallback** — if validation rejects an answer, the system returns evidence-only content rather than presenting an unverified generated claim.

The retriever/reasoner can retry retrieval using expanded query variants, with a configurable retry limit.

## 6. Grounding

The generation prompt instructs the LLM to use only facts supplied in the context, avoid guessing, and explicitly report unsupported requested information.

The system also supports partial information. Therefore, if one part of a question is answerable and another part is unavailable, the answerable part is returned and the unavailable part is reported separately.

## 7. Evaluation Results

### Retrieval Evaluation

- True Positives: 6
- True Negatives: 3
- False Positives: 1
- False Negatives: 0
- Precision: 85.71%
- Recall: 100.00%
- F1 Score: 92.31%

### RAG Pipeline Evaluation

- Tests passed: 5/5
- Success rate: 100%

The RAG tests covered:
1. Known question
2. Paraphrased known question
3. Unknown question
4. Partial information
5. Unrelated question

## 8. Observed Evaluation Limitation

One retrieval test for sick-leave entitlement returned a document chunk even though the evaluation expected the information to be unavailable. This produced one false positive and resulted in retrieval precision of 85.71%.

The complete RAG pipeline nevertheless passed its unknown-question and grounding tests. This observation is retained as an evaluation limitation rather than being hidden.

## 9. Conclusion

The project demonstrates a complete enterprise RAG workflow with semantic retrieval, relevance filtering, grounded local LLM generation, and explicit handling of missing information.
