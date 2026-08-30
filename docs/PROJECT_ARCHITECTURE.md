# Project Architecture

## Agentic RAG architecture

```text
Streamlit UI
     |
     v
Agent Orchestrator
     |
     v
Planner Agent
     |
     v
Retriever / Reasoner Agent
     |
     +---- insufficient evidence ----> query rewrite / retry
     |
     v
Generator Agent
     |
     v
Validator Agent
     |
     +---- unsupported ----> evidence-only correction
                                  |
                                  v
                              Validator again
     |
     v
Verified / Safe Answer

Document path:
Upload -> Format Loader -> Heading-aware Chunking
-> SentenceTransformer Embeddings -> ChromaDB
-> Semantic Retrieval -> Evidence Context
```

## Agent responsibilities

- **Planner Agent**: decomposes complex questions and determines a retrieval strategy.
- **Retriever / Reasoner Agent**: retrieves evidence, checks relevance/sufficiency, and retries with rewritten queries.
- **Generator Agent**: generates an answer from retrieved evidence.
- **Validator Agent**: verifies generated claims against the evidence.
- **Orchestrator**: coordinates all agents and enforces safe exception and validation behavior.

## Exception and validation policy

The validation path is **fail-closed**. A validator exception cannot be interpreted as successful validation. If LLM validation and deterministic fallback validation both fail, the result remains unverified and a controlled fallback is returned.

## Deployment

The reference deployment is a locally deployable Streamlit application using Ollama/Llama 3.2 for local LLM inference. Cloud deployment is a future enhancement and is not represented as an already-completed deployment.

