# Capstone Requirement Mapping

| Capstone requirement | Project implementation |
|---|---|
| Project foundation | Python application structure, configuration and requirements |
| User interaction | Streamlit document upload and natural-language questions |
| Multi-format ingestion | PDF, TXT, CSV and XLSX |
| Data preparation | Heading-aware document chunking |
| Vector knowledge store | Sentence Transformers + ChromaDB |
| Intelligent retrieval | Top-K, distance filtering, grounding checks and retry |
| RAG | Retrieved evidence + local LLM generation |
| Agent-based reasoning | Planner, Retriever/Reasoner, Generator, Validator and Orchestrator |
| Reliability/safety | Input checks, exception handling, deterministic fallbacks and fail-closed validation |
| Deployment/documentation | Local Streamlit + Ollama setup, architecture, evaluation and limitations |

The project is documented according to the supplied capstone specification and does not claim cloud deployment that has not been performed.
