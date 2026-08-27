# Project Architecture

## 1. High-Level Architecture

```text
                    ┌──────────────────────┐
                    │      Streamlit UI    │
                    └──────────┬───────────┘
                               │
                    Upload / Ask Question
                               │
              ┌────────────────┴────────────────┐
              │                                 │
       Document Ingestion                 User Question
              │                                 │
       Text Extraction                    Embedding
              │                                 │
          Chunking                              │
              │                                 │
          Embedding                             │
              │                                 │
              └──────────────┐     ┌─────────────┘
                             ▼     ▼
                         ┌────────────┐
                         │  ChromaDB  │
                         └─────┬──────┘
                               │
                       Semantic Retrieval
                               │
                           Top-K Filter
                               │
                       Distance Threshold
                               │
                        Relevant Context
                               │
                        Prompt + Context
                               │
                         Ollama / LLM
                               │
                         Grounded Answer
                               ▼
                        Streamlit Response
```

## 2. Main Components

### Streamlit

Provides the user interface for document interaction and question answering.

### Ingestion

Loads supported documents, extracts text, creates chunks, generates embeddings, and stores vector data.

### ChromaDB

Stores document embeddings and supports semantic similarity search.

### Retrieval

Converts a user question into an embedding, searches ChromaDB, applies Top-K retrieval, and filters results using the configured distance threshold.

### Generation

Builds a grounding-oriented prompt and sends the relevant context to the local LLM through Ollama.

### Evaluation

Contains tests for retrieval behavior and complete RAG pipeline behavior.

## 3. Data Flow

### Ingestion flow

```text
Document
   ↓
Extract Text
   ↓
Create Chunks
   ↓
Generate Embeddings
   ↓
Store in ChromaDB
```

### Query flow

```text
User Question
   ↓
Generate Query Embedding
   ↓
ChromaDB Similarity Search
   ↓
Top-K Candidates
   ↓
Distance Threshold
   ↓
Relevant Context
   ↓
Grounding Prompt
   ↓
LLM
   ↓
Final Answer
```
