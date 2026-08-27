# RAG Pipeline

## 1. Document Processing

The uploaded document is converted into text and divided into smaller chunks.

Chunking makes it possible to retrieve focused sections rather than supplying an entire document to the LLM.

## 2. Embeddings

The project uses:

```text
all-MiniLM-L6-v2
```

The model converts document chunks into numerical vectors. The same embedding model is used for user questions so that semantic similarity can be calculated.

## 3. ChromaDB

The generated vectors are stored in ChromaDB.

ChromaDB is used as the vector store for semantic retrieval.

## 4. Semantic Search

When a user asks a question:

```text
Question
  ↓
Question Embedding
  ↓
Similarity Search
  ↓
Candidate Chunks
```

## 5. Top-K

The current retrieval configuration uses:

```text
Top-K = 3
```

This means the retrieval stage can return up to three candidate chunks.

## 6. Distance Threshold

The current configured threshold is:

```text
Distance Threshold = 1.0
```

Retrieved chunks with a distance within the configured threshold are eligible as relevant context; results beyond the threshold are rejected.

## 7. Grounded Prompting

The prompt instructs the LLM to:

- Use only facts in the supplied context.
- Never use outside knowledge.
- Never guess or invent facts.
- Answer each requested part independently.
- Report unavailable information explicitly.
- Keep the answer concise.

## 8. Missing and Partial Information

For an unavailable question, the application returns a not-found response.

For a mixed question, the application answers the supported part and reports the unsupported part.

Example:

```text
Question:
What are the annual leave and maternity leave policies?

Result:
Annual leave: Employees are entitled to 24 days of annual leave every year.

Information about maternity leave policies was not found in the provided documents.
```

## 9. Failure Types

### Retrieval Failure

The required information exists in the document but the relevant chunk is not retrieved.

### Grounding / Generation Failure

Relevant context is retrieved but the generated answer is unsupported, incorrect, or unrelated to the retrieved context.

### Hallucination

An answer contains information that is not supported by the provided document context. The grounding prompt is designed to reduce this behavior.
