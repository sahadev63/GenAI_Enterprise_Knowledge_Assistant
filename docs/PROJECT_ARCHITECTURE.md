# Project Architecture

## 1. High-Level Agentic Architecture

```text
                              ┌──────────────────────┐
                              │      Streamlit UI    │
                              └──────────┬───────────┘
                                         │
                              Upload / Ask Question
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                         │
             Document Ingestion                         User Question
                    │                                         │
             Text Extraction                            Planner Agent
                    │                                         │
          Heading-aware Chunking                       Subtasks / Strategy
                    │                                         │
                Embedding                                     ▼
                    │                              Retriever / Reasoner Agent
                    ▼                                         │
               ┌───────────┐                         ┌─────────┴─────────┐
               │ ChromaDB  │◄────────────────────────│ Semantic Retrieval │
               └─────┬─────┘                         └─────────┬─────────┘
                     │                                           │
                     │                                  Sufficiency Check
                     │                                      │       │
                     │                                     NO      YES
                     │                                      │       │
                     │                              Query Rewrite   ▼
                     │                                   │     Generator Agent
                     │                                   └──────►    │
                     │                                              ▼
                     │                                       Validator Agent
                     │                                              │
                     └──────────────────────────────────────────────┤
                                                                    ▼
                                                            Verified Answer
                                                                    │
                                                                    ▼
                                                              Streamlit UI
```

## 2. Main Components

### Planner Agent
Creates an execution plan and splits multi-part questions into independent subtasks.

### Retriever / Reasoner Agent
Performs semantic retrieval, relevance filtering, evidence grounding and
sufficiency reasoning. It can rewrite the query and retry retrieval when evidence
is insufficient.

### Generator Agent
Uses only retrieved evidence to produce a concise answer.

### Validator Agent
Checks factual claims in the generated answer against retrieved evidence.

### Orchestrator
Coordinates Planner → Retriever/Reasoner → Generator → Validator and returns a
traceable result.

### Ingestion
Supports PDF, TXT, CSV and XLSX extraction, followed by heading-aware chunking
and embedding generation.

### ChromaDB
Stores document embeddings and metadata and provides semantic similarity search.

## 3. Query Flow

```text
User Question
   ↓
Input Validation
   ↓
Planner Agent
   ↓
Subtasks
   ↓
Retriever / Reasoner Agent
   ↓
Top-K + Distance Threshold + Grounding
   ↓
Sufficient? ── No ──► Query Rewrite ──► Retry
   │
  Yes
   ↓
Generator Agent
   ↓
Validator Agent
   ↓
Supported? ── No ──► Evidence-only correction
   │
  Yes
   ↓
Verified Answer + Sources + Trace
```

## 4. Code Structure

```text
app/
├── agents/
│   ├── common.py
│   ├── models.py
│   ├── planner_agent.py
│   ├── retriever_reasoner_agent.py
│   ├── generator_agent.py
│   ├── validator_agent.py
│   ├── orchestrator.py
│   └── test_agents.py
├── ingestion/
├── retrieval/
├── generation/
├── rag/
└── evaluation/
```

## 5. Backward Compatibility

Existing callers can continue using:

```python
from app.rag.rag_pipeline import answer_question
```

For source metadata, validation status and execution trace, use:

```python
from app.rag.rag_pipeline import answer_question_with_trace
```
