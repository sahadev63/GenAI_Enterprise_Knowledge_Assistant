# GenAI Enterprise Knowledge Assistant

An **Agentic Retrieval-Augmented Generation (Agentic RAG)** enterprise knowledge assistant that allows users to upload enterprise documents and ask natural-language questions.

The system combines document ingestion, semantic retrieval, autonomous planning/reasoning, grounded generation, and answer validation using a local LLM (Ollama/Llama 3.2).

## Agentic workflow

```text
User Question
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
                              validate again
     |
     v
Verified / Safe Answer
```

### Reliability and exception handling

The orchestrator handles planner, retrieval, generation, and validation failures explicitly. Validation is **fail-closed**: if validation cannot be completed, the answer is not marked as verified. Rejected answers can be corrected from retrieved evidence and validated again.

## 1. Project Overview

The system follows an agentic RAG architecture:

User Question
    ↓
Planner Agent
    ↓
Retriever / Reasoner Agent
    ↓
Generator Agent
    ↓
Validator Agent
    ↓
Verified Grounded Answer

The original deterministic RAG pipeline remains available through the same
`answer_question()` API, while the new orchestration path adds planning,
retrieval sufficiency checks, controlled retries, and answer validation.

User Question
    ↓
Question Embedding
    ↓
ChromaDB Semantic Search
    ↓
Top-K Retrieval
    ↓
Distance Threshold Filtering
    ↓
Relevant Context
    ↓
Prompt + Context
    ↓
Ollama / Llama 3.2
    ↓
Grounded Answer

The application is designed to avoid answering from general knowledge when
the required information is not available in the provided documents.

---

## 2. Key Features

- Upload enterprise documents
- Supported formats:
  - PDF
  - TXT
  - CSV
  - XLSX
- Automatic document text extraction
- Text chunking
- Semantic embeddings
- ChromaDB vector storage
- Semantic similarity search
- Deterministic answerability guard for clear evidence
- Top-K retrieval
- Distance threshold filtering
- Grounded answer generation
- Planner, Retriever/Reasoner, Generator and Validator agents
- Retrieval sufficiency checks and controlled query retries
- Evidence-backed answer validation
- Agent execution trace and source display
- Missing-information handling
- Partial-information handling
- Local LLM using Ollama
- Streamlit web interface
- Retrieval and answer evaluation
- Project status dashboard

---

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Application development |
| Streamlit | Web interface |
| Sentence Transformers | Text embeddings |
| all-MiniLM-L6-v2 | Embedding model |
| ChromaDB | Vector database |
| Ollama | Local LLM integration |
| Llama 3.2 3B | Answer generation |
| PyPDF | PDF extraction |
| Pandas | CSV/data processing |
| OpenPyXL | XLSX processing |

---

## 4. Agentic RAG Pipeline

### Step 1: Document Upload

The user uploads a supported document through the Streamlit interface.

### Step 2: Text Extraction

Text is extracted from the uploaded document.

### Step 3: Chunking

Large document text is divided into smaller chunks.

This allows the retrieval system to search for specific pieces of
information instead of passing the entire document to the LLM.

### Step 4: Embedding

Each chunk is converted into a numerical vector using:

`all-MiniLM-L6-v2`

The same embedding model is used to convert the user's question into a
query vector.

### Step 5: Vector Storage

Document chunk embeddings are stored in ChromaDB.

### Step 6: Semantic Retrieval

The user's question is converted into an embedding and compared with the
stored document embeddings.

The most similar chunks are retrieved.

### Step 7: Top-K Retrieval

The system retrieves a limited number of candidate chunks.

Example:

`Top-K = 3`

means up to three candidate chunks are retrieved.

### Step 8: Distance Threshold

Retrieved chunks are filtered using a distance threshold.

Current project configuration:

`Distance Threshold = 1.0`

Chunks whose distance is greater than the configured threshold are rejected.

### Step 9: Context Construction

Only relevant retrieved information is supplied to the LLM.

### Step 10: Grounded Generation

The LLM generates an answer using the supplied context.

The system instructs the model not to invent information that is not
available in the documents.

---

## 5. Agent Architecture

```text
                         User Question
                              │
                              ▼
                     ┌─────────────────┐
                     │  Planner Agent  │
                     └────────┬────────┘
                              │
                  one or more subtasks
                              │
                              ▼
              ┌────────────────────────────┐
              │ Retriever / Reasoner Agent │
              └─────────────┬──────────────┘
                            │
                    evidence sufficient?
                       /             \
                     NO              YES
                     │                │
                rewrite query        ▼
                     │        ┌───────────────┐
                     └───────►│ Generator     │
                              │ Agent         │
                              └───────┬───────┘
                                      │
                                      ▼
                              ┌───────────────┐
                              │ Validator     │
                              │ Agent         │
                              └───────┬───────┘
                                      │
                                      ▼
                              Verified Answer
```

### Planner Agent

Breaks a complex user request into independent subtasks. The local LLM is
asked for a JSON plan; a deterministic fallback is used if the model is
unavailable or returns invalid JSON.

### Retriever / Reasoner Agent

Embeds each subtask, retrieves Top-K evidence, applies the configured distance
threshold and a grounding check, then asks the local LLM whether the evidence
is sufficient. If evidence is insufficient, it retries with query-expanded
variants, up to the configured retry limit.

### Generator Agent

Generates an answer using only the evidence selected by the retriever/reasoner.

### Validator Agent

Checks the generated answer against the retrieved evidence. Unsupported output
is not accepted as a verified answer. A deterministic evidence-only fallback is
used when validation cannot be completed by the local LLM.

### Orchestrator

`AgentOrchestrator` coordinates all agents and returns the answer together with
the execution trace, evidence metadata, retrieval attempts and validation
status.

## 6. Grounding and Missing Information

The system handles missing information explicitly.

For example, if the document contains:

"Employees are entitled to 24 days of annual leave every year."

and the user asks:

"What is the company's maternity leave policy?"

the system should not invent a maternity leave policy.

Instead, it returns that the information was not found in the provided
documents.

For a question containing both available and unavailable information, the
system supports partial answers.

Example:

Question:

"What are the annual leave and maternity leave policies?"

Expected behavior:

Annual leave: Employees are entitled to 24 days of annual leave every year.

Maternity leave: Information about maternity leave policies was not found
in the provided documents.

---

## 7. Project Structure

```text
GenAI_Enterprise_Knowledge_Assistant/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── app/
│   ├── agents/
│   │   ├── planner_agent.py
│   │   ├── retriever_reasoner_agent.py
│   │   ├── generator_agent.py
│   │   ├── validator_agent.py
│   │   └── orchestrator.py
│   ├── ingestion/
│   ├── retrieval/
│   ├── generation/
│   ├── rag/
│   └── evaluation/
├── data/
│   ├── documents/
│   ├── evaluation/
│   └── vectorstore/
└── docs/
    ├── CAPSTONE_REPORT.md
    ├── CAPSTONE_REPORT.docx
    ├── CAPSTONE_REQUIREMENT_MAPPING.md
    ├── PROJECT_ARCHITECTURE.md
    ├── RAG_PIPELINE.md
    ├── EVALUATION_REPORT.md
    └── SETUP_AND_DEPLOYMENT.md
```

## Documentation

Detailed project documentation is available in the `docs/` directory.

- [Capstone Report](docs/CAPSTONE_REPORT.md)
- [Project Architecture](docs/PROJECT_ARCHITECTURE.md)
- [RAG Pipeline](docs/RAG_PIPELINE.md)
- [Evaluation Report](docs/EVALUATION_REPORT.md)
- [Setup and Deployment](docs/SETUP_AND_DEPLOYMENT.md)

The documentation is based on the current implementation and verified test results.
