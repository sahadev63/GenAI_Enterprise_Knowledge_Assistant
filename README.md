# GenAI Enterprise Knowledge Assistant

A Retrieval-Augmented Generation (RAG) based enterprise knowledge assistant
that allows users to upload documents and ask questions using natural language.

The application retrieves relevant information from the uploaded documents
using semantic search and generates grounded answers using a local LLM.

---

## 1. Project Overview

The system follows a RAG architecture:

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
- Top-K retrieval
- Distance threshold filtering
- Grounded answer generation
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

## 4. RAG Pipeline

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

## 5. Grounding and Missing Information

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

## 6. Project Structure

```text
GenAI_Enterprise_Knowledge_Assistant/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
├── app/
│   ├── ingestion/
│   │   ├── embedding.py
│   │   └── ...
│   │
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   └── ...
│   │
│   ├── generation/
│   │   ├── llm.py
│   │   ├── prompt.py
│   │   └── ...
│   │
│   ├── rag/
│   │   ├── rag_pipeline.py
│   │   └── ...
│   │
│   └── evaluation/
│       └── ...
│
└── data/
    ├── documents/
    ├── evaluation/
    └── vectorstore/

## Documentation

Detailed project documentation is available in the `docs/` directory.

- [Capstone Report](docs/CAPSTONE_REPORT.md)
- [Project Architecture](docs/PROJECT_ARCHITECTURE.md)
- [RAG Pipeline](docs/RAG_PIPELINE.md)
- [Evaluation Report](docs/EVALUATION_REPORT.md)
- [Setup and Deployment](docs/SETUP_AND_DEPLOYMENT.md)

The documentation is based on the current implementation and verified test results.
