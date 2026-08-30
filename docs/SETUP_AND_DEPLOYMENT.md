# Setup and Deployment

## 1. Prerequisites

Install:

- Python
- Ollama

The required Python packages are listed in `requirements.txt`.

## 2. Create Virtual Environment

From the project root:

```bash
python -m venv .venv
```

Activate on Windows:

```bash
.venv\Scripts\activate
```

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

## 4. Install the LLM

Use Ollama and pull the configured model:

```bash
ollama pull llama3.2:3b
```

Verify:

```bash
ollama list
```

## 5. Rebuild the Vector Store (Optional)

The submission contains a persisted ChromaDB vector store for the sample
documents. If the vector store is removed or you want to rebuild it:

```bash
python -m app.ingestion.clean_vector_store
python -m app.ingestion.index_documents
```

## 6. Run the Application

From the project root:

```bash
streamlit run app.py
```

## 7. Run Evaluation Tests

Retrieval evaluation:

```bash
python -m app.evaluation.test_evaluation
```

Retrieval test:

```bash
python -m app.retrieval.test_retrieval
```

RAG pipeline test:

```bash
python -m app.rag.test_rag_pipeline
```

## 8. Expected Validation

The current verified project produced:

```text
Retrieval Precision : 85.71%
Retrieval Recall    : 100.00%
Retrieval F1 Score  : 92.31%

RAG Pipeline:
Passed              : 5/5
Success Rate        : 100.00%
```

## 9. Submission Packaging

The `.venv` directory should not be included in the submission ZIP.

Also exclude:

```text
.env
__pycache__/
*.pyc
.vscode/
.idea/
```

The submission ZIP should contain the source code, documentation, requirements, evaluation files, sample documents, and the project's persisted vector store when required by the application.


## Capstone Deployment Statement

The application is locally deployable with Streamlit and Ollama/Llama 3.2. The end-to-end workflow is:

`Document Ingestion -> Chunking -> Embeddings -> ChromaDB -> Retrieval -> Agent Planning/Reasoning -> Grounded Generation -> Validation -> Safe Answer`

The project does not claim an already-completed cloud deployment; cloud hosting is a future enhancement.

### Reliability

Planner, retrieval, generation, and validation failures are handled explicitly. Validation is fail-closed so an exception cannot turn an unverified answer into a verified answer.
