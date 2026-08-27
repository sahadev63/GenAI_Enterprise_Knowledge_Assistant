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

## 5. Run the Application

From the project root:

```bash
streamlit run app.py
```

## 6. Run Evaluation Tests

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

## 7. Expected Validation

The current verified project produced:

```text
Retrieval Precision : 85.71%
Retrieval Recall    : 100.00%
Retrieval F1 Score  : 92.31%

RAG Pipeline:
Passed              : 5/5
Success Rate        : 100.00%
```

## 8. Submission Packaging

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
