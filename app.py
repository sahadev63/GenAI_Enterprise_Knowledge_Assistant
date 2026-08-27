import os

import streamlit as st

from app.config import RAG_DISTANCE_THRESHOLD, RAG_TOP_K
from app.ingestion.document_loader import load_document
from app.ingestion.embedding import generate_embedding
from app.ingestion.text_chunker import chunk_text
from app.rag.rag_pipeline import answer_question
from app.retrieval.vector_store import add_document, get_collection_stats


st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Enterprise AI Knowledge Assistant")

st.write(
    "Upload enterprise documents and ask questions using a grounded "
    "Retrieval-Augmented Generation (RAG) pipeline."
)

st.subheader("📄 Document Upload")

uploaded_files = st.file_uploader(
    "Upload your documents",
    type=["pdf", "txt", "csv", "xlsx"],
    accept_multiple_files=True,
)

if uploaded_files:
    os.makedirs("data/documents", exist_ok=True)

    st.subheader("Uploaded Documents")

    for uploaded_file in uploaded_files:
        file_path = os.path.join(
            "data",
            "documents",
            uploaded_file.name,
        )

        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())

        try:
            document = load_document(file_path)
            chunks = chunk_text(document["text"])

            for index, chunk in enumerate(chunks):
                embedding = generate_embedding(chunk)

                document_id = f"{uploaded_file.name}_chunk_{index}"

                metadata = {
                    "file_name": uploaded_file.name,
                    "chunk_index": index,
                }

                add_document(
                    document_id=document_id,
                    text=chunk,
                    embedding=embedding,
                    metadata=metadata,
                )

            st.success(
                f"Successfully processed: {document['file_name']} "
                f"({len(chunks)} chunks)"
            )

            with st.expander(f"View {document['file_name']} details"):
                st.write(f"File type: {document['file_type']}")
                st.write(
                    f"Characters extracted: {document['character_count']}"
                )
                st.write(f"Chunks created: {len(chunks)}")

                st.write("### Extracted text preview")
                st.text(document["text"][:5000])

                st.write("### Chunks")
                for index, chunk in enumerate(chunks, start=1):
                    st.write(f"**Chunk {index}**")
                    st.text(chunk)

        except Exception as error:
            st.error(
                f"Failed to process {uploaded_file.name}: {error}"
            )


st.divider()

st.subheader("🔍 Ask a Question")

with st.form("question_form"):
    question = st.text_input(
        "Enter your question",
        placeholder="e.g. How many annual leave days are available?",
    )

    search_clicked = st.form_submit_button("🤖 Ask AI")

if search_clicked and question.strip():
    with st.spinner("Searching knowledge base and generating answer..."):
        try:
            answer = answer_question(
                question,
                n_results=RAG_TOP_K,
                distance_threshold=RAG_DISTANCE_THRESHOLD,
            )

            st.subheader("🤖 AI Answer")
            st.write(answer)

        except Exception as error:
            st.error(f"Failed to generate answer: {error}")


st.subheader("Project Status")

document_count, chunk_count = get_collection_stats()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Documents", document_count)

with col2:
    st.metric("Knowledge Chunks", chunk_count)

with col3:
    st.metric("AI Status", "Configured")
