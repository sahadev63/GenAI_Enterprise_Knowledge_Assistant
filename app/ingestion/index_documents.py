"""Index all supported documents in data/documents into ChromaDB.

Run from the project root:
    python -m app.ingestion.index_documents
"""

from pathlib import Path

from app.ingestion.document_loader import SUPPORTED_EXTENSIONS, load_document
from app.ingestion.embedding import generate_embedding
from app.ingestion.text_chunker import chunk_text
from app.retrieval.vector_store import add_document


DOCUMENTS_DIR = Path("data/documents")


def index_documents(directory: Path = DOCUMENTS_DIR) -> tuple[int, int]:
    document_count = 0
    chunk_count = 0

    for file_path in sorted(directory.iterdir()):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        document = load_document(str(file_path))
        chunks = chunk_text(document["text"])

        for index, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)
            add_document(
                document_id=f"{document['document_id']}_chunk_{index}",
                text=chunk,
                embedding=embedding,
                metadata={
                    "document_id": document["document_id"],
                    "file_name": document["file_name"],
                    "file_type": document["file_type"],
                    "chunk_index": index,
                    "chunk_count": len(chunks),
                },
            )
            chunk_count += 1

        document_count += 1
        print(f"Indexed: {document['file_name']} ({len(chunks)} chunks)")

    print(f"Indexed documents: {document_count}")
    print(f"Indexed chunks: {chunk_count}")
    return document_count, chunk_count


if __name__ == "__main__":
    index_documents()
