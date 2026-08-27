from app.ingestion.document_loader import load_document
from app.ingestion.text_chunker import chunk_text
from app.ingestion.embedding import generate_embedding
from app.retrieval.vector_store import add_document

file_path = "data/documents/test_policy.pdf"


document = load_document(file_path)

chunks = chunk_text(document["text"])


for index, chunk in enumerate(chunks):

    embedding = generate_embedding(chunk)

    document_id = f"{document['document_id']}_chunk_{index}"

    add_document(
    document_id=document_id,
    text=chunk,
    embedding=embedding,
    metadata={
        "file_name": document["file_name"],
        "chunk_index": index,
    },
)

    print(f"Stored: {document_id}")
