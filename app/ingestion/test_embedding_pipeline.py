from app.ingestion.document_loader import load_document
from app.ingestion.text_chunker import chunk_text
from app.ingestion.embedding import generate_embedding

file_path = "data/documents/test_policy.pdf"


document = load_document(file_path)

print("Document:", document["file_name"])
print("Character count:", document["character_count"])


chunks = chunk_text(document["text"])

print("Number of chunks:", len(chunks))


for index, chunk in enumerate(chunks):

    embedding = generate_embedding(chunk)

    print()
    print("Chunk:", index + 1)
    print("Chunk characters:", len(chunk))
    print("Embedding dimensions:", len(embedding))