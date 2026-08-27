import chromadb


client = chromadb.PersistentClient(
    path="data/vectorstore"
)

collection = client.get_collection(
    name="enterprise_documents"
)

collection.delete(
    ids=["test_policy.pdf_chunk_0"]
)

print("Removed old duplicate record.")

print("Remaining records:", collection.count())

results = collection.get(
    include=["documents", "metadatas"]
)

print("\nRemaining IDs:")
for document_id in results["ids"]:
    print(document_id)