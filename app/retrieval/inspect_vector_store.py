import chromadb


client = chromadb.PersistentClient(
    path="data/vectorstore"
)

collection = client.get_collection(
    name="enterprise_documents"
)

print("Total records:", collection.count())

results = collection.get(
    include=["documents", "metadatas"]
)

print("\nIDs:")
for document_id in results["ids"]:
    print(document_id)

print("\nMetadata:")
for metadata in results["metadatas"]:
    print(metadata)