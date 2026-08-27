import chromadb

client = chromadb.PersistentClient(
    path="data/vectorstore"
)

collection = client.get_or_create_collection(
    name="enterprise_documents"
)


def add_document(
    document_id: str,
    text: str,
    embedding,
    metadata: dict,
):
    collection.upsert(
        ids=[document_id],
        documents=[text],
        embeddings=[embedding],
        metadatas=[metadata],
    )


def search_documents(
    query_embedding,
    n_results: int = 3,
    distance_threshold: float | None = None,
):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
    )

    if distance_threshold is not None and results["documents"][0]:
        distances = results["distances"][0]

        filtered_indexes = [
            i
            for i, distance in enumerate(distances)
            if distance <= distance_threshold
        ]

        results["documents"][0] = [
            results["documents"][0][i]
            for i in filtered_indexes
        ]
        results["metadatas"][0] = [
            results["metadatas"][0][i]
            for i in filtered_indexes
        ]
        results["distances"][0] = [
            results["distances"][0][i]
            for i in filtered_indexes
        ]

    return results


def get_collection_stats() -> tuple[int, int]:
    """
    Return (unique_documents, total_chunks) currently stored in ChromaDB.
    """
    total_chunks = collection.count()

    if total_chunks == 0:
        return 0, 0

    data = collection.get(include=["metadatas"])
    metadatas = data.get("metadatas") or []

    file_names = {
        metadata.get("file_name")
        for metadata in metadatas
        if metadata and metadata.get("file_name")
    }

    return len(file_names), total_chunks
