import sys
from types import ModuleType
from unittest.mock import MagicMock, patch


def test_reset_collection_recreates_cosine_collection():
    # Keep this regression test runnable even in a minimal CI environment
    # where optional project dependencies have not yet been installed.
    fake_chromadb = ModuleType("chromadb")
    fake_chromadb.PersistentClient = MagicMock()

    with patch.dict(sys.modules, {"chromadb": fake_chromadb}):
        # Import after the dependency shim is installed.
        sys.modules.pop("app.retrieval.vector_store", None)
        from app.retrieval import vector_store

        client = MagicMock()
        with patch.object(vector_store, "_get_client", return_value=client):
            result = vector_store.reset_collection()

        client.delete_collection.assert_called_once_with(
            name=vector_store.COLLECTION_NAME
        )
        client.create_collection.assert_called_once_with(
            name=vector_store.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        assert result is client.create_collection.return_value
