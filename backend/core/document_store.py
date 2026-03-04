import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_DIM
)

def get_document_store():
    """
    Create and return a Haystack QdrantDocumentStore.
    This is the single source of truth for document storage.
    """
    from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
    from haystack.utils import Secret

    store = QdrantDocumentStore(
        url=QDRANT_URL,
        api_key=Secret.from_token(QDRANT_API_KEY),
        index=COLLECTION_NAME,
        embedding_dim=EMBEDDING_DIM,
        recreate_index=False,
        return_embedding=True,
        wait_result_from_api=True
    )

    return store

def get_document_count(store) -> int:
    """Return how many documents are in the store."""
    return store.count_documents()

def clear_document_store(store):
    """
    Delete all documents from the store.
    Used for testing or when user wants to reset.
    """
    store.delete_documents(delete_all=True)
    print("Document store cleared")


# test connection
if __name__ == "__main__":
    print("Testing Qdrant Document Store connection...")
    store = get_document_store()
    count = get_document_count(store)
    print(f"Connected successfully")
    print(f"Documents in store: {count}")
    print(f"Collection: {COLLECTION_NAME}")