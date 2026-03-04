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
    # get all document ids and delete them
    from qdrant_client import QdrantClient
    from configs.settings import QDRANT_URL, QDRANT_API_KEY, COLLECTION_NAME

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    # delete and recreate the collection
    client.delete_collection(COLLECTION_NAME)
    print(f"Collection {COLLECTION_NAME} deleted")

    # recreate it fresh
    from qdrant_client.models import Distance, VectorParams
    from configs.settings import EMBEDDING_DIM

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE
        )
    )
    print(f"Collection {COLLECTION_NAME} recreated fresh")


# test connection
if __name__ == "__main__":
    print("Testing Qdrant Document Store connection...")
    store = get_document_store()
    count = get_document_count(store)
    print(f"Connected successfully")
    print(f"Documents in store: {count}")
    print(f"Collection: {COLLECTION_NAME}")