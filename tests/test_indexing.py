import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.document_store import get_document_store
from backend.pipeline.indexing import index_document
from backend.tools.document_search import DocumentSearchTool


def test_full_flow():
    print("FULL INDEXING AND RETRIEVAL TEST")
    print("=" * 60)

    # step 1 - connect to store
    print("\nStep 1: Connecting to Qdrant...")
    store = get_document_store()
    print(f"Documents currently in store: {store.count_documents()}")

    # step 2 - search the indexed document
    print("\nStep 2: Searching indexed documents...")
    search_tool = DocumentSearchTool()

    test_queries = [
        "What is Agentic RAG?",
        "How does machine learning work?",
        "What is natural language processing?",
        "Who invented the telephone?"  # not in document - should return low scores
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        result = search_tool.format_for_agent(query)
        print(result[:400])
        print()


if __name__ == "__main__":
    test_full_flow()