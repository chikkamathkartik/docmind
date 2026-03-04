"""
Test retrieval pipeline with multiple queries.
Run with: python tests/test_retrieval.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core.document_store import get_document_store
from backend.pipeline.retrieval import retrieve_documents
from backend.tools.document_search import DocumentSearchTool


def test_retrieval():
    print("RETRIEVAL PIPELINE TEST")
    print("=" * 60)

    store = get_document_store()
    count = store.count_documents()
    print(f"Documents in store: {count}\n")

    if count == 0:
        print("No documents found. Run indexing first.")
        return

    # test queries with expected relevance
    test_cases = [
        ("What is Agentic RAG?", "HIGH relevance expected"),
        ("How does machine learning work?", "HIGH relevance expected"),
        ("What is deep learning?", "MEDIUM relevance expected"),
        ("Who invented the telephone?", "LOW relevance expected"),
        ("What is the capital of France?", "LOW relevance expected"),
    ]

    correct = 0
    total = len(test_cases)

    for query, expectation in test_cases:
        print(f"Query      : {query}")
        print(f"Expectation: {expectation}")

        results = retrieve_documents(query, store)

        if results["success"] and results["results"]:
            top_score = results["results"][0]["score"]
            top_content = results["results"][0]["content"][:80]

            bar = "█" * int(top_score * 20)
            print(f"Top Score  : {top_score} {bar}")
            print(f"Top Result : {top_content}...")

            # simple relevance check
            if "HIGH" in expectation and top_score > 0.3:
                print("Result     : ✅ CORRECT")
                correct += 1
            elif "LOW" in expectation and top_score < 0.3:
                print("Result     : ✅ CORRECT")
                correct += 1
            else:
                print("Result     : ⚠️ CHECK MANUALLY")
        else:
            print(f"Error: {results.get('message', 'Unknown error')}")

        print()

    print("=" * 60)
    print(f"Retrieval Accuracy: {correct}/{total} correct")
    print("=" * 60)


if __name__ == "__main__":
    test_retrieval()