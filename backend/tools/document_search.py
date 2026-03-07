"""
Tool 1: Document Search
Uses the Haystack retrieval pipeline to search indexed documents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL
)


class DocumentSearchTool:
    """
    Searches uploaded documents using Haystack retrieval pipeline.
    This is Tool 1 in the agent's tool registry.
    """

    name = "document_search"
    description = """Search through uploaded documents to find relevant
    information. Use this tool when the question is about content that
    might be in the user's documents. Input should be a search query."""

    def __init__(self):
        self.document_store = None
        self.bm25_store = None
        self._initialize()

    def _initialize(self):
        """Connect to Qdrant document store and BM25 store."""
        from backend.core.document_store import get_document_store
        from backend.core.bm25_store import BM25Store

        self.document_store = get_document_store()
        self.bm25_store = BM25Store()

    def run(self, query: str) -> dict:
        """
        Search documents using hybrid search.
        Combines dense vector search with BM25 keyword search.
        """
        from backend.pipeline.hybrid_search import hybrid_search

        if self.document_store.count_documents() == 0:
            return {
                "success": False,
                "message": "No documents uploaded yet.",
                "results": []
            }

        return hybrid_search(query, self.document_store, self.bm25_store)

    def format_for_agent(self, query: str) -> str:
        """
        Run hybrid search and format results for agent.
        """
        from backend.pipeline.hybrid_search import format_hybrid_results_for_agent

        if self.document_store.count_documents() == 0:
            return "No documents uploaded yet. Please upload documents first."

        return format_hybrid_results_for_agent(
            query,
            self.document_store,
            self.bm25_store
        )


# quick test
if __name__ == "__main__":
    tool = DocumentSearchTool()
    print("Testing Document Search Tool\n")

    queries = [
        "What is Agentic RAG?",
        "How does machine learning work?",
    ]

    for query in queries:
        print(f"Query: {query}")
        print("-" * 50)
        result = tool.format_for_agent(query)
        print(result[:500])
        print()