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
        self._initialize()

    def _initialize(self):
        """Connect to Qdrant document store."""
        from backend.core.document_store import get_document_store
        self.document_store = get_document_store()

    def run(self, query: str) -> dict:
        """
        Search documents for the given query.
        Returns top matching chunks with scores.
        """
        from backend.pipeline.retrieval import retrieve_documents

        # check if any documents exist
        if self.document_store.count_documents() == 0:
            return {
                "success": False,
                "message": "No documents uploaded yet.",
                "results": []
            }

        return retrieve_documents(query, self.document_store)

    def format_for_agent(self, query: str) -> str:
        """
        Run search and format results as text the agent can read.
        """
        from backend.pipeline.retrieval import format_results_for_llm

        result = self.run(query)

        if not result["success"]:
            return f"Document search failed: {result['message']}"

        if not result["results"]:
            return "No relevant documents found for this query."

        # add confidence warning if scores are low
        top_score = result["results"][0]["score"] if result["results"] else 0
        warning = ""
        if top_score < 0.3:
            warning = "⚠️ Low confidence — results may not be relevant.\n\n"

        output = warning
        output += f"Found {result['total_found']} relevant chunks "
        output += f"in {result['time_taken']}s:\n\n"

        for i, doc in enumerate(result["results"]):
            output += f"[Source {i+1}]\n"
            output += f"File   : {doc['source']}\n"
            output += f"Page   : {doc['page']}\n"
            output += f"Score  : {doc['score']}\n"
            output += f"Content: {doc['content']}\n"
            output += "-" * 40 + "\n"

        return output


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