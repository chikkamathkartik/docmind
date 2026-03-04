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
    Searches uploaded documents using vector similarity.
    This is Tool 1 in the agent's tool registry.
    """

    name = "document_search"
    description = """Search through uploaded documents to find relevant 
    information. Use this tool when the question is about content that 
    might be in the user's documents. Input should be a search query string."""

    def __init__(self):
        self.embedding_model = None
        self.qdrant_client = None
        self._initialize()

    def _initialize(self):
        """Load embedding model and connect to Qdrant."""
        print("Initializing Document Search Tool...")

        # load embedding model
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        # connect to Qdrant
        from qdrant_client import QdrantClient
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )

        print("Document Search Tool ready")

    def run(self, query: str) -> dict:
        """
        Search documents for the given query.
        Returns top matching chunks with their sources.
        """
        try:
            # check if collection exists and has documents
            collections = [
                c.name for c in
                self.qdrant_client.get_collections().collections
            ]

            if COLLECTION_NAME not in collections:
                return {
                    "success": False,
                    "message": "No documents have been uploaded yet.",
                    "results": []
                }

            # embed the query
            query_embedding = self.embedding_model.encode([query])[0]

            # search Qdrant
            results = self.qdrant_client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_embedding.tolist(),
                limit=TOP_K_RETRIEVAL,
                with_payload=True
            )

            # format results
            formatted_results = []
            for point in results.points:
                formatted_results.append({
                    "content": point.payload.get("content", ""),
                    "source": point.payload.get("source", "unknown"),
                    "page": point.payload.get("page", "N/A"),
                    "score": round(point.score, 4)
                })

            return {
                "success": True,
                "query": query,
                "results": formatted_results,
                "total_found": len(formatted_results)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Search failed: {str(e)}",
                "results": []
            }

    def format_for_agent(self, query: str) -> str:
        """
        Run search and format results as text the agent can read.
        This is what the agent actually sees as the tool output.
        """
        result = self.run(query)

        if not result["success"]:
            return f"Document search failed: {result['message']}"

        if not result["results"]:
            return "No relevant documents found for this query."

        # format as readable text for the agent
        output = f"Found {result['total_found']} relevant chunks:\n\n"
        for i, r in enumerate(result["results"]):
            output += f"[Source {i+1}]\n"
            output += f"File: {r['source']} | Page: {r['page']}\n"
            output += f"Relevance Score: {r['score']}\n"
            output += f"Content: {r['content']}\n"
            output += "-" * 40 + "\n"

        return output


# quick test
if __name__ == "__main__":
    tool = DocumentSearchTool()
    print("\nTesting document search tool...")
    print("Note: Returns empty if no documents uploaded to Qdrant yet\n")
    result = tool.format_for_agent("What is machine learning?")
    print(result)