"""
Haystack Retrieval Pipeline
Takes a user query and returns the most relevant chunks.

Pipeline flow:
Query → TextEmbedder → EmbeddingRetriever (Qdrant) → Results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import (
    EMBEDDING_MODEL,
    TOP_K_RETRIEVAL,
    TOP_K_RERANK
)


def build_retrieval_pipeline(document_store):
    """
    Build and return the Haystack retrieval pipeline.
    Takes a query string and returns top matching chunks.
    """
    from haystack import Pipeline
    from haystack.components.embedders import SentenceTransformersTextEmbedder
    from haystack.components.rankers import SentenceTransformersSimilarityRanker
    from haystack_integrations.components.retrievers.qdrant import (
        QdrantEmbeddingRetriever
    )

    pipeline = Pipeline()

    # embed the query using same model as documents
    pipeline.add_component("embedder", SentenceTransformersTextEmbedder(
        model=EMBEDDING_MODEL
    ))

    # retrieve top k chunks from Qdrant
    pipeline.add_component("retriever", QdrantEmbeddingRetriever(
        document_store=document_store,
        top_k=TOP_K_RETRIEVAL
    ))

    # rerank results using cross encoder for better accuracy
    pipeline.add_component("reranker", SentenceTransformersSimilarityRanker(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_k=TOP_K_RERANK,
    scale_score=True
    ))

    # connect components
    pipeline.connect("embedder.embedding", "retriever.query_embedding")
    pipeline.connect("retriever.documents", "reranker.documents")

    return pipeline


def retrieve_documents(query: str, document_store) -> dict:
    """
    Retrieve relevant documents for a query.
    Returns ranked chunks with scores and metadata.
    """
    import time

    start_time = time.time()

    try:
        pipeline = build_retrieval_pipeline(document_store)

        result = pipeline.run({
            "embedder": {"text": query},
            "reranker": {"query": query}
        })

        documents = result["reranker"]["documents"]
        elapsed = time.time() - start_time

        # format results cleanly
        formatted = []
        for doc in documents:
            formatted.append({
                "content": doc.content,
                "score": round(doc.score, 4),
                "source": doc.meta.get("file_name",
                          doc.meta.get("source", "unknown")),
                "page": doc.meta.get("page_number",
                        doc.meta.get("page", "N/A"))
            })

        return {
            "success": True,
            "query": query,
            "results": formatted,
            "total_found": len(formatted),
            "time_taken": round(elapsed, 2)
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": str(e),
            "results": [],
            "total_found": 0
        }


def format_results_for_llm(results: dict) -> str:
    """
    Format retrieval results as context for the LLM prompt.
    This is what gets sent to the agent.
    """
    if not results["success"] or not results["results"]:
        return "No relevant documents found."

    context = ""
    for i, doc in enumerate(results["results"]):
        context += f"[Document {i+1}]\n"
        context += f"Source : {doc['source']}\n"
        context += f"Page   : {doc['page']}\n"
        context += f"Score  : {doc['score']}\n"
        context += f"Content: {doc['content']}\n"
        context += "-" * 40 + "\n"

    return context


# test the retrieval pipeline
if __name__ == "__main__":
    from backend.core.document_store import get_document_store

    print("Testing Retrieval Pipeline")
    print("=" * 60)

    store = get_document_store()
    count = store.count_documents()
    print(f"Documents in store: {count}")

    if count == 0:
        print("\nNo documents in store.")
        print("Run backend/pipeline/indexing.py first.")
        sys.exit(1)

    # test queries
    test_queries = [
        "What is Agentic RAG?",
        "How does machine learning work?",
        "What is natural language processing?",
        "Who invented the telephone?"
    ]

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 50)

        results = retrieve_documents(query, store)

        if results["success"]:
            print(f"Found    : {results['total_found']} chunks")
            print(f"Time     : {results['time_taken']}s")
            print()

            for i, doc in enumerate(results["results"]):
                bar = "█" * int(doc["score"] * 30)
                print(f"Rank {i+1} | Score: {doc['score']} | {bar}")
                print(f"Source : {doc['source']}")
                print(f"Content: {doc['content'][:100]}...")
                print()
        else:
            print(f"Error: {results['message']}")