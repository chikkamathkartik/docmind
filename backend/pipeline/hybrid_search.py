import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import TOP_K_RETRIEVAL, TOP_K_RERANK


def reciprocal_rank_fusion(
    dense_results: list,
    sparse_results: list,
    k: int = 60
) -> list:
    """
    Merge two ranked lists using Reciprocal Rank Fusion.
    RRF score = sum(1 / (k + rank)) for each result across lists.

    k=60 is the standard value from the original RRF paper.
    """
    scores = {}
    doc_map = {}

    # score dense results
    for rank, doc in enumerate(dense_results):
        key = doc["content"][:100]  # use content as key
        if key not in scores:
            scores[key] = 0
            doc_map[key] = doc
        scores[key] += 1 / (k + rank + 1)

    # score sparse results
    for rank, doc in enumerate(sparse_results):
        key = doc["content"][:100]
        if key not in scores:
            scores[key] = 0
            doc_map[key] = doc
        scores[key] += 1 / (k + rank + 1)

    # sort by combined RRF score
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)

    # build final results
    final_results = []
    for key in sorted_keys:
        doc = doc_map[key].copy()
        doc["rrf_score"] = round(scores[key], 6)
        final_results.append(doc)

    return final_results


def hybrid_search(
    query: str,
    document_store,
    bm25_store,
    top_k: int = None
) -> dict:
    """
    Perform hybrid search combining dense and sparse retrieval.

    query: user's question
    document_store: Qdrant document store
    bm25_store: BM25 index store
    top_k: number of results to return

    Returns merged and ranked results.
    """
    import time

    if top_k is None:
        top_k = TOP_K_RERANK

    start_time = time.time()

    # ── Dense retrieval ──────────────────────────────
    from backend.pipeline.retrieval import retrieve_documents

    dense_result = retrieve_documents(query, document_store)
    dense_docs = dense_result.get("results", [])

    # ── Sparse retrieval (BM25) ──────────────────────
    sparse_docs = bm25_store.search(query, top_k=TOP_K_RETRIEVAL)

    # ── Merge using RRF ──────────────────────────────
    merged = reciprocal_rank_fusion(dense_docs, sparse_docs)

    # take top k
    final = merged[:top_k]
    elapsed = time.time() - start_time

    return {
        "success": True,
        "query": query,
        "results": final,
        "total_found": len(final),
        "dense_count": len(dense_docs),
        "sparse_count": len(sparse_docs),
        "time_taken": round(elapsed, 2)
    }


def format_hybrid_results_for_agent(query: str, document_store, bm25_store) -> str:
    """
    Run hybrid search and format results for the agent.
    """
    result = hybrid_search(query, document_store, bm25_store)

    if not result["success"] or not result["results"]:
        return "No relevant documents found."

    # confidence warning
    top_score = result["results"][0].get("rrf_score", 0)
    output = ""

    if top_score < 0.01:
        output += "⚠️ Low confidence — results may not be relevant.\n\n"

    output += f"Found {result['total_found']} results "
    output += f"(dense: {result['dense_count']}, "
    output += f"sparse: {result['sparse_count']}) "
    output += f"in {result['time_taken']}s:\n\n"

    for i, doc in enumerate(result["results"]):
        output += f"[Source {i+1}]\n"
        output += f"File   : {doc.get('source', 'unknown')}\n"
        output += f"Page   : {doc.get('page', 'N/A')}\n"
        output += f"Score  : {doc.get('rrf_score', 0)}\n"
        output += f"Content: {doc.get('content', '')}\n"
        output += "-" * 40 + "\n"

    return output


# test hybrid search
if __name__ == "__main__":
    from backend.core.document_store import get_document_store
    from backend.core.bm25_store import BM25Store

    print("Testing Hybrid Search")
    print("=" * 60)

    store = get_document_store()
    bm25 = BM25Store()

    print(f"Qdrant documents : {store.count_documents()}")
    print(f"BM25 documents   : {bm25.get_count()}")

    if store.count_documents() == 0:
        print("\nNo documents in store.")
        print("Run backend/pipeline/indexing.py first.")
        import sys
        sys.exit(1)

    # if BM25 is empty but Qdrant has docs
    # add sample docs to BM25 for testing
    if bm25.get_count() == 0:
        print("\nBM25 index empty — adding sample docs for test...")
        bm25.add_documents([
            {
                "content": "Agentic RAG extends standard RAG by adding autonomous agent layer",
                "source": "sample_ai_doc.txt",
                "page": 1
            },
            {
                "content": "Machine learning automatically learns from experience without explicit programming",
                "source": "sample_ai_doc.txt",
                "page": 2
            },
            {
                "content": "Natural language processing helps computers understand human language",
                "source": "sample_ai_doc.txt",
                "page": 3
            }
        ])

    queries = [
        "What is Agentic RAG?",
        "How does machine learning work?",
        "Who invented the telephone?"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 50)
        result = hybrid_search(query, store, bm25)

        print(f"Dense results  : {result['dense_count']}")
        print(f"Sparse results : {result['sparse_count']}")
        print(f"Final results  : {result['total_found']}")
        print(f"Time taken     : {result['time_taken']}s")
        print()

        for i, doc in enumerate(result["results"][:3]):
            print(f"Rank {i+1} | RRF Score: {doc['rrf_score']}")
            print(f"Source : {doc.get('source', 'unknown')}")
            print(f"Content: {doc.get('content', '')[:80]}...")
            print()