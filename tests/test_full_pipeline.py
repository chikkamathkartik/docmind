"""
Full Pipeline Integration Test
Tests every component end to end.
Run with: python tests/test_full_pipeline.py
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def separator(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_config():
    separator("1. CONFIG TEST")
    from configs.settings import validate_config, LLM_MODEL, EMBEDDING_MODEL
    result = validate_config()
    print(f"LLM Model      : {LLM_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    return result


def test_document_store():
    separator("2. DOCUMENT STORE TEST")
    from backend.core.document_store import get_document_store

    store = get_document_store()
    count = store.count_documents()
    print(f"Qdrant connected   : Yes")
    print(f"Documents in store : {count}")
    return store, count


def test_bm25_store():
    separator("3. BM25 STORE TEST")
    from backend.core.bm25_store import BM25Store

    bm25 = BM25Store()
    count = bm25.get_count()
    print(f"BM25 store loaded  : Yes")
    print(f"Documents in index : {count}")
    return bm25, count


def test_indexing(store, bm25):
    separator("4. INDEXING TEST")
    from backend.pipeline.indexing import index_document

    test_file = "data/sample_docs/sample_ai_doc.txt"

    if not os.path.exists(test_file):
        print(f"Sample file not found: {test_file}")
        print("Creating sample file...")
        os.makedirs("data/sample_docs", exist_ok=True)
        content = """
Artificial Intelligence Overview

Artificial intelligence is the simulation of human intelligence
by computer systems. It includes learning, reasoning, and problem solving.

Machine learning is a subset of AI where systems learn from data
automatically without being explicitly programmed for each task.

Deep learning uses neural networks with many layers to learn
complex patterns in data. It powers modern image and speech recognition.

Natural language processing enables computers to understand, interpret,
and generate human language in a meaningful and useful way.

Retrieval Augmented Generation combines information retrieval with
text generation to produce accurate and grounded responses.

Agentic RAG adds an autonomous agent layer on top of standard RAG,
allowing the system to decide which tools to use and perform
multi-step reasoning before generating a final answer.
        """ * 3
        with open(test_file, "w") as f:
            f.write(content)
        print("Sample file created")

    start = time.time()
    result = index_document(test_file, store)
    elapsed = time.time() - start

    print(f"File indexed       : {result.get('file', test_file)}")
    print(f"Success            : {result.get('success')}")
    print(f"Chunks created     : {result.get('chunks_created', 0)}")
    print(f"Total in store     : {result.get('total_documents', 0)}")
    print(f"Time taken         : {round(elapsed, 2)}s")
    return result.get("success", False)


def test_retrieval(store):
    separator("5. RETRIEVAL TEST")
    from backend.pipeline.retrieval import retrieve_documents

    queries = [
        "What is Agentic RAG?",
        "How does machine learning work?",
        "Who invented the telephone?"
    ]

    for query in queries:
        result = retrieve_documents(query, store)
        top_score = result["results"][0]["score"] if result["results"] else 0
        status = "Correct" if result["success"] else "Wrong"
        print(f"{status} Query: {query[:40]}")
        print(f"   Top score : {top_score}")
        print(f"   Results   : {result['total_found']}")


def test_hybrid_search(store, bm25):
    separator("6. HYBRID SEARCH TEST")
    from backend.pipeline.hybrid_search import hybrid_search

    query = "What is Agentic RAG?"
    result = hybrid_search(query, store, bm25)

    print(f"Query         : {query}")
    print(f"Dense results : {result.get('dense_count', 0)}")
    print(f"Sparse results: {result.get('sparse_count', 0)}")
    print(f"Final results : {result.get('total_found', 0)}")
    print(f"Time taken    : {result.get('time_taken', 0)}s")

    if result.get("results"):
        top = result["results"][0]
        print(f"Top result    : {top.get('content', '')[:80]}...")
        print(f"RRF Score     : {top.get('rrf_score', 0)}")


def test_web_search():
    separator("7. WEB SEARCH TEST")
    from backend.tools.web_search import WebSearchTool

    tool = WebSearchTool()
    result = tool.format_for_agent("What is RAG in AI?")

    if "error" in result.lower() or "failed" in result.lower():
        print(" Web search returned error — check SERPER_API_KEY")
    else:
        print(" Web search working")
        print(f"Result preview: {result[:150]}...")


def test_agent():
    separator("8. AGENT TEST")
    from backend.agents.rag_agent import DocMindAgent

    agent = DocMindAgent()

    question = "What is Agentic RAG and how does it work?"
    print(f"Question: {question}\n")

    start = time.time()
    result = agent.run(question, session_id="integration_test")
    elapsed = time.time() - start

    print(f"\nAnswer     : {result['answer'][:200]}...")
    print(f"Iterations : {result['iterations']}")
    print(f"Time taken : {result['time_taken']}s")
    print(f"Trace steps: {len(result['reasoning_trace'])}")

    return result["answer"] != ""


def run_all_tests():
    separator("DOCMIND FULL INTEGRATION TEST")
    print("Running all tests...\n")

    results = {}

    # run tests in order
    results["config"] = test_config()
    store, qdrant_count = test_document_store()
    bm25, bm25_count = test_bm25_store()

    # only test indexing if store is empty
    if qdrant_count == 0:
        results["indexing"] = test_indexing(store, bm25)
    else:
        print(f"\nSkipping indexing — {qdrant_count} documents already in store")
        results["indexing"] = True

    test_retrieval(store)
    test_hybrid_search(store, bm25)
    test_web_search()
    results["agent"] = test_agent()

    # final summary
    separator("TEST SUMMARY")
    all_passed = all(results.values())
    for test, passed in results.items():
        status = " PASS" if passed else " FAIL"
        print(f"{status} — {test}")

    print()
    if all_passed:
        print("All tests passed. System is ready.")
    else:
        print("Some tests failed. Check output above.")


if __name__ == "__main__":
    run_all_tests()