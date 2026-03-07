import sys
import os
import pickle
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class BM25Store:
    """
    Maintains a BM25 index of all indexed documents.
    Saved to disk so it persists between sessions.
    """

    def __init__(self, index_path: str = "data/bm25_index.pkl"):
        self.index_path = index_path
        self.documents = []  # list of document dicts
        self.bm25_index = None
        self._load_or_create()

    def _load_or_create(self):
        """Load existing index or create fresh one."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "rb") as f:
                    data = pickle.load(f)
                    self.documents = data.get("documents", [])
                    self.bm25_index = data.get("bm25_index", None)
                print(f"BM25 index loaded: {len(self.documents)} documents")
            except Exception:
                print("Could not load BM25 index, creating fresh")
                self.documents = []
                self.bm25_index = None
        else:
            print("Creating fresh BM25 index")
            self.documents = []
            self.bm25_index = None

    def _save(self):
        """Save index to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "bm25_index": self.bm25_index
            }, f)

    def _rebuild_index(self):
        """Rebuild BM25 index from stored documents."""
        from rank_bm25 import BM25Okapi

        if not self.documents:
            self.bm25_index = None
            return

        # tokenize all documents
        tokenized = [
            doc["content"].lower().split()
            for doc in self.documents
        ]

        self.bm25_index = BM25Okapi(tokenized)

    def add_documents(self, documents: list):
        """
        Add documents to the BM25 index.
        documents: list of dicts with content and metadata
        """
        self.documents.extend(documents)
        self._rebuild_index()
        self._save()
        print(f"BM25 index updated: {len(self.documents)} documents")

    def search(self, query: str, top_k: int = 10) -> list:
        """
        Search using BM25 keyword matching.
        Returns top_k most relevant documents.
        """
        if not self.bm25_index or not self.documents:
            return []

        # tokenize query
        tokenized_query = query.lower().split()

        # get BM25 scores
        scores = self.bm25_index.get_scores(tokenized_query)

        # get top k indices
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # only include docs with positive score
                results.append({
                    "content": self.documents[idx]["content"],
                    "source": self.documents[idx].get("source", "unknown"),
                    "page": self.documents[idx].get("page", "N/A"),
                    "bm25_score": float(scores[idx])
                })

        return results

    def clear(self):
        """Clear all documents from the index."""
        self.documents = []
        self.bm25_index = None
        if os.path.exists(self.index_path):
            os.remove(self.index_path)
        print("BM25 index cleared")

    def get_count(self) -> int:
        """Return number of documents in index."""
        return len(self.documents)


# test
if __name__ == "__main__":
    store = BM25Store()
    print(f"Documents in BM25 index: {store.get_count()}")

    # add test documents
    test_docs = [
        {
            "content": "Agentic RAG uses autonomous agents to decide which tools to use",
            "source": "test.txt",
            "page": 1
        },
        {
            "content": "Machine learning is a subset of artificial intelligence",
            "source": "test.txt",
            "page": 2
        },
        {
            "content": "The Eiffel Tower is located in Paris France",
            "source": "test.txt",
            "page": 3
        }
    ]

    store.add_documents(test_docs)

    # test search
    queries = [
        "What is Agentic RAG?",
        "machine learning AI",
        "Eiffel Tower location"
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        results = store.search(query, top_k=2)
        for r in results:
            print(f"  Score: {r['bm25_score']:.4f} | {r['content'][:60]}")