import sys
import os
import time
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import EMBEDDING_MODEL

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    Returns a value between -1 and 1.
    1 means identical, 0 means unrelated, -1 means opposite.
    """
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

def load_embedding_model():
    """Load the sentence transformer model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    print("This may take a minute the first time (downloading model)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded successfully\n")
    return model

def embed_texts(model, texts):
    """Convert a list of texts into vectors."""
    start_time = time.time()
    embeddings = model.encode(texts, show_progress_bar=False)
    elapsed = time.time() - start_time
    print(f"Embedded {len(texts)} texts in {elapsed:.2f} seconds")
    print(f"Each vector has {len(embeddings[0])} dimensions\n")
    return embeddings

def print_similarity_matrix(texts, embeddings):
    """Print similarity scores between all pairs of texts."""
    print("="*70)
    print("SIMILARITY MATRIX")
    print("="*70)
    print("Score closer to 1.0 = very similar")
    print("Score closer to 0.0 = unrelated")
    print("="*70 + "\n")

    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            score = cosine_similarity(embeddings[i], embeddings[j])
            bar = "█" * int(score * 20)
            print(f"Text {i+1} vs Text {j+1}: {score:.4f}  {bar}")
            print(f"  Text {i+1}: {texts[i][:60]}...")
            print(f"  Text {j+1}: {texts[j][:60]}...")
            print()

def run_retrieval_simulation(model, query, documents):
    """
    Simulate what RAG retrieval looks like.
    Given a query, find the most relevant document.
    """
    print("="*70)
    print("RETRIEVAL SIMULATION")
    print("="*70)
    print(f"Query: {query}\n")

    # embed query and all documents
    all_texts = [query] + documents
    all_embeddings = model.encode(all_texts, show_progress_bar=False)

    query_embedding = all_embeddings[0]
    doc_embeddings = all_embeddings[1:]

    # calculate similarity of each document to query
    scores = []
    for i, doc_emb in enumerate(doc_embeddings):
        score = cosine_similarity(query_embedding, doc_emb)
        scores.append((score, i, documents[i]))

    # sort by score highest first
    scores.sort(reverse=True)

    print("Documents ranked by relevance to query:")
    print("-" * 70)
    for rank, (score, idx, doc) in enumerate(scores):
        bar = "█" * int(score * 30)
        print(f"Rank {rank+1} | Score: {score:.4f} | {bar}")
        print(f"        {doc[:100]}")
        print()

def main():
    # load model
    model = load_embedding_model()

    # ---- TEST 1: Basic similarity ----
    print("TEST 1: Basic Similarity Between Sentences")
    print("-" * 70)

    test_sentences = [
        "The cat sat on the mat.",
        "A feline rested on the rug.",
        "The stock market crashed today.",
        "Dogs are loyal and friendly animals.",
        "Machine learning is a subset of artificial intelligence.",
        "AI systems can learn patterns from large datasets."
    ]

    print("Sentences being tested:")
    for i, s in enumerate(test_sentences):
        print(f"  Text {i+1}: {s}")
    print()

    embeddings = embed_texts(model, test_sentences)
    print_similarity_matrix(test_sentences, embeddings)

    # ---- TEST 2: RAG Retrieval Simulation ----
    print("\n\nTEST 2: RAG Retrieval Simulation")
    print("-" * 70)
    print("This simulates exactly what happens in your RAG pipeline")
    print("when a user asks a question.\n")

    # simulate a small document store
    document_chunks = [
        "RAG stands for Retrieval Augmented Generation. It combines search with language models.",
        "The Eiffel Tower is located in Paris, France and was built in 1889.",
        "Python is a popular programming language known for its simple syntax.",
        "RAG systems retrieve relevant documents before generating an answer.",
        "The Great Wall of China stretches over 13,000 miles.",
        "Vector databases store embeddings and allow fast similarity search.",
        "FastAPI is a modern Python web framework for building APIs.",
        "Embeddings are numerical representations of text in vector space.",
    ]

    # test queries
    queries = [
        "How does RAG work?",
        "Where is the Eiffel Tower?",
        "What is a vector database?",
    ]

    for query in queries:
        run_retrieval_simulation(model, query, document_chunks)
        print("\n")

    # ---- TEST 3: Embedding Speed ----
    print("TEST 3: Embedding Speed Test")
    print("-" * 70)
    print("Testing how fast embeddings are generated on your machine...\n")

    test_sizes = [10, 50, 100]
    sample_text = "This is a sample sentence for speed testing purposes."

    for size in test_sizes:
        texts = [sample_text] * size
        start = time.time()
        model.encode(texts, show_progress_bar=False)
        elapsed = time.time() - start
        print(f"{size} texts embedded in {elapsed:.2f} seconds ({elapsed/size*1000:.1f}ms per text)")

if __name__ == "__main__":
    main()