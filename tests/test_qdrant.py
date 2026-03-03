import sys
import os
import time
import uuid

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import (
    QDRANT_URL,
    QDRANT_API_KEY,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    EMBEDDING_DIM
)

# sample documents simulating chunked content
# in the real pipeline these come from actual uploaded PDFs
SAMPLE_DOCUMENTS = [
    {
        "id": str(uuid.uuid4()),
        "content": "Retrieval Augmented Generation (RAG) combines search with language models to produce accurate answers.",
        "metadata": {"source": "ai_guide.pdf", "page": 1}
    },
    {
        "id": str(uuid.uuid4()),
        "content": "Vector databases store high dimensional embeddings and support fast similarity search.",
        "metadata": {"source": "ai_guide.pdf", "page": 2}
    },
    {
        "id": str(uuid.uuid4()),
        "content": "The Eiffel Tower was built in 1889 and stands 330 meters tall in Paris, France.",
        "metadata": {"source": "landmarks.pdf", "page": 1}
    },
    {
        "id": str(uuid.uuid4()),
        "content": "Python is a high level programming language known for readability and simplicity.",
        "metadata": {"source": "programming.pdf", "page": 1}
    },
    {
        "id": str(uuid.uuid4()),
        "content": "Haystack is an open source framework for building production grade RAG pipelines.",
        "metadata": {"source": "ai_guide.pdf", "page": 3}
    },
    {
        "id": str(uuid.uuid4()),
        "content": "Qdrant is a vector database optimized for storing and searching embeddings at scale.",
        "metadata": {"source": "ai_guide.pdf", "page": 4}
    },
    {
        "id": str(uuid.uuid4()),
        "content": "Large language models like Llama and GPT are trained on massive text datasets.",
        "metadata": {"source": "ai_guide.pdf", "page": 5}
    },
    {
        "id": str(uuid.uuid4()),
        "content": "The Great Wall of China stretches over 13000 miles across northern China.",
        "metadata": {"source": "landmarks.pdf", "page": 2}
    },
    {
        "id": str(uuid.uuid4()),
        "content": "FastAPI is a modern Python web framework for building APIs with automatic documentation.",
        "metadata": {"source": "programming.pdf", "page": 2}
    },
    {
        "id": str(uuid.uuid4()),
        "content": "Sentence transformers convert text into dense vector representations capturing semantic meaning.",
        "metadata": {"source": "ai_guide.pdf", "page": 6}
    },
]

def load_embedding_model():
    """Load the sentence transformer model."""
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded\n")
    return model

def connect_to_qdrant():
    """Connect to Qdrant cloud cluster."""
    from qdrant_client import QdrantClient

    print("Connecting to Qdrant...")
    print(f"URL: {QDRANT_URL}")

    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

    # test connection by listing collections
    collections = client.get_collections()
    print(f"Connection successful")
    print(f"Existing collections: {[c.name for c in collections.collections]}\n")

    return client

def create_collection(client):
    """Create a fresh Qdrant collection for testing."""
    from qdrant_client.models import Distance, VectorParams

    # use a test collection name so we don't mess up the real one
    test_collection = f"{COLLECTION_NAME}_test"

    # delete if exists so we start fresh every test run
    existing = [c.name for c in client.get_collections().collections]
    if test_collection in existing:
        client.delete_collection(test_collection)
        print(f"Deleted existing test collection: {test_collection}")

    # create new collection
    client.create_collection(
        collection_name=test_collection,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,  # 384 for all-MiniLM-L6-v2
            distance=Distance.COSINE  # use cosine similarity
        )
    )

    print(f"Created collection: {test_collection}")
    print(f"Vector dimensions : {EMBEDDING_DIM}")
    print(f"Distance metric   : Cosine\n")

    return test_collection

def store_documents(client, collection_name, model, documents):
    """Embed all documents and store them in Qdrant."""
    from qdrant_client.models import PointStruct

    print("="*60)
    print("STORING DOCUMENTS IN QDRANT")
    print("="*60)

    # extract just the text content for embedding
    texts = [doc["content"] for doc in documents]

    # embed all documents at once
    print(f"Embedding {len(texts)} document chunks...")
    start = time.time()
    embeddings = model.encode(texts, show_progress_bar=True)
    elapsed = time.time() - start
    print(f"Embedding complete in {elapsed:.2f} seconds\n")

    # create Qdrant points
    points = []
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        point = PointStruct(
            id=i,  # simple integer ID
            vector=embedding.tolist(),
            payload={
                "content": doc["content"],
                "source": doc["metadata"]["source"],
                "page": doc["metadata"]["page"],
                "original_id": doc["id"]
            }
        )
        points.append(point)

    # upload to Qdrant
    client.upsert(
        collection_name=collection_name,
        points=points
    )

    # verify storage
    count = client.count(collection_name=collection_name)
    print(f"Successfully stored {count.count} documents in Qdrant")
    print()

def search_documents(client, collection_name, model, query, top_k=3):
    """
    Search for documents similar to the query.
    This is exactly what the retrieval pipeline does.
    """
    from qdrant_client.models import SearchRequest

    # embed the query
    query_embedding = model.encode([query])[0]

    # search Qdrant using new API
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding.tolist(),
        limit=top_k,
        with_payload=True
    )

    return results.points

def run_search_tests(client, collection_name, model):
    """Run several test queries and display results."""
    print("="*60)
    print("RETRIEVAL TESTS")
    print("="*60)
    print("Querying Qdrant with test questions...\n")

    test_queries = [
        "How does RAG work?",
        "What is a vector database?",
        "Tell me about famous landmarks",
        "What programming frameworks exist for building APIs?",
        "How are language models trained?",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 50)

        start = time.time()
        results = search_documents(client, collection_name, model, query, top_k=3)
        elapsed = time.time() - start

        for rank, result in enumerate(results):
            score = result.score
            content = result.payload["content"]
            source = result.payload["source"]
            page = result.payload["page"]

            bar = "█" * int(score * 25)
            print(f"  Rank {rank+1} | Score: {score:.4f} | {bar}")
            print(f"  Source : {source} | Page {page}")
            print(f"  Content: {content[:80]}...")
            print()

        print(f"Search time: {elapsed*1000:.0f}ms\n")

def run_full_rag_simulation(client, collection_name, model):
    """
    Simulate the complete RAG flow:
    question → retrieve chunks → build prompt → LLM answer
    This is a preview of what Week 2 onwards builds properly.
    """
    from groq import Groq
    from configs.settings import GROQ_API_KEY, LLM_MODEL

    print("="*60)
    print("FULL RAG SIMULATION")
    print("="*60)
    print("This combines retrieval + LLM into one flow\n")

    client_groq = Groq(api_key=GROQ_API_KEY)

    question = "What tools and frameworks are used for building RAG systems?"
    print(f"Question: {question}\n")

    # step 1 - retrieve relevant chunks
    print("Step 1: Retrieving relevant chunks from Qdrant...")
    results = search_documents(client, collection_name, model, question, top_k=3)

    print(f"Retrieved {len(results)} chunks:")
    for i, r in enumerate(results):
        print(f"  Chunk {i+1} (score {r.score:.3f}): {r.payload['content'][:60]}...")
    print()

    # step 2 - build context from retrieved chunks
    print("Step 2: Building context for LLM prompt...")
    context = ""
    for r in results:
        context += f"Source: {r.payload['source']} | Page: {r.payload['page']}\n"
        context += f"Content: {r.payload['content']}\n"
        context += "---\n"
    print("Context built\n")

    # step 3 - send to LLM
    print("Step 3: Sending to Groq LLM...")
    start = time.time()

    response = client_groq.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": """You are a helpful research assistant. 
Answer using ONLY the context provided. 
Always cite your sources."""
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\nQuestion: {question}"
            }
        ],
        temperature=0.1,
        max_tokens=300
    )

    elapsed = time.time() - start
    answer = response.choices[0].message.content

    print(f"\nFinal Answer:")
    print("="*60)
    print(answer)
    print("="*60)
    print(f"\nTotal RAG time: {elapsed:.2f} seconds")

def cleanup(client, collection_name):
    """Delete the test collection after we're done."""
    client.delete_collection(collection_name)
    print(f"\nCleaned up test collection: {collection_name}")

def main():
    print("\nDOCMIND - Qdrant Vector Database Tests")
    print("="*60 + "\n")

    # check config
    if not QDRANT_URL or not QDRANT_API_KEY:
        print("ERROR: Qdrant credentials not found in .env file")
        print("Go to cloud.qdrant.io and get your URL and API key")
        sys.exit(1)

    # load embedding model
    model = load_embedding_model()

    # connect to Qdrant
    client = connect_to_qdrant()

    # create test collection
    collection_name = create_collection(client)

    # store sample documents
    store_documents(client, collection_name, model, SAMPLE_DOCUMENTS)

    # run retrieval tests
    run_search_tests(client, collection_name, model)

    # run full RAG simulation
    run_full_rag_simulation(client, collection_name, model)

    # cleanup
    cleanup(client, collection_name)

    print("\nAll Qdrant tests complete")
    print("Go to cloud.qdrant.io to see your cluster dashboard")

if __name__ == "__main__":
    main()