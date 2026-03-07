import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# Groq LLM config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")
LLM_TEMPERATURE  = 0.1
LLM_MAX_TOKENS   = 600

# Qdrant vector database config
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "docmind")

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Embedding model config
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2"
)
EMBEDDING_DIM = 384  # dimension for all-MiniLM-L6-v2

# Retrieval config
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", 10))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", 5))

# Chunking config
CHUNK_SIZE = 5        # sentences per chunk
CHUNK_OVERLAP = 2     # sentence overlap between chunks

# File upload config
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx"}
MAX_FILE_SIZE_MB = 50

BM25_INDEX_PATH  = "data/bm25_index.pkl"
BM25_CHUNK_SIZE  = 500

AGENT_MAX_ITERATIONS = 4
AGENT_MAX_MEMORY_TURNS = 5
SESSION_TIMEOUT_SECONDS = 3600



# validate that critical keys are present
def validate_config():
    missing = []
    if not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if not QDRANT_URL:
        missing.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")
    if not SERPER_API_KEY:
        missing.append("SERPER_API_KEY")
    if missing:
        print(f"Missing API keys: {missing}")
        print("Add them to your .env file")
    else:
        print("All API keys loaded")

    return len(missing) == 0

if __name__ == "__main__":
    print("DocMind Configuration")
    print("=" * 40)
    validate_config()
    print("All config variables loaded successfully")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Top K Retrieval: {TOP_K_RETRIEVAL}")
    print(f"Top K Rerank   : {TOP_K_RERANK}")
    print(f"Chunk Size     : {CHUNK_SIZE}")
    print(f"Max File Size  : {MAX_FILE_SIZE_MB}MB")
    print(f"Serper API Key: {SERPER_API_KEY[:8]}..." if SERPER_API_KEY else "Serper API Key: NOT FOUND")