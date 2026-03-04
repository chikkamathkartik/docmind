import os
from dotenv import load_dotenv

# load environment variables from .env file
load_dotenv()

# Groq LLM config
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.1-70b-versatile")

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
        raise ValueError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

if __name__ == "__main__":
    validate_config()
    print("All config variables loaded successfully")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Qdrant URL: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Serper API Key: {SERPER_API_KEY[:8]}..." if SERPER_API_KEY else "Serper API Key: NOT FOUND")