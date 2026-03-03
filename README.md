# Docmind
A production grade multi document RAG system built with Haystack, Qdrant, Groq, and RAGAS evaluation
# DocMind — Intelligent Multi Document Research Assistant

A production grade RAG (Retrieval Augmented Generation) system that lets 
users upload multiple documents and ask questions across all of them. 
Built with Haystack 2.0, Qdrant, Groq (Llama 3.1 70B), and evaluated 
with RAGAS.

## Features
- Multi document ingestion (PDF, TXT, DOCX)
- Hybrid search (BM25 + Dense Retrieval)
- Cross encoder reranking
- Conversation memory
- Confidence scores and hallucination detection
- RAGAS evaluation pipeline

## Tech Stack
- Pipeline: Haystack 2.0
- Vector DB: Qdrant Cloud
- LLM: Groq (Llama 3.1 70B) / Ollama (local)
- Embeddings: all-MiniLM-L6-v2
- Backend: FastAPI
- Frontend: Streamlit

## Project Structure
docmind/
├── backend/
│   ├── main.py
│   ├── pipeline/
│   │   ├── indexing.py
│   │   ├── retrieval.py
│   │   └── haystack_config.py
│   ├── core/
│   │   ├── document_store.py
│   │   └── embedder.py
│   └── utils/
│       └── file_handler.py
├── frontend/
│   └── app.py
├── data/sample_docs/
├── tests/
├── configs/
│   └── settings.py
├── .env.example
├── requirements.txt
└── README.md

## Setup Instructions
1. Clone the repo
   git clone https://github.com/chikkamathkartik/docmind.git
   cd docmind

2. Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Mac/Linux
   venv\Scripts\activate     # Windows

3. Install dependencies
   pip install -r requirements.txt

4. Set up environment variables
   cp .env.example .env
   # Fill in your API keys in .env

5. Run the application
   uvicorn backend.main:app --reload  # backend
   streamlit run frontend/app.py      # frontend

## Team
- Kartik — Backend & ML Pipeline
- Vikas — LLM Integration & Frontend
- Goutam — QA & Evaluation
- Riya — Report & Docs