# DocMind — Agentic RAG Research Assistant

An intelligent multi-document research agent that uses the ReAct 
(Reasoning + Acting) loop to answer questions. The agent decides 
whether to search uploaded documents, search the web, summarize 
content, or verify answers — dynamically choosing the right tool 
for each question.

## What Makes This Agentic

Unlike standard RAG systems that follow fixed steps, DocMind uses 
an AI agent that:
- Thinks about the question before acting
- Chooses from multiple tools dynamically
- Can loop multiple times before answering
- Shows its full reasoning trace to the user
- Verifies its own answers for hallucinations

## Features
- Multi-document ingestion (PDF, TXT, DOCX)
- Hybrid search (BM25 + Dense Retrieval)
- Cross-encoder reranking
- Web search when documents are insufficient
- Document summarization
- Hallucination detection via Answer Verifier
- Conversation memory across turns
- Full agent reasoning trace visible in UI
- RAGAS evaluation pipeline

## Agent Tools

- **Document Search** — searches uploaded documents via Qdrant
- **Web Search** — searches internet via Serper API
- **Summarizer** — summarizes entire documents on request
- **Answer Verifier** — checks answers against source content

## Tech Stack
| Component | Tool |
|---|---|
| Agent Framework | Haystack Agents + ReAct |
| LLM (Agent Brain) | Groq — Llama 3.3 70B |
| Embeddings | all-MiniLM-L6-v2 |
| Vector DB | Qdrant Cloud |
| Web Search | Serper API |
| Reranker | cross-encoder/ms-marco |
| Backend | FastAPI |
| Frontend | Streamlit |
| Evaluation | RAGAS |
| Deployment | Streamlit Cloud |

## Project Structure
```
docmind/
├── backend/
│   ├── main.py                    ← FastAPI app
│   ├── agents/
│   │   ├── rag_agent.py           ← Agent brain + ReAct loop
│   │   └── agent_memory.py        ← Conversation memory
│   ├── tools/
│   │   ├── document_search.py     ← Tool 1
│   │   ├── web_search.py          ← Tool 2
│   │   ├── summarizer.py          ← Tool 3
│   │   └── answer_verifier.py     ← Tool 4
│   ├── pipeline/
│   │   ├── indexing.py            ← Document ingestion
│   │   ├── retrieval.py           ← Vector retrieval
│   │   └── haystack_config.py     ← Haystack configs
│   ├── core/
│   │   ├── document_store.py      ← Qdrant connection
│   │   └── embedder.py            ← Embedding model
│   └── utils/
│       └── file_handler.py        ← File upload utilities
├── frontend/
│   └── app.py                     ← Streamlit UI
├── data/
│   └── sample_docs/               ← Test documents
├── tests/
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_llm.py
│   ├── test_qdrant.py
│   ├── test_indexing.py
│   ├── test_retrieval.py
│   └── ragas_eval.py
├── configs/
│   └── settings.py
├── .env.example
├── requirements.txt
└── README.md
```

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