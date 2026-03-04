"""
DocMind FastAPI Backend
All API routes for the Agentic RAG system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import time

from configs.settings import validate_config
from backend.utils.file_handler import (
    validate_file,
    save_upload,
    delete_upload,
    get_uploaded_files
)
from backend.core.document_store import (
    get_document_store,
    get_document_count,
    clear_document_store
)
from backend.pipeline.indexing import index_document
from backend.agents.rag_agent import DocMindAgent

# validate config on startup
validate_config()

# initialize FastAPI app
app = FastAPI(
    title="DocMind API",
    description="Agentic RAG Research Assistant API",
    version="1.0.0"
)

# allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# initialize shared resources
print("Initializing DocMind...")
document_store = get_document_store()
agent = DocMindAgent()
print("DocMind ready")


# ─────────────────────────────────────────
# REQUEST MODELS
# ─────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str
    session_id: Optional[str] = "default"


class DeleteDocumentRequest(BaseModel):
    file_id: str


# ─────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "app": "DocMind Agentic RAG",
        "version": "1.0.0",
        "documents_indexed": get_document_count(document_store)
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy",
        "document_store": "connected",
        "documents_indexed": get_document_count(document_store),
        "agent": "ready",
        "timestamp": time.time()
    }


# ─────────────────────────────────────────
# DOCUMENT ROUTES
# ─────────────────────────────────────────

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and index a document.
    Accepts PDF, TXT, DOCX files.
    """
    try:
        # read file content
        content = await file.read()

        # validate file
        validation = validate_file(file.filename, len(content))
        if not validation["valid"]:
            raise HTTPException(
                status_code=400,
                detail=validation["message"]
            )

        # save to disk
        saved = save_upload(content, file.filename)

        # index into Qdrant
        result = index_document(saved["file_path"], document_store)

        if not result["success"]:
            # clean up saved file if indexing failed
            delete_upload(saved["file_path"])
            raise HTTPException(
                status_code=500,
                detail=f"Indexing failed: {result['message']}"
            )

        return {
            "success": True,
            "message": f"Successfully uploaded and indexed {file.filename}",
            "file_id": saved["file_id"],
            "filename": file.filename,
            "chunks_created": result["chunks_created"],
            "total_documents": result["total_documents"],
            "time_taken": result["time_taken"]
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all uploaded documents."""
    files = get_uploaded_files()
    return {
        "success": True,
        "documents": files,
        "total_files": len(files),
        "total_chunks": get_document_count(document_store)
    }


@app.delete("/documents")
async def clear_documents():
    """Clear all uploaded documents and reset vector store."""
    try:
        # clear vector store
        clear_document_store(document_store)

        # clear uploaded files
        import shutil
        upload_dir = "data/uploads"
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
            os.makedirs(upload_dir)

        return {
            "success": True,
            "message": "All documents cleared successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────
# AGENT ROUTES
# ─────────────────────────────────────────

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    """
    Ask a question to the agent.
    Returns answer with reasoning trace and sources.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    try:
        # run agent
        result = agent.run(request.question)

        return {
            "success": True,
            "question": result["question"],
            "answer": result["answer"],
            "reasoning_trace": result["reasoning_trace"],
            "iterations": result["iterations"],
            "time_taken": result["time_taken"],
            "sources": _extract_sources(result["reasoning_trace"])
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-memory")
async def clear_memory():
    """Clear agent conversation memory."""
    agent.clear_memory()
    return {
        "success": True,
        "message": "Conversation memory cleared"
    }


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def _extract_sources(reasoning_trace: list) -> list:
    """
    Extract source documents from the reasoning trace.
    Returns a clean list of sources used.
    """
    sources = []
    for step in reasoning_trace:
        if step.get("type") == "observation":
            content = step.get("content", "")
            # extract source lines from tool output
            for line in content.split("\n"):
                if "File:" in line or "Source:" in line:
                    source = line.replace("File:", "").replace(
                        "Source:", ""
                    ).strip()
                    if source and source not in sources:
                        sources.append(source)
    return sources


# ─────────────────────────────────────────
# RUN SERVER
# ─────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )