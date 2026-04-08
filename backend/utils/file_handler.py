"""
File Handler Utility
Handles file upload validation, storage, and tracking.
"""

import os
import uuid
import json
from pathlib import Path

UPLOAD_DIR = "data/uploads"
REGISTRY_PATH = "data/document_registry.json"

os.makedirs(UPLOAD_DIR, exist_ok=True)


def validate_file(filename: str, file_size_bytes: int) -> dict:
    """Validate file extension and size."""
    from configs.settings import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {
            "valid": False,
            "message": f"File type '{ext}' not supported. Use PDF, TXT, or DOCX."
        }

    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        return {
            "valid": False,
            "message": f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."
        }

    if file_size_bytes == 0:
        return {
            "valid": False,
            "message": "File is empty."
        }

    return {"valid": True, "message": "Valid"}


def save_upload(file_content: bytes, filename: str) -> dict:
    """Save uploaded file to disk with unique ID."""
    file_id = str(uuid.uuid4())[:8]
    safe_name = f"{file_id}_{filename}"
    file_path = os.path.join(UPLOAD_DIR, safe_name)

    with open(file_path, "wb") as f:
        f.write(file_content)

    return {
        "file_id": file_id,
        "file_path": file_path,
        "original_name": filename,
        "saved_name": safe_name,
        "size_bytes": len(file_content),
        "size_mb": round(len(file_content) / (1024 * 1024), 2)
    }


def register_document(file_id: str, filename: str, chunks: int):
    """Register an indexed document in the local registry."""
    registry = load_registry()
    registry[file_id] = {
        "file_id": file_id,
        "filename": filename,
        "chunks": chunks,
        "uploaded_at": __import__("time").time()
    }
    save_registry(registry)


def load_registry() -> dict:
    """Load document registry from disk."""
    if os.path.exists(REGISTRY_PATH):
        try:
            with open(REGISTRY_PATH, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_registry(registry: dict):
    """Save document registry to disk."""
    os.makedirs("data", exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def get_uploaded_files() -> list:
    """Return list of all registered documents."""
    registry = load_registry()
    return list(registry.values())


def clear_registry():
    """Clear document registry."""
    if os.path.exists(REGISTRY_PATH):
        os.remove(REGISTRY_PATH)


def delete_upload(file_path: str) -> bool:
    """Delete an uploaded file from disk."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception:
        return False