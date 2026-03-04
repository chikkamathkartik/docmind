"""
File Handler Utility
Handles file upload validation and temporary storage.
"""

import sys
import os
import shutil
import uuid
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB


def validate_file(filename: str, file_size_bytes: int) -> dict:
    """
    Validate uploaded file extension and size.
    Returns dict with success status and message.
    """
    # check extension
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {
            "valid": False,
            "message": f"File type {ext} not allowed. Allowed types: {ALLOWED_EXTENSIONS}"
        }

    # check size
    max_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    if file_size_bytes > max_bytes:
        return {
            "valid": False,
            "message": f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
        }

    return {"valid": True, "message": "File is valid"}


def save_upload(file_content: bytes, filename: str) -> dict:
    """
    Save uploaded file to data/uploads directory.
    Returns the saved file path and a unique file ID.
    """
    # create uploads directory if it doesn't exist
    upload_dir = "data/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    # generate unique ID for this file
    file_id = str(uuid.uuid4())[:8]
    safe_filename = f"{file_id}_{filename}"
    file_path = os.path.join(upload_dir, safe_filename)

    # save file
    with open(file_path, "wb") as f:
        f.write(file_content)

    return {
        "file_id": file_id,
        "file_path": file_path,
        "original_name": filename,
        "saved_name": safe_filename,
        "size_bytes": len(file_content)
    }


def delete_upload(file_path: str) -> bool:
    """Delete an uploaded file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False
    except Exception:
        return False


def get_uploaded_files() -> list:
    """List all uploaded files."""
    upload_dir = "data/uploads"
    if not os.path.exists(upload_dir):
        return []

    files = []
    for filename in os.listdir(upload_dir):
        filepath = os.path.join(upload_dir, filename)
        files.append({
            "filename": filename,
            "filepath": filepath,
            "size_bytes": os.path.getsize(filepath),
            "size_mb": round(os.path.getsize(filepath) / (1024*1024), 2)
        })

    return files


# create uploads directory on import
os.makedirs("data/uploads", exist_ok=True)