import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    COLLECTION_NAME
)


def build_indexing_pipeline(document_store):
    from haystack import Pipeline
    from haystack.components.converters import (
        PyPDFToDocument,
        TextFileToDocument,
        DOCXToDocument
    )
    from haystack.components.preprocessors import (
        DocumentSplitter,
        DocumentCleaner
    )
    from haystack.components.embedders import (
        SentenceTransformersDocumentEmbedder
    )
    from haystack.components.writers import DocumentWriter
    from haystack.components.routers import FileTypeRouter
    from haystack.components.joiners import DocumentJoiner

    pipeline = Pipeline()

    pipeline.add_component("router", FileTypeRouter(mime_types=[
        "application/pdf",
        "text/plain",
        "application/vnd.openxmlformats-officedocument"
        ".wordprocessingml.document"
    ]))

    pipeline.add_component("pdf_converter", PyPDFToDocument())
    pipeline.add_component("txt_converter", TextFileToDocument())
    pipeline.add_component("docx_converter", DOCXToDocument())
    pipeline.add_component("joiner", DocumentJoiner())
    pipeline.add_component("cleaner", DocumentCleaner(
        remove_empty_lines=True,
        remove_extra_whitespaces=True
    ))
    pipeline.add_component("splitter", DocumentSplitter(
        split_by="sentence",
        split_length=CHUNK_SIZE,
        split_overlap=CHUNK_OVERLAP
    ))
    pipeline.add_component("embedder", SentenceTransformersDocumentEmbedder(
        model=EMBEDDING_MODEL,
        progress_bar=True
    ))
    pipeline.add_component("writer", DocumentWriter(
        document_store=document_store
    ))

    pipeline.connect("router.application/pdf", "pdf_converter.sources")
    pipeline.connect("router.text/plain", "txt_converter.sources")
    pipeline.connect(
        "router.application/vnd.openxmlformats-officedocument"
        ".wordprocessingml.document",
        "docx_converter.sources"
    )
    pipeline.connect("pdf_converter.documents", "joiner.documents")
    pipeline.connect("txt_converter.documents", "joiner.documents")
    pipeline.connect("docx_converter.documents", "joiner.documents")
    pipeline.connect("joiner.documents", "cleaner.documents")
    pipeline.connect("cleaner.documents", "splitter.documents")
    pipeline.connect("splitter.documents", "embedder.documents")
    pipeline.connect("embedder.documents", "writer.documents")

    return pipeline


def index_document(file_path: str, document_store) -> dict:
    import time
    from pathlib import Path
    from haystack.components.preprocessors import DocumentCleaner
    from pathlib import Path


    print(f"Indexing: {file_path}")
    start_time = time.time()

    if not os.path.exists(file_path):
        return {
            "success": False,
            "message": f"File not found: {file_path}",
            "chunks_created": 0
        }

    count_before = document_store.count_documents()
    pipeline = build_indexing_pipeline(document_store)

    try:
        pipeline.run({
            "router": {
            "sources": [file_path],
            "meta": [{"file_name": Path(file_path).name, "source": Path(file_path).name}]
        }
    })

        count_after = document_store.count_documents()
        chunks_created = count_after - count_before
        elapsed = time.time() - start_time

        print(f"Successfully indexed: {Path(file_path).name}")
        print(f"Chunks created      : {chunks_created}")
        print(f"Time taken          : {elapsed:.2f} seconds")

        return {
            "success": True,
            "file": Path(file_path).name,
            "chunks_created": chunks_created,
            "total_documents": count_after,
            "time_taken": round(elapsed, 2)
        }

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": str(e),
            "chunks_created": 0
        }


if __name__ == "__main__":
    from backend.core.document_store import get_document_store

    print("Testing Indexing Pipeline")
    print("=" * 50)

    store = get_document_store()
    print(f"Documents before: {store.count_documents()}")

    sample_dir = "data/sample_docs"
    test_file = os.path.join(sample_dir, "sample_ai_doc.txt")

    if not os.path.exists(test_file):
        os.makedirs(sample_dir, exist_ok=True)
        sample_text = """
Artificial Intelligence and Machine Learning

Artificial intelligence is the simulation of human intelligence
processes by computer systems including learning and reasoning.

Machine learning is a subset of AI that provides systems the ability
to automatically learn and improve from experience without being
explicitly programmed. It focuses on development of programs that
access data and use it to learn for themselves.

Deep learning uses neural networks with many layers to learn
representations of data with multiple levels of abstraction.
It has achieved remarkable results in image recognition and NLP.

Natural Language Processing is a branch of AI that helps computers
understand, interpret and manipulate human language. It draws from
computer science and computational linguistics.

Retrieval Augmented Generation or RAG is an AI framework that
retrieves facts from an external knowledge base to ground large
language models on accurate and up-to-date information.

Agentic RAG extends standard RAG by adding an autonomous agent
layer that decides which tools to use and performs multi-step
reasoning before generating a final answer.
        """ * 4
        with open(test_file, "w") as f:
            f.write(sample_text)
        print(f"Created sample file: {test_file}")

    result = index_document(test_file, store)

    print(f"\nFinal Result:")
    print(f"Success        : {result['success']}")
    print(f"Chunks created : {result.get('chunks_created', 0)}")
    print(f"Total in store : {result.get('total_documents', 0)}")
    print(f"Time taken     : {result.get('time_taken', 0)}s")