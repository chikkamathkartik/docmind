import sys
import os

# add project root to path so we can import configs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import CHUNK_SIZE, CHUNK_OVERLAP

def load_txt_file(filepath):
    """Load a plain text file and return its content."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    return content

def load_pdf_file(filepath):
    """Load a PDF file and return its text content."""
    import fitz  # this is PyMuPDF
    doc = fitz.open(filepath)
    full_text = ""
    for page_num, page in enumerate(doc):
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    doc.close()
    return full_text

def chunk_text(text, chunk_size=500, overlap=50):
    """
    Split text into overlapping chunks.
    chunk_size: number of characters per chunk
    overlap: number of characters to overlap between chunks
    """
    chunks = []
    start = 0
    chunk_index = 0

    while start < len(text):
        # get chunk of specified size
        end = start + chunk_size

        # if not at end of text, try to break at a sentence boundary
        if end < len(text):
            # look for the last period, question mark, or newline before the end
            last_period = text.rfind(".", start, end)
            last_question = text.rfind("?", start, end)
            last_newline = text.rfind("\n", start, end)

            # use the latest natural break point we can find
            break_point = max(last_period, last_question, last_newline)

            if break_point > start + (chunk_size // 2):
                # only use break point if it's at least halfway through the chunk
                end = break_point + 1

        chunk_text = text[start:end].strip()

        if chunk_text:  # only add non-empty chunks
            chunks.append({
                "index": chunk_index,
                "content": chunk_text,
                "char_count": len(chunk_text),
                "start_pos": start,
                "end_pos": end
            })
            chunk_index += 1

        # move start forward, minus overlap
        start = end - overlap

    return chunks

def analyze_chunks(chunks):
    """Print statistics about the chunks."""
    total_chars = sum(c["char_count"] for c in chunks)
    avg_chars = total_chars / len(chunks) if chunks else 0
    min_chars = min(c["char_count"] for c in chunks) if chunks else 0
    max_chars = max(c["char_count"] for c in chunks) if chunks else 0

    print("\n" + "="*60)
    print("CHUNKING STATISTICS")
    print("="*60)
    print(f"Total chunks created : {len(chunks)}")
    print(f"Total characters     : {total_chars}")
    print(f"Average chunk size   : {avg_chars:.0f} chars")
    print(f"Smallest chunk       : {min_chars} chars")
    print(f"Largest chunk        : {max_chars} chars")
    print("="*60)

def print_chunks(chunks, show_first_n=5):
    """Print the first N chunks so we can visually inspect them."""
    print(f"\nShowing first {show_first_n} chunks:\n")
    for chunk in chunks[:show_first_n]:
        print(f"--- Chunk {chunk['index']} ---")
        print(f"Characters: {chunk['char_count']}")
        print(f"Content preview: {chunk['content'][:150]}...")
        print()

def main():
    # find a test file
    sample_dir = "data/sample_docs"
    test_files = []

    if os.path.exists(sample_dir):
        for f in os.listdir(sample_dir):
            if f.endswith(".txt") or f.endswith(".pdf"):
                test_files.append(os.path.join(sample_dir, f))

    if not test_files:
        print("No test files found in data/sample_docs/")
        print("Creating a sample text file for testing...")

        # create a sample text file automatically
        os.makedirs(sample_dir, exist_ok=True)
        sample_text = """
        Artificial Intelligence (AI) is intelligence demonstrated by machines, 
        as opposed to the natural intelligence displayed by animals including humans. 
        AI research has been defined as the field of study of intelligent agents, 
        which refers to any system that perceives its environment and takes actions 
        that maximize its chance of achieving its goals.

        Machine learning is a method of data analysis that automates analytical 
        model building. It is based on the idea that systems can learn from data, 
        identify patterns and make decisions with minimal human intervention.
        Machine learning algorithms include supervised learning, unsupervised learning,
        and reinforcement learning.

        Natural Language Processing (NLP) is a subfield of linguistics, computer 
        science, and artificial intelligence concerned with the interactions between 
        computers and human language, in particular how to program computers to 
        process and analyze large amounts of natural language data. The goal is a 
        computer capable of understanding the contents of documents, including the 
        contextual nuances of the language within them.

        Deep learning is part of a broader family of machine learning methods based 
        on artificial neural networks with representation learning. Learning can be 
        supervised, semi-supervised or unsupervised. Deep learning architectures such 
        as deep neural networks, recurrent neural networks, convolutional neural 
        networks and transformers have been applied to fields including computer 
        vision, speech recognition, natural language processing, and more.

        Retrieval Augmented Generation (RAG) is an AI framework for retrieving facts 
        from an external knowledge base to ground large language models on the most 
        accurate, up-to-date information and to give users insight into the generative 
        process. RAG improves large language model performance by allowing them to 
        use information that was not in their training data.
        """ * 3  # repeat 3 times to make it longer

        with open(f"{sample_dir}/sample_ai_article.txt", "w") as f:
            f.write(sample_text)

        test_files = [f"{sample_dir}/sample_ai_article.txt"]
        print(f"Created sample file: {test_files[0]}\n")

    # process each test file
    for filepath in test_files:
        print(f"\nProcessing: {filepath}")
        print("-" * 60)

        # load the file
        if filepath.endswith(".pdf"):
            text = load_pdf_file(filepath)
            print(f"Loaded PDF - {len(text)} total characters")
        else:
            text = load_txt_file(filepath)
            print(f"Loaded TXT - {len(text)} total characters")

        # chunk with different sizes to compare
        print("\nTesting chunk size 300 characters:")
        chunks_300 = chunk_text(text, chunk_size=300, overlap=30)
        analyze_chunks(chunks_300)

        print("\nTesting chunk size 500 characters:")
        chunks_500 = chunk_text(text, chunk_size=500, overlap=50)
        analyze_chunks(chunks_500)

        print("\nTesting chunk size 1000 characters:")
        chunks_1000 = chunk_text(text, chunk_size=1000, overlap=100)
        analyze_chunks(chunks_1000)

        # show actual chunk content for 500 char size
        print("\nActual chunk content preview (500 char chunks):")
        print_chunks(chunks_500, show_first_n=3)

if __name__ == "__main__":
    main()