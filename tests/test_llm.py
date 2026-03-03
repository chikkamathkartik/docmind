"""
Day 3 - Script 1: Test Groq LLM Connection
Run this with: python tests/test_llm.py
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.settings import GROQ_API_KEY, LLM_MODEL

def test_basic_connection():
    """Test that Groq API is reachable and responding."""
    from groq import Groq

    print("="*60)
    print("TEST 1: Basic Groq Connection")
    print("="*60)

    client = Groq(api_key=GROQ_API_KEY)

    print(f"Model: {LLM_MODEL}")
    print("Sending test message...\n")

    start = time.time()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": "Say exactly this and nothing else: Groq connection successful."
            }
        ],
        temperature=0,
        max_tokens=20
    )

    elapsed = time.time() - start

    answer = response.choices[0].message.content
    print(f"Response    : {answer}")
    print(f"Time taken  : {elapsed:.2f} seconds")
    print(f"Tokens used : {response.usage.total_tokens}")
    print()

def test_rag_style_prompt():
    """
    Test the exact prompt format we will use in our RAG pipeline.
    This is the most important test — we want to make sure the LLM
    follows our instructions and only answers from context.
    """
    from groq import Groq

    print("="*60)
    print("TEST 2: RAG Style Prompt")
    print("="*60)

    client = Groq(api_key=GROQ_API_KEY)

    # this is the exact prompt template we will use in the real pipeline
    system_prompt = """You are an expert research assistant. 
Answer questions using ONLY the context provided below.
If the answer is not found in the context, say exactly:
'I cannot find this information in the provided documents.'
Always mention which source your answer comes from."""

    # simulate retrieved chunks from our vector database
    fake_context = """
Source: ai_textbook.pdf | Page: 3
Content: Retrieval Augmented Generation (RAG) is a technique that 
combines information retrieval with text generation. It first retrieves 
relevant documents from a knowledge base, then uses those documents as 
context for generating an answer.
---
Source: ai_textbook.pdf | Page: 5  
Content: The main advantage of RAG over pure language models is that 
it can access up-to-date information and cite its sources, reducing 
hallucination significantly.
---
Source: ml_guide.pdf | Page: 12
Content: Vector databases like Qdrant and ChromaDB store embeddings 
and allow fast similarity search across millions of documents.
"""

    # test 3 different questions
    questions = [
        "What is RAG and how does it work?",
        "What are the advantages of using RAG?",
        "Who invented the telephone?",  # this answer is NOT in context
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 50)

        start = time.time()

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{fake_context}\n\nQuestion: {question}"}
            ],
            temperature=0.1,
            max_tokens=300
        )

        elapsed = time.time() - start
        answer = response.choices[0].message.content

        print(f"Answer: {answer}")
        print(f"Time  : {elapsed:.2f} seconds")
        print()

def test_hallucination_guard():
    """
    Test what happens when we ask something not in the context.
    The LLM should refuse to answer instead of making something up.
    This is critical for our hallucination detection feature.
    """
    from groq import Groq

    print("="*60)
    print("TEST 3: Hallucination Guard")
    print("="*60)
    print("Asking questions that are NOT in the context.")
    print("A well-behaved LLM should refuse to answer.\n")

    client = Groq(api_key=GROQ_API_KEY)

    system_prompt = """You are a helpful assistant. Answer ONLY from 
the context provided. If the answer is not in the context, respond with:
'I cannot find this information in the provided documents.'
Do NOT use your general knowledge."""

    # very limited context about only one topic
    context = """
Source: company_policy.pdf | Page: 1
Content: Our company was founded in 2010. We have 500 employees.
Our main office is in Bangalore, India.
"""

    # questions - some answerable, some not
    test_cases = [
        ("When was the company founded?", "SHOULD answer - in context"),
        ("How many employees does the company have?", "SHOULD answer - in context"),
        ("What is the capital of France?", "SHOULD REFUSE - not in context"),
        ("Who is the CEO?", "SHOULD REFUSE - not in context"),
    ]

    for question, expectation in test_cases:
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0,
            max_tokens=150
        )

        answer = response.choices[0].message.content
        print(f"Question    : {question}")
        print(f"Expectation : {expectation}")
        print(f"LLM Answer  : {answer[:100]}")
        print()

def main():
    print("\nDOCMIND - Groq LLM Tests")
    print("="*60 + "\n")

    # check API key exists
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY not found in .env file")
        print("Go to console.groq.com and get your API key")
        sys.exit(1)

    print(f"API Key found: {GROQ_API_KEY[:8]}...")
    print()

    test_basic_connection()
    test_rag_style_prompt()
    test_hallucination_guard()

    print("="*60)
    print("All LLM tests complete")
    print("="*60)

if __name__ == "__main__":
    main()