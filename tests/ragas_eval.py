"""
RAGAS Evaluation Script
Measures RAG system quality with standard metrics.

Metrics:
- Faithfulness     : Is the answer grounded in retrieved context?
- Answer Relevancy : Does the answer actually address the question?
- Context Precision: Are the retrieved chunks relevant to the question?
- Context Recall   : Are all relevant chunks being retrieved?

Run with: python tests/ragas_eval.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time


# ─────────────────────────────────────────
# TEST DATASET
# ─────────────────────────────────────────

# These are question-answer pairs for the sample AI document
# Add more pairs as you test with real documents
TEST_DATASET = [
    {
        "question": "What is Agentic RAG?",
        "ground_truth": (
            "Agentic RAG extends standard RAG by adding an autonomous "
            "agent layer that decides which tools to use and performs "
            "multi-step reasoning before generating a final answer."
        )
    },
    {
        "question": "What is machine learning?",
        "ground_truth": (
            "Machine learning is a subset of AI that provides systems "
            "the ability to automatically learn and improve from "
            "experience without being explicitly programmed."
        )
    },
    {
        "question": "What is natural language processing?",
        "ground_truth": (
            "Natural language processing is a branch of AI that helps "
            "computers understand, interpret and manipulate human language."
        )
    },
    {
        "question": "What is deep learning?",
        "ground_truth": (
            "Deep learning uses neural networks with many layers to "
            "learn representations of data with multiple levels of "
            "abstraction."
        )
    },
    {
        "question": "What is retrieval augmented generation?",
        "ground_truth": (
            "RAG is an AI framework that retrieves facts from an external "
            "knowledge base to ground large language models on accurate "
            "and up-to-date information."
        )
    }
]


# ─────────────────────────────────────────
# MANUAL EVALUATION (No API needed)
# ─────────────────────────────────────────

def calculate_word_overlap(text1: str, text2: str) -> float:
    """Simple word overlap score between two texts."""
    stop_words = {
        "the", "a", "an", "is", "it", "in", "of", "to", "and",
        "or", "that", "this", "was", "are", "for", "with", "as",
        "be", "has", "have", "by", "on", "at", "from", "which"
    }

    words1 = set(
        w.lower().strip(".,?!;:")
        for w in text1.split()
        if len(w) > 3 and w.lower() not in stop_words
    )
    words2 = set(
        w.lower().strip(".,?!;:")
        for w in text2.split()
        if len(w) > 3 and w.lower() not in stop_words
    )

    if not words1 or not words2:
        return 0.0

    intersection = words1.intersection(words2)
    union = words1.union(words2)
    return round(len(intersection) / len(union), 4)


def evaluate_pipeline():
    """
    Run evaluation on the test dataset.
    Measures retrieval quality and answer quality.
    """
    from backend.core.document_store import get_document_store
    from backend.core.bm25_store import BM25Store
    from backend.pipeline.hybrid_search import hybrid_search
    from backend.agents.rag_agent import DocMindAgent

    print("RAGAS-Style Evaluation")
    print("=" * 60)
    print(f"Test questions: {len(TEST_DATASET)}")
    print()

    store = get_document_store()
    bm25 = BM25Store()
    agent = DocMindAgent()

    if store.count_documents() == 0:
        print("No documents in store. Run indexing first.")
        return

    results = []

    for i, test_case in enumerate(TEST_DATASET):
        question = test_case["question"]
        ground_truth = test_case["ground_truth"]

        print(f"Q{i+1}: {question}")
        print("-" * 50)

        start = time.time()

        # get retrieval results
        retrieval_result = hybrid_search(question, store, bm25)
        retrieved_chunks = retrieval_result.get("results", [])

        # get agent answer
        agent_result = agent.run(question, session_id="eval")
        answer = agent_result.get("answer", "")
        confidence = agent_result.get("confidence", {})

        elapsed = time.time() - start

        # calculate metrics

        # 1. context precision — are retrieved chunks relevant?
        context_precision = 0.0
        if retrieved_chunks:
            chunk_scores = []
            for chunk in retrieved_chunks[:3]:
                overlap = calculate_word_overlap(
                    chunk.get("content", ""),
                    question + " " + ground_truth
                )
                chunk_scores.append(overlap)
            context_precision = sum(chunk_scores) / len(chunk_scores)

        # 2. faithfulness — word overlap between answer and retrieved chunks
        context_text = " ".join([
            chunk.get("content", "")
            for chunk in retrieved_chunks[:3]
        ])
        faithfulness = calculate_word_overlap(answer, context_text)

        # 3. answer relevancy — does answer match ground truth?
        answer_relevancy = calculate_word_overlap(answer, ground_truth)

        # 4. recall@3 — does ground truth appear in top 3 chunks?
        recall_at_3 = 0.0
        for chunk in retrieved_chunks[:3]:
            overlap = calculate_word_overlap(
                chunk.get("content", ""),
                ground_truth
            )
            if overlap > 0.2:
                recall_at_3 = 1.0
                break

        result = {
            "question": question,
            "answer": answer[:100],
            "context_precision": round(context_precision, 4),
            "faithfulness": round(faithfulness, 4),
            "answer_relevancy": round(answer_relevancy, 4),
            "recall_at_3": recall_at_3,
            "confidence_label": confidence.get("label", "Unknown"),
            "confidence_pct": confidence.get("percentage", 0),
            "time_taken": round(elapsed, 2)
        }

        results.append(result)

        print(f"Answer         : {answer[:80]}...")
        print(f"Context Prec.  : {context_precision:.2%}")
        print(f"Faithfulness   : {faithfulness:.2%}")
        print(f"Answer Relev.  : {answer_relevancy:.2%}")
        print(f"Recall@3       : {'✅' if recall_at_3 else '❌'}")
        print(f"Confidence     : {confidence.get('label')} "
              f"({confidence.get('percentage')}%)")
        print(f"Time           : {elapsed:.2f}s")
        print()

    # summary
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    avg_precision = sum(r["context_precision"] for r in results) / len(results)
    avg_faithfulness = sum(r["faithfulness"] for r in results) / len(results)
    avg_relevancy = sum(r["answer_relevancy"] for r in results) / len(results)
    recall_at_3_score = sum(r["recall_at_3"] for r in results) / len(results)
    avg_time = sum(r["time_taken"] for r in results) / len(results)

    print(f"Context Precision  : {avg_precision:.2%}")
    print(f"Faithfulness       : {avg_faithfulness:.2%}")
    print(f"Answer Relevancy   : {avg_relevancy:.2%}")
    print(f"Recall@3           : {recall_at_3_score:.2%}")
    print(f"Avg Response Time  : {avg_time:.2f}s")
    print()

    # save results to file
    output_path = "tests/eval_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "summary": {
                "context_precision": round(avg_precision, 4),
                "faithfulness": round(avg_faithfulness, 4),
                "answer_relevancy": round(avg_relevancy, 4),
                "recall_at_3": round(recall_at_3_score, 4),
                "avg_response_time": round(avg_time, 2),
                "total_questions": len(TEST_DATASET)
            },
            "individual_results": results
        }, f, indent=2)

    print(f"Results saved to: {output_path}")
    print()

    # grade the system
    if recall_at_3_score >= 0.7 and avg_faithfulness >= 0.5:
        print(" System performance: GOOD — ready for report")
    elif recall_at_3_score >= 0.5:
        print("  System performance: ACCEPTABLE — consider improvements")
    else:
        print(" System performance: NEEDS IMPROVEMENT")

    return results


if __name__ == "__main__":
    evaluate_pipeline()