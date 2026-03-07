"""
Confidence Scorer
Calculates how confident the system is in each answer.
Based on retrieval scores and answer-context overlap.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class ConfidenceScorer:
    """
    Calculates confidence scores for RAG answers.

    Three signals are combined:
    1. Retrieval score  — how similar are retrieved chunks to query
    2. Coverage score   — how much of the answer is grounded in context
    3. Source agreement — do multiple sources agree on the answer
    """

    def __init__(self):
        self.weights = {
            "retrieval": 0.5,
            "coverage": 0.35,
            "source_agreement": 0.15
        }

    def calculate_retrieval_score(self, search_results: list) -> float:
        """
        Score based on how relevant retrieved chunks are.
        Uses top-3 results to avoid noise from low ranked chunks.
        """
        if not search_results:
            return 0.0

        # get top 3 scores
        top_results = search_results[:3]

        scores = []
        for result in top_results:
            # handle both rrf_score and regular score
            score = result.get("rrf_score",
                    result.get("score", 0))
            scores.append(float(score))

        if not scores:
            return 0.0

        # weighted average — top result matters most
        if len(scores) == 1:
            return min(scores[0] * 10, 1.0)  # normalize RRF scores
        elif len(scores) == 2:
            weighted = (scores[0] * 0.6 + scores[1] * 0.4)
        else:
            weighted = (
                scores[0] * 0.5 +
                scores[1] * 0.3 +
                scores[2] * 0.2
            )

        # normalize — RRF scores are small (0.01-0.03), regular scores 0-1
        max_score = max(scores)
        if max_score < 0.1:
            # RRF scores — normalize to 0-1
            return min(weighted * 30, 1.0)
        else:
            # regular cosine scores — already 0-1
            return min(weighted, 1.0)

    def calculate_coverage_score(
        self,
        answer: str,
        context_chunks: list
    ) -> float:
        """
        Score based on how much of the answer is grounded in context.
        Simple word overlap — checks if answer words appear in context.
        """
        if not answer or not context_chunks:
            return 0.0

        # build full context string
        full_context = " ".join([
            chunk.get("content", "") for chunk in context_chunks
        ]).lower()

        if not full_context:
            return 0.0

        # tokenize answer into meaningful words
        stop_words = {
            "the", "a", "an", "is", "it", "in", "of", "to",
            "and", "or", "that", "this", "was", "are", "for",
            "with", "as", "be", "has", "have", "by", "on", "at",
            "from", "which", "but", "not", "they", "their", "can"
        }

        answer_words = [
            w.lower().strip(".,?!;:")
            for w in answer.split()
            if len(w) > 3 and w.lower() not in stop_words
        ]

        if not answer_words:
            return 0.5

        # count how many answer words appear in context
        found = sum(1 for w in answer_words if w in full_context)
        coverage = found / len(answer_words)

        return round(min(coverage, 1.0), 4)

    def calculate_source_agreement(self, search_results: list) -> float:
        """
        Score based on whether multiple sources support the answer.
        Multiple sources agreeing = higher confidence.
        """
        if not search_results:
            return 0.0

        # count unique sources
        sources = set()
        for result in search_results:
            source = result.get("source", result.get("file", "unknown"))
            if source and source != "unknown":
                sources.add(source)

        num_sources = len(sources)
        num_results = len(search_results)

        # more sources = higher agreement score
        if num_results == 0:
            return 0.0
        elif num_sources >= 3:
            return 1.0
        elif num_sources == 2:
            return 0.75
        elif num_sources == 1 and num_results >= 3:
            return 0.6  # multiple chunks from same source
        elif num_sources == 1:
            return 0.4
        else:
            return 0.2

    def score(
        self,
        answer: str,
        search_results: list,
        query: str = ""
    ) -> dict:
        """
        Calculate overall confidence score for an answer.

        Returns a dict with:
        - overall_score: 0.0 to 1.0
        - percentage: 0 to 100
        - label: High / Medium / Low
        - breakdown: individual component scores
        - warning: message if confidence is low
        - color: for UI display
        """
        # calculate components
        retrieval = self.calculate_retrieval_score(search_results)
        coverage = self.calculate_coverage_score(answer, search_results)
        agreement = self.calculate_source_agreement(search_results)

        # weighted combination
        overall = (
            retrieval  * self.weights["retrieval"] +
            coverage   * self.weights["coverage"] +
            agreement  * self.weights["source_agreement"]
        )
        overall = round(min(overall, 1.0), 4)
        percentage = round(overall * 100, 1)

        # label and color
        if overall >= 0.7:
            label = "High"
            color = "green"
            warning = None
        elif overall >= 0.4:
            label = "Medium"
            color = "orange"
            warning = "Answer may be partially supported by documents."
        else:
            label = "Low"
            color = "red"
            warning = (
                "Low confidence — answer may not be well supported "
                "by the provided documents. Consider uploading more "
                "relevant documents or rephrasing your question."
            )

        return {
            "overall_score": overall,
            "percentage": percentage,
            "label": label,
            "color": color,
            "warning": warning,
            "breakdown": {
                "retrieval_score": round(retrieval, 4),
                "coverage_score": round(coverage, 4),
                "source_agreement": round(agreement, 4)
            },
            "sources_used": list(set([
                r.get("source", r.get("file", "unknown"))
                for r in search_results
            ]))
        }

    def format_for_display(self, confidence: dict) -> str:
        """Format confidence score as readable text."""
        pct = confidence["percentage"]
        label = confidence["label"]
        color_emoji = {"green": "🟢", "orange": "🟡", "red": "🔴"}
        emoji = color_emoji.get(confidence["color"], "⚪")

        output = f"{emoji} Confidence: {label} ({pct}%)\n"

        breakdown = confidence["breakdown"]
        output += f"  Retrieval relevance : "
        output += f"{round(breakdown['retrieval_score']*100)}%\n"
        output += f"  Answer grounding    : "
        output += f"{round(breakdown['coverage_score']*100)}%\n"
        output += f"  Source agreement    : "
        output += f"{round(breakdown['source_agreement']*100)}%\n"

        if confidence["warning"]:
            output += f"\n {confidence['warning']}\n"

        return output


# test
if __name__ == "__main__":
    scorer = ConfidenceScorer()

    print("Testing Confidence Scorer")
    print("=" * 60)

    # simulate high confidence scenario
    good_results = [
        {
            "content": "Agentic RAG uses autonomous agents to decide tools",
            "source": "doc1.pdf",
            "score": 0.92
        },
        {
            "content": "The agent performs multi-step reasoning before answering",
            "source": "doc1.pdf",
            "score": 0.85
        },
        {
            "content": "RAG combines retrieval with generation for accuracy",
            "source": "doc2.pdf",
            "score": 0.78
        }
    ]
    good_answer = (
        "Agentic RAG uses autonomous agents to decide which tools "
        "to use and performs multi-step reasoning before answering."
    )

    print("\nScenario 1 — High Confidence:")
    result = scorer.score(good_answer, good_results)
    print(scorer.format_for_display(result))

    # simulate low confidence scenario
    bad_results = [
        {
            "content": "The weather today is sunny and warm",
            "source": "doc1.pdf",
            "score": 0.12
        }
    ]
    bad_answer = "I cannot find specific information about this topic."

    print("\nScenario 2 — Low Confidence:")
    result = scorer.score(bad_answer, bad_results)
    print(scorer.format_for_display(result))

    # simulate medium confidence scenario
    medium_results = [
        {
            "content": "Machine learning learns patterns from data automatically",
            "source": "doc1.pdf",
            "score": 0.65
        },
        {
            "content": "AI systems can improve performance through training",
            "source": "doc1.pdf",
            "score": 0.55
        }
    ]
    medium_answer = (
        "Machine learning is a technique where systems automatically "
        "learn patterns from training data to improve performance."
    )

    print("\nScenario 3 — Medium Confidence:")
    result = scorer.score(medium_answer, medium_results)
    print(scorer.format_for_display(result))