"""
Tool 4: Answer Verifier
Checks if the generated answer is grounded in the retrieved context.
Flags hallucinated claims not found in source documents.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import GROQ_API_KEY, LLM_MODEL


class AnswerVerifierTool:
    """
    Verifies that an answer is grounded in the retrieved context.
    This is Tool 4 in the agent's tool registry.
    """

    name = "answer_verifier"
    description = """Verify if an answer is supported by the retrieved
    documents. Use this tool when you want to check if your answer
    is grounded in the source documents and not hallucinated.
    Input should be: 'ANSWER: [answer] CONTEXT: [context]'"""

    def __init__(self):
        from groq import Groq
        self.llm = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL

    def run(self, answer: str, context_chunks: list) -> dict:
        """
        Verify an answer against context chunks.
        Returns verification result with grounding score.
        """
        if not answer or not context_chunks:
            return {
                "verified": False,
                "grounding_score": 0.0,
                "issues": ["No answer or context provided"],
                "verdict": "Cannot verify"
            }

        # build context string
        context = "\n---\n".join([
            chunk.get("content", "")
            for chunk in context_chunks[:3]
        ])

        prompt = f"""You are a fact checker. Check if the answer is 
supported by the context. Be strict.

CONTEXT:
{context}

ANSWER TO VERIFY:
{answer}

Respond in this exact format:
GROUNDING: [0-100 score of how well answer is grounded in context]
VERDICT: [SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED]
ISSUES: [list any claims in the answer not found in context, or NONE]

Your response:"""

        try:
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=200
            )

            result_text = response.choices[0].message.content.strip()
            return self._parse_verification(result_text)

        except Exception as e:
            return {
                "verified": False,
                "grounding_score": 0.0,
                "issues": [str(e)],
                "verdict": "Verification failed"
            }

    def _parse_verification(self, text: str) -> dict:
        """Parse the LLM verification response."""
        grounding_score = 0.5
        verdict = "UNKNOWN"
        issues = []

        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("GROUNDING:"):
                try:
                    grounding_score = float(
                        line.split(":")[1].strip()
                    ) / 100
                except Exception:
                    pass
            elif line.startswith("VERDICT:"):
                verdict = line.split(":", 1)[1].strip()
            elif line.startswith("ISSUES:"):
                issues_text = line.split(":", 1)[1].strip()
                if issues_text.upper() != "NONE":
                    issues = [issues_text]

        return {
            "verified": verdict == "SUPPORTED",
            "grounding_score": round(grounding_score, 2),
            "verdict": verdict,
            "issues": issues
        }

    def format_for_agent(
        self,
        answer: str,
        context_chunks: list
    ) -> str:
        """Format verification result for the agent."""
        result = self.run(answer, context_chunks)

        output = f"Verification Result:\n"
        output += f"Verdict         : {result['verdict']}\n"
        output += f"Grounding Score : {result['grounding_score']*100:.0f}%\n"

        if result["issues"]:
            output += f"Issues Found    : {', '.join(result['issues'])}\n"
        else:
            output += f"Issues Found    : None\n"

        if not result["verified"]:
            output += (
                "\n⚠️ Warning: Some claims may not be fully "
                "supported by the source documents."
            )

        return output


# test
if __name__ == "__main__":
    tool = AnswerVerifierTool()

    print("Testing Answer Verifier")
    print("=" * 60)

    # test 1 — well grounded answer
    context = [
        {
            "content": (
                "Agentic RAG uses autonomous agents with tools "
                "including document search and web search to "
                "perform multi-step reasoning."
            )
        }
    ]
    answer = (
        "Agentic RAG uses autonomous agents that can use "
        "document search and web search tools."
    )

    print("Test 1 — Well grounded answer:")
    print(f"Answer : {answer}")
    result = tool.format_for_agent(answer, context)
    print(result)

    # test 2 — hallucinated answer
    context2 = [
        {
            "content": (
                "Machine learning is a subset of AI that learns "
                "patterns from data."
            )
        }
    ]
    answer2 = (
        "Machine learning was invented by Alan Turing in 1950 "
        "at Cambridge University."
    )

    print("\nTest 2 — Potentially hallucinated answer:")
    print(f"Answer : {answer2}")
    result2 = tool.format_for_agent(answer2, context2)
    print(result2)