"""
Tool 4: Answer Verifier
Checks whether the agent's answer is supported by 
the retrieved source documents. Flags hallucinations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import GROQ_API_KEY, LLM_MODEL

class AnswerVerifierTool:
    """
    Verifies that generated answers are grounded in sources.
    This is Tool 4 in the agent's tool registry.
    """

    name = "answer_verifier"
    description = """Verify whether an answer is supported by the 
    source documents. Use this after generating an answer to check 
    for hallucinations. Input should be the answer and sources 
    combined as a string."""

    def __init__(self):
        from groq import Groq
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL

    def run(self, answer: str, sources: list) -> dict:
        """
        Verify the answer against the provided sources.

        answer: the generated answer string
        sources: list of source chunks used to generate the answer
        """
        try:
            # combine sources into one block
            sources_text = ""
            for i, source in enumerate(sources):
                if isinstance(source, dict):
                    sources_text += f"Source {i+1}: {source.get('content', source)}\n"
                else:
                    sources_text += f"Source {i+1}: {source}\n"

            verification_prompt = f"""You are a fact-checking assistant.

Given the following SOURCE DOCUMENTS and an ANSWER, determine if 
every claim in the answer is supported by the source documents.

SOURCE DOCUMENTS:
{sources_text}

ANSWER TO VERIFY:
{answer}

Respond in this exact format:
VERDICT: [SUPPORTED / PARTIALLY SUPPORTED / NOT SUPPORTED]
CONFIDENCE: [HIGH / MEDIUM / LOW]
UNSUPPORTED CLAIMS: [list any claims not found in sources, or 'None']
EXPLANATION: [one sentence explanation]"""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": verification_prompt
                    }
                ],
                temperature=0,
                max_tokens=300
            )

            verification_text = response.choices[0].message.content

            # parse the response
            verdict = "UNKNOWN"
            confidence = "LOW"
            unsupported = "None"
            explanation = ""

            for line in verification_text.split("\n"):
                if line.startswith("VERDICT:"):
                    verdict = line.replace("VERDICT:", "").strip()
                elif line.startswith("CONFIDENCE:"):
                    confidence = line.replace("CONFIDENCE:", "").strip()
                elif line.startswith("UNSUPPORTED CLAIMS:"):
                    unsupported = line.replace(
                        "UNSUPPORTED CLAIMS:", ""
                    ).strip()
                elif line.startswith("EXPLANATION:"):
                    explanation = line.replace("EXPLANATION:", "").strip()

            # determine if hallucination risk exists
            is_hallucination = verdict in [
                "PARTIALLY SUPPORTED",
                "NOT SUPPORTED"
            ]

            return {
                "success": True,
                "verdict": verdict,
                "confidence": confidence,
                "unsupported_claims": unsupported,
                "explanation": explanation,
                "is_hallucination_risk": is_hallucination,
                "full_verification": verification_text
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Verification failed: {str(e)}",
                "is_hallucination_risk": False
            }

    def format_for_agent(self, answer: str, sources: list) -> str:
        """
        Run verification and format result for agent.
        """
        result = self.run(answer, sources)

        if not result["success"]:
            return f"Verification failed: {result['message']}"

        # choose emoji based on verdict
        if result["verdict"] == "SUPPORTED":
            icon = "✅"
        elif result["verdict"] == "PARTIALLY SUPPORTED":
            icon = "⚠️"
        else:
            icon = "❌"

        output = f"\n{icon} Verification Result: {result['verdict']}\n"
        output += f"Confidence: {result['confidence']}\n"

        if result["unsupported_claims"] != "None":
            output += f"Unsupported Claims: {result['unsupported_claims']}\n"

        output += f"Explanation: {result['explanation']}\n"

        if result["is_hallucination_risk"]:
            output += "\nWARNING: This answer may contain information "
            output += "not found in the source documents.\n"

        return output


# quick test
if __name__ == "__main__":
    tool = AnswerVerifierTool()
    print("Testing Answer Verifier Tool...\n")

    # test 1 - answer that IS supported
    print("Test 1: Answer that IS supported by sources")
    print("-" * 50)
    answer1 = "RAG combines retrieval with language models to generate accurate answers."
    sources1 = [
        {
            "content": """Retrieval Augmented Generation combines 
            information retrieval with text generation to produce 
            accurate grounded answers."""
        }
    ]
    print(tool.format_for_agent(answer1, sources1))

    # test 2 - answer that is NOT supported
    print("\nTest 2: Answer that is NOT supported by sources")
    print("-" * 50)
    answer2 = "RAG was invented by Google in 2019 and first used in Gmail."
    sources2 = [
        {
            "content": """RAG is a technique for grounding language 
            model responses in retrieved documents."""
        }
    ]
    print(tool.format_for_agent(answer2, sources2))