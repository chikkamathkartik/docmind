import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import GROQ_API_KEY, LLM_MODEL

class SummarizerTool:
    """
    Summarizes documents or large pieces of text.
    This is Tool 3 in the agent's tool registry.
    """

    name = "summarizer"
    description = """Summarize a document or large piece of text. 
    Use this tool when the user explicitly asks for a summary of 
    a document, or when retrieved content is too long and needs 
    to be condensed. Input should be the text to summarize."""

    def __init__(self):
        from groq import Groq
        self.client = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL

    def run(self, text: str, style: str = "concise") -> dict:
        """
        Summarize the given text.
        style: 'concise' for short summary, 'detailed' for longer one
        """
        try:
            if style == "concise":
                instruction = """Create a concise summary in 3-5 bullet 
                points covering the main ideas. Be clear and direct."""
            else:
                instruction = """Create a detailed summary covering all 
                major points, key arguments, and important details. 
                Use clear paragraphs."""

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": instruction
                    },
                    {
                        "role": "user",
                        "content": f"Please summarize the following text:\n\n{text}"
                    }
                ],
                temperature=0.3,
                max_tokens=500
            )

            summary = response.choices[0].message.content

            return {
                "success": True,
                "summary": summary,
                "original_length": len(text),
                "summary_length": len(summary),
                "compression_ratio": round(len(summary) / len(text), 2)
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Summarization failed: {str(e)}",
                "summary": ""
            }

    def format_for_agent(self, text: str) -> str:
        """
        Run summarization and format result for agent.
        """
        result = self.run(text)

        if not result["success"]:
            return f"Summarization failed: {result['message']}"

        output = "Document Summary:\n"
        output += "=" * 40 + "\n"
        output += result["summary"]
        output += "\n" + "=" * 40
        output += f"\nOriginal: {result['original_length']} chars"
        output += f" → Summary: {result['summary_length']} chars"
        output += f" (compression: {result['compression_ratio']})"

        return output


# quick test
if __name__ == "__main__":
    tool = SummarizerTool()
    print("Testing Summarizer Tool...\n")

    sample_text = """
    Artificial Intelligence (AI) is transforming industries across 
    the globe. From healthcare to finance, AI systems are being 
    deployed to automate tasks, improve decision making, and 
    uncover insights from large datasets. Machine learning, a 
    subset of AI, allows systems to learn from data without being 
    explicitly programmed. Deep learning, which uses neural networks 
    with many layers, has achieved remarkable results in image 
    recognition, natural language processing, and game playing.
    
    Large language models like GPT and Llama represent the latest 
    advancement in AI, capable of understanding and generating 
    human-like text across virtually any topic. These models are 
    trained on massive datasets and can be fine-tuned for specific 
    tasks. Retrieval Augmented Generation combines these powerful 
    language models with information retrieval systems, allowing 
    AI to answer questions based on specific documents rather than 
    just its training data, significantly reducing hallucinations 
    and improving accuracy.
    """ * 2

    result = tool.format_for_agent(sample_text)
    print(result)