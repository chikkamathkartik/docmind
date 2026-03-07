"""
DocMind RAG Agent - Improved Brain
Implements the ReAct loop with better tool selection
and session memory support.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import GROQ_API_KEY, LLM_MODEL
from backend.tools.document_search import DocumentSearchTool
from backend.tools.web_search import WebSearchTool
from backend.tools.summarizer import SummarizerTool
from backend.tools.answer_verifier import AnswerVerifierTool
from backend.agents.agent_memory import memory_manager
from backend.core.confidence_scorer import ConfidenceScorer


class DocMindAgent:
    """
    Improved Agentic RAG agent with session memory
    and smarter tool selection.
    """

    def __init__(self):
        print("Initializing DocMind Agent...")

        # initialize tools
        self.tools = {
            "document_search": DocumentSearchTool(),
            "web_search": WebSearchTool(),
            "summarizer": SummarizerTool(),
            "answer_verifier": AnswerVerifierTool()
        }
        
        self.confidence_scorer = ConfidenceScorer()

        # initialize LLM
        from groq import Groq
        self.llm = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL

        # agent config
        self.max_iterations = 4

        print("DocMind Agent ready\n")

    def _get_system_prompt(self, memory_context: str = "") -> str:
        """Improved system prompt with clearer instructions."""
        base_prompt = """You are DocMind, an intelligent research \
assistant using the ReAct approach.

AVAILABLE TOOLS:
1. document_search - Search uploaded documents (USE FIRST for most questions)
2. web_search - Search the internet (USE when documents lack the answer)
3. summarizer - Summarize long text (USE when asked to summarize)
4. answer_verifier - Verify answer accuracy (USE for important claims)

RESPONSE FORMAT - use EXACTLY one of these formats:

Format A - To use a tool:
THOUGHT: [your reasoning - which tool and why]
ACTION: [exact tool name from list above]
INPUT: [your search query or text]

Format B - When you have the final answer:
THOUGHT: [your final reasoning]
FINAL ANSWER: [complete answer with source citations]

RULES:
- Always use document_search FIRST before web_search
- Keep thoughts concise - one or two sentences maximum
- If document_search returns low confidence, then try web_search
- Always cite sources in your final answer
- If nothing found anywhere, say so clearly
- Never make up information"""

        if memory_context:
            base_prompt += f"\n\n{memory_context}"

        return base_prompt

    def _parse_response(self, response: str) -> dict:
        """Parse agent response into structured format."""
        response = response.strip()

        if "FINAL ANSWER:" in response:
            thought = ""
            if "THOUGHT:" in response:
                thought = response.split("THOUGHT:")[1].split(
                    "FINAL ANSWER:"
                )[0].strip()
            answer = response.split("FINAL ANSWER:")[1].strip()
            return {
                "type": "final",
                "thought": thought,
                "answer": answer
            }

        if "ACTION:" in response and "INPUT:" in response:
            thought = ""
            if "THOUGHT:" in response:
                thought = response.split("THOUGHT:")[1].split(
                    "ACTION:"
                )[0].strip()
            action = response.split("ACTION:")[1].split(
                "INPUT:"
            )[0].strip()
            input_text = response.split("INPUT:")[1].strip()
            return {
                "type": "action",
                "thought": thought,
                "action": action,
                "input": input_text
            }

        # default to final answer if format not recognized
        return {
            "type": "final",
            "thought": "",
            "answer": response
        }

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute a tool and return its output."""
        tool_name = tool_name.strip().lower()

        if tool_name not in self.tools:
            available = list(self.tools.keys())
            return f"Tool '{tool_name}' not found. Available: {available}"

        tool = self.tools[tool_name]

        try:
            if tool_name == "document_search":
                return tool.format_for_agent(tool_input)
            elif tool_name == "web_search":
                return tool.format_for_agent(tool_input)
            elif tool_name == "summarizer":
                return tool.format_for_agent(tool_input)
            elif tool_name == "answer_verifier":
                return tool.format_for_agent(tool_input, [])
            else:
                return f"Tool {tool_name} execution not implemented"
        except Exception as e:
            return f"Tool error: {str(e)}"

    def run(self, question: str, session_id: str = "default") -> dict:
        """
        Run the agent on a question.
        Uses session memory for conversation context.
        """
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"Session : {session_id}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # get session memory
        session = memory_manager.get_session(session_id)
        memory_context = session.format_for_prompt(n=3)

        # build messages
        messages = [
            {
                "role": "system",
                "content": self._get_system_prompt(memory_context)
            },
            {
                "role": "user",
                "content": question
            }
        ]

        reasoning_trace = []
        final_answer = None
        iteration = 0

        # ReAct loop
        while iteration < self.max_iterations:
            iteration += 1
            print(f"--- Iteration {iteration} ---")

            try:
                response = self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=600
                )
                agent_output = response.choices[0].message.content

            except Exception as e:
                print(f"LLM error: {e}")
                final_answer = "I encountered an error. Please try again."
                break

            parsed = self._parse_response(agent_output)

            # log thought
            if parsed["thought"]:
                print(f"THOUGHT: {parsed['thought']}")
                reasoning_trace.append({
                    "iteration": iteration,
                    "type": "thought",
                    "content": parsed["thought"]
                })

            # final answer
            if parsed["type"] == "final":
                final_answer = parsed["answer"]
                print(f"FINAL ANSWER: {final_answer[:200]}...")
                reasoning_trace.append({
                    "iteration": iteration,
                    "type": "final_answer",
                    "content": final_answer
                })
                break

            # tool call
            if parsed["type"] == "action":
                tool_name = parsed["action"]
                tool_input = parsed["input"]

                print(f"ACTION: {tool_name}")
                print(f"INPUT : {tool_input[:80]}...")

                reasoning_trace.append({
                    "iteration": iteration,
                    "type": "action",
                    "tool": tool_name,
                    "input": tool_input
                })

                # execute tool
                observation = self._execute_tool(tool_name, tool_input)
                print(f"OBSERVATION: {observation[:150]}...\n")

                reasoning_trace.append({
                    "iteration": iteration,
                    "type": "observation",
                    "tool": tool_name,
                    "content": observation[:500]
                })

                # add to message history
                messages.append({
                    "role": "assistant",
                    "content": agent_output
                })
                messages.append({
                    "role": "user",
                    "content": f"OBSERVATION: {observation}"
                })

        # handle max iterations
        if not final_answer:
            final_answer = (
                "I was unable to find a complete answer. "
                "Please try rephrasing your question."
            )

        elapsed = time.time() - start_time

        # save to session memory
        session.add_turn(
            question=question,
            answer=final_answer,
            reasoning_trace=reasoning_trace
        )
        
        # calculate confidence score
        search_results = []
        for step in reasoning_trace:
            if (step.get("type") == "observation" and
                    step.get("tool") == "document_search"):
                content = step.get("content", "")
                # parse Score lines from hybrid search output
                for line in content.split("\n"):
                    line = line.strip()
                    if line.startswith("Score"):
                        try:
                            score_val = float(line.split(":")[1].strip())
                            search_results.append({
                                "rrf_score": score_val,
                                "source": "document"
                            })
                        except Exception:
                            pass

        # if no scores parsed, use default medium score
        # so retrieval that worked doesn't show as low confidence
        if not search_results and any(
            step.get("tool") == "document_search"
            for step in reasoning_trace
            if step.get("type") == "observation"
        ):
            search_results = [{"rrf_score": 0.015, "source": "document"}] * 3

        confidence = self.confidence_scorer.score(
            final_answer,
            search_results
        )

        return {
            "question": question,
            "answer": final_answer,
            "reasoning_trace": reasoning_trace,
            "iterations": iteration,
            "time_taken": round(elapsed, 2),
            "session_id": session_id,
            "confidence": confidence
        }

    def clear_memory(self, session_id: str = "default"):
        """Clear memory for a specific session."""
        memory_manager.clear_session(session_id)
        print(f"Memory cleared for session: {session_id}")


# test
if __name__ == "__main__":
    agent = DocMindAgent()

    # test with session memory
    session_id = "test_session"

    questions = [
        "What is Agentic RAG?",
        "How is it different from standard RAG?",  # follow up question
    ]

    for question in questions:
        result = agent.run(question, session_id=session_id)
        print(f"\n{'='*60}")
        print(f"Answer     : {result['answer'][:300]}...")
        print(f"Iterations : {result['iterations']}")
        print(f"Time       : {result['time_taken']}s")
        print()