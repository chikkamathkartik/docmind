"""
DocMind RAG Agent - The Brain
Implements the ReAct (Reasoning + Acting) loop.
The agent decides which tools to use based on the question.
"""

import sys
import os
import json
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import GROQ_API_KEY, LLM_MODEL
from backend.tools.document_search import DocumentSearchTool
from backend.tools.web_search import WebSearchTool
from backend.tools.summarizer import SummarizerTool
from backend.tools.answer_verifier import AnswerVerifierTool

class DocMindAgent:
    """
    The main Agentic RAG agent.
    Uses ReAct loop to answer questions using multiple tools.
    """

    def __init__(self):
        print("Initializing DocMind Agent...")

        # initialize all tools
        self.tools = {
            "document_search": DocumentSearchTool(),
            "web_search": WebSearchTool(),
            "summarizer": SummarizerTool(),
            "answer_verifier": AnswerVerifierTool()
        }

        # initialize LLM
        from groq import Groq
        self.llm = Groq(api_key=GROQ_API_KEY)
        self.model = LLM_MODEL

        # agent config
        self.max_iterations = 5  # prevent infinite loops
        self.conversation_history = []

        print("DocMind Agent ready\n")

    def _get_system_prompt(self) -> str:
        """
        The system prompt that defines the agent's behavior.
        This is the most critical piece of the entire agent.
        """
        return """You are DocMind, an intelligent research assistant 
that uses the ReAct (Reasoning + Acting) approach to answer questions.

You have access to these tools:
1. document_search - Search uploaded documents for relevant information
2. web_search - Search the internet for current information
3. summarizer - Summarize long documents or text
4. answer_verifier - Verify if an answer is supported by sources

To use a tool, respond in this EXACT format:
THOUGHT: [your reasoning about what to do next]
ACTION: [tool_name]
INPUT: [your input to the tool]

When you have enough information to answer, respond in this EXACT format:
THOUGHT: [your final reasoning]
FINAL ANSWER: [your complete answer with source citations]

Rules:
- Always start by thinking about which tool is most appropriate
- Use document_search first for questions about uploaded documents
- Use web_search when documents don't have the answer
- Always verify important answers with answer_verifier
- Never make up information - only use what tools return
- Always cite your sources in the final answer
- If no relevant information found anywhere, say so clearly"""

    def _parse_agent_response(self, response: str) -> dict:
        """
        Parse the agent's response to extract thought, action, and input.
        """
        response = response.strip()

        # check if this is a final answer
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

        # check if this is a tool call
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

        # if format not recognized, treat as final answer
        return {
            "type": "final",
            "thought": "",
            "answer": response
        }

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """
        Execute the specified tool with the given input.
        Returns the tool output as a string.
        """
        if tool_name not in self.tools:
            return f"Error: Tool '{tool_name}' not found. Available tools: {list(self.tools.keys())}"

        tool = self.tools[tool_name]

        try:
            if tool_name == "document_search":
                return tool.format_for_agent(tool_input)
            elif tool_name == "web_search":
                return tool.format_for_agent(tool_input)
            elif tool_name == "summarizer":
                return tool.format_for_agent(tool_input)
            elif tool_name == "answer_verifier":
                # for verifier, split input into answer and sources
                return tool.format_for_agent(tool_input, [])
            else:
                return f"Tool {tool_name} not implemented"
        except Exception as e:
            return f"Tool execution error: {str(e)}"

    def run(self, question: str) -> dict:
        """
        Run the agent on a question using the ReAct loop.
        Returns the final answer with full reasoning trace.
        """
        print(f"\n{'='*60}")
        print(f"Agent processing: {question}")
        print(f"{'='*60}\n")

        start_time = time.time()

        # build messages for LLM
        messages = [
            {"role": "system", "content": self._get_system_prompt()}
        ]

        # add conversation history for memory
        for turn in self.conversation_history[-3:]:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({
                "role": "assistant",
                "content": turn["answer"]
            })

        # add current question
        messages.append({"role": "user", "content": question})

        # reasoning trace - stores every thought and action
        reasoning_trace = []
        final_answer = None
        iteration = 0

        # ReAct loop
        while iteration < self.max_iterations:
            iteration += 1
            print(f"--- Iteration {iteration} ---")

            # get agent response
            response = self.llm.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                max_tokens=800
            )

            agent_output = response.choices[0].message.content
            parsed = self._parse_agent_response(agent_output)

            # log the thought
            if parsed["thought"]:
                print(f"THOUGHT: {parsed['thought']}")
                reasoning_trace.append({
                    "iteration": iteration,
                    "type": "thought",
                    "content": parsed["thought"]
                })

            # if final answer reached
            if parsed["type"] == "final":
                final_answer = parsed["answer"]
                print(f"\nFINAL ANSWER: {final_answer}")
                reasoning_trace.append({
                    "iteration": iteration,
                    "type": "final_answer",
                    "content": final_answer
                })
                break

            # if tool call
            if parsed["type"] == "action":
                tool_name = parsed["action"].strip()
                tool_input = parsed["input"].strip()

                print(f"ACTION: {tool_name}")
                print(f"INPUT: {tool_input[:100]}...")

                reasoning_trace.append({
                    "iteration": iteration,
                    "type": "action",
                    "tool": tool_name,
                    "input": tool_input
                })

                # execute the tool
                observation = self._execute_tool(tool_name, tool_input)
                print(f"OBSERVATION: {observation[:200]}...\n")

                reasoning_trace.append({
                    "iteration": iteration,
                    "type": "observation",
                    "tool": tool_name,
                    "content": observation[:500]
                })

                # add to message history so agent sees the result
                messages.append({
                    "role": "assistant",
                    "content": agent_output
                })
                messages.append({
                    "role": "user",
                    "content": f"OBSERVATION: {observation}"
                })

        # handle max iterations reached without final answer
        if not final_answer:
            final_answer = """I was unable to find a complete answer 
            within the allowed steps. Please try rephrasing your 
            question or uploading more relevant documents."""

        elapsed = time.time() - start_time

        # save to conversation history
        self.conversation_history.append({
            "question": question,
            "answer": final_answer
        })

        return {
            "question": question,
            "answer": final_answer,
            "reasoning_trace": reasoning_trace,
            "iterations": iteration,
            "time_taken": round(elapsed, 2)
        }

    def clear_memory(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("Conversation memory cleared")


# test the agent
if __name__ == "__main__":
    agent = DocMindAgent()

    # test questions
    test_questions = [
        "What is Agentic RAG and how is it different from standard RAG?",
        "What are the latest developments in AI in 2024?",
    ]

    for question in test_questions:
        result = agent.run(question)

        print(f"\n{'='*60}")
        print("RESULT SUMMARY")
        print(f"{'='*60}")
        print(f"Question   : {result['question']}")
        print(f"Answer     : {result['answer'][:300]}...")
        print(f"Iterations : {result['iterations']}")
        print(f"Time       : {result['time_taken']} seconds")
        print(f"Steps taken: {len(result['reasoning_trace'])}")
        print()