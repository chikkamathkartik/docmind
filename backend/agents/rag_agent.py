"""
DocMind RAG Agent - Improved Brain
"""

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from configs.settings import LLM_PROVIDER, OLLAMA_BASE_URL, LLM_MODEL, GROQ_API_KEY

from backend.tools.document_search import DocumentSearchTool
from backend.tools.summarizer import SummarizerTool
from backend.tools.answer_verifier import AnswerVerifierTool
from backend.agents.agent_memory import memory_manager
from backend.core.confidence_scorer import ConfidenceScorer


class DocMindAgent:

    def __init__(self):
        print("Initializing DocMind Agent...")

        self.tools = {
            "document_search": DocumentSearchTool(),
            "summarizer": SummarizerTool(),
            "answer_verifier": AnswerVerifierTool()
        }

        self.confidence_scorer = ConfidenceScorer()

        from groq import Groq
        self.llm = Groq(api_key=GROQ_API_KEY)

        self.max_iterations = 4

        print("DocMind Agent ready\n")

    def _get_system_prompt(self, memory_context: str = "") -> str:
        base_prompt = """You are DocMind, an intelligent biomedical research assistant.

Use ONLY the uploaded documents to answer.
Always follow the ReAct format strictly.
"""

        if memory_context:
            base_prompt += f"\n\n{memory_context}"

        return base_prompt

    def _parse_response(self, response: str) -> dict:
        response = response.strip()

        if "FINAL ANSWER:" in response:
            thought = ""
            if "THOUGHT:" in response:
                thought = response.split("THOUGHT:")[1].split("FINAL ANSWER:")[0].strip()
            answer = response.split("FINAL ANSWER:")[1].strip()
            return {"type": "final", "thought": thought, "answer": answer}

        if "ACTION:" in response and "INPUT:" in response:
            thought = ""
            if "THOUGHT:" in response:
                thought = response.split("THOUGHT:")[1].split("ACTION:")[0].strip()
            action = response.split("ACTION:")[1].split("INPUT:")[0].strip()
            input_text = response.split("INPUT:")[1].strip()
            return {"type": "action", "thought": thought, "action": action, "input": input_text}

        return {"type": "final", "thought": "", "answer": response}

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        tool_name = tool_name.strip().lower()

        if tool_name not in self.tools:
            return f"Tool '{tool_name}' not found."

        try:
            if tool_name == "document_search":
                return self.tools[tool_name].format_for_agent(tool_input)
            elif tool_name == "summarizer":
                return self.tools[tool_name].format_for_agent(tool_input)
            elif tool_name == "answer_verifier":
                return self.tools[tool_name].format_for_agent(tool_input, [])
        except Exception as e:
            return f"Tool error: {str(e)}"

    def _call_llm(self, messages):
        """🔥 Handles both Groq and Ollama safely"""

        try:
            # ✅ GROQ
            if LLM_PROVIDER == "groq":
                response = self.llm.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages,
                    temperature=0.1,
                    max_tokens=600
                )
                return response.choices[0].message.content

            # ✅ OLLAMA
            elif LLM_PROVIDER == "ollama":
                import requests

                # convert messages → prompt
                full_prompt = ""
                for msg in messages:
                    full_prompt += f"{msg['role'].upper()}: {msg['content']}\n"

                response = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": LLM_MODEL,
                        "prompt": full_prompt,
                        "stream": False
                    },
                    timeout=60
                )

                # 🔍 DEBUG (IMPORTANT)
                print("\n=== OLLAMA DEBUG ===")
                print("STATUS:", response.status_code)
                print("TEXT:", response.text[:300])
                print("====================\n")

                # ✅ SAFE JSON PARSE
                try:
                    result = response.json()
                    return result.get("response", "")
                except Exception:
                    return "ERROR"

        except Exception as e:
            print(f"LLM error: {e}")
            return "ERROR"

    def run(self, question: str, session_id: str = "default") -> dict:

        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print(f"Session : {session_id}")
        print(f"{'='*60}\n")

        start_time = time.time()

        session = memory_manager.get_session(session_id)
        memory_context = session.format_for_prompt(n=3)

        messages = [
            {"role": "system", "content": self._get_system_prompt(memory_context)},
            {"role": "user", "content": question}
        ]

        reasoning_trace = []
        final_answer = None

        for iteration in range(1, self.max_iterations + 1):

            print(f"--- Iteration {iteration} ---")

            agent_output = self._call_llm(messages)

            if agent_output == "ERROR" or not agent_output:
                final_answer = "LLM connection failed. Check ngrok/Colab."
                break

            parsed = self._parse_response(agent_output)

            if parsed["thought"]:
                reasoning_trace.append({
                    "iteration": iteration,
                    "type": "thought",
                    "content": parsed["thought"]
                })

            if parsed["type"] == "final":
                final_answer = parsed["answer"]
                break

            if parsed["type"] == "action":
                tool_name = parsed["action"]
                tool_input = parsed["input"]

                observation = self._execute_tool(tool_name, tool_input)

                reasoning_trace.append({
                    "iteration": iteration,
                    "type": "action",
                    "tool": tool_name,
                    "input": tool_input
                })

                reasoning_trace.append({
                    "iteration": iteration,
                    "type": "observation",
                    "tool": tool_name,
                    "content": observation[:500]
                })

                messages.append({"role": "assistant", "content": agent_output})
                messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})

        if not final_answer:
            final_answer = "Could not find complete answer."

        elapsed = time.time() - start_time

        session.add_turn(question, final_answer, reasoning_trace)

        confidence = self.confidence_scorer.score(final_answer, [])

        return {
            "question": question,
            "answer": final_answer,
            "reasoning_trace": reasoning_trace,
            "time_taken": round(elapsed, 2),
            "confidence": confidence
        }