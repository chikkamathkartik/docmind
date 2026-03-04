"""
Agent Memory
Manages conversation history across turns so the agent
remembers previous questions and answers.
"""

import json
import time

class AgentMemory:
    """
    Stores and retrieves conversation history.
    Keeps last N turns to avoid context window overflow.
    """

    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self.history = []
        self.session_start = time.time()

    def add_turn(self, question: str, answer: str, 
                 reasoning_trace: list = None):
        """Add a conversation turn to memory."""
        turn = {
            "turn_id": len(self.history) + 1,
            "question": question,
            "answer": answer,
            "reasoning_trace": reasoning_trace or [],
            "timestamp": time.time()
        }
        self.history.append(turn)

        # keep only last max_turns
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

    def get_recent_history(self, n: int = 3) -> list:
        """Get the n most recent conversation turns."""
        return self.history[-n:]

    def format_for_prompt(self, n: int = 3) -> str:
        """Format recent history as text for the agent prompt."""
        recent = self.get_recent_history(n)
        if not recent:
            return ""

        formatted = "Previous conversation:\n"
        for turn in recent:
            formatted += f"User: {turn['question']}\n"
            formatted += f"Assistant: {turn['answer'][:200]}...\n\n"

        return formatted

    def clear(self):
        """Clear all conversation history."""
        self.history = []

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "total_turns": len(self.history),
            "session_duration": round(time.time() - self.session_start, 2),
            "max_turns": self.max_turns
        }