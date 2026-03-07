import time
from typing import Dict, List


class SessionMemory:
    """
    Manages conversation history for a single session.
    """

    def __init__(self, session_id: str, max_turns: int = 5):
        self.session_id = session_id
        self.max_turns = max_turns
        self.history = []
        self.created_at = time.time()
        self.last_active = time.time()

    def add_turn(
        self,
        question: str,
        answer: str,
        reasoning_trace: list = None
    ):
        """Add a conversation turn."""
        self.history.append({
            "turn_id": len(self.history) + 1,
            "question": question,
            "answer": answer,
            "reasoning_trace": reasoning_trace or [],
            "timestamp": time.time()
        })

        # keep only last max_turns
        if len(self.history) > self.max_turns:
            self.history = self.history[-self.max_turns:]

        self.last_active = time.time()

    def get_recent(self, n: int = 3) -> list:
        """Get n most recent turns."""
        return self.history[-n:]

    def format_for_prompt(self, n: int = 3) -> str:
        """Format recent history as text for agent prompt."""
        recent = self.get_recent(n)
        if not recent:
            return ""

        formatted = "Previous conversation:\n"
        for turn in recent:
            formatted += f"User: {turn['question']}\n"
            formatted += f"Assistant: {turn['answer'][:150]}...\n\n"

        return formatted

    def clear(self):
        """Clear conversation history."""
        self.history = []

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "session_id": self.session_id,
            "total_turns": len(self.history),
            "created_at": self.created_at,
            "last_active": self.last_active,
            "age_seconds": round(time.time() - self.created_at, 2)
        }


class MemoryManager:
    """
    Manages multiple sessions.
    Each user session gets its own memory.
    Automatically cleans up old sessions.
    """

    def __init__(
        self,
        max_sessions: int = 100,
        session_timeout: int = 3600
    ):
        self.sessions: Dict[str, SessionMemory] = {}
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout  # seconds

    def get_session(self, session_id: str) -> SessionMemory:
        """
        Get or create a session.
        Automatically cleans up expired sessions.
        """
        self._cleanup_expired()

        if session_id not in self.sessions:
            self.sessions[session_id] = SessionMemory(session_id)

        return self.sessions[session_id]

    def clear_session(self, session_id: str):
        """Clear a specific session."""
        if session_id in self.sessions:
            self.sessions[session_id].clear()

    def delete_session(self, session_id: str):
        """Delete a session entirely."""
        if session_id in self.sessions:
            del self.sessions[session_id]

    def _cleanup_expired(self):
        """Remove sessions that have been inactive too long."""
        now = time.time()
        expired = [
            sid for sid, session in self.sessions.items()
            if now - session.last_active > self.session_timeout
        ]
        for sid in expired:
            del self.sessions[sid]

    def get_all_stats(self) -> list:
        """Get stats for all active sessions."""
        return [
            session.get_stats()
            for session in self.sessions.values()
        ]

    def get_total_sessions(self) -> int:
        """Return number of active sessions."""
        return len(self.sessions)


# global memory manager instance
# imported by the agent and FastAPI
memory_manager = MemoryManager()