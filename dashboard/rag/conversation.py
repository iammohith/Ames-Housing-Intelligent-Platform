"""
Conversation Manager — Session memory for multi-turn chat.
"""

from __future__ import annotations

from typing import Dict, List


class ConversationManager:
    """Manages chat history per session for contextual follow-up questions."""

    def __init__(self, max_history: int = 20):
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str):
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history :]

    def get_context_window(self, n_recent: int = 5) -> str:
        """Get recent conversation context for follow-up awareness."""
        recent = self.history[-n_recent:]
        parts = []
        for msg in recent:
            prefix = "User" if msg["role"] == "user" else "Assistant"
            parts.append(f"{prefix}: {msg['content']}")
        return "\n".join(parts)

    def format_for_prompt(self, window_size: int = 5) -> str:
        recent = self.history[-window_size * 2:]  # pairs of user+assistant
        turns, i = [], 0
        while i + 1 < len(recent):
            u = recent[i]; a = recent[i+1]
            n = i // 2 + 1
            turns.append(f"[Turn {n}] User: {u['content']}\nAssistant: {a['content']}")
            i += 2
        return "\n".join(turns)

    def clear(self):
        self.history = []
