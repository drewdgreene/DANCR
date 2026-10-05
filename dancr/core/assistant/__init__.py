"""The Assistant: a conversational front door to DANCR's deterministic engine.

The model proposes (intent, plans, narration, next questions); the engine proves (every number, join, test
and chart comes from an executed step with a node id). Nothing in this package imports Qt: the window, the
command line and any future front end share it.
"""
from __future__ import annotations

from .client import (
    ChatResult,
    FakeProvider,
    Message,
    ModelSettings,
    OpenAIProvider,
    Provider,
    ProviderError,
    ProviderPaused,
    ToolCall,
    ToolSpec,
    Usage,
    provider_for,
)

__all__ = [
    "ChatResult", "FakeProvider", "Message", "ModelSettings", "OpenAIProvider", "Provider",
    "ProviderError", "ProviderPaused", "ToolCall", "ToolSpec", "Usage", "provider_for",
    "AssistantReply", "AssistantSession", "Thread", "load_thread", "save_thread",
]

_LAZY = {"AssistantReply", "AssistantSession", "Thread", "load_thread", "save_thread"}


def __getattr__(name: str):
    if name in _LAZY:
        from . import session, store
        return {"AssistantReply": session.AssistantReply, "AssistantSession": session.AssistantSession,
                "Thread": store.Thread, "load_thread": store.load_thread, "save_thread": store.save_thread}[name]
    raise AttributeError(name)
