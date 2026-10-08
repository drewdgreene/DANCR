"""Model clients for the Assistant.

One small interface, so the engine does not care which model answers: an OpenAI-style
``/v1/chat/completions`` endpoint (Fireworks, OpenAI, OpenRouter, a local server), or a scripted fake
used by the tests. The model only ever *proposes*; every number comes from a DANCR run (see
``session.py`` and ``tools.py``).

Nothing here touches Qt or the project: the client sends messages and returns the model's reply.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

log = logging.getLogger("dancr.assistant")

DEFAULT_BASE_URL = "https://api.fireworks.ai/inference/v1"
DEFAULT_MODEL = "accounts/fireworks/models/deepseek-v4p1-flash"
DEFAULT_TIMEOUT = 120.0


class ProviderError(Exception):
    """The model could not be reached, or refused the request, with a plain reason."""


class ProviderPaused(ProviderError):
    """The key has no funds left, or is rate-limited. The Assistant pauses; the project is safe."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str = ""


@dataclass
class Message:
    role: str                       # system | user | assistant | tool
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str = ""

    def to_wire(self) -> dict[str, Any]:
        """As the OpenAI-style API wants it."""
        m: dict[str, Any] = {"role": self.role, "content": self.content or None}
        if self.role == "assistant" and self.tool_calls:
            m["tool_calls"] = [{"id": c.id, "type": "function",
                                "function": {"name": c.name, "arguments": c.raw_arguments or json.dumps(c.arguments)}}
                               for c in self.tool_calls]
        if self.role == "tool":
            m["tool_call_id"] = self.tool_call_id
        return m


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]

    def to_wire(self) -> dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description,
                                                 "parameters": self.parameters}}


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.cached_tokens += other.cached_tokens


@dataclass
class ChatResult:
    message: Message
    usage: Usage = field(default_factory=Usage)
    finish_reason: str = ""


@dataclass
class ModelSettings:
    """Where the model is and how it is called. The key is never written into a project file."""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    api_key: str = ""
    temperature: float = 1.0
    top_p: float = 0.95
    max_tokens: int = 4096
    timeout: float = DEFAULT_TIMEOUT

    @property
    def configured(self) -> bool:
        return bool(self.api_key.strip() and self.base_url.strip() and self.model.strip())

    @classmethod
    def from_env(cls) -> "ModelSettings":
        return cls(
            base_url=os.environ.get("DANCR_ASSISTANT_BASE_URL") or DEFAULT_BASE_URL,
            model=os.environ.get("DANCR_ASSISTANT_MODEL") or DEFAULT_MODEL,
            api_key=os.environ.get("DANCR_ASSISTANT_API_KEY") or os.environ.get("FIREWORKS_API_KEY") or "",
        )


class Provider:
    """What the Assistant needs from a model: one call, with tools, that may come back with tool calls."""

    name = "provider"

    def __init__(self) -> None:
        # A thread-safe flag a worker thread can check while the GUI thread asks a call to stop (Stop button,
        # Escape). A plain bool happens to work in CPython, but an Event makes the intent explicit and the
        # happens-before visible, so a provider on another runtime cannot miss the write.
        self._cancelled = threading.Event()

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec], settings: ModelSettings,
             on_delta: Callable[[str], None] | None = None) -> ChatResult:
        """One call. ``on_delta`` receives the reply's text as it streams in (a provider may ignore it and
        return the whole reply at once)."""
        raise NotImplementedError

    def cancel(self) -> None:
        """Ask an in-flight call to stop. Best-effort; may do nothing."""
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()


class FakeProvider(Provider):
    """A scripted model for tests and for running the window with no key: replies are queued in order.

    A reply may be a :class:`ChatResult`, a :class:`Message` (usage is zero), a plain string (a final reply),
    or a callable ``(messages, tools) -> ChatResult | Message | str`` asked for the next reply."""
    name = "fake"

    def __init__(self, replies: Sequence[Any] | Callable[..., Any] | None = None) -> None:
        super().__init__()
        self._replies = list(replies) if isinstance(replies, (list, tuple)) else None
        self._respond = replies if callable(replies) else None
        self.calls: list[list[Message]] = []
        self._n = 0

    def next_reply(self, messages: Sequence[Message], tools: Sequence[ToolSpec]) -> ChatResult:
        if self._respond is not None:
            r = self._respond(messages, tools)
        elif self._replies:
            r = self._replies[min(self._n, len(self._replies) - 1)]
        else:
            r = "I can help, but no reply was scripted."
        self._n += 1
        if isinstance(r, ChatResult):
            return r
        if isinstance(r, Message):
            return ChatResult(r)
        return ChatResult(Message("assistant", str(r)))

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec], settings: ModelSettings,
             on_delta: Callable[[str], None] | None = None) -> ChatResult:
        self.calls.append(list(messages))
        if self.cancelled:
            raise ProviderError("stopped")
        result = self.next_reply(messages, tools)
        if on_delta and result.message.content:
            on_delta(result.message.content)          # the scripted reply, as one delta: tests the streaming path
        return result


MAX_ATTEMPTS = 3                # a transient network error or a 5xx is retried this many times
RETRY_DELAY = 0.4               # seconds before the first retry; grows linearly


class OpenAIProvider(Provider):
    """Any OpenAI-style ``/v1/chat/completions`` endpoint, over httpx (included)."""
    name = "openai-compatible"

    def __init__(self, settings: ModelSettings) -> None:
        super().__init__()
        self.settings = settings
        self._active: Any = None                  # the client of a call in flight, so cancel() can close it
        self._active_lock = threading.Lock()

    def cancel(self) -> None:
        """Ask a call to stop; closing the in-flight client aborts a request already waiting on the socket."""
        super().cancel()
        with self._active_lock:
            client = self._active
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001 - closing is best-effort
                pass

    def _client(self, settings: ModelSettings):
        try:
            import httpx
        except ImportError:
            raise ProviderError("Talking to a model needs the 'httpx' package, which is missing from this build. "
                                "Reinstall DANCR (or, in a source checkout, run 'uv sync').") from None
        return httpx.Client(timeout=settings.timeout)          # the call's settings, not the constructor's

    @staticmethod
    def _raise_status(resp: Any) -> None:
        """A non-2xx reply as the right error. Call ``resp.read()`` first for a streamed response."""
        code = resp.status_code
        if code in (401, 403):
            raise ProviderError("The model rejected the key. Check the key in the Assistant settings")
        if code == 402:
            raise ProviderPaused("The key is out of funds. Add funds or use your own key to continue")
        if code == 429:
            raise ProviderPaused("The model is rate-limited right now. Wait a moment and try again")
        if code >= 400:
            raise ProviderError(f"The model returned an error ({code}): {_short(resp.text)}")

    def chat(self, messages: Sequence[Message], tools: Sequence[ToolSpec], settings: ModelSettings,
             on_delta: Callable[[str], None] | None = None) -> ChatResult:
        import time

        import httpx
        if self.cancelled:
            raise ProviderError("stopped")
        body: dict[str, Any] = {
            "model": settings.model,
            "messages": [m.to_wire() for m in messages],
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_tokens": settings.max_tokens,
        }
        if tools:
            body["tools"] = [t.to_wire() for t in tools]
            body["tool_choice"] = "auto"
        if on_delta is not None:
            body["stream"] = True
        url = settings.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {settings.api_key}", "Content-Type": "application/json"}
        client = self._client(settings)
        with self._active_lock:
            self._active = client
        try:
            for attempt in range(MAX_ATTEMPTS):
                if self.cancelled:
                    raise ProviderError("stopped")
                state = {"emitted": False}          # did anything stream out? never retry after partial text
                try:
                    if on_delta is not None:
                        with client.stream("POST", url, json=body, headers=headers) as resp:
                            if resp.status_code >= 500 and attempt < MAX_ATTEMPTS - 1:
                                resp.read()
                                time.sleep(RETRY_DELAY * (attempt + 1))   # a transient server error: try again
                                continue
                            if resp.status_code >= 400:
                                resp.read()
                                self._raise_status(resp)
                            if "event-stream" not in resp.headers.get("content-type", "").lower():
                                resp.read()                  # a server that ignored `stream` and sent one JSON reply
                                try:
                                    return _parse(resp.json())
                                except ValueError as e:
                                    raise ProviderError(f"The model returned a reply that is not JSON: "
                                                        f"{_short(resp.text)}") from e
                            return _consume_stream(resp, on_delta, state)
                    resp = client.post(url, json=body, headers=headers)
                    if resp.status_code >= 500 and attempt < MAX_ATTEMPTS - 1:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    self._raise_status(resp)
                    try:
                        data = resp.json()
                    except ValueError as e:
                        raise ProviderError(f"The model returned a reply that is not JSON: {_short(resp.text)}") from e
                    return _parse(data)
                except (httpx.HTTPError, RuntimeError) as e:     # a closed client raises RuntimeError
                    if self.cancelled:
                        raise ProviderError("stopped") from e
                    if attempt < MAX_ATTEMPTS - 1 and not state["emitted"]:
                        time.sleep(RETRY_DELAY * (attempt + 1))
                        continue
                    if isinstance(e, httpx.TimeoutException):
                        raise ProviderError("The model took too long to answer. Try again, or raise the timeout in "
                                            "settings") from e
                    raise ProviderError(f"Could not reach the model at {settings.base_url}: {e}") from e
            raise ProviderError(f"Could not reach the model at {settings.base_url}")     # the attempts ran out
        finally:
            with self._active_lock:
                self._active = None
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass


def _parse(data: dict[str, Any]) -> ChatResult:
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError("The model returned no reply")
    choice = choices[0]
    msg = choice.get("message") or {}
    calls = [_tool_call(tc.get("id") or f"call_{i}", (tc.get("function") or {}).get("name"),
                        (tc.get("function") or {}).get("arguments") or "{}")
             for i, tc in enumerate(msg.get("tool_calls") or [])]
    message = Message("assistant", str(msg.get("content") or ""), calls)
    u = data.get("usage") or {}
    details = u.get("prompt_tokens_details") or {}
    usage = Usage(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0),
                  int(u.get("total_tokens") or 0), int(details.get("cached_tokens") or 0))
    return ChatResult(message, usage, str(choice.get("finish_reason") or ""))


def _tool_call(call_id: Any, name: Any, raw: Any) -> ToolCall:
    if not isinstance(raw, str):
        raw = json.dumps(raw)
    try:
        args = json.loads(raw)
        if not isinstance(args, dict):
            args = {"value": args}
    except (ValueError, TypeError):
        args = {}
    return ToolCall(id=str(call_id), name=str(name or ""), arguments=args, raw_arguments=raw)


def _consume_stream(resp: Any, on_delta: Callable[[str], None], state: dict[str, bool]) -> ChatResult:
    """Read an OpenAI-style SSE stream: content deltas go to ``on_delta`` as they arrive; the tool calls,
    finish reason and usage (if the server sends it) are assembled into the one reply the session expects."""
    content: list[str] = []
    slots: dict[int, dict[str, str]] = {}          # tool-call index -> {id, name, args}
    finish = ""
    usage = Usage()
    for line in resp.iter_lines():
        line = line.strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except ValueError:
            continue
        u = chunk.get("usage")
        if isinstance(u, dict):
            details = u.get("prompt_tokens_details") or {}
            usage = Usage(int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0),
                          int(u.get("total_tokens") or 0), int(details.get("cached_tokens") or 0))
        choices = chunk.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        if choice.get("finish_reason"):
            finish = str(choice["finish_reason"])
        delta = choice.get("delta") or {}
        piece = delta.get("content")
        if piece:
            content.append(str(piece)); state["emitted"] = True; on_delta(str(piece))
        for tc in delta.get("tool_calls") or []:
            slot = slots.setdefault(int(tc.get("index", 0) or 0), {"id": "", "name": "", "args": ""})
            if tc.get("id"):
                slot["id"] = str(tc["id"])
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["name"] = str(fn["name"])
            if fn.get("arguments"):
                slot["args"] += str(fn["arguments"])
            state["emitted"] = True
    if not content and not slots and not finish:
        raise ProviderError("The model returned no reply")
    calls = [_tool_call(slots[i]["id"] or f"call_{i}", slots[i]["name"], slots[i]["args"] or "{}")
             for i in sorted(slots)]
    return ChatResult(Message("assistant", "".join(content), calls), usage, finish)


def _short(text: str, limit: int = 300) -> str:
    t = " ".join(str(text or "").split())
    return t[:limit] + ("…" if len(t) > limit else "")


def provider_for(settings: ModelSettings, provider: Provider | None = None) -> Provider:
    """The provider to use: one passed in (tests, a window with a fake), else an OpenAI-style client.

    A fake is used automatically when ``DANCR_ASSISTANT_FAKE`` is set, so the window can be driven with no key."""
    if provider is not None:
        return provider
    if os.environ.get("DANCR_ASSISTANT_FAKE"):
        return FakeProvider()
    return OpenAIProvider(settings)
