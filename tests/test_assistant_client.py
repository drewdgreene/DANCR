"""The Assistant's model client: wire format, parsing, error mapping, and the scripted fake."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from dancr.core.assistant.client import (
    FakeProvider,
    Message,
    ModelSettings,
    OpenAIProvider,
    ProviderError,
    ProviderPaused,
    ToolCall,
    ToolSpec,
    Usage,
    provider_for,
)


def test_message_wire_format_for_tool_calls():
    m = Message("assistant", "", [ToolCall("c1", "get_stats", {"node": "src"})])
    w = m.to_wire()
    assert w["role"] == "assistant"
    assert w["tool_calls"][0]["function"]["name"] == "get_stats"
    assert json.loads(w["tool_calls"][0]["function"]["arguments"]) == {"node": "src"}
    tool = Message("tool", "ok", tool_call_id="c1").to_wire()
    assert tool["tool_call_id"] == "c1" and tool["role"] == "tool"


def test_fake_provider_scripts_and_records():
    p = FakeProvider(["hello", Message("assistant", "second")])
    r1 = p.chat([Message("user", "hi")], [], ModelSettings())
    r2 = p.chat([Message("user", "again")], [], ModelSettings())
    assert r1.message.content == "hello"
    assert r2.message.content == "second"
    assert len(p.calls) == 2


def _server(payload: dict, status: int = 200):
    """A tiny in-process OpenAI-style endpoint; returns (base_url, received-bodies list)."""
    seen: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            seen.append(json.loads(self.rfile.read(n) or b"{}"))
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    return base, seen, httpd


REPLY = {"choices": [{"finish_reason": "tool_calls", "message": {
    "role": "assistant", "content": None,
    "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "get_stats", "arguments": "{\"node\": \"src\"}"}}]}}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
              "prompt_tokens_details": {"cached_tokens": 6}}}


def test_openai_provider_parses_reply_and_sends_tools():
    base, seen, httpd = _server(REPLY)
    try:
        s = ModelSettings(base_url=base, model="m", api_key="k")
        r = OpenAIProvider(s).chat([Message("user", "hi")], [ToolSpec("get_stats", "stats", {"type": "object"})], s)
        assert r.message.tool_calls[0].name == "get_stats"
        assert r.message.tool_calls[0].arguments == {"node": "src"}
        assert r.usage == Usage(10, 4, 14, 6)
        assert seen[0]["model"] == "m" and seen[0]["tools"][0]["function"]["name"] == "get_stats"
    finally:
        httpd.shutdown()


def test_openai_provider_maps_bad_key():
    base, _, httpd = _server({"error": "no"}, status=401)
    try:
        with pytest.raises(ProviderError) as e:
            OpenAIProvider(ModelSettings(base_url=base, model="m", api_key="k")).chat([], [], ModelSettings(base_url=base, model="m", api_key="k"))
        assert "key" in str(e.value).lower()
    finally:
        httpd.shutdown()


def test_openai_provider_pauses_when_out_of_funds():
    base, _, httpd = _server({"error": "payment"}, status=402)
    try:
        with pytest.raises(ProviderPaused):
            OpenAIProvider(ModelSettings(base_url=base, model="m", api_key="k")).chat([], [], ModelSettings(base_url=base, model="m", api_key="k"))
    finally:
        httpd.shutdown()


def test_openai_provider_pauses_when_rate_limited():
    base, _, httpd = _server({"error": "slow down"}, status=429)
    try:
        with pytest.raises(ProviderPaused):
            OpenAIProvider(ModelSettings(base_url=base, model="m", api_key="k")).chat([], [], ModelSettings(base_url=base, model="m", api_key="k"))
    finally:
        httpd.shutdown()


def test_openai_provider_retries_a_transient_5xx():
    seen: list[int] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            seen.append(len(seen))
            status = 500 if len(seen) == 1 else 200
            body = json.dumps(REPLY).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    try:
        s = ModelSettings(base_url=base, model="m", api_key="k")
        r = OpenAIProvider(s).chat([Message("user", "hi")], [], s)
        assert r.message.tool_calls[0].name == "get_stats" and len(seen) == 2
    finally:
        httpd.shutdown()


def test_openai_provider_streams_text_and_assembles_a_tool_call():
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a): pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"choices": [{"delta": {"content": "Hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1",
                                                        "function": {"name": "get_stats", "arguments": "{\"node\""}}]}}]},
                {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ": \"src\"}"}}]},
                              "finish_reason": "tool_calls"}]},
            ]
            for ch in chunks:
                self.wfile.write(b"data: " + json.dumps(ch).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    try:
        s = ModelSettings(base_url=base, model="m", api_key="k")
        pieces: list[str] = []
        r = OpenAIProvider(s).chat([Message("user", "hi")], [ToolSpec("get_stats", "x", {"type": "object"})], s,
                                  on_delta=pieces.append)
        assert "".join(pieces) == "Hello" and r.message.content == "Hello"
        assert r.message.tool_calls[0].name == "get_stats"
        assert r.message.tool_calls[0].arguments == {"node": "src"}
        assert r.finish_reason == "tool_calls"
    finally:
        httpd.shutdown()


def test_openai_provider_falls_back_when_a_server_ignores_stream():
    base, _seen, httpd = _server(REPLY)          # replies as application/json, ignoring stream=True
    try:
        s = ModelSettings(base_url=base, model="m", api_key="k")
        pieces: list[str] = []
        r = OpenAIProvider(s).chat([Message("user", "hi")], [], s, on_delta=pieces.append)
        assert r.message.tool_calls[0].name == "get_stats"
        assert pieces == []                      # it did not stream, but the reply still arrived
    finally:
        httpd.shutdown()


def test_openai_provider_stops_when_cancelled():
    p = OpenAIProvider(ModelSettings(base_url="http://127.0.0.1:9/v1", model="m", api_key="k"))
    p.cancel()
    with pytest.raises(ProviderError) as e:
        p.chat([Message("user", "hi")], [], p.settings)
    assert "stopped" in str(e.value).lower()


def test_clean_error_redacts_a_connection_secret():
    from dancr.core.assistant.tools import _clean_error
    assert "****" in _clean_error(Exception("could not connect to postgres://u:hunter2@host/db"))


def test_provider_for_uses_a_passed_provider(monkeypatch):
    fake = FakeProvider(["x"])
    assert provider_for(ModelSettings(), fake) is fake


def test_provider_for_fake_env(monkeypatch):
    monkeypatch.setenv("DANCR_ASSISTANT_FAKE", "1")
    assert isinstance(provider_for(ModelSettings()), FakeProvider)


def test_settings_from_env(monkeypatch):
    monkeypatch.setenv("DANCR_ASSISTANT_API_KEY", "abc")
    monkeypatch.setenv("DANCR_ASSISTANT_MODEL", "custom")
    s = ModelSettings.from_env()
    assert s.api_key == "abc" and s.model == "custom" and s.configured
