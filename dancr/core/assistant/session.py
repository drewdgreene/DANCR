"""One conversation with the Assistant.

A turn is: send the project profile plus the thread to the model; the model may call read tools (which run
against the engine) and ends by proposing an answer spec, a list of steps, or a plain reply. The session
does not change the project — it returns a :class:`AssistantReply` the window applies. Every number in a
reply is checked against what the tools actually returned; anything unbacked is flagged, never presented as
fact.
"""
from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from ..understand import DataModel
from .client import ChatResult, Message, ModelSettings, OpenAIProvider, Provider, ProviderError, ProviderPaused, Usage
from .context import data_block, profile_json, project_profile
from .prompts import system_message
from .store import Thread, Turn
from .tools import ToolRunner, tool_result_text

log = logging.getLogger("dancr.assistant")

MAX_ROUNDS = 12                 # model -> tools rounds in one turn
MAX_TOOL_CALLS = 40
TOOL_REPEAT_LIMIT = 2           # the same call, same arguments: after this many, nudge instead of running it
TOOL_NAME_LIMIT = 8             # any one tool, this many times in a turn, then nudge
MAX_PRIOR_TURNS = 12
MAX_REPLY = 4000

_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


@dataclass
class AssistantReply:
    text: str = ""
    kind: str = "text"                       # text | answer | answers | steps | choice | edits | error | paused
    proposal: dict[str, Any] | None = None
    flags: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)   # the exact figures no tool result backed
    allowed: list[str] = field(default_factory=list)      # figures this turn's tools/profile did back
    usage: dict[str, int] = field(default_factory=dict)
    tool_calls: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.kind not in ("error", "paused")


class AssistantSession:
    """A conversation against one snapshot of a project. Rebuild it when the project changes."""

    def __init__(self, pipeline, executor, model: DataModel, provider: Provider, settings: ModelSettings, *,
                 allow_samples: bool = False, focus: str | None = None, thread: Thread | None = None,
                 run: bool = True, max_rounds: int = MAX_ROUNDS, max_tool_calls: int = MAX_TOOL_CALLS) -> None:
        self.pipeline = pipeline
        self.executor = executor
        self.model = model
        self.provider = provider
        self.settings = settings
        self.focus = focus
        self.thread = thread if thread is not None else Thread()
        self.max_rounds = max_rounds
        self.max_tool_calls = max_tool_calls
        self.runner = ToolRunner(pipeline, executor, model, allow_samples=allow_samples, run=run)
        self.usage = Usage()
        self.last_connections: list[dict[str, Any]] = []
        self.session_cost = 0.0

    # ---------------------------------------------------------------- context
    def profile(self) -> dict[str, Any]:
        return project_profile(self.pipeline, self.model, node=self.focus)

    def _messages(self, user_text: str) -> list[Message]:
        msgs = [Message("system", system_message()),
                Message("user", data_block("project_profile", profile_json(self.profile())))]
        for turn in self.thread.turns[-MAX_PRIOR_TURNS:]:
            if turn.role == "user":
                msgs.append(Message("user", turn.text))
            elif turn.role == "assistant" and (turn.text or turn.finding):
                extra = []
                if turn.finding:
                    extra.append(f"[engine finding: {turn.finding}]")
                if turn.answer:
                    extra.append(f"[built answer {turn.answer}]")
                msgs.append(Message("assistant", (turn.text + "\n" + " ".join(extra)).strip()))
        msgs.append(Message("user", user_text))
        return msgs

    # ---------------------------------------------------------------- the turn
    def turn(self, user_text: str, *, cancel: threading.Event | None = None,
             on_event: Callable[[dict[str, Any]], None] | None = None) -> AssistantReply:
        """Ask the model, run whatever read tools it calls, and return what it proposes (or its reply).

        ``on_event`` is called from the working thread with small dicts so a window can show what is happening:
        ``{"status": text}`` and ``{"tool": name}``."""
        user_text = str(user_text or "").strip()
        if not user_text:
            return AssistantReply(kind="text", text="")
        if isinstance(self.provider, OpenAIProvider) and not self.settings.configured:
            return AssistantReply(kind="paused", text="No model key is set. Open the Assistant settings and add a key.")
        messages = self._messages(user_text)
        schemas = self.runner.schemas()
        # What counts as a source for a figure: what the engine's tools returned this turn, plus the engine's
        # own statistics in the profile (counts, ranges, distinct) — but not the profile's category *values*,
        # which are data, so a number copied from a cell is still unverified.
        # Engine statistics back a figure; so do the person's own words (restating what they wrote is not a
        # claim by the model). What the model itself wrote — a tool argument echoed back, a proposal reply or
        # title — never counts, or a made-up number could be laundered through a tool call.
        allowed_texts: list[str] = [_profile_stats_text(self.profile()), user_text]
        allowed_texts += [t.text for t in self.thread.turns[-MAX_PRIOR_TURNS:] if t.role == "user"]
        prior = self._prior_allowed()
        called: list[str] = []
        proposal: dict[str, Any] | None = None
        connections: list[dict[str, Any]] = []
        final_text = ""
        finish = ""
        extra_flags: list[str] = []
        seen: dict[str, int] = {}          # name+arguments -> how many times this turn
        by_name: dict[str, int] = {}       # tool name -> how many times this turn
        emit = on_event or (lambda _s: None)
        try:
            for _round in range(self.max_rounds):
                if cancel is not None and cancel.is_set():
                    self.provider.cancel()
                    return AssistantReply(kind="paused", text="Stopped.", flags=["stopped"], usage=self._usage())
                emit({"status": "Thinking" if _round else "Reviewing what the engine returned"})
                result: ChatResult = self.provider.chat(
                    messages, schemas, self.settings, on_delta=lambda piece: emit({"delta": piece}))
                self.usage.add(result.usage)
                finish = result.finish_reason or finish
                msg = result.message
                messages.append(msg)
                if not msg.tool_calls:
                    final_text = msg.content or ""
                    break
                if len(called) + len(msg.tool_calls) > self.max_tool_calls:
                    final_text = ("I used up the checks allowed in one turn before finishing. Ask me to carry on, "
                                  "or tell me which part to do first.")
                    extra_flags.append("budget")
                    break
                executed = 0
                for call in msg.tool_calls:
                    called.append(call.name)
                    sig = call.name + json.dumps(call.arguments, sort_keys=True, default=str)
                    seen[sig] = seen.get(sig, 0) + 1
                    by_name[call.name] = by_name.get(call.name, 0) + 1
                    if seen[sig] > TOOL_REPEAT_LIMIT or by_name[call.name] > TOOL_NAME_LIMIT:
                        emit({"status": "Nudging a repeated check"})
                        messages.append(Message("tool", tool_result_text(call.name, {
                            "note": f"You already called {call.name} with these arguments and have the result above. "
                                    "Do not call it again: act on what you have — call propose or propose_edits, "
                                    "or say in one line what you still need."}), tool_call_id=call.id))
                        continue
                    emit({"tool": call.name})
                    executed += 1
                    outcome = self.runner.call(call.name, call.arguments)
                    text = tool_result_text(call.name, outcome.content)
                    allowed_texts.append(text if outcome.evidence is None else outcome.evidence)
                    messages.append(Message("tool", text, tool_call_id=call.id))
                    if call.name == "list_connections" and isinstance(outcome.content.get("connections"), list):
                        connections = outcome.content["connections"]       # keep the engine's map, to persist
                    if outcome.terminal and outcome.proposal is not None:
                        proposal = outcome.proposal
                if proposal is not None:
                    break
                if executed == 0:
                    # every call this round was a repeat: the model is looping
                    final_text = ("I kept asking the engine the same thing without moving forward. "
                                  "Tell me which step to change, or say “try a different way”.")
                    extra_flags.append("stuck")
                    break
            else:
                final_text = ("I did not finish within the step budget. Ask me to carry on, "
                              "or tell me which part to do first.")
                extra_flags.append("budget")
        except ProviderPaused as e:
            return AssistantReply(kind="paused", text=str(e), usage=self._usage())
        except ProviderError as e:
            return AssistantReply(kind="error", text="", error=str(e), usage=self._usage())

        if finish == "length":
            extra_flags.append("truncated")          # the model was cut off mid-answer
        if connections:
            self.last_connections = connections
        if proposal is not None:
            text = str(proposal.get("reply") or "").strip() or final_text
            kind = "text" if proposal.get("kind") == "text" else str(proposal.get("kind"))
        else:
            text, kind = final_text.strip(), "text"
        text = text[:MAX_REPLY]
        allowed = _allowed_numbers(allowed_texts)
        # The proposal's assumptions and follow-up questions are shown to the person and saved, so check their
        # figures too — a fabricated number must not slip through in a chip.
        checked = text
        if proposal is not None:
            for _key in ("assumptions", "next_questions"):
                for _item in (proposal.get(_key) or []):
                    checked += "\n" + str(_item)
        flags, unverified = _flags_for(checked, allowed, prior)
        return AssistantReply(text=text, kind=kind, proposal=proposal, flags=flags + extra_flags,
                              unverified=unverified, allowed=sorted(allowed)[:800],
                              usage=self._usage(), tool_calls=called)

    def record(self, user_text: str, reply: AssistantReply, *, finding: str = "", node: str | None = None,
               answer: str | None = None, new_user: bool = True) -> None:
        """Keep the turn in the thread (the caller saves the project). ``new_user=False`` replaces an existing
        reply — used by Regenerate, which keeps the user's question and discards the answer it is re-asking."""
        prop = reply.proposal or {}
        if self.last_connections:
            self.thread.connections = self.last_connections
        if new_user:
            self.thread.add(Turn("user", str(user_text)[:MAX_REPLY]))
        self.thread.add(Turn("assistant", reply.text, kind=reply.kind, proposal=reply.proposal,
                             finding=finding, node=node, answer=answer,
                             assumptions=[str(a) for a in (prop.get("assumptions") or [])],
                             next_questions=[str(q) for q in (prop.get("next_questions") or [])],
                             flags=list(reply.flags), unverified=list(reply.unverified),
                             allowed=list(reply.allowed), usage=dict(reply.usage)))

    def synthesize(self, findings: str, *,
                   on_event: Callable[[dict[str, Any]], None] | None = None) -> AssistantReply:
        """A short, plain-language synthesis over the engine's findings after a build: what the data shows and the
        hypotheses it supports. The model writes it, but every figure is checked against the findings text."""
        emit = on_event or (lambda _s: None)
        if isinstance(self.provider, OpenAIProvider) and not self.settings.configured:
            return AssistantReply(kind="paused", text="No model key is set.")
        prompt = ("These analyses were just built and run. The engine's finding for each is below. Write a short "
                  "synthesis for the person: what the data shows and which hypotheses it supports, in three to "
                  "five sentences, then up to three next questions. Use only figures that appear below or in the "
                  "profile.\n\n" + findings)
        messages = self._messages(prompt)
        allowed = [_profile_stats_text(self.profile()), findings]
        allowed += [t.text for t in self.thread.turns[-MAX_PRIOR_TURNS:] if t.role == "user"]
        try:
            emit({"status": "Interpreting the results"})
            result: ChatResult = self.provider.chat(messages, [], self.settings,
                                                    on_delta=lambda piece: emit({"delta": piece}))
        except ProviderPaused as e:
            return AssistantReply(kind="paused", text=str(e), usage=self._usage())
        except ProviderError as e:
            return AssistantReply(kind="error", text="", error=str(e), usage=self._usage())
        self.usage.add(result.usage)
        text = (result.message.content or "").strip()[:MAX_REPLY]
        allowed_set = _allowed_numbers(allowed)
        flags, unverified = _flags_for(text, allowed_set, self._prior_allowed())
        return AssistantReply(text=text, kind="text", flags=flags, unverified=unverified,
                              allowed=sorted(allowed_set)[:800], usage=self._usage())

    def _prior_allowed(self) -> set[str]:
        """The figures earlier turns' tools backed, so a model restating a verified number is not flagged."""
        out: set[str] = set()
        for t in self.thread.turns:
            if t.role == "assistant" and t.allowed:
                out.update(t.allowed)
        return out

    def _usage(self) -> dict[str, int]:
        return {"prompt_tokens": self.usage.prompt_tokens, "completion_tokens": self.usage.completion_tokens,
                "total_tokens": self.usage.total_tokens, "cached_tokens": self.usage.cached_tokens}


def _allowed_numbers(texts: Sequence[str]) -> set[str]:
    """Every number a tool result contained, plus the plain forms a reply may repeat it as
    (50.0 -> "50", 1,234 -> "1234"): a figure counts as backed when it names one of these."""
    out: set[str] = set()
    for t in texts:
        for m in _NUMBER.finditer(t):
            for form in _number_forms(m.group(0)):
                out.add(form)
    return out


def _number_forms(tok: str) -> set[str]:
    forms = {tok, tok.replace(",", "")}
    try:
        f = float(tok.replace(",", ""))
        if f.is_integer():
            forms.add(str(int(f)))
        forms.add(f"{f:g}")
    except (TypeError, ValueError):
        pass
    return forms


_DATEISH = re.compile(r"\d{4}-\d{2}-\d{2}")


def _profile_stats_text(profile: dict[str, Any]) -> str:
    """The profile as text with cell data dropped: the engine's own statistics (counts, ranges, distinct, dates)
    are backed figures, but a value copied from a cell is not. So the category ``values`` are dropped, and a
    text column's ``min``/``max`` (which are lexicographic cell values) too; a number or a date is kept."""
    def strip(o: Any) -> Any:
        if isinstance(o, dict):
            out: dict[str, Any] = {}
            for k, v in o.items():
                if k == "values":
                    continue
                if k in ("min", "max") and isinstance(v, str) and not _DATEISH.search(v):
                    continue
                out[k] = strip(v)
            return out
        if isinstance(o, list):
            return [strip(x) for x in o]
        return o
    return json.dumps(strip(profile), default=str, ensure_ascii=False)


def _flags_for(text: str, allowed: set[str], prior: set[str]) -> tuple[list[str], list[str]]:
    """Split a reply's figures into (flags, the exact unbacked tokens).

    A figure is backed when any of its written forms (``50`` ≡ ``50.0`` ≡ ``1,234``) appears in a tool result,
    the profile's statistics, or an earlier turn. Only a single-digit integer is treated as a structural count
    ("2 tables") rather than a claim about the data; anything larger, or with a decimal or a percent, that no
    tool backed is flagged and named."""
    if not text:
        return [], []
    unverified: list[str] = []
    for m in _NUMBER.finditer(text):
        tok = m.group(0)
        forms = _number_forms(tok)
        if forms & allowed or forms & prior:
            continue
        core = tok.replace(",", "").lstrip("-")
        if "." not in tok and "," not in tok and len(core) <= 1:
            continue
        unverified.append(tok)
    return (["unverified-figure"] if unverified else []), unverified
