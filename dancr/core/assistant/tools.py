"""The Assistant's hands: the tools it may call.

Every tool is a thin wrapper over machinery DANCR already has — the data model, the answer engine, the
executor, the step registry. Nothing here computes a result of its own, so a tool result is always the
engine's own words. The only tool that shapes a reply is :data:`PROPOSE`, and it does not change the
project: it validates a spec or a list of steps and returns what the engine would build (or the error).
The window applies it, undoably, after the person approves.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ..dtypes import json_safe
from ..registry import registry
from ..understand import DataModel, Table
from .client import ToolSpec
from .context import clean, table_profile
from .prompts import answer_reference_text

log = logging.getLogger("dancr.assistant")

PROPOSE = "propose"
ASK_CHOICE = "ask_choice"
MAX_TOOL_CHARS = 24_000          # a tool result longer than this is cut, so one call cannot flood the thread
MAX_LIST = 60


@dataclass
class ToolOutcome:
    content: dict[str, Any]
    terminal: bool = False
    proposal: dict[str, Any] | None = None
    # Text that may back a figure in the reply: the engine-computed part of the result. ``None`` means the whole
    # content; ``""`` means the result computes no data (a parser, a proposal) and must not back any number, so a
    # model cannot launder a made-up figure by echoing it through a tool argument.
    evidence: str | None = None


def _str_list(value: Any) -> list[str]:
    """A model-supplied list of strings, tolerating only an actual list (a bare string is not iterated letter by
    letter into a confusing column list)."""
    if not isinstance(value, (list, tuple)):
        return []
    return [str(v) for v in value if str(v).strip()]


def tool_result_text(name: str, content: dict[str, Any]) -> str:
    """A tool result as the model reads it: JSON, inside a labelled DATA block, capped in length."""
    text = json.dumps(json_safe(content), ensure_ascii=False, default=str)
    if len(text) > MAX_TOOL_CHARS:
        text = text[:MAX_TOOL_CHARS] + "…"
    from .context import data_block
    safe_name = "".join(ch for ch in str(name) if ch.isalnum() or ch in "_-") or "tool"
    return data_block(f'tool_result name="{safe_name}"', text)


@dataclass
class ToolRunner:
    """Answers the model's tool calls against one snapshot of the project."""
    pipe: Any                                   # a Pipeline (a snapshot, never the live document)
    executor: Any                               # an Executor over that snapshot
    model: DataModel
    allow_samples: bool = False
    run: bool = True
    logger: Any = log
    _calls: int = field(default=0, init=False)

    # ---------------------------------------------------------------- schemas
    def schemas(self) -> list[ToolSpec]:
        def spec(name: str, description: str, props: dict[str, Any] | None = None, required: list[str] | None = None) -> ToolSpec:
            return ToolSpec(name, description,
                            {"type": "object", "properties": props or {}, "required": required or []})

        node = {"node": {"type": "string", "description": "the table's step id (see list_tables)"}}
        return [
            spec("list_tables", "The tables in the project and a one-line summary of each (shape, rows, time column, measures, categories)."),
            spec("list_steps", "Every step in the project's map, in order: id, type, the title the person sees, what feeds it and what it feeds, and its status. Use this to rename or retitle steps."),
            spec("describe_table", "Full profile of one table: every column's name, role, type, range, blanks and category values.", node, ["node"]),
            spec("get_stats", "Exact statistics for a table's columns (count, blanks, mean, std, min, quartiles, max). Runs the table if needed.",
                 {**node, "columns": {"type": "array", "items": {"type": "string"}}}, ["node"]),
            spec("get_sample", "A few real rows of a table, to see how values look. Off unless the person allows sample rows.",
                 {**node, "rows": {"type": "integer"}, "columns": {"type": "array", "items": {"type": "string"}}}, ["node"]),
            spec("read_question", "Ask DANCR's deterministic reader to read a question in the person's own words; it returns a spec or why it could not.",
                 {"text": {"type": "string"}}, ["text"]),
            spec("search_knowledge", "Search the project's own text index (a 'Build search index' step) and return the "
                                     "closest passages, each with the source columns it came from. Use it to answer "
                                     "questions from the project's documents and notes. Confidential/restricted "
                                     "passages are withheld.",
                 {**node, "query": {"type": "string"}, "k": {"type": "integer"}}, ["query"]),
            spec("suggest_answers", "The answers DANCR can build for these tables on its own, best first. Each carries a ready spec.",
                 {"focus": {"type": "string", "description": "optional step id to focus on"}}),
            spec("answer_reference", "The answer recipes and the spec keys each accepts."),
            spec("list_node_types", "The catalogue of pipeline steps, optionally filtered by a word.",
                 {"query": {"type": "string"}}),
            spec("formula_reference", "The formula language for calculated columns."),
            spec("node_status", "A step's status: rows, columns, messages, and its finding (fit equation, PASS/FAIL, gap counts…).", node, ["node"]),
            spec("list_connections", "How the tables relate (the engine's own detection): each link, stack or time alignment, with its match percentage and cardinality. More than one link for a pair means the choice is ambiguous.",
                 {"from": {"type": "string", "description": "optional table node id"}, "to": {"type": "string", "description": "optional table node id"}}),
            spec(ASK_CHOICE, "Ask the person to choose, when the data leaves more than one reasonable option (which key links two tables, which column a word means). Never guess; ask.",
                 {"question": {"type": "string", "description": "the choice, in one line"},
                  "options": {"type": "array", "items": {"type": "string"}, "description": "two to six plain options"},
                  "reply": {"type": "string"}}, ["question", "options"]),
            spec(PROPOSE, "Propose what to build. Validates the spec(s) against the engine and returns what it would "
                          "build, or the error. Does not change the project. Call it once when ready. For an open-ended "
                          "request (explore this, find interesting things, find hypotheses), pass several specs in "
                          "`answers` in one call — each becomes its own Answer and chart, so the person gets the whole "
                          "investigation at once.",
                 {"reply": {"type": "string", "description": "one or two plain sentences to the person; no numbers unless a tool result gave one"},
                  "answer": {"type": "object", "description": "a single spec for the deterministic answer engine (preferred)"},
                  "answers": {"type": "array", "items": {"type": "object"},
                              "description": "several specs at once for an open-ended request; each becomes its own Answer and chart (preferred there)"},
                  "steps": {"type": "array", "items": {"type": "object"},
                            "description": "manual steps [{type, title, params, after, port}] when no recipe fits"},
                  "assumptions": {"type": "array", "items": {"type": "string"}},
                  "next_questions": {"type": "array", "items": {"type": "string"}}}),
            spec("propose_edits", "Propose changes to the project itself (not a dataflow answer): rename steps, change a step's settings, rename a column for display, or add/update an Input. One call for the whole batch. Validates against the project and returns what would change, or the error. The window applies it, undoably, after the person approves.",
                 {"reply": {"type": "string", "description": "one plain sentence"},
                  "edits": {"type": "array", "items": {"type": "object"},
                            "description": "each edit is {\"op\": \"rename\", \"node\": id, \"title\": \"…\"} | "
                                           "{\"op\": \"set_params\", \"node\": id, \"params\": {…}} | "
                                           "{\"op\": \"set_input\", \"name\": \"…\", \"value\": …, \"unit\": \"…\"} | "
                                           "{\"op\": \"column_label\", \"column\": name, \"label\": \"…\", \"unit\": \"…\"}"},
                  "assumptions": {"type": "array", "items": {"type": "string"}},
                  "next_questions": {"type": "array", "items": {"type": "string"}}},
                 ["edits"]),
        ]

    # ---------------------------------------------------------------- dispatch
    def call(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        self._calls += 1
        fn = getattr(self, f"_t_{name}", None)
        if fn is None:
            return ToolOutcome({"error": f"No tool called {name!r}", "known": [s.name for s in self.schemas()]})
        try:
            return fn(args or {})
        except Exception as e:  # noqa: BLE001 - an engine error is information for the model, not a crash
            self.logger.debug("assistant tool %s failed", name, exc_info=True)
            return ToolOutcome({"error": _clean_error(e)})

    def _require_table(self, node: Any) -> Table:
        t = self.model.table(str(node or ""))
        if t is None:
            raise ValueError(f"No table called {node!r}. Tables: {list(self.model.tables)}")
        return t

    def _write_problem(self, node_type: Any, params: dict[str, Any]) -> str | None:
        """Why a proposed step's output path is not allowed (outside the project folder, or over a source file),
        or None. Only steps that write files are checked; a source's read path may be anywhere."""
        if getattr(node_type, "kind", None) != "sink":
            return None
        from pathlib import Path as _Path
        from ...headless import _safety
        for prm in node_type.params:
            if prm.kind in ("path", "dir") and str(params.get(prm.name) or "").strip():
                target = _Path(str(params[prm.name]).strip()).expanduser()
                if not target.is_absolute():
                    target = self.pipe.directory / target
                why = _safety.unsafe_write(self.pipe, target, self.pipe.directory)
                if why:
                    return why
        return None

    # ---------------------------------------------------------------- read tools
    def _t_list_tables(self, args: dict[str, Any]) -> ToolOutcome:
        out = []
        for nid, t in self.model.tables.items():
            row: dict[str, Any] = {"node": nid, "title": clean(t.title), "shape": t.shape}
            if t.rows is not None:
                row["rows"] = t.rows
            if t.time:
                row["time"] = clean(t.time)
            row["measures"] = [clean(c.name) for c in t.measures][:MAX_LIST]
            row["categories"] = [clean(c.name) for c in t.categories][:MAX_LIST]
            if t.geo:
                row["place"] = {k: clean(v) for k, v in t.geo.items()}
            out.append(row)
        if self.model.skipped:
            out.append({"not_read": {clean(k): clean(v) for k, v in self.model.skipped.items()}})
        return ToolOutcome({"tables": out})

    def _t_list_steps(self, args: dict[str, Any]) -> ToolOutcome:
        """Every step in the project's map, in order. This is what the person sees on the canvas."""
        order = self.pipe.topological_order()
        out = []
        for nid in order:
            n = self.pipe.nodes[nid]
            try:
                nt = registry.get(n.type)
            except KeyError:
                continue
            st = self.executor.state(nid)
            row: dict[str, Any] = {"id": nid, "type": n.type, "label": nt.label, "category": nt.category,
                                   "title": clean(n.title), "status": st.status}
            ins = self.pipe.inputs_of(nid)
            if ins:
                row["inputs"] = {p: [clean(s) for s in srcs] for p, srcs in ins.items()}
            outs = self.pipe.outputs_of(nid)
            if outs:
                row["feeds"] = [clean(o) for o in outs]
            if st.rows is not None:
                row["rows"] = st.rows
            out.append(row)
        return ToolOutcome({"steps": out[:MAX_LIST], "count": len(out)})

    def _t_describe_table(self, args: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome(table_profile(self._require_table(args.get("node"))))

    def _t_get_stats(self, args: dict[str, Any]) -> ToolOutcome:
        node = str(args.get("node") or "")
        self._require_table(node)                              # a clear error before touching the executor
        from ...views.stats import column_summary
        from ...headless import result_frame
        columns = _str_list(args.get("columns")) or None
        with result_frame(self.pipe, self.executor, node, run=self.run) as lf:
            df = column_summary(lf, columns)
        return ToolOutcome({"node": node, "stats": json_safe(df.to_dicts())})

    def _t_get_sample(self, args: dict[str, Any]) -> ToolOutcome:
        if not self.allow_samples:
            return ToolOutcome({"disabled": "Sample rows are turned off. The person can allow them in the Assistant "
                                             "settings; until then use get_stats and the profile."})
        node = str(args.get("node") or "")
        self._require_table(node)
        rows = max(1, min(int(args.get("rows") or 5), 20))
        cols = _str_list(args.get("columns")) or None
        from ...headless import result_frame, select_columns
        with result_frame(self.pipe, self.executor, node, run=self.run) as lf:
            df = select_columns(lf, cols).head(rows).collect(engine="streaming")
        return ToolOutcome({"node": node, "rows": json_safe(df.to_dicts()),
                            "note": "Real sample rows were sent to the model because sample rows are allowed."})

    def _t_search_knowledge(self, args: dict[str, Any]) -> ToolOutcome:
        """Search the project's own text index. Restricted passages are withheld (never bypassable by the model)."""
        from ..rag import search_knowledge
        query = str(args.get("query") or "").strip()
        if not query:
            return ToolOutcome({"error": "search_knowledge needs a query"})
        node = str(args.get("node") or "") or None
        k = max(1, min(int(args.get("k") or 5), 20))
        try:
            out = search_knowledge(self.pipe, query, node=node, k=k, executor=self.executor)
        except ValueError as e:
            return ToolOutcome({"error": _clean_error(e)})
        return ToolOutcome(out)

    def _t_read_question(self, args: dict[str, Any]) -> ToolOutcome:
        from ..ask import ask
        asked = ask(self.model, str(args.get("text") or ""))
        return ToolOutcome(json_safe(asked.to_dict()), evidence="")   # a parse, not a data fact

    def _t_suggest_answers(self, args: dict[str, Any]) -> ToolOutcome:
        from ..recipes import suggest
        focus = args.get("focus")
        if focus and focus not in self.pipe.nodes:
            raise ValueError(f"No step called {focus!r}")
        sugs = suggest(self.model, str(focus) if focus else None)[:MAX_LIST]
        return ToolOutcome({"suggestions": [s.to_dict() for s in sugs]})

    def _t_answer_reference(self, args: dict[str, Any]) -> ToolOutcome:
        return ToolOutcome({"reference": answer_reference_text()})

    def _t_list_node_types(self, args: dict[str, Any]) -> ToolOutcome:
        q = str(args.get("query") or "").strip().lower()
        types = [t.to_json() for t in registry.all()
                 if not q or q in t.key.lower() or q in t.label.lower() or q in t.category.lower()]
        return ToolOutcome({"node_types": types[:MAX_LIST], "total": len(types)})

    def _t_formula_reference(self, args: dict[str, Any]) -> ToolOutcome:
        from ..expr import function_docs
        return ToolOutcome({"functions": [{"name": n, "doc": d} for n, d in function_docs()]})

    def _t_node_status(self, args: dict[str, Any]) -> ToolOutcome:
        node = str(args.get("node") or "")
        if node not in self.pipe.nodes:
            raise ValueError(f"No step called {node!r}. Steps: {list(self.pipe.nodes)[:40]}")
        from ...headless import node_record
        return ToolOutcome(json_safe(node_record(self.pipe, self.executor.state(node))))

    def _t_list_connections(self, args: dict[str, Any]) -> ToolOutcome:
        want_from, want_to = str(args.get("from") or ""), str(args.get("to") or "")
        out = []
        for r in self.model.relations:
            tables = list(r.tables)
            if want_from and want_from not in tables:
                continue
            if want_to and want_to not in tables:
                continue
            row: dict[str, Any] = {"kind": r.kind, "tables": [clean(t) for t in tables], "why": clean(r.why, 200)}
            for k in ("left_on", "right_on", "cardinality", "tolerance"):
                if getattr(r, k, None):
                    row[k] = clean(getattr(r, k))
            if r.match_pct:
                row["match_pct"] = round(r.match_pct, 1)
            if r.shared:
                row["shared_columns"] = [clean(s) for s in r.shared[:20]]
            out.append(row)
        return ToolOutcome({"connections": out[:MAX_LIST], "count": len(out)})

    # ---------------------------------------------------------------- the proposal
    def _t_ask_choice(self, args: dict[str, Any]) -> ToolOutcome:
        question = clean(args.get("question") or "", 400)
        options = [clean(o, 200) for o in (args.get("options") or []) if str(o).strip()][:6]
        if not question or len(options) < 2:
            return ToolOutcome({"ok": False, "error": "ask_choice needs a question and at least two options"})
        reply = clean(args.get("reply") or question, 1000)
        return ToolOutcome({"ok": True, "asked": question}, terminal=True,
                           proposal={"kind": "choice", "question": question, "options": options, "reply": reply},
                           evidence="")   # the question and options are the model's own words, never evidence

    def _t_propose(self, args: dict[str, Any]) -> ToolOutcome:
        reply = clean(args.get("reply") or "", 2000)
        specs: list[dict[str, Any]] = []
        if isinstance(args.get("answer"), dict) and args["answer"]:
            specs.append(args["answer"])
        for s in (args.get("answers") or [])[:8]:
            if isinstance(s, dict) and s:
                specs.append(s)
        steps = args.get("steps")
        assumptions = [clean(a, 300) for a in (args.get("assumptions") or []) if str(a).strip()][:10]
        nexts = [clean(q, 200) for q in (args.get("next_questions") or []) if str(q).strip()][:5]
        if specs:
            from ..recipes import plan as make_plan
            plans = []
            for i, spec in enumerate(specs):
                try:
                    plans.append(make_plan(self.model, spec))
                except (ValueError, KeyError, TypeError) as e:      # PlanError is a ValueError
                    where = f"answers[{i}]" if len(specs) > 1 else "answer"
                    return ToolOutcome({"ok": False, "error": f"{where}: {_clean_error(e)}",
                                        "hint": "Fix the spec and call propose again, or use read_question on the person's words."})
            built = [{"type": s.type, "title": s.title} for p in plans for s in p.new_steps]
            if len(plans) == 1:
                p = plans[0]
                proposal = {"kind": "answer", "spec": json_safe(p.config), "title": p.title, "view": p.view,
                            "reply": reply, "assumptions": assumptions, "next_questions": nexts,
                            "why": p.why, "steps": built}
                return ToolOutcome({"ok": True, "will_build": built, "title": p.title}, terminal=True,
                                   proposal=proposal, evidence="")
            proposal = {"kind": "answers", "specs": [json_safe(p.config) for p in plans],
                        "titles": [p.title for p in plans], "views": [p.view for p in plans],
                        "reply": reply, "assumptions": assumptions, "next_questions": nexts,
                        "why": [p.why for p in plans]}
            return ToolOutcome({"ok": True, "will_build": [{"title": p.title} for p in plans]}, terminal=True,
                               proposal=proposal, evidence="")   # the specs are the model's, so they back no figure
        if isinstance(steps, list) and steps:
            checked = self._validate_steps(steps)
            if isinstance(checked, dict):                      # an error to hand back
                return ToolOutcome({"ok": False, **checked})
            proposal = {"kind": "steps", "steps": checked, "reply": reply, "assumptions": assumptions, "next_questions": nexts}
            return ToolOutcome({"ok": True, "will_build": [{"type": s["type"], "title": s["title"]} for s in checked]},
                               terminal=True, proposal=proposal, evidence="")
        return ToolOutcome({"ok": True, "text_only": True}, terminal=True,
                           proposal={"kind": "text", "reply": reply, "assumptions": assumptions, "next_questions": nexts},
                           evidence="")

    def _t_propose_edits(self, args: dict[str, Any]) -> ToolOutcome:
        reply = clean(args.get("reply") or "", 2000)
        raw = args.get("edits")
        if not isinstance(raw, list) or not raw:
            return ToolOutcome({"ok": False, "error": "propose_edits needs a non-empty list of edits"})
        checked = self._validate_edits(raw)
        if isinstance(checked, dict):
            return ToolOutcome({"ok": False, **checked})
        assumptions = [clean(a, 300) for a in (args.get("assumptions") or []) if str(a).strip()][:10]
        nexts = [clean(q, 200) for q in (args.get("next_questions") or []) if str(q).strip()][:5]
        proposal = {"kind": "edits", "edits": checked, "reply": reply, "assumptions": assumptions,
                    "next_questions": nexts, "title": "Changes to the project"}
        return ToolOutcome({"ok": True, "will_change": [e["summary"] for e in checked]}, terminal=True, proposal=proposal,
                           evidence="")   # the summaries echo the model's own edit values

    def _validate_edits(self, edits: list[Any]) -> list[dict[str, Any]] | dict[str, Any]:
        """Check a batch of pipeline edits against the project, without changing anything."""
        known = set(self.pipe.nodes)
        columns = {c.name for t in self.model.tables.values() for c in t.columns}
        out: list[dict[str, Any]] = []
        for i, e in enumerate(edits):
            if not isinstance(e, dict):
                return {"error": f"Edit {i} is not an object"}
            op = str(e.get("op") or "")
            if op == "rename":
                nid = str(e.get("node") or "")
                if nid not in known:
                    return {"error": f"Edit {i}: no step called {nid!r}. Steps: {list(known)[:40]}"}
                title = clean(e.get("title") or "", 120)
                if not title:
                    return {"error": f"Edit {i}: a rename needs a title"}
                out.append({"op": "rename", "node": nid, "title": title,
                            "summary": f"{self.pipe.nodes[nid].title} → {title}"})
            elif op == "set_params":
                nid = str(e.get("node") or "")
                if nid not in known:
                    return {"error": f"Edit {i}: no step called {nid!r}"}
                nt = registry.get(self.pipe.nodes[nid].type)
                changes = e.get("params")
                if not isinstance(changes, dict) or not changes:
                    return {"error": f"Edit {i} ({nid}): set_params needs a params object"}
                merged = dict(self.pipe.nodes[nid].params)
                for k, v in changes.items():
                    if k not in {p.name for p in nt.params}:
                        return {"error": f"Edit {i} ({nid}): unknown setting {k!r}"}
                    try:
                        merged[k] = nt.param(k).coerce(v)
                    except ValueError as ex:
                        return {"error": f"Edit {i} ({nid}.{k}): {ex}"}
                why = self._write_problem(nt, merged)
                if why:
                    return {"error": f"Edit {i} ({nid}): {why}"}
                out.append({"op": "set_params", "node": nid, "params": changes,
                            "summary": f"{self.pipe.nodes[nid].title}: " + ", ".join(f"{k}={v}" for k, v in changes.items())})
            elif op == "set_input":
                name = clean(e.get("name") or "", 60)
                if not name:
                    return {"error": f"Edit {i}: set_input needs a name"}
                unit = clean(e.get("unit") or "", 20)
                note = clean(e.get("note") or "", 200)
                try:                                    # validated on a throwaway project, as the window applies it
                    from ..model import Pipeline
                    Pipeline().set_input(name, e.get("value"), unit, note)
                except Exception as ex:  # noqa: BLE001 - a bad input is handed back to the model to fix
                    return {"error": f"Edit {i}: {ex}"}
                out.append({"op": "set_input", "name": name, "value": e.get("value"),
                            "unit": unit, "note": note, "summary": f"input {name} = {e.get('value')}"})
            elif op == "column_label":
                col = str(e.get("column") or "").strip()
                if not col:
                    return {"error": f"Edit {i}: column_label needs a column"}
                if columns and col not in columns:
                    near = [c for c in columns if col.lower() in c.lower()][:5]
                    return {"error": f"Edit {i}: no column called {col!r}", "did_you_mean": near}
                out.append({"op": "column_label", "column": col, "label": clean(e.get("label") or "", 80),
                            "unit": clean(e.get("unit") or "", 20),
                            "summary": f"column {col} → “{e.get('label') or col}”"})
            else:
                return {"error": f"Edit {i}: unknown op {op!r}. Use rename, set_params, set_input or column_label"}
        return out

    def _validate_steps(self, steps: list[Any]) -> list[dict[str, Any]] | dict[str, Any]:
        """Check a hand-built plan against the registry, without changing anything. Returns the cleaned steps,
        or {'error': ...} to hand back to the model."""
        known = set(self.pipe.nodes)
        out: list[dict[str, Any]] = []
        for i, raw in enumerate(steps):
            if not isinstance(raw, dict):
                return {"error": f"Step {i} is not an object"}
            tkey = str(raw.get("type") or "")
            if not registry.has(tkey):
                near = [t.key for t in registry.all() if tkey and tkey[:3] in t.key][:5]
                return {"error": f"Step {i}: unknown step type {tkey!r}", "did_you_mean": near}
            nt = registry.get(tkey)
            try:
                params = nt.normalize_params(raw.get("params") or {}, strict=True)
            except ValueError as e:
                return {"error": f"Step {i} ({tkey}): {e}"}
            why = self._write_problem(nt, params)
            if why:
                return {"error": f"Step {i} ({tkey}): {why}"}
            new_id = str(raw.get("id") or f"assistant_{i + 1}")
            if new_id in known or any(s["id"] == new_id for s in out):
                return {"error": f"Step {i}: id {new_id!r} already exists"}
            after = raw.get("after")
            if after and after not in known and not any(s.get("id") == after for s in out):
                return {"error": f"Step {i} ({tkey}) follows {after!r}, which is neither in the project nor an earlier proposed step"}
            out.append({"type": tkey, "title": clean(raw.get("title") or nt.label, 120), "params": params,
                        "after": str(after) if after else None, "port": raw.get("port"), "id": new_id})
        return out


def _clean_error(e: BaseException) -> str:
    from ..secrets import redact
    msg = redact(str(e)).strip().strip("'\"")        # a connector error may embed a DSN or URL with a secret
    return msg[:600] or type(e).__name__
