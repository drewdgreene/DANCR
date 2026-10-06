"""Attestations and verification: a portable, checkable record that a result reproduces.

An **attestation** is a self-contained JSON record of a project's exact state: the engine and library
versions, the code fingerprint, each source file's size and content sample, and, for every step, its plan
hash, the content hash of its materialised output, its row count, schema, and the numbers its report found.
Re-running the project and comparing the same fields gives a verdict, so a claim can be checked by a third
party without trusting the machine that made it.

This reuses the run manifest (``fair.manifest``) for the provenance half and the executor's plan hashes and
cache for the rest. Pure core, no Qt: the command line, the MCP server and the window all call this.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

from .dtypes import json_safe
from .executor import CODE_FINGERPRINT, Executor
from .fair._common import ATTESTATION_KIND, ATTESTATION_VERSION, now as _now, version as _version
from .fair.manifest import run_manifest
from .registry import registry


# ----------------------------------------------------------------- small helpers
def canonical(obj: Any) -> str:
    """A stable JSON text for hashing: keys sorted, no incidental whitespace, non-JSON values stringified."""
    return json.dumps(json_safe(obj), sort_keys=True, default=str, ensure_ascii=False, separators=(",", ":"))


def _hash12(obj: Any) -> str:
    return hashlib.sha1(canonical(obj).encode("utf-8")).hexdigest()[:12]


def _finite(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _close(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


# ----------------------------------------------------------------- output hash
def output_hash(executor: Executor, node_id: str) -> str | None:
    """A stable content hash of a step's materialised result, or None for a step that stores no table.

    Order-independent (the sum of per-row struct hashes) and streaming (never materialises the table), so it
    costs one aggregate pass over the result no matter how large. Schema, row count and the aggregate together
    pin the content: any change to a value, a type or the number of rows moves the hash."""
    st = executor.state(node_id)
    if st.status != "done" or not st.output:
        return None
    try:
        lf = pl.scan_parquet(st.output)
        schema = [[n, str(t)] for n, t in lf.collect_schema().items()]
        rows = st.rows
        if rows is None:
            rows = int(lf.select(pl.len()).collect(engine="streaming")[0, 0])
        acc = 0
        if schema:
            acc = int(lf.select(pl.struct(pl.all()).hash(seed=0).sum()).collect(engine="streaming")[0, 0])
        return _hash12({"schema": schema, "rows": int(rows), "acc": str(acc)})
    except Exception:  # noqa: BLE001 - a result that cannot be hashed is simply not verified by content
        return None


# ----------------------------------------------------------------- report digest
def flatten_report(report: dict[str, Any] | None, _depth: int = 0, _prefix: str = "") -> dict[str, Any]:
    """A step's report as a flat ``{path: scalar}`` map, for comparing two runs with a tolerance.

    Lists are indexed (capped), so a fit table, a list of group results or a set of statistics all compare
    element by element. Only finite numbers, booleans and short strings are kept; the finding's sentence is
    stored separately by :func:`build_attestation`."""
    out: dict[str, Any] = {}
    if _depth > 8 or report is None:
        return out
    if isinstance(report, dict):
        for k, v in report.items():
            out.update(flatten_report(v, _depth + 1, f"{_prefix}.{k}" if _prefix else str(k)))
    elif isinstance(report, (list, tuple)):
        for i, v in enumerate(report[:500]):
            out.update(flatten_report(v, _depth + 1, f"{_prefix}[{i}]"))
    elif isinstance(report, bool):
        out[_prefix] = report
    elif isinstance(report, (int, float)):
        f = _finite(report)
        if f is not None:
            out[_prefix] = f
    elif isinstance(report, str) and len(report) <= 300:
        out[_prefix] = report
    return out


def _finding(report: dict[str, Any] | None) -> str:
    return str(((report or {}).get("finding") or {}).get("statement") or "")


# ----------------------------------------------------------------- attestation
def build_attestation(pipe, executor: Executor | None = None, *, states: dict[str, Any] | None = None,
                      hash_outputs: bool = True, run: bool = True) -> dict[str, Any]:
    """A complete, self-contained record of the project's current state (see the module docstring).

    A materialised step that has not been computed is run first (so its content can be hashed), exactly as the
    knowledge-base export computes a table when statistics are asked for. ``states`` lets a caller that just ran
    the project reuse its ``NodeState`` map (and skips that run). ``hash_outputs=False`` skips the content hash
    of each result (a cheaper record that still checks plan hashes and findings)."""
    ex = executor if executor is not None else Executor(pipe)
    if states is None:
        if run:
            try:
                need = [nid for nid in pipe.topological_order()
                        if registry.get(pipe.nodes[nid].type).materialize and ex.state(nid).status != "done"]
                if need:
                    ex.run(targets=need)
            except Exception:  # noqa: BLE001 - a failing step is recorded as it is, not raised here
                pass
        states = ex.states()
    man = run_manifest(pipe, ex, states=states)

    for entry in man["nodes"]:
        nid = entry["id"]
        node = pipe.nodes[nid]
        nt = registry.get(node.type)
        st = states.get(nid) or ex.state(nid)
        entry["materialize"] = bool(nt.materialize)
        entry["output_hash"] = (output_hash(ex, nid) if hash_outputs and nt.materialize else None)
        entry["report"] = flatten_report(st.report)
        entry["finding"] = _finding(st.report)

    answers = []
    for a in pipe.answers:
        answers.append({"id": a.id, "title": a.title, "terminal": a.terminal,
                        "spec": a.spec, "spec_hash": _hash12(a.spec)})

    att: dict[str, Any] = {**man, "kind": ATTESTATION_KIND, "version": ATTESTATION_VERSION, "answers": answers}
    att["attestation_hash"] = _hash12({k: v for k, v in att.items() if k not in ("attestation_hash", "generated_at")})
    return json_safe(att)


def attestation_hash(attestation: dict[str, Any]) -> str:
    """The recorded hash of an attestation, or one computed from its content (ignoring the volatile fields)."""
    recorded = attestation.get("attestation_hash")
    if isinstance(recorded, str) and recorded:
        return recorded
    return _hash12({k: v for k, v in attestation.items() if k not in ("attestation_hash", "generated_at")})


def dump_attestation(attestation: dict[str, Any]) -> str:
    """An attestation as the JSON text written to a file."""
    return json.dumps(json_safe(attestation), indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n"


def load_attestation(reference: Any) -> dict[str, Any]:
    """An attestation from a dict, a path to a JSON file, or the JSON text itself. A run manifest is accepted
    too: it verifies on plan hashes and findings, without the output content hashes."""
    if isinstance(reference, dict):
        return reference
    text = ""
    if isinstance(reference, (str, Path)):
        p = Path(str(reference)).expanduser()
        if p.is_file():
            text = p.read_text(encoding="utf-8")
        else:
            text = str(reference)
    try:
        data = json.loads(text)
    except ValueError as e:
        raise ValueError(f"That is not an attestation or a run manifest (not valid JSON): {e}") from e
    if not isinstance(data, dict):
        raise ValueError("An attestation is a JSON object, a JSON file, or its text")
    return data


# ----------------------------------------------------------------- verification
def _check(scope: str, name: str, expected: Any, actual: Any, ok: bool | None, node_id: str | None = None) -> dict[str, Any]:
    c: dict[str, Any] = {"scope": scope, "name": name, "match": ok, "expected": expected, "actual": actual}
    if node_id is not None:
        c["id"] = node_id
    return c


def _diff_reports(expected: dict[str, Any], actual: dict[str, Any], tol: float) -> list[dict[str, Any]]:
    diffs: list[dict[str, Any]] = []
    for k in sorted(set(expected) | set(actual)):
        a, b = expected.get(k), actual.get(k)
        if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
            if not _close(float(a), float(b), tol):
                diffs.append({"path": k, "expected": a, "actual": b})
        elif a != b:
            diffs.append({"path": k, "expected": a, "actual": b})
    return diffs


def verify_pipeline(pipe, reference: Any, *, mode: str = "stored", strict_sources: bool = False,
                    tol: float = 1e-9, hash_outputs: bool = True, executor: Executor | None = None) -> dict[str, Any]:
    """Re-run a project and compare it to an attestation (or run manifest).

    ``mode`` ``stored`` computes only steps not in the cache; ``rerun`` recomputes everything (the strong
    claim). ``strict_sources`` turns a changed source file into a mismatch rather than a notice. Returns the
    verify report: a list of checks, the mismatches, notices, and a verdict
    (``verified`` | ``mismatch`` | ``incomplete`` | ``engine-changed``)."""
    if mode not in ("stored", "rerun"):
        raise ValueError("mode must be 'stored' or 'rerun'")
    ref = load_attestation(reference)
    ex = executor if executor is not None else Executor(pipe)

    checks: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    notices: list[dict[str, Any]] = []
    incomplete: list[str] = []

    ref_engine = ref.get("engine") or {}
    ref_fp = ref_engine.get("fingerprint")
    engine_changed = bool(ref_fp) and ref_fp != CODE_FINGERPRINT
    checks.append(_check("engine", "code_fingerprint", ref_fp, CODE_FINGERPRINT, (ref_fp == CODE_FINGERPRINT)))
    if engine_changed:
        notices.append({"scope": "engine", "message": "the DANCR code or a compute library changed since this "
                        "attestation; plan hashes are expected to differ. Re-baseline to compare data again."})

    # ---- compute what is needed
    ref_nodes = {str(n.get("id")): n for n in ref.get("nodes") or [] if isinstance(n, dict) and n.get("id")}
    targets = [nid for nid in ref_nodes if nid in pipe.nodes and ref_nodes[nid].get("materialize", True)]
    try:
        if mode == "rerun":
            ex.run(targets=targets or None, force=True)
        else:
            need = [nid for nid in targets if ex.state(nid).status != "done"]
            if need:
                ex.run(targets=need)
    except Exception as e:  # noqa: BLE001 - a failing run is reported as a notice, not raised
        notices.append({"scope": "run", "message": str(e)})
    states = ex.states()

    cur = build_attestation(pipe, ex, states=states, hash_outputs=hash_outputs)
    checks.append(_check("engine", "engine_version", ref_engine.get("version"), _version(), None))
    cur_nodes = {n["id"]: n for n in cur["nodes"]}
    cur_sources: dict[str, dict[str, Any]] = {}
    for s in cur["sources"]:
        for f in s.get("files") or []:
            cur_sources[f.get("path")] = f

    # ---- sources
    ref_sources: dict[str, dict[str, Any]] = {}
    for s in ref.get("sources") or []:
        for f in s.get("files") or []:
            ref_sources[f.get("path")] = f
    for path, rf in ref_sources.items():
        cf = cur_sources.get(path)
        if cf is None:
            incomplete.append(f"source missing: {path}")
            checks.append(_check("source", path, rf.get("sample"), None, False))
            continue
        if cf.get("missing"):
            incomplete.append(f"source unreadable: {path}")
            checks.append(_check("source", path, rf.get("sample"), None, False))
            continue
        same = (rf.get("size") == cf.get("size") and rf.get("sample") == cf.get("sample"))
        checks.append(_check("source", path, rf.get("sample"), cf.get("sample"), same))
        if not same:
            item = {"scope": "source", "name": path, "message": "the source file changed (size or content sample)"}
            (mismatches if strict_sources else notices).append(item)
    for path in cur_sources:
        if path not in ref_sources:
            notices.append({"scope": "source", "name": path, "message": "a source file not in the attestation"})

    # ---- nodes
    if not engine_changed:
        for nid, rn in ref_nodes.items():
            cn = cur_nodes.get(nid)
            if cn is None:
                incomplete.append(f"step missing: {nid}")
                checks.append(_check("node", "present", True, False, False, nid))
                continue
            same_plan = rn.get("hash") == cn.get("hash")
            checks.append(_check("node", "plan_hash", rn.get("hash"), cn.get("hash"), same_plan, nid))
            if not same_plan:
                mismatches.append({"scope": "node", "name": "plan_hash", "id": nid,
                                   "expected": rn.get("hash"), "actual": cn.get("hash")})
            # an attestation records a content hash per result; a plain run manifest does not, so only compare
            # it when the reference carries one (a missing key means an older record, not a changed result)
            if hash_outputs and "output_hash" in rn and rn.get("materialize", True):
                same_out = rn.get("output_hash") == cn.get("output_hash")
                checks.append(_check("node", "output_hash", rn.get("output_hash"), cn.get("output_hash"), same_out, nid))
                if not same_out and rn.get("output_hash") and cn.get("output_hash"):
                    mismatches.append({"scope": "node", "name": "output_hash", "id": nid,
                                       "expected": rn.get("output_hash"), "actual": cn.get("output_hash")})
            same_rows = rn.get("rows") == cn.get("rows")
            checks.append(_check("node", "rows", rn.get("rows"), cn.get("rows"), same_rows, nid))
            # reports/findings exist only in an attestation; a manifest records none, so skip that comparison
            if "report" in rn:
                diffs = _diff_reports(rn.get("report") or {}, cn.get("report") or {}, tol)
                checks.append(_check("finding", "report", None, None, not diffs, nid))
                if diffs:
                    mismatches.append({"scope": "finding", "id": nid, "diffs": diffs[:50],
                                       "expected": rn.get("finding"), "actual": cn.get("finding")})
    else:
        for nid in ref_nodes:
            checks.append(_check("node", "plan_hash", ref_nodes[nid].get("hash"), None, None, nid))

    # ---- answers
    ref_answers = {a.get("id"): a for a in ref.get("answers") or [] if isinstance(a, dict)}
    cur_answers = {a.get("id"): a for a in cur.get("answers") or []}
    for aid, ra in ref_answers.items():
        ca = cur_answers.get(aid)
        same = ca is not None and ra.get("spec_hash") == ca.get("spec_hash")
        checks.append(_check("answer", "spec_hash", ra.get("spec_hash"), (ca or {}).get("spec_hash"), same, aid))
        if not same:
            mismatches.append({"scope": "answer", "name": "spec_hash", "id": aid,
                               "expected": ra.get("spec_hash"), "actual": (ca or {}).get("spec_hash")})

    if incomplete:
        verdict = "incomplete"
    elif mismatches:
        verdict = "mismatch"
    elif engine_changed:
        verdict = "engine-changed"
    else:
        verdict = "verified"

    matched = sum(1 for c in checks if c.get("match") is True)
    return json_safe({
        "kind": "dancr.verify",
        "version": 1,
        "ok": verdict == "verified",
        "verdict": verdict,
        "mode": mode,
        "reference": {"attestation_hash": attestation_hash(ref),
                      "attestation_hash_actual": cur.get("attestation_hash")},
        "checks": checks,
        "mismatches": mismatches,
        "notices": notices,
        "incomplete": incomplete,
        "summary": f"{len(checks)} checks: {matched} match, {len(mismatches)} mismatch"
                   + (f", {len(incomplete)} missing" if incomplete else ""),
    })
