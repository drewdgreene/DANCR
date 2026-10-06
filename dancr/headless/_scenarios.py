"""Run a project across many scenarios (roadmap F2).

A scenario is a named set of Input values. Each scenario runs on a private clone of
the project (the project file is never changed), with the existing cache, leases
and write confinement, so a scenario that recomputes one input does not disturb the
others. Results are one output per scenario plus a combined table (a `scenario`
column), and every scenario records its plan hash and output hash — the evidence
block that :func:`dancr.core.verify.build_attestation` folds in and
:func:`verify_pipeline` checks. Deterministic: the same spec and data give the same
scenarios and the same hashes.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import polars as pl

from ..core import Pipeline
from ..core.executor import Executor
from ..core.model import Pipeline as _Pipeline
from ..core.registry import in_dancr_folder
from ..core.scenarios import from_spec
from ..core.verify import output_hash
from ._safety import unsafe_write
from ._atomic import write_text_atomic

SCENARIO_EXT = ("csv", "parquet", "xlsx")


def scenario_set(spec: dict[str, Any] | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A scenario spec (or an explicit list) as the list of ``{"id", "inputs"}`` the runner uses."""
    scenarios = from_spec(spec) if isinstance(spec, dict) else list(spec or [])
    ids = [str(s.get("id")) for s in scenarios]
    if len(set(ids)) != len(ids):
        raise ValueError("Scenario ids must be unique")
    if not scenarios:
        raise ValueError("No scenarios to run")
    return scenarios


def run_scenarios(p: Pipeline, spec: dict[str, Any] | list[dict[str, Any]], *, target: str | None = None,
                  out_dir: Path | str = "scenarios", ext: str = "csv", jobs: int = 1, force: bool = False,
                  combined: bool = True, manifest: Path | str | None = None, on_event: Any = None) -> dict[str, Any]:
    """Run ``target`` once per scenario, writing an output each and a combined table. Returns a record whose
    ``evidence`` block is the per-scenario plan/output hashes for an attestation. Writes stay inside the project
    folder, never into ``.dancr`` and never over a source file."""
    from datetime import datetime
    from ..core.nodes.outputs import write_table
    emit = on_event or (lambda e: None)
    folder = p.directory.resolve()
    scenarios = scenario_set(spec)
    if not target:
        order = p.topological_order()
        if not order:
            raise ValueError("The project has no steps to run")
        target = order[-1]
    if target not in p.nodes:
        raise ValueError(f"No step called {target!r}. Steps: {list(p.nodes)}")
    ext = str(ext or "csv").lower().lstrip(".")
    if ext not in SCENARIO_EXT:
        raise ValueError(f"Save as one of: {', '.join(SCENARIO_EXT)}")
    destination = Path(out_dir).expanduser()
    destination = (destination if destination.is_absolute() else folder / destination).resolve()
    if not destination.is_relative_to(folder):
        raise ValueError(f"Scenario results must be written inside the project folder {folder}, not {destination}")
    if in_dancr_folder(destination, folder):
        raise ValueError(f"Won't write scenario results into DANCR's own .dancr folder: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    def one(sc: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
        sid = str(sc["id"])
        clone = _Pipeline.from_dict(copy.deepcopy(p.to_dict()), p.path)
        for name, value in (sc.get("inputs") or {}).items():
            clone.set_input(name, value)
        ex = Executor(clone)
        st = ex.run(targets=[target], force=force, sweep=False).get(target)
        rec: dict[str, Any] = {"id": sid, "inputs": dict(sc.get("inputs") or {}), "status": (st.status if st else "idle"),
                               "plan_hash": ex.safe_hash(target)}
        if st is not None and st.status == "done" and st.output:
            out_file = destination / f"{sid}.{ext}"
            why = unsafe_write(clone, out_file, folder)
            if why:
                rec.update(status="failed", error=why)
                emit({"type": "scenario", "id": sid, "status": "failed"})
                return rec, None
            write_table(pl.scan_parquet(st.output), out_file)
            rec.update(rows=st.rows, output=str(out_file), output_hash=output_hash(ex, target))
            emit({"type": "scenario", "id": sid, "status": "done"})
            return rec, st.output
        rec["error"] = st.error if st is not None else "not run"
        emit({"type": "scenario", "id": sid, "status": rec["status"]})
        return rec, None

    results: list[dict[str, Any]] = []
    ready: list[tuple[str, str]] = []
    if int(jobs or 1) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=int(jobs)) as pool:
            for rec, out in pool.map(one, scenarios):
                results.append(rec)
                if out:
                    ready.append((rec["id"], out))
    else:
        for sc in scenarios:
            rec, out = one(sc)
            results.append(rec)
            if out:
                ready.append((rec["id"], out))
    results.sort(key=lambda r: r["id"])
    ready.sort(key=lambda pair: pair[0])

    combined_path: Path | None = None
    if combined and ready:
        frames = [pl.scan_parquet(out).with_columns(pl.lit(sid, dtype=pl.Utf8).alias("scenario"))
                  for sid, out in ready]
        candidate = destination / f"combined.{ext}"
        if not unsafe_write(p, candidate, folder):
            write_table(pl.concat(frames, how="diagonal_relaxed"), candidate)
            combined_path = candidate

    evidence = {"kind": "dancr.scenarios", "version": 1,
                "scenarios": [{"id": r["id"], "inputs": r["inputs"], "plan_hash": r.get("plan_hash"),
                               "output_hash": r.get("output_hash"), "rows": r.get("rows"),
                               "status": r["status"]} for r in results]}
    record = {
        "kind": "dancr.scenarios", "version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "project": {"name": p.name, "file": p.path.name if p.path else None},
        "target": target, "out_dir": str(destination), "ext": ext, "count": len(results),
        "ok": all(r["status"] == "done" for r in results), "scenarios": results,
        "combined": str(combined_path) if combined_path else None, "evidence": evidence,
    }
    if manifest is not None:
        from ..core.fair import dump as _dump_json
        write_text_atomic(manifest, _dump_json(record))
        record["manifest"] = str(manifest)
    return record
