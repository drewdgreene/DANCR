"""Report: several charts and tables on one page (HTML, plus a PDF) for sending to a colleague."""
from __future__ import annotations

import os

import base64
import html

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ._common import private_temp
from ..registry import NodeType, InputSpec, Ctx, NodeResult, registry
from ..dtypes import strip_time_zones


def _fmt(v: Any) -> str:
    from ...views.table import format_value
    return html.escape(format_value(v))


def _table_html(df: pl.DataFrame, max_rows: int, total: int | None = None) -> str:
    head = "".join(f"<th>{html.escape(c)}</th>" for c in df.columns)
    rows = []
    for row in df.head(max_rows).iter_rows():
        rows.append("<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in row) + "</tr>")
    total = df.height if total is None else total
    more = f"<p class='muted'>Showing the first {max_rows:,} of {total:,} rows.</p>" if total > max_rows else ""
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>{more}"


def _item_html(i: int, lf: pl.LazyFrame, meta: dict[str, Any], params: dict[str, Any], columns: dict[str, dict] | None,
               project_inputs: dict[str, Any] | None = None) -> list[str]:
    from ...views.render import render_chart
    from ...views.stats import column_summary
    max_rows = int(params.get("max_rows") or 30)
    parts: list[str] = []
    heading = html.escape(meta.get("title") or f"Item {i + 1}")
    parts.append(f"<h2>{heading}</h2>")
    rep = meta.get("report") or {}
    if meta.get("node_type") == "check_limits" and rep.get("verdict"):
        cls = {"PASS": "pass", "FAIL": "fail"}.get(rep["verdict"], "none")
        what = (f"{rep.get('outside', 0):,} of {rep.get('checked', 0):,} values outside" if cls != "none"
                else "no row has a value to check against")
        parts.append(f"<p class='verdict {cls}'>{rep['verdict']}: {what} {html.escape(str(rep.get('limit', '')))}</p>")
    for m in meta.get("messages") or []:
        parts.append(f"<p class='muted'>{html.escape(str(m))}</p>")
    shown = {k: v for k, v in rep.items() if k not in ("fits", "parameters") and not isinstance(v, (list, dict))}
    if shown:
        parts.append("<table class='kv'>" + "".join(f"<tr><th>{html.escape(k.replace('_', ' '))}</th><td>{_fmt(v)}</td></tr>" for k, v in shown.items()) + "</table>")
    node_type = meta.get("node_type")
    if node_type in ("chart", "map"):
        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / "c.png"
            try:
                if node_type == "map":
                    from ...views.render import render_map
                    render_map(lf, meta.get("params") or {}, png, width=1400, height=800, columns=columns, inputs=project_inputs)
                    alt = "map"
                else:
                    render_chart(lf, meta.get("params") or {}, png, width=1400, height=620, columns=columns, inputs=project_inputs)
                    alt = "chart"
                data = base64.b64encode(png.read_bytes()).decode()
                parts.append(f"<img src='data:image/png;base64,{data}' alt='{html.escape(alt)}'>")
            except Exception as e:
                parts.append(f"<p class='error'>Could not draw this {node_type}: {html.escape(str(e))}</p>")
    else:
        n = int(lf.select(pl.len()).collect(engine="streaming")[0, 0])
        df = strip_time_zones(lf.head(max_rows).collect(engine="streaming"))
        if columns:
            df = df.rename(_display_names(df.columns, columns))
        parts.append(f"<p class='muted'>{n:,} rows × {len(df.columns)} columns</p>")
        parts.append(_table_html(df, max_rows, n))
        if n > max_rows and params.get("include_stats", True):
            try:
                parts.append("<h3>Summary statistics</h3>")
                parts.append(_table_html(column_summary(lf), 100))
            except Exception:
                pass
    return parts


def _title(c: str, columns: dict[str, dict] | None) -> str:
    from ...views.table import column_title
    return column_title(c, columns)


def _provenance_html(item_meta: list[dict[str, Any]]) -> str:
    """A short, honest provenance footer: the steps the report drew on, the source files behind them, the
    tool that made it, and how to re-check it. Full hashes come from ``dancr verify`` / ``dancr proof``; this
    block never claims more than it can show."""
    from dancr import __version__
    steps: list[str] = []
    sources: list[str] = []
    for m in item_meta or []:
        title = str(m.get("title") or m.get("node") or "").strip()
        nt = str(m.get("node_type") or "").strip()
        if title:
            steps.append(f"{title} ({nt})" if nt else title)
        p = (m.get("params") or {}).get("path")
        if isinstance(p, str) and p.strip():
            sources.append(Path(p).name)
    bits = ["<h3>Provenance</h3>"]
    bits.append(f"<p class='muted'>Made with DANCR {html.escape(str(__version__))} on {datetime.now():%Y-%m-%d %H:%M}"
                + (f" · {len(steps)} step(s)." if steps else ".") + "</p>")
    if steps:
        bits.append("<p class='muted'>Steps: " + html.escape(", ".join(steps)) + "</p>")
    if sources:
        bits.append("<p class='muted'>Sources: " + html.escape(", ".join(dict.fromkeys(sources))) + "</p>")
    bits.append("<p class='muted'>Re-check this result: <code>dancr verify &lt;project&gt;.json "
                "--manifest &lt;attestation&gt;.json</code></p>")
    return "<div class='provenance'>" + "".join(bits) + "</div>"


def build_report(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any], item_meta: list[dict[str, Any]],
                 columns: dict[str, dict] | None = None, project_inputs: dict[str, Any] | None = None) -> str:
    """Return a self-contained HTML document. item_meta: one dict per input in order with keys
    title, node_type, params (the chart node's params if it is a chart), messages, report.
    `blocks` (optional) orders text and items: [{"type": "text", "text": ...}, {"type": "item", "index": 0}, ...]."""
    title = (params.get("title") or "Report").strip()
    notes = (params.get("notes") or "").strip()
    company = (params.get("company") or "").strip()
    author = (params.get("author") or "").strip()
    parts = []
    if company:
        parts.append(f"<p class='company'>{html.escape(company)}</p>")
    parts.append(f"<h1>{html.escape(title)}</h1>")
    by = f" · {html.escape(author)}" if author else ""
    parts.append(f"<p class='muted'>{datetime.now():%Y-%m-%d %H:%M}{by} · made with DANCR</p>")
    if notes:
        parts.append("<div class='notes'>" + "".join(f"<p>{html.escape(line)}</p>" for line in notes.splitlines() if line.strip()) + "</div>")
    frames = inputs.get("items") or []
    # A block names its item by source node id, which survives connections being reordered or
    # re-added; older projects stored a positional index, so fall back to that.
    node_index = {m.get("node"): i for i, m in enumerate(item_meta) if m.get("node")}
    verdicts = [(m.get("title"), (m.get("report") or {}).get("verdict")) for m in item_meta if (m.get("report") or {}).get("verdict")]
    if verdicts:
        found = {v for _, v in verdicts}
        # a check that had nothing to compare is neither a pass nor a fail
        overall = "FAIL" if "FAIL" in found else ("PASS" if found == {"PASS"} else "NOT EVERYTHING CHECKED")
        cls = {"PASS": "pass", "FAIL": "fail"}.get(overall, "none")
        parts.append(f"<p class='verdict {cls}'>Overall: {overall}</p>")
    blocks = params.get("blocks") or [{"type": "item", "index": i} for i in range(len(frames))]
    used = set()
    for b in blocks:
        if b.get("type") == "text":
            txt = str(b.get("text") or "")
            parts.append("<div class='text'>" + "".join(f"<p>{html.escape(line)}</p>" for line in txt.splitlines() if line.strip()) + "</div>")
        elif b.get("type") == "heading":
            parts.append(f"<h2>{html.escape(str(b.get('text') or ''))}</h2>")
        else:
            i = node_index.get(b.get("node")) if b.get("node") is not None else None
            if i is None:
                try:
                    i = int(b.get("index", -1))
                except (TypeError, ValueError):
                    continue
            if i is not None and 0 <= i < len(frames):
                used.add(i)
                parts += _item_html(i, frames[i], item_meta[i] if i < len(item_meta) else {}, params, columns, project_inputs)
    for i in range(len(frames)):
        if i not in used:
            parts += _item_html(i, frames[i], item_meta[i] if i < len(item_meta) else {}, params, columns, project_inputs)
    if params.get("include_proof", False):
        parts.append(_provenance_html(item_meta))
    css = """
    .company { font-size: 13px; letter-spacing: 1px; text-transform: uppercase; color: #555; margin-bottom: 0; }
    .verdict { font-weight: 700; padding: 6px 10px; border-radius: 4px; display: inline-block; }
    .verdict.pass { background: #dcfce7; color: #166534; } .verdict.fail { background: #fee2e2; color: #991b1b; }
    .verdict.none { background: #f1f5f9; color: #334155; }
    .text p { margin: 6px 0; }
    body { font-family: -apple-system, 'Segoe UI', 'Adwaita Sans', 'Noto Sans', Helvetica, Arial, sans-serif; color: #1c1c1e; max-width: 1100px; margin: 32px auto; padding: 0 24px; line-height: 1.45; }
    h1 { font-size: 26px; margin-bottom: 4px; } h2 { font-size: 18px; margin-top: 36px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
    h3 { font-size: 14px; color: #555; } .muted { color: #6b6b73; font-size: 13px; } .error { color: #b91c1c; }
    .notes { background: #f6f6f8; border-left: 3px solid #c9c9cf; padding: 8px 14px; margin: 12px 0; }
    img { max-width: 100%; height: auto; border: 1px solid #e6e6ea; }
    table { border-collapse: collapse; font-size: 12.5px; margin: 8px 0; } th, td { border: 1px solid #e3e3e8; padding: 4px 8px; text-align: right; }
    th { background: #f2f2f4; text-align: left; } td:first-child, th:first-child { text-align: left; }
    table.kv th { width: 200px; } @media print { body { margin: 0; } h2 { page-break-before: auto; } img { page-break-inside: avoid; } }
    .provenance { margin-top: 40px; border-top: 1px solid #ddd; padding-top: 8px; } .provenance h3 { font-size: 13px; color: #555; margin: 8px 0 4px; }
    .provenance code { background: #f2f2f4; padding: 1px 4px; border-radius: 3px; }
    """
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>{''.join(parts)}</body></html>"


def _apply(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    frames = inputs.get("items") or []
    if not frames:
        raise ValueError("Connect the charts and tables you want in the report")
    path = (params.get("path") or "").strip()
    if not path:
        raise ValueError("Choose where to save the report (a .html file)")
    out = ctx.resolve_output(path)
    if out.suffix.lower() not in (".html", ".htm"):
        raise ValueError("Save the report as a .html file (it opens in any browser and prints to PDF)")
    if ctx.preview:
        return NodeResult(frames[0], messages=[f"Will write {out.name} when the project runs"])
    meta = getattr(ctx, "item_meta", None) or []
    doc = build_report(ctx, inputs, params, meta, getattr(ctx, "columns", None), ctx.inputs)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = private_temp(out)
    try:
        with open(tmp, "x", encoding="utf-8") as f:          # "x": never write through a file (or link) already there
            f.write(doc)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    msgs = [f"Saved report to {out}"]
    rep: dict[str, Any] = {"path": str(out), "items": len(frames)}
    files = [out]
    if params.get("pdf", True):
        from ...views.pdf import html_to_pdf
        try:
            target = ctx.resolve_output(str(out.with_suffix(".pdf")))     # the PDF is a file the step writes too
            tmp_pdf = private_temp(target)
            try:
                html_to_pdf(doc, tmp_pdf)
                os.replace(tmp_pdf, target)
            finally:
                tmp_pdf.unlink(missing_ok=True)
            pdf = target
            msgs.append(f"PDF: {pdf}")
            rep["pdf"] = str(pdf)
            files.append(Path(pdf))
        except Exception as e:  # PDF is a convenience; never fail the report for it
            ctx.logger.warning("PDF not written for %s: %s", out, e)
            msgs.append(f"PDF not written ({e})")
    return NodeResult(frames[0], messages=msgs, report=rep, files=files)


def _display_names(names: list[str], columns: dict) -> dict[str, str]:
    """Column headings by display label; two columns with the same label keep their names beside it."""
    titles = {c: _title(c, columns) for c in names}
    counts: dict[str, int] = {}
    for t in titles.values():
        counts[t] = counts.get(t, 0) + 1
    out = {}
    for c, t in titles.items():
        if counts[t] > 1 and t != c:
            t = f"{t} [{c}]"
        if t != c:
            out[c] = t
    return out


registry.register(NodeType(
    key="report", label="Report", category="Share", icon="▣",
    description="Put several charts and tables on one page to send to a colleague. Saves an HTML file that opens in any browser and prints to PDF.",
    apply=_apply,
    kind="sink",
    materialize=False,
    inputs=[InputSpec("items", "Charts and tables", multiple=True)],
    summary=lambda p: (p.get("title") or "untitled report") + (f" → {Path(p['path']).name}" if p.get("path") else ""),
    params=[
        Param("title", "Title", "text", default="", required=True, placeholder="e.g. Monthly summary, June 2024"),
        Param("path", "Save as", "path", required=True, help="An .html file (a PDF is saved next to it)"),
        Param("notes", "Introduction", "text", default="", help="A few lines shown at the top of the report"),
        Param("company", "Company or organisation (letterhead)", "text", default=""),
        Param("author", "Prepared by", "text", default=""),
        Param("blocks", "Layout", "blocks", default=[], advanced=True, help="Order of text and connected items"),
        Param("pdf", "Also save a PDF", "bool", default=True, advanced=True),
        Param("max_rows", "Rows to show per table", "int", default=30, min=5, max=500, advanced=True),
        Param("include_stats", "Add summary statistics under long tables", "bool", default=True, advanced=True),
        Param("include_proof", "Add a provenance footer (how to re-check the report)", "bool", default=True, advanced=True),
    ],
))
