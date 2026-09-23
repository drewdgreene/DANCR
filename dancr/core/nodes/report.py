"""Report: several charts and tables on one page (HTML, plus a PDF) for sending to a colleague."""
from __future__ import annotations

import base64
import html

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
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


def _item_html(i: int, lf: pl.LazyFrame, meta: dict[str, Any], params: dict[str, Any], columns: dict[str, dict] | None) -> list[str]:
    from ...views.render import render_chart
    from ...views.stats import column_summary
    max_rows = int(params.get("max_rows") or 30)
    parts: list[str] = []
    heading = html.escape(meta.get("title") or f"Item {i + 1}")
    parts.append(f"<h2>{heading}</h2>")
    rep = meta.get("report") or {}
    if meta.get("node_type") == "check_limits" and rep.get("verdict"):
        cls = "pass" if rep["verdict"] == "PASS" else "fail"
        parts.append(f"<p class='verdict {cls}'>{rep['verdict']}: {rep.get('outside', 0):,} of {rep.get('rows', 0):,} rows outside {html.escape(str(rep.get('limit', '')))}</p>")
    for m in meta.get("messages") or []:
        parts.append(f"<p class='muted'>{html.escape(str(m))}</p>")
    shown = {k: v for k, v in rep.items() if k not in ("fits", "parameters") and not isinstance(v, (list, dict))}
    if shown:
        parts.append("<table class='kv'>" + "".join(f"<tr><th>{html.escape(k.replace('_', ' '))}</th><td>{_fmt(v)}</td></tr>" for k, v in shown.items()) + "</table>")
    if meta.get("node_type") == "chart":
        with tempfile.TemporaryDirectory() as td:
            png = Path(td) / "c.png"
            try:
                render_chart(lf, meta.get("params") or {}, png, width=1400, height=620, columns=columns)
                data = base64.b64encode(png.read_bytes()).decode()
                parts.append(f"<img src='data:image/png;base64,{data}' alt='{heading}'>")
            except Exception as e:
                parts.append(f"<p class='error'>Could not draw this chart: {html.escape(str(e))}</p>")
    else:
        n = int(lf.select(pl.len()).collect(engine="streaming")[0, 0])
        df = strip_time_zones(lf.head(max_rows).collect(engine="streaming"))
        if columns:
            df = df.rename({c: _title(c, columns) for c in df.columns if _title(c, columns) != c})
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
    m = (columns or {}).get(c) or {}
    label = m.get("label") or c
    return f"{label} ({m['unit']})" if m.get("unit") else label


def build_report(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any], item_meta: list[dict[str, Any]],
                 columns: dict[str, dict] | None = None) -> str:
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
    verdicts = [(m.get("title"), (m.get("report") or {}).get("verdict")) for m in item_meta if (m.get("report") or {}).get("verdict")]
    if verdicts:
        overall = "PASS" if all(v == "PASS" for _, v in verdicts) else "FAIL"
        parts.append(f"<p class='verdict {'pass' if overall == 'PASS' else 'fail'}'>Overall: {overall}</p>")
    blocks = params.get("blocks") or [{"type": "item", "index": i} for i in range(len(frames))]
    used = set()
    for b in blocks:
        if b.get("type") == "text":
            txt = str(b.get("text") or "")
            parts.append("<div class='text'>" + "".join(f"<p>{html.escape(line)}</p>" for line in txt.splitlines() if line.strip()) + "</div>")
        elif b.get("type") == "heading":
            parts.append(f"<h2>{html.escape(str(b.get('text') or ''))}</h2>")
        else:
            try:
                i = int(b.get("index", -1))
            except (TypeError, ValueError):
                continue
            if 0 <= i < len(frames):
                used.add(i)
                parts += _item_html(i, frames[i], item_meta[i] if i < len(item_meta) else {}, params, columns)
    for i in range(len(frames)):
        if i not in used:
            parts += _item_html(i, frames[i], item_meta[i] if i < len(item_meta) else {}, params, columns)
    css = """
    .company { font-size: 13px; letter-spacing: 1px; text-transform: uppercase; color: #555; margin-bottom: 0; }
    .verdict { font-weight: 700; padding: 6px 10px; border-radius: 4px; display: inline-block; }
    .verdict.pass { background: #dcfce7; color: #166534; } .verdict.fail { background: #fee2e2; color: #991b1b; }
    .text p { margin: 6px 0; }
    body { font-family: -apple-system, 'Segoe UI', 'Adwaita Sans', 'Noto Sans', Helvetica, Arial, sans-serif; color: #1c1c1e; max-width: 1100px; margin: 32px auto; padding: 0 24px; line-height: 1.45; }
    h1 { font-size: 26px; margin-bottom: 4px; } h2 { font-size: 18px; margin-top: 36px; border-bottom: 1px solid #ddd; padding-bottom: 4px; }
    h3 { font-size: 14px; color: #555; } .muted { color: #6b6b73; font-size: 13px; } .error { color: #b91c1c; }
    .notes { background: #f6f6f8; border-left: 3px solid #c9c9cf; padding: 8px 14px; margin: 12px 0; }
    img { max-width: 100%; height: auto; border: 1px solid #e6e6ea; }
    table { border-collapse: collapse; font-size: 12.5px; margin: 8px 0; } th, td { border: 1px solid #e3e3e8; padding: 4px 8px; text-align: right; }
    th { background: #f2f2f4; text-align: left; } td:first-child, th:first-child { text-align: left; }
    table.kv th { width: 200px; } @media print { body { margin: 0; } h2 { page-break-before: auto; } img { page-break-inside: avoid; } }
    """
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{css}</style></head><body>{''.join(parts)}</body></html>"


def _apply(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    frames = inputs.get("items") or []
    if not frames:
        raise ValueError("Connect the charts and tables you want in the report")
    path = (params.get("path") or "").strip()
    if not path:
        raise ValueError("Choose where to save the report (a .html file)")
    out = ctx.resolve(path)
    if out.suffix.lower() not in (".html", ".htm"):
        raise ValueError("Save the report as a .html file (it opens in any browser and prints to PDF)")
    if ctx.preview:
        return NodeResult(frames[0], messages=[f"Will write {out.name} when the pipeline runs"])
    meta = getattr(ctx, "item_meta", None) or []
    doc = build_report(ctx, inputs, params, meta, getattr(ctx, "columns", None))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f".{out.name}.tmp")
    try:
        tmp.write_text(doc, encoding="utf-8")
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)
    msgs = [f"Saved report to {out}"]
    rep: dict[str, Any] = {"path": str(out), "items": len(frames)}
    if params.get("pdf", True):
        from ...views.pdf import html_to_pdf
        try:
            pdf = html_to_pdf(doc, out.with_suffix(".pdf"))
            msgs.append(f"PDF: {pdf}")
            rep["pdf"] = str(pdf)
        except Exception as e:  # PDF is a convenience; never fail the report for it
            ctx.logger.warning("PDF not written for %s: %s", out, e)
            msgs.append(f"PDF not written ({e})")
    return NodeResult(frames[0], messages=msgs, report=rep)


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
    ],
))
