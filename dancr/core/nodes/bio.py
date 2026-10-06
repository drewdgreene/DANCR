"""Bio / lab ingestion: read non-tabular scientific files as ordinary DANCR tables.

Sequence, variant, feature, GenBank and PLINK files are not tables, so ``load_file`` refuses them; these
source steps turn each into a table with real semantics — one row per sequence / variant / feature / marker —
which the rest of DANCR can then filter, join, chart and report like any other table. Pure standard library
plus Polars, so they ship in the one install and add no dependency.
"""
from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Any

import polars as pl

from ..params import Param
from ..registry import NodeType, Ctx, NodeResult, registry
from .load import effective_ext


# --------------------------------------------------------------------------- shared
def _path(ctx: Ctx, params: dict[str, Any]) -> Path:
    raw = params.get("path")
    if not raw or not str(raw).strip():
        raise ValueError("Choose a file to load")
    p = ctx.resolve(str(raw))
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    if p.is_dir():
        raise ValueError(f"{p.name} is a folder. Choose a sequence, variant, feature or marker file inside it")
    if p.stat().st_size == 0:
        raise ValueError(f"{p.name} is empty")
    return p


def _text(path: Path) -> list[str]:
    """The file's lines, transparently through a .gz / .bgz. The records are read matching the file's own
    encoding, so a plain text scientific file always reads."""
    if path.suffix.lower() in (".gz", ".bgz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            return fh.read().splitlines()
    return path.read_text(encoding="utf-8", errors="replace").splitlines()


def _frame(cols: dict[str, list], order: list[str]) -> pl.LazyFrame:
    return pl.DataFrame({k: list(cols.get(k, [])) for k in order}).lazy()


def _trim(rows: list[Any], limit: Any) -> list[Any]:
    n = int(limit or 0)
    return rows[:n] if n > 0 else rows


def _num(text: Any) -> float | None:
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _int(text: Any) -> int | None:
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _summary(p: dict[str, Any]) -> str:
    return Path(str(p.get("path") or "")).name or "no file chosen"


# extension -> the bio step that reads it, so a dropped file gets the right loader instead of a refusal
_BIO_NODE_BY_EXT = {
    ".fa": "load_sequences", ".fasta": "load_sequences", ".fna": "load_sequences", ".faa": "load_sequences",
    ".ffn": "load_sequences", ".fq": "load_sequences", ".fastq": "load_sequences",
    ".vcf": "load_variants",
    ".gff": "load_features", ".gff3": "load_features", ".gtf": "load_features", ".bed": "load_features",
    ".gb": "load_genbank", ".gbk": "load_genbank", ".genbank": "load_genbank",
    ".ped": "load_markers", ".bim": "load_markers", ".fam": "load_markers",
}


def bio_node_for(path: str | Path) -> str | None:
    """The bio source step that reads this file, by extension (seeing through .gz), or None if it is not a bio format."""
    return _BIO_NODE_BY_EXT.get(effective_ext(Path(str(path))))


# --------------------------------------------------------------------------- sequences
def _load_sequences(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    path = _path(ctx, params)
    lines = _text(path)
    fmt = str(params.get("format") or "auto")
    if fmt == "auto":
        first = next((ln.strip() for ln in lines if ln.strip()), "")
        fmt = "fastq" if first.startswith("@") else "fasta"
    records: list[tuple[str, str, str]] = []
    if fmt == "fastq":
        i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1
                continue
            if not lines[i].startswith("@"):
                break
            header = lines[i][1:].strip()
            seq = lines[i + 1].strip() if i + 1 < len(lines) else ""
            records.append((*header.partition(" ")[::2], seq))
            i += 4
    else:
        cur_id = cur_desc = None
        seq: list[str] = []
        for ln in lines:
            if ln.startswith(">"):
                if cur_id is not None:
                    records.append((cur_id, cur_desc, "".join(seq)))
                cur_id, _, cur_desc = ln[1:].strip().partition(" ")
                seq = []
            elif cur_id is not None:
                seq.append(ln.strip())
        if cur_id is not None:
            records.append((cur_id, cur_desc, "".join(seq)))
        if not records and any(ln.strip() for ln in lines):
            raise ValueError(f"{path.name} does not look like FASTA (no '>' record lines)")
    records = _trim(records, params.get("limit"))
    ids = [r[0] for r in records]
    seqs = [r[2] for r in records]
    cols = {"id": ids, "description": [r[1] for r in records],
            "length": [len(s) for s in seqs],
            "gc_percent": [round(100.0 * sum(s.upper().count(b) for b in "GC") / len(s), 1) if s else None for s in seqs]}
    order = ["id", "description", "length", "gc_percent"]
    if bool(params.get("include_sequence", True)):
        cols["sequence"] = seqs
        order.append("sequence")
    label = "FASTQ" if fmt == "fastq" else "FASTA"
    return NodeResult(_frame(cols, order), report={"format": fmt, "sequences": len(ids)},
                      messages=[f"Read {len(ids):,} {label} sequence{'s' if len(ids) != 1 else ''}"])


registry.register(NodeType(
    key="load_sequences", label="Load sequences (FASTA/FASTQ)", category="Get data", icon="dna",
    kind="source", inputs=[],
    description="Read a FASTA or FASTQ file as one row per sequence: id, description, length, GC%, and the sequence itself.",
    apply=_load_sequences, summary=_summary,
    params=[
        Param("path", "File", "path", required=True, help="A FASTA (.fa, .fasta, .fna) or FASTQ (.fq, .fastq) file"),
        Param("format", "Format", "choice", default="auto", choices=[("auto", "detect"), ("fasta", "FASTA"), ("fastq", "FASTQ")]),
        Param("include_sequence", "Keep the sequence column", "bool", default=True, advanced=True),
        Param("limit", "Read at most this many sequences", "int", default=0, min=0, advanced=True, help="0 = all"),
    ],
))


# --------------------------------------------------------------------------- variants
def _parse_info(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for kv in text.split(";"):
        if not kv:
            continue
        k, _, v = kv.partition("=")
        out[k] = _coerce(v) if v != "" else True
    return out


def _coerce(v: str) -> Any:
    """An INFO value as a number when it is one (DP=20, AF=0.5), else the text."""
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _dosage(gt: Any) -> int | None:
    """The number of non-reference alleles in a genotype ('0/1'->1, '1|1'->2, './.'->None)."""
    if not isinstance(gt, str) or gt in ("", "."):
        return None
    alleles = re.split(r"[/|]", gt)
    n = 0
    for a in alleles:
        if a in ("", "."):
            return None
        if a != "0":
            n += 1
    return n


def _load_variants(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    path = _path(ctx, params)
    lines = _text(path)
    info_on = bool(params.get("info", True))
    layout = str(params.get("samples") or "none")          # none | genotype | full
    limit = int(params.get("limit") or 0)
    samples: list[str] = []
    info_keys: list[str] = []
    fmt_keys: list[str] = []
    rows: list[dict[str, Any]] = []
    variants = 0
    for ln in lines:
        if ln.startswith("##"):
            continue
        if ln.startswith("#"):
            parts = ln.lstrip("#").split("\t")
            if parts and parts[0].upper() == "CHROM" and len(parts) > 9:
                samples = parts[9:]
            continue
        if not ln.strip():
            continue
        f = ln.split("\t")
        if len(f) < 8:
            continue
        base: dict[str, Any] = {
            "chrom": f[0], "pos": _int(f[1]), "id": f[2] if f[2] not in ("", ".") else None,
            "ref": f[3], "alt": f[4], "qual": _num(f[5]), "filter": f[6] if f[6] not in ("", ".") else None,
        }
        info = _parse_info(f[7]) if (info_on and f[7] not in ("", ".")) else {}
        for k in info:
            if k not in info_keys:
                info_keys.append(k)
        if layout == "none" or not samples:
            rows.append({**base, **info})
        else:
            fmt = f[8].split(":") if len(f) > 8 else []
            keys = ["gt"] if layout == "genotype" else [k.lower() for k in fmt]
            for key in keys:
                if key not in fmt_keys:
                    fmt_keys.append(key)
            if "dosage" not in fmt_keys:
                fmt_keys.append("dosage")
            for si, sample in enumerate(samples):
                call = f[9 + si].split(":") if len(f) > 9 + si else []
                rec = {**base, **info, "sample": sample}
                for ki, key in enumerate(fmt):
                    rec[key.lower()] = call[ki] if ki < len(call) else None
                rec["dosage"] = _dosage(rec.get("gt"))
                rows.append(rec)
        variants += 1
        if limit and variants >= limit:
            break
    if layout == "none":
        order = ["chrom", "pos", "id", "ref", "alt", "qual", "filter", *info_keys]
    else:
        order = ["chrom", "pos", "id", "ref", "alt", "qual", "filter", *info_keys, "sample", *fmt_keys]
    cols = {k: [r.get(k) for r in rows] for k in order}
    if not rows:                                            # keep the header shape even with no records
        rows = []
    msg = (f"Read {variants:,} variant{'s' if variants != 1 else ''}"
           + (f" in long form ({len(rows):,} variant×sample rows, {len(samples)} samples)" if layout != "none" and samples else ""))
    return NodeResult(_frame(cols, order), report={"variants": variants, "samples": samples, "info_fields": info_keys},
                      messages=[msg])


registry.register(NodeType(
    key="load_variants", label="Load variants (VCF)", category="Get data", icon="table",
    kind="source", inputs=[],
    description="Read a VCF as one row per variant: chrom, pos, id, ref, alt, qual, filter, INFO fields split into "
                "columns, and — with Samples set — genotype rows (sample, gt, …).",
    apply=_load_variants, summary=_summary,
    params=[
        Param("path", "File", "path", required=True, help="A .vcf (or .vcf.gz unzipped) variant file"),
        Param("info", "Split INFO into columns", "bool", default=True),
        Param("samples", "Genotypes", "choice", default="none",
              choices=[("none", "one row per variant"), ("genotype", "one row per variant and sample (GT)"),
                       ("full", "one row per variant and sample (every FORMAT field)")]),
        Param("limit", "Read at most this many variants", "int", default=0, min=0, advanced=True, help="0 = all"),
    ],
))


# --------------------------------------------------------------------------- features (GFF/GTF/BED)
def _gff_attrs(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if re.search(r'\S+\s+"', text):                        # GTF: key "value"; key2 "value2";
        for m in re.finditer(r'(\w+)\s+"([^"]*)"', text):
            out[m.group(1)] = m.group(2)
    else:                                                  # GFF3: key=value;key2=value2
        for kv in text.split(";"):
            if not kv.strip():
                continue
            k, _, v = kv.partition("=")
            out[k.strip()] = v.strip()
    return out


def _load_features(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    path = _path(ctx, params)
    lines = _text(path)
    fmt = str(params.get("format") or "auto")
    if fmt == "auto":
        base = effective_ext(path)
        if base == ".bed":
            fmt = "bed"
        elif base in (".gff", ".gff3", ".gtf"):
            fmt = "gff"
        else:
            first = next((ln for ln in lines if ln.strip() and not ln.startswith("#")), "")
            fmt = "gff" if len(first.split("\t")) >= 8 else "bed"
    lift = [str(k) for k in (params.get("attributes") or ["ID", "Name", "Parent", "gene_id", "transcript_id", "gene", "product"])]
    limit = int(params.get("limit") or 0)
    rows: list[dict[str, Any]] = []
    if fmt == "bed":
        for ln in lines:
            if not ln.strip() or ln.startswith(("#", "track", "browser")):
                continue
            f = re.split(r"\s+", ln.strip())
            if len(f) < 3:
                continue
            rows.append({"chrom": f[0], "chrom_start": _int(f[1]), "chrom_end": _int(f[2]),
                         "name": f[3] if len(f) > 3 else None, "score": _num(f[4]) if len(f) > 4 else None,
                         "strand": f[5] if len(f) > 5 else None})
        order = ["chrom", "chrom_start", "chrom_end", "name", "score", "strand"]
    else:
        for ln in lines:
            if not ln.strip() or ln.startswith("#"):
                continue
            f = ln.split("\t")
            if len(f) < 9:
                continue
            attrs = _gff_attrs(f[8])
            row: dict[str, Any] = {
                "seqid": f[0], "source": f[1] if f[1] != "." else None, "type": f[2],
                "start": _int(f[3]), "end": _int(f[4]), "score": _num(f[5]),
                "strand": f[6] if f[6] != "." else None, "phase": _int(f[7]),
                "attributes": f[8],
            }
            for k in lift:
                row[k] = attrs.get(k)
            rows.append(row)
        order = ["seqid", "source", "type", "start", "end", "score", "strand", "phase", *lift, "attributes"]
    rows = _trim(rows, limit)
    kind_word = "BED interval" if fmt == "bed" else "feature"
    return NodeResult(_frame({k: [r.get(k) for r in rows] for k in order}, order),
                      report={"format": fmt, "features": len(rows)},
                      messages=[f"Read {len(rows):,} {kind_word}{'s' if len(rows) != 1 else ''}"])


registry.register(NodeType(
    key="load_features", label="Load features (GFF/GTF/BED)", category="Get data", icon="table",
    kind="source", inputs=[],
    description="Read a GFF3, GTF or BED file as one row per feature: position, type, strand, and named "
                "attributes (ID, Name, Parent, gene_id, …) lifted into their own columns.",
    apply=_load_features, summary=_summary,
    params=[
        Param("path", "File", "path", required=True, help="A .gff3, .gff, .gtf or .bed file"),
        Param("format", "Format", "choice", default="auto", choices=[("auto", "detect"), ("gff", "GFF/GTF"), ("bed", "BED")]),
        Param("attributes", "Attributes to lift into columns", "text_list",
              default=["ID", "Name", "Parent", "gene_id", "transcript_id", "gene", "product"], advanced=True),
        Param("limit", "Read at most this many features", "int", default=0, min=0, advanced=True, help="0 = all"),
    ],
))


# --------------------------------------------------------------------------- GenBank
def _load_genbank(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    path = _path(ctx, params)
    lines = _text(path)
    records: list[dict[str, Any]] = []
    locus: str | None = None
    locus_len: int | None = None
    in_features = False
    cur: dict[str, Any] | None = None

    def flush() -> None:
        if cur is not None:
            records.append(cur)

    for ln in lines:
        if ln.startswith("LOCUS"):
            flush()
            cur = None
            in_features = True                       # features follow the header block, with or without a FEATURES line
            parts = ln.split()
            locus = parts[1] if len(parts) > 1 else None
            m = re.search(r"(\d+)\s+bp", ln)
            locus_len = int(m.group(1)) if m else None
        elif ln.startswith("FEATURES"):
            in_features = True
        elif ln.startswith("ORIGIN") or ln.startswith("//") or ln.startswith("CONTIG"):
            flush()
            cur = None
            in_features = False
        elif in_features:
            m = re.match(r"^ {5}(\S+)\s*(.*)$", ln)
            if m and not ln.lstrip().startswith("/"):
                flush()
                cur = {"locus": locus, "record_length": locus_len, "type": m.group(1),
                       "loc": m.group(2).strip(), "quals": {}}
            elif cur is not None:
                s = ln.strip()
                if s.startswith("/"):
                    k, _, v = s[1:].partition("=")
                    cur["quals"][k] = v.strip().strip('"')
                elif s:
                    cur["loc"] = (cur["loc"] + " " + s).strip()
    flush()
    lift = [str(k) for k in (params.get("qualifiers") or ["gene", "product", "note", "locus_tag", "label"])]
    rows: list[dict[str, Any]] = []
    for r in _trim(records, params.get("limit")):
        loc = r["loc"]
        # a location may be a compound: join(10..90,200..290). Every range counts, so the span is min..max and
        # the length is the sum of the parts (a join of two 81/91 bp blocks is 172 bp, not 81).
        ranges = [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*\.\.\s*(\d+)", loc)]
        if ranges:
            start, end = min(a for a, _ in ranges), max(b for _, b in ranges)
            length: int | None = sum(b - a + 1 for a, b in ranges)
        else:
            single = re.search(r"(\d+)", loc)
            start = end = int(single.group(1)) if single else None
            length = 1 if start is not None else None
        row: dict[str, Any] = {
            "locus": r["locus"], "record_length": r["record_length"], "type": r["type"],
            "start": start, "end": end, "length": length,
            "strand": "-" if loc.lstrip().startswith("complement") else "+",
        }
        for k in lift:
            row[k] = r["quals"].get(k)
        row["qualifiers"] = "; ".join(f"{k}={v}" for k, v in r["quals"].items())
        rows.append(row)
    order = ["locus", "record_length", "type", "start", "end", "length", "strand", *lift, "qualifiers"]
    return NodeResult(_frame({k: [r.get(k) for r in rows] for k in order}, order),
                      report={"features": len(rows)},
                      messages=[f"Read {len(rows):,} feature{'s' if len(rows) != 1 else ''} from GenBank"])


registry.register(NodeType(
    key="load_genbank", label="Load GenBank features", category="Get data", icon="table",
    kind="source", inputs=[],
    description="Read a GenBank file as one row per feature: locus, type, position, strand, and common "
                "qualifiers (gene, product, note, …) lifted into their own columns.",
    apply=_load_genbank, summary=_summary,
    params=[
        Param("path", "File", "path", required=True, help="A .gb, .gbk or .genbank file"),
        Param("qualifiers", "Qualifiers to lift into columns", "text_list",
              default=["gene", "product", "note", "locus_tag", "label"], advanced=True),
        Param("limit", "Read at most this many features", "int", default=0, min=0, advanced=True, help="0 = all"),
    ],
))


# --------------------------------------------------------------------------- PLINK markers
def _load_markers(ctx: Ctx, inputs: dict[str, list[pl.LazyFrame]], params: dict[str, Any]) -> NodeResult:
    path = _path(ctx, params)                              # the .map (or .ped) file
    ped_path: Path | None = None
    if path.suffix.lower() == ".ped":
        ped_path, map_path = path, path.with_suffix(".map")
    else:
        map_path = path
        sibling = path.with_suffix(".ped")
        ped_path = sibling if sibling.exists() else None
    if not map_path.exists():
        raise ValueError(f"{path.name}: expected a PLINK .map file next to it ({map_path.name})")

    markers = []
    for ln in _text(map_path):
        if not ln.strip() or ln.startswith("#"):
            continue
        f = re.split(r"\s+", ln.strip())
        if len(f) < 4:
            continue
        markers.append({"chrom": f[0], "marker": f[1], "cm": _num(f[2]), "pos": _int(f[3])})
    markers = _trim(markers, params.get("limit"))

    cols_out = {k: [m.get(k) for m in markers] for k in ("chrom", "marker", "cm", "pos")}
    order = ["chrom", "marker", "cm", "pos"]
    samples: list[str] = []
    if ped_path is not None and ped_path.exists() and bool(params.get("genotypes", True)):
        geno: list[list[str]] = [[] for _ in markers]
        seen: dict[str, int] = {}
        for ln in _text(ped_path):
            if not ln.strip():
                continue
            f = re.split(r"\s+", ln.strip())
            if len(f) < 6:
                continue
            famid, iid = f[0], f[1]
            name = iid or famid
            if name in seen:
                seen[name] += 1
                name = f"{name}_{seen[name]}"
            else:
                seen[name] = 0
            samples.append(name)
            alleles = f[6:]
            for mi, m in enumerate(markers):
                a1 = alleles[2 * mi] if 2 * mi < len(alleles) else ""
                a2 = alleles[2 * mi + 1] if 2 * mi + 1 < len(alleles) else ""
                geno[mi].append((a1 + a2) if (a1 or a2) else None)
        for mi, name in enumerate(samples):
            cols_out[name] = [geno[m][mi] for m in range(len(markers))]
            order.append(name)
    return NodeResult(_frame(cols_out, order),
                      report={"markers": len(markers), "samples": samples},
                      messages=[f"Read {len(markers):,} marker{'s' if len(markers) != 1 else ''}"
                                + (f" with genotypes for {len(samples)} samples" if samples else " (positions only)")])


registry.register(NodeType(
    key="load_markers", label="Load markers (PLINK)", category="Get data", icon="table",
    kind="source", inputs=[],
    description="Read a PLINK .map (and its .ped, when present) as one row per marker: chrom, marker, cM, bp, "
                "and one genotype column per sample (alleles joined, e.g. AG).",
    apply=_load_markers, summary=_summary,
    params=[
        Param("path", "File", "path", required=True, help="A PLINK .map (or .ped — its .map is read from the same name)"),
        Param("genotypes", "Include genotypes from the .ped file", "bool", default=True),
        Param("limit", "Read at most this many markers", "int", default=0, min=0, advanced=True, help="0 = all"),
    ],
))
