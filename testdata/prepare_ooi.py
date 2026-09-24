"""Turn the raw OOI BOTPT files into one CSV per probe.

Raw lines look like
    NANO,P,2024/05/31 23:59:58.100,2253.543287,2.750924099
interleaved with LILY/IRIS tilt-meter lines. We keep only the 20 Hz NANO
pressure records: time, pressure (psia), temperature (C).

    python testdata/prepare_ooi.py            # writes testdata/probe_MJ03F.csv, probe_MJ03E.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import polars as pl

RAW = Path(__file__).resolve().parent / "raw"
OUT = Path(__file__).resolve().parent


def prepare(site: str) -> None:
    files = sorted((RAW / site).glob("*.dat"))
    if not files:
        print(f"no files for {site}"); return
    t0 = time.time()
    lf = pl.scan_csv(files, has_header=False, separator=",", truncate_ragged_lines=True, ignore_errors=True,
                     schema={"c1": pl.Utf8, "c2": pl.Utf8, "c3": pl.Utf8, "c4": pl.Utf8, "c5": pl.Utf8},
                     infer_schema=False, quote_char=None)
    lf = (lf.filter((pl.col("c1") == "NANO") & (pl.col("c2") == "P"))
            .select([pl.col("c3").str.strip_chars().alias("time"),
                     pl.col("c4").cast(pl.Float64, strict=False).alias("pressure_psia"),
                     pl.col("c5").cast(pl.Float64, strict=False).alias("temperature_c")]))
    out = OUT / f"probe_{site}.csv"
    lf.sink_csv(out)
    n = pl.scan_csv(out).select(pl.len()).collect()[0, 0]
    print(f"{site}: {n:,} rows -> {out} ({out.stat().st_size / 1e9:.2f} GB) in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for site in (sys.argv[1:] or ["MJ03F", "MJ03E"]):
        prepare(site)
