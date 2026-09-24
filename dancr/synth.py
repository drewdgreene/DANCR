"""Synthetic two-probe pressure dataset with known ground truth.

    python -m dancr.synth out_dir --hours 1 --rate 20

Probe A is the sensitive reference. Probe B = a*A + b + slow drift + more
noise, with a few dropouts and spikes. Both are written as CSV with a
timestamp column, plus a JSON file describing the truth.
"""
from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl


def make_probe_data(hours: float = 1.0, rate: float = 20.0, seed: int = 1, start: datetime | None = None,
                    chunk_hours: float = 6.0):
    """Yield (name, DataFrame) chunks for probe A and B, plus a truth dict (last)."""
    rng = np.random.default_rng(seed)
    start = start or datetime(2024, 6, 1)
    n_total = int(hours * 3600 * rate)
    truth = {
        "rate_hz": rate, "hours": hours, "start": start.isoformat(),
        "b_slope": 1.0025, "b_offset": -12.35, "drift_per_day": 0.02,
        "noise_a": 0.002, "noise_b": 0.015, "tide_amp": 0.9, "tide_period_h": 12.42,
        "gaps": [], "spikes_b": 0, "rows_a": 0, "rows_b": 0,
    }
    # gaps: fixed set based on seed
    n_gaps = max(1, int(hours / 12)) if hours >= 2 else 1
    gap_starts = np.sort(rng.uniform(0.1, 0.9, n_gaps)) * n_total
    gap_lens = rng.integers(max(1, int(rate * 5)), max(2, int(rate * 120)), n_gaps)  # 5 s to 2 min
    for gs, gl in zip(gap_starts, gap_lens):
        truth["gaps"].append({"start_row": int(gs), "rows": int(gl), "start": (start + timedelta(seconds=gs / rate)).isoformat(),
                              "seconds": float(gl / rate)})
    chunk = int(chunk_hours * 3600 * rate)
    rows_a = rows_b = 0
    spikes = 0
    for c0 in range(0, n_total, chunk):
        idx = np.arange(c0, min(n_total, c0 + chunk))
        t_s = idx / rate
        base = 2250.0 + truth["tide_amp"] * np.sin(2 * np.pi * t_s / (truth["tide_period_h"] * 3600)) \
            + 0.15 * np.sin(2 * np.pi * t_s / 86400.0)
        a = base + rng.normal(0, truth["noise_a"], len(idx))
        b = truth["b_slope"] * base + truth["b_offset"] + truth["drift_per_day"] * (t_s / 86400.0) \
            + rng.normal(0, truth["noise_b"], len(idx))
        # spikes in B
        k = rng.random(len(idx)) < 1e-5
        b[k] += rng.choice([-1, 1], k.sum()) * rng.uniform(2, 8, k.sum())
        spikes += int(k.sum())
        keep = np.ones(len(idx), bool)
        for g in truth["gaps"]:
            keep &= ~((idx >= g["start_row"]) & (idx < g["start_row"] + g["rows"]))
        ts = np.array(start, dtype="datetime64[us]") + (t_s * 1e6).astype("timedelta64[us]")
        temp = 2.75 + 0.01 * np.sin(2 * np.pi * t_s / 86400.0)
        dfa = pl.DataFrame({"time": ts[keep], "pressure_psi": a[keep].astype(np.float64), "temp_c": temp[keep]})
        # probe B has its own dropouts: shift gap by a bit and add one extra
        keep_b = np.ones(len(idx), bool)
        for g in truth["gaps"]:
            keep_b &= ~((idx >= g["start_row"] + 37) & (idx < g["start_row"] + 37 + g["rows"] // 2))
        dfb = pl.DataFrame({"time": ts[keep_b], "pressure_psi": b[keep_b].astype(np.float64), "temp_c": (temp + 0.5)[keep_b]})
        rows_a += len(dfa)
        rows_b += len(dfb)
        yield "A", dfa
        yield "B", dfb
    truth["rows_a"], truth["rows_b"], truth["spikes_b"] = rows_a, rows_b, spikes
    yield "truth", truth


def write_dataset(out_dir: Path, hours: float, rate: float, seed: int = 1, fmt: str = "csv") -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = {}
    if fmt == "csv":
        first = {"A": True, "B": True}
        for name, df in make_probe_data(hours, rate, seed):
            if name == "truth":
                truth = df
                break
            path = out_dir / f"probe_{name}.csv"
            with open(path, "a" if not first[name] else "w", newline="") as f:
                df.with_columns(pl.col("time").dt.strftime("%Y-%m-%d %H:%M:%S%.3f")).write_csv(f, include_header=first[name])
            first[name] = False
    else:
        # stream each chunk to its own temp file, then combine lazily, so memory stays flat for long runs
        with tempfile.TemporaryDirectory(prefix="dancr-synth-") as td:
            parts: dict[str, list[str]] = {"A": [], "B": []}
            for name, df in make_probe_data(hours, rate, seed):
                if name == "truth":
                    truth = df
                    break
                p = Path(td) / f"{name}_{len(parts[name])}.parquet"
                df.write_parquet(p)
                parts[name].append(str(p))
            for name, ps in parts.items():
                if ps:
                    pl.scan_parquet(ps).sink_parquet(out_dir / f"probe_{name}.parquet")
    (out_dir / "truth.json").write_text(json.dumps(truth, indent=2))
    return truth


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_dir")
    ap.add_argument("--hours", type=float, default=1.0)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--format", choices=["csv", "parquet"], default="csv")
    a = ap.parse_args()
    t = write_dataset(Path(a.out_dir), a.hours, a.rate, a.seed, a.format)
    print(json.dumps({k: v for k, v in t.items() if k != "gaps"}, indent=2), f"gaps={len(t['gaps'])}")


if __name__ == "__main__":
    main()
