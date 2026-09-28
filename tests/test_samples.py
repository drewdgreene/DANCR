"""Sample data, templates and the synthetic probe dataset."""
import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.samples import EXAMPLES, TEMPLATES, build_template, write_example, write_sample


def test_templates_build_and_run(tmp_path):
    from dancr.core.executor import Executor
    data = write_sample(tmp_path, rows=8000)
    for t in TEMPLATES:
        p = Pipeline(t["key"]); p.path = tmp_path / f"{t['key']}.json"
        build_template(t["key"], p, data)
        assert p.problems() == [], (t["key"], p.problems())
        res = Executor(p).run()
        bad = [s for s in res.values() if s.status != "done"]
        assert not bad, (t["key"], [(s.node_id, s.error) for s in bad])


def test_a_file_named_like_the_sample_is_never_used_or_overwritten(tmp_path):
    from dancr.core.samples import write_sample, SAMPLE_NAME
    theirs = tmp_path / SAMPLE_NAME
    theirs.write_text("mine\n1\n")
    ours = write_sample(tmp_path, rows=3000)
    assert ours != theirs and theirs.read_text() == "mine\n1\n"
    assert write_sample(tmp_path, rows=3000) == ours           # our own sample of that size is reused
    with pytest.raises(ValueError):
        write_sample(tmp_path, rows=1)


def test_synthetic_truth_describes_probe_b(tmp_path):
    from dancr.synth import write_dataset
    t = write_dataset(tmp_path, hours=0.5, rate=20.0, seed=3)
    b = pl.read_csv(tmp_path / "probe_B.csv", try_parse_dates=True)
    assert len(t["gaps_b"]) == len(t["gaps"]) and b.height == t["rows_b"]
    g = t["gaps_b"][0]
    gap = b["time"].diff().dt.total_milliseconds().max() / 1000
    assert gap == pytest.approx(g["seconds"] + 1 / 20.0, abs=0.01)


def test_synth_parquet_streams_and_matches_csv(tmp_path):
    from dancr.synth import write_dataset
    csv = write_dataset(tmp_path / "csv", hours=0.05, rate=20.0, seed=3, fmt="csv")
    pq = write_dataset(tmp_path / "pq", hours=0.05, rate=20.0, seed=3, fmt="parquet")
    assert pq["rows_a"] == csv["rows_a"] and pq["rows_b"] == csv["rows_b"]
    assert pl.read_parquet(tmp_path / "pq" / "probe_A.parquet").height == pq["rows_a"]
    assert pl.read_parquet(tmp_path / "pq" / "probe_B.parquet").height == pq["rows_b"]


def test_synth_low_rate_does_not_crash(tmp_path):
    from dancr.synth import write_dataset
    t = write_dataset(tmp_path, hours=1.0, rate=0.001, seed=1, fmt="parquet")
    assert pl.read_parquet(tmp_path / "probe_A.parquet").height == t["rows_a"]


@pytest.mark.parametrize("key", [e["key"] for e in EXAMPLES])
def test_examples_read_left_to_right(tmp_path, key):
    """Every step of an example sits right of the steps it reads from (the report after all it shows), and no
    two steps sit on each other, so the map needs no tidying before it can be read."""
    import json
    p = json.loads(write_example(key, tmp_path).read_text())
    at = {n["id"]: (n["x"], n["y"]) for n in p["nodes"]}
    assert [(e["source"], e["target"]) for e in p["edges"] if at[e["source"]][0] >= at[e["target"]][0]] == []
    spots = list(at.items())
    assert [(a, b) for i, (a, (ax, ay)) in enumerate(spots) for b, (bx, by) in spots[i + 1:]
            if abs(ax - bx) < 230 and abs(ay - by) < 130] == []
