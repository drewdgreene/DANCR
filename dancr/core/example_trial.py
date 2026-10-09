"""The "Maize trial & data stewardship" starter pack.

A single, finished demonstration of what DANCR is for, on domain-accurate but entirely fictional
agricultural-research data. It writes a whole starter *repository*:

- **Maize trial & data stewardship** (the main project, built by the generic example builder) — a
  breeder's plain-language questions over a multi-site yield trial.
- **Stewardship audit** — the data-steward's checklist: id uniqueness, controlled vocabularies,
  provenance completeness and cross-file references, each finding named.
- **Knowledge base** — an offline, deterministic search index over SOPs/notes with an IP guardrail
  (a restricted document withheld by default).
- **START-HERE.md** — a one-page guided walkthrough.

Everything is generated from a fixed seed, so the same example always comes out the same. No
organisation, brand or product is named: identifiers are invented (LINE-, EVT-, TRL-, SITE-) and the
only real references are public data standards (MIAPPE, the Crop Ontology, the public regulation of
genetically engineered plants).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

SEED = 42
SITES = [("SITE-A", "North Farm", 42.03, -93.63),
         ("SITE-B", "River Bottom", 41.67, -93.70),
         ("SITE-C", "Ridge Field", 41.99, -93.47)]
YEARS = (2025, 2026)
LINES = [f"LINE-{i:04d}" for i in range(1, 21)]
CHECKS = ["CHECK-01", "CHECK-02", "CHECK-03", "CHECK-04"]
TREATMENTS = ("rainfed", "irrigated", "drought_stress")
TREAT_MULT = {"rainfed": 1.0, "irrigated": 1.14, "drought_stress": 0.72}

MAIN_FILES = ["trials/plots.csv", "trials/phenotype_measurements.csv",
              "trials/trial_metadata.csv", "environment/weather_daily.csv"]


# --------------------------------------------------------------------------- the data
def _write_weather(directory: Path, rng: np.random.Generator) -> None:
    rows: dict[str, list] = {"date": [], "site": [], "tmin_c": [], "tmax_c": [], "rain_mm": []}
    for year in YEARS:
        days = [date(year, 5, 1) + timedelta(days=i) for i in range(184)]     # May 1 – Oct 31
        for sid, _name, _lat, _lon in SITES:
            shift = {"SITE-A": 0.0, "SITE-B": -0.6, "SITE-C": 0.5}[sid]
            for d in days:
                doy = d.timetuple().tm_yday
                season = 0.5 + 0.5 * np.sin(2 * np.pi * (doy - 130) / 365.0)
                tmax = 16 + 14 * season + shift + rng.normal(0, 2.0)
                tmin = tmax - (10 + rng.uniform(0, 4))
                rain = float(rng.choice([0.0, 0.0, 0.0, 1.5, 6.0, 18.0], p=[.55, .2, .1, .08, .05, .02]))
                rows["date"].append(d); rows["site"].append(sid)
                rows["tmin_c"].append(round(float(tmin), 1)); rows["tmax_c"].append(round(float(tmax), 1))
                rows["rain_mm"].append(round(rain, 1))
    pl.DataFrame(rows).write_csv(directory / "environment" / "weather_daily.csv")


def _write_trials(directory: Path, rng: np.random.Generator) -> None:
    trials, meta = [], []
    for yi, year in enumerate(YEARS):
        for si, (sid, sname, lat, lon) in enumerate(SITES):
            tid = f"TRL-{year}-{si + 1:02d}"
            design = "RCBD"
            # planted seed a deliberately ambiguous year on one trial (a stewardship defect)
            stated_year = year if tid != "TRL-2025-03" else 2026
            trials.append((tid, year, sid, lat, lon))
            meta.append({"trial_id": tid, "site": sid, "site_name": sname, "year": stated_year,
                         "design": design, "reps": 3})
    plots: dict[str, list] = {"plot_id": [], "trial_id": [], "site": [], "line": [], "rep": [], "plot_index": [], "treatment": []}
    meas: dict[str, list] = {"plot_id": [], "trial_id": [], "site": [], "line": [], "rep": [], "treatment": [], "measured_on": [],
            "grain_yield_q_ha": [], "plant_height_cm": [], "days_to_anthesis": [],
            "disease_severity_pct": [], "lodging_pct": []}
    entries = LINES + CHECKS
    line_base = {ln: float(rng.uniform(72, 94)) for ln in entries}
    line_height = {ln: float(rng.uniform(235, 260)) for ln in entries}
    for tid, year, sid, lat, lon in trials:
        season_date = date(year, 8, 20) + timedelta(days=int(rng.integers(-10, 11)))
        for rep in (1, 2, 3):
            treatment = TREATMENTS[rep - 1]
            order = list(rng.permutation(entries))
            for idx, ln in enumerate(order, start=1):
                plot_id = f"{tid}-R{rep}-P{idx:02d}"
                plots["plot_id"].append(plot_id); plots["trial_id"].append(tid); plots["site"].append(sid)
                plots["line"].append(ln); plots["rep"].append(rep); plots["plot_index"].append(idx)
                plots["treatment"].append(treatment)
                site_shift = {"SITE-A": 1.5, "SITE-B": -1.0, "SITE-C": 0.0}[sid]
                y = line_base[ln] * TREAT_MULT[treatment] + site_shift + rng.normal(0, 2.6)
                h = line_height[ln] + rng.normal(0, 7.0)
                meas["plot_id"].append(plot_id); meas["trial_id"].append(tid); meas["site"].append(sid)
                meas["line"].append(ln); meas["rep"].append(rep); meas["treatment"].append(treatment)
                meas["measured_on"].append(season_date)
                meas["grain_yield_q_ha"].append(round(float(y), 2))
                meas["plant_height_cm"].append(round(float(h), 1))
                meas["days_to_anthesis"].append(int(round(74 + rng.normal(0, 3))))
                meas["disease_severity_pct"].append(round(float(max(0.0, rng.normal(7, 4))), 1))
                meas["lodging_pct"].append(round(float(max(0.0, rng.normal(3, 3))), 1))
    # seed the stewardship defects a real dataset always has
    n = len(meas["grain_yield_q_ha"])
    for i in rng.choice(n, size=23, replace=False):            # missing yields
        meas["grain_yield_q_ha"][int(i)] = None
    for i in rng.choice(n, size=2, replace=False):             # extreme outliers
        v = meas["grain_yield_q_ha"][int(i)]
        if v is not None:
            meas["grain_yield_q_ha"][int(i)] = round(float(v) + 45.0, 2)
    meas["grain_yield_q_ha"][3] = 401.0                        # a slipped decimal point
    directory.joinpath("trials").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(plots).write_csv(directory / "trials" / "plots.csv")
    pl.DataFrame(meas).write_csv(directory / "trials" / "phenotype_measurements.csv")
    pl.DataFrame(meta).write_csv(directory / "trials" / "trial_metadata.csv")


def _write_registry(directory: Path, rng: np.random.Generator) -> None:
    entries = LINES + CHECKS
    rows: dict[str, list] = {"canonical_name": [], "registry_id": [], "lims_id": [], "crop": [], "status": []}
    for i, ln in enumerate(entries, start=1):
        rows["canonical_name"].append(ln)
        rows["registry_id"].append(f"REG-{i:04d}")
        rows["lims_id"].append(f"LIMS-{10000 + i * 7}")       # a different namespace from registry_id
        rows["crop"].append("Zea mays")
        rows["status"].append("active" if i % 5 else "retired")
    rows["status"][5] = "Unknown"                             # a value outside the controlled set (defect)
    for k in rows:                                            # an extra row carrying a duplicate registry id (defect)
        rows[k].append(rows[k][3])
    rows["lims_id"][-1] = "LIMS-10999"
    directory.joinpath("germplasm").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_csv(directory / "germplasm" / "line_registry.csv")


def _write_editing(directory: Path, rng: np.random.Generator) -> None:
    n = 24
    events: dict[str, list] = {"event_id": [], "lims_id": [], "registry_id": [], "construct_id": [], "host_line": [], "program": []}
    for i in range(1, n + 1):
        events["event_id"].append(f"EVT-{i:04d}")
        events["lims_id"].append(f"LIMS-{20000 + i * 5}")
        events["registry_id"].append(f"REG-{(i % 20) + 1:04d}")
        events["construct_id"].append(f"CON-{(i % 8) + 1:04d}")
        events["host_line"].append((LINES + CHECKS)[i % len(LINES + CHECKS)])
        events["program"].append("DP-2026")
    events["registry_id"][-1] = "REG-9999"                    # a reference that points nowhere (defect)
    directory.joinpath("gene_editing").mkdir(parents=True, exist_ok=True)
    pl.DataFrame(events).write_csv(directory / "gene_editing" / "transformation_events.csv")

    codes = ["CO_322:0000401", "CO_322:0000482", "CO_322:0000531", "CO_322:0000595"]
    names = ["grain yield", "plant height", "days to anthesis", "drought susceptibility"]
    traits: dict[str, list] = {"event_id": [], "trait_reference": [], "trait_reference_type": [], "effect_direction": [], "effect_size_pct": []}
    for i in range(1, n + 1):
        free = (i % 3 == 0)                                   # a third written as free text (defect)
        j = i % 4
        traits["event_id"].append(f"EVT-{i:04d}")
        traits["trait_reference"].append(names[j] if free else codes[j])
        traits["trait_reference_type"].append("free_text" if free else "ontology_code")
        traits["effect_direction"].append(["increase", "decrease", "no_effect"][i % 3])
        traits["effect_size_pct"].append(round(float(rng.normal(0, 5)), 1))
    pl.DataFrame(traits).write_csv(directory / "gene_editing" / "edit_to_trait_map.csv")


def _write_governance(directory: Path, rng: np.random.Generator) -> None:
    directory.joinpath("governance").mkdir(parents=True, exist_ok=True)
    prov: dict[str, list] = {"record_id": [], "entity": [], "activity": [], "agent": [], "pipeline_version": [], "timestamp": []}
    for i in range(1, 31):
        prov["record_id"].append(f"PROV-{i:04d}")
        prov["entity"].append(["phenotype_measurements.csv", "edit_outcomes.csv", "variants.vcf"][i % 3])
        prov["activity"].append(["data_entry", "qc", "analysis"][i % 3])
        prov["agent"].append(["pipeline", "J. Rivera", "pipeline"][i % 3])
        prov["pipeline_version"].append("" if i % 5 == 0 else f"4.{i % 5}.0.0")   # blank on some (defect)
        prov["timestamp"].append(f"2026-{1 + i % 9:02d}-{1 + i % 27:02d}")
    pl.DataFrame(prov).write_csv(directory / "governance" / "provenance_log.csv")

    vocab: dict[str, list] = {"vocabulary": [], "term_id": [], "label": [], "definition": []}
    for term, label, definition in [
            ("CO_322:0000401", "Grain yield", "Grain yield measured as plot weight at maturity"),
            ("CO_322:0000482", "Plant height", "Plant height measured as stem length, soil to tassel tip"),
            ("CO_322:0000531", "Days to anthesis", "Days from sowing to 50% anthesis"),
            ("CO_322:0000595", "Drought susceptibility", "Drought susceptibility index")]:
        vocab["vocabulary"].append("CO_322"); vocab["term_id"].append(term)
        vocab["label"].append(label); vocab["definition"].append(definition)
    pl.DataFrame(vocab).write_csv(directory / "governance" / "controlled_vocabularies.csv")


def _write_knowledge(directory: Path, rng: np.random.Generator) -> None:
    """A small SOP / literature / regulatory corpus for the offline search index. One document is restricted."""
    docs = [
        ("faq_line_ids", "Are LIMS and registry line ids interchangeable? No. The LIMS id and the registry "
                         "id live in different namespaces and must be cross-referenced through the registry "
                         "table; they are not interchangeable.", ""),
        ("grain_yield_definition", "Grain yield is reported in quintals per hectare (q/ha), adjusted to 15.5% "
                                   "moisture. The canonical trait code for grain yield is CO_322:0000401.", ""),
        ("miappe_metadata", "Phenotyping datasets follow MIAPPE: every dataset carries the study, the "
                            "environment, the plants and the observed variables with their units.", ""),
        ("crispr_scoring", "The standard SpCas9 PAM is NGG. Guide scoring follows the CFD and MIT conventions "
                           "for on-target and off-target prediction.", ""),
        ("trait_ontology_codes", "Traits are recorded with Crop Ontology codes (for example CO_322:0000401 for "
                                 "grain yield), never as free text, so datasets stay interoperable.", ""),
        ("sop_regen_v3", "SOP: transformation and regeneration. Tissue is cultured, selected, and regenerated; "
                         "events are recorded in the transformation event registry with a construct id and a "
                         "host line. Only validated events are advanced.", ""),
        ("drought_susceptibility", "Drought susceptibility is abbreviated DSI in the field notes; tolerance is "
                                   "DT. Both map to the trait code CO_322:0000595.", ""),
        ("regulatory_dossier", "RESTRICTED - regulatory dossier for the gene-editing program. Contains "
                               "unpublished construct designs, jurisdictions and filing strategy. Not for "
                               "external distribution.", "restricted"),
    ]
    ai = directory / "ai_knowledge"; ai.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"doc": [d[0] for d in docs], "text": [d[1] for d in docs],
                  "sensitivity": [d[2] for d in docs]}).write_csv(ai / "corpus.csv")


def _audit_project(directory: Path):
    from .model import Pipeline
    p = Pipeline("Stewardship audit")
    p.path = directory / "Stewardship audit.json"
    y = 200.0

    def load(rel: str, nid: str) -> str:
        return p.add_node("load_file", title=Path(rel).stem, params={"path": rel},
                          x=60.0, y=y, id=nid).id

    registry = load("germplasm/line_registry.csv", "registry")
    events = load("gene_editing/transformation_events.csv", "events")
    traits = load("gene_editing/edit_to_trait_map.csv", "traits")
    prov = load("governance/provenance_log.csv", "provenance")
    meas = load("trials/phenotype_measurements.csv", "measurements")

    def check(src: str, contract: dict, title: str, y: float = 200.0, refs: list[str] | None = None) -> str:
        cid = p.add_node("check_contract", title=title,
                         params={"contract": json.dumps(contract), "output": "issues"}, x=400.0, y=y).id
        p.connect(src, cid, "in")
        for r in refs or []:
            p.connect(r, cid, "references")
        return cid

    checks = [
        check(registry, {"columns": {"registry_id": {"kind": "text", "required": True, "unique": True},
                                     "canonical_name": {"kind": "text", "required": True},
                                     "status": {"kind": "text", "allowed": ["active", "retired"]}}},
              "Registry: unique ids, allowed status", y=40.0),
        check(prov, {"columns": {"pipeline_version": {"kind": "text", "required": True}}},
              "Provenance: every record has a version", y=150.0),
        check(traits, {"columns": {"trait_reference_type": {"kind": "text", "allowed": ["ontology_code"]}}},
              "Traits: ontology codes, not free text", y=260.0),
        check(events, {"references": [{"column": "registry_id", "to_node": registry, "to_column": "registry_id"}]},
              "Events: registry references resolve", y=370.0, refs=[registry]),
    ]
    qc = p.add_node("check_data", title="Measurements: blanks, duplicates, outliers",
                    params={"columns": []}, x=400.0, y=480.0).id
    p.connect(meas, qc, "in")
    checks.append(qc)

    rep = p.add_node("report", title="Stewardship audit report", id="report", x=760.0, y=200.0,
                     params={"title": "Stewardship audit", "path": "Stewardship audit report.html",
                             "pdf": False,
                             "notes": "Every data-contract failure and every quality flag, in one place. "
                                      "This is the data-steward's checklist run as steps, so it repeats next season."}).id
    for c in checks:
        p.connect(c, rep, "items")
    p.save()
    return p


def _cleanup_project(directory: Path):
    """The steward's second half: repair the defects at the source, then prove the repair holds."""
    from .model import Pipeline
    p = Pipeline("Data cleanup")
    p.path = directory / "Data cleanup.json"

    def load(rel: str, nid: str, y: float) -> str:
        return p.add_node("load_file", title=Path(rel).stem, params={"path": rel}, x=60.0, y=y, id=nid).id

    registry = load("germplasm/line_registry.csv", "registry", 40.0)
    events = load("gene_editing/transformation_events.csv", "events", 220.0)
    meas = load("trials/phenotype_measurements.csv", "measurements", 400.0)

    # registry: correct the status typo, drop the duplicate id, then assert both hold
    fix_status = p.add_node("fix_values", title="Correct the status code", id="fix_status",
                            params={"fixes": [{"row": 6, "column": "status", "value": "active",
                                               "was": "Unknown", "note": "outside the controlled vocabulary"}]},
                            x=380.0, y=40.0).id
    p.connect(registry, fix_status)
    dedup = p.add_node("remove_duplicates", title="One row per registry id", id="dedup",
                       params={"columns": ["registry_id"], "keep": "first"}, x=700.0, y=40.0).id
    p.connect(fix_status, dedup)
    reg_ok = dedup
    reg_check = p.add_node("check_contract", title="Registry is clean", id="reg_check",
                           params={"output": "issues",
                                   "contract": json.dumps({"columns": {
                                       "registry_id": {"kind": "text", "required": True, "unique": True},
                                       "status": {"kind": "text", "allowed": ["active", "retired"]}}})},
                           x=1020.0, y=40.0).id
    p.connect(reg_ok, reg_check, "in")

    # events: repair the reference that pointed nowhere, then prove every reference resolves
    fix = p.add_node("fix_values", title="Repair the broken registry reference", id="fix_ref",
                     params={"fixes": [{"row": 24, "column": "registry_id", "value": "REG-0001",
                                        "was": "REG-9999", "note": "the reference pointed at an id that does not exist"}]},
                     x=380.0, y=220.0).id
    p.connect(events, fix)
    ev_check = p.add_node("check_contract", title="Every event points at a real line", id="ev_check",
                          params={"output": "issues",
                                  "contract": json.dumps({"references": [
                                      {"column": "registry_id", "to_node": reg_ok, "to_column": "registry_id"}]})},
                          x=700.0, y=220.0).id
    p.connect(fix, ev_check, "in")
    p.connect(reg_ok, ev_check, "references")

    # measurements: fix the slipped decimal, drop rows with no yield, save a clean copy
    fixm = p.add_node("fix_values", title="Fix the slipped decimal point", id="fix_yield",
                      params={"fixes": [{"row": 4, "column": "grain_yield_q_ha", "value": 40.1,
                                         "was": 401.0, "note": "10x too high"}]},
                      x=380.0, y=400.0).id
    p.connect(meas, fixm)
    drop = p.add_node("fix_missing", title="Drop plots with no yield", id="drop_blank",
                      params={"method": "drop", "columns": ["grain_yield_q_ha"]}, x=700.0, y=400.0).id
    p.connect(fixm, drop)
    p.add_node("export", title="Save the clean copy", id="save",
               params={"path": "trials/phenotype_measurements_clean.csv"}, x=1020.0, y=400.0)
    p.connect(drop, "save")

    rep = p.add_node("report", title="Data cleanup report", id="report", x=1340.0, y=220.0,
                     params={"title": "Data cleanup — from defects to clean data",
                             "path": "Data cleanup report.html", "pdf": False,
                             "notes": "The steward repairs each defect at the source and the contract checks "
                                      "confirm the repair holds. The cleaned table is saved beside the raw one."}).id
    for nid in (reg_check, ev_check, drop):
        p.connect(nid, rep, "items")
    p.save()
    return p


def _trial_project(directory: Path):
    """A genotype trial fitted as a randomised complete block design: BLUPs and heritability per trial."""
    from .model import Pipeline
    p = Pipeline("Trial analysis")
    p.path = directory / "Trial analysis.json"
    src = p.add_node("load_file", title="phenotype_measurements",
                     params={"path": "trials/phenotype_measurements.csv"}, x=60.0, y=200.0, id="plots").id
    fix = p.add_node("fix_values", title="Fix the slipped decimal", id="fix",
                     params={"fixes": [{"row": 4, "column": "grain_yield_q_ha", "value": 40.1, "was": 401.0,
                                        "note": "10x too high"}]}, x=380.0, y=200.0).id
    p.connect(src, fix)
    ta = p.add_node("trial_analysis", title="BLUPs by trial", id="blups", x=700.0, y=200.0,
                    params={"value": "grain_yield_q_ha", "genotype": "line", "block": "rep", "group": "trial_id"}).id
    p.connect(fix, ta)
    rep = p.add_node("report", title="Trial analysis report", id="report", x=1020.0, y=200.0,
                     params={"title": "Trial analysis (randomised complete block design)",
                             "path": "Trial analysis report.html", "pdf": False,
                             "notes": "Each trial fitted as a randomised complete block design: the block (rep) "
                                      "effects are accounted for, the genetic and residual variance are estimated, "
                                      "and each line gets a BLUP with the trial's heritability. This is the analysis "
                                      "a breeder runs before advancing a line."}).id
    p.connect(ta, rep, "items")
    p.save()
    return p


def _knowledge_project(directory: Path):
    from .model import Pipeline
    p = Pipeline("Knowledge base")
    p.path = directory / "Knowledge base.json"
    docs = p.add_node("load_file", title="corpus", params={"path": "ai_knowledge/corpus.csv"}, id="docs",
                      x=60.0, y=200.0).id
    idx = p.add_node("build_index", title="Build search index",
                     params={"text_column": "text", "chunk_chars": 400}, id="index", x=360.0, y=200.0).id
    p.connect(docs, idx, "items")
    search = p.add_node("retrieve", title="Search the SOPs and notes",
                        params={"query": "canonical trait code for grain yield", "k": 3, "retriever": "hybrid"},
                        id="search", x=660.0, y=120.0).id
    p.connect(idx, search, "in")
    # a shareable copy: keep every passage that is not restricted, then hand it on
    share = p.add_node("keep_rows", title="A shareable copy (no restricted rows)", id="shareable",
                       params={"mode": "keep",
                               "conditions": {"match": "all", "rules": [{"column": "sensitivity", "op": "ne", "value": "restricted"}]}},
                       x=360.0, y=340.0).id
    p.connect(docs, share)
    p.add_node("export", title="Save the shareable corpus", id="save_corpus",
               params={"path": "ai_knowledge/corpus_shareable.csv"}, x=660.0, y=340.0)
    p.connect(share, "save_corpus")
    p.add_node("report", title="Knowledge base report", id="report", x=960.0, y=120.0,
               params={"title": "Knowledge base (guarded search)",
                       "path": "Knowledge base report.html", "pdf": False,
                       "notes": "Offline, deterministic search over the programme's own notes. The restricted "
                                "regulatory dossier is withheld by default; turn on 'Include restricted' on the "
                                "Search step (or pass allow_restricted) to reveal it. A shareable copy with the "
                                "restricted rows removed is saved beside the raw corpus."})
    p.connect(search, "report", "items")
    p.connect(share, "report", "items")
    p.save()
    return p


def build_siblings(main, directory: Path | str) -> None:
    """Write the pack's sibling projects (the stewardship audit and the guarded knowledge base)."""
    directory = Path(directory)
    if not (directory / "Stewardship audit.json").exists():
        _audit_project(directory)
    if not (directory / "Knowledge base.json").exists():
        _knowledge_project(directory)
    if not (directory / "Data cleanup.json").exists():
        _cleanup_project(directory)
    if not (directory / "Trial analysis.json").exists():
        _trial_project(directory)
    _write_walkthrough(directory)


def _write_walkthrough(directory: Path) -> None:
    text = """# Maize trial & data stewardship — start here

A finished, runnable example. Everything below is on fictional data for a maize breeding programme; no
real company, brand or product is named. It is here to show what DANCR does end to end.

## 1. The trial project (opened for you)

`Maize trial & data stewardship.json` — four tables (plots, plot measurements, trial metadata, daily
weather) and six questions typed in plain English. Open the **Answers** in the rail; each one is real
steps you can click into and change.

Try typing these, and compare with the answer already there:

- `t test grain yield rainfed vs drought_stress` — DANCR picks the test itself (Welch's t-test) and
  gives the difference, the p-value and an effect size, in one sentence.
- `average grain yield by treatment` — a pivot-style breakdown.
- `relationship between grain yield and plant height` — a scatter with the fitted equation and R².
- `top 5 lines by grain yield`
- `check the measurements` — blanks, duplicates, a slipped decimal point, outliers.
- `grain yield per month` — a time trend.

Change any step's settings and press Run: only the steps whose data changed recompute.

## 2. Stewardship audit.json

The data-steward's checklist as steps: id uniqueness, an allowed-value check on the status codes, a
provenance-completeness check, a cross-file reference check (do the gene-edit events point at a real
registry entry?), and a quality pass over the measurements. Run it and open
`Stewardship audit report.html`. It finds the defects planted on purpose.

## 3. Data cleanup.json

The other half of stewardship: repair the defects **at the source**, then prove the repair holds. A
`Fix values` step corrects a status code, `Remove duplicates` drops the duplicate registry id, a
`Fix values` step repairs the event reference that pointed nowhere, and the measurements get a slipped
decimal corrected and the plots with no yield dropped. Two `Check data contract` steps then report
**zero** issues, and the clean table is saved beside the raw one. Open `Data cleanup report.html`.

## 4. Trial analysis.json

The breeder's question: which line is best, once the blocks are accounted for? This project fits each
trial as a **randomised complete block design** — a mixed model with the block (rep) fixed and the line
random — estimates the genetic and residual variance by REML, and gives every line a **BLUP** (a shrunken
estimate) with the trial's **heritability**. Open `Trial analysis report.html`. This is the analysis a
breeder runs before advancing a line, and for a balanced design it reproduces the exact ANOVA estimates.

## 5. Knowledge base.json

An offline, deterministic search index over the programme's SOPs, notes and one restricted regulatory
dossier. Run it: the hybrid retriever answers "canonical trait code for grain yield" with the source
passages, and the **restricted dossier is withheld by default** — the report says how many were
withheld. Turn on *Include restricted* on the Search step to reveal it. The same guard withholds
restricted rows from an agent's reads (`get_sample`, `get_stats`, a rendered chart or map), the command
line, and every export, unless `allow_restricted` is passed.

## 6. Prove it, package it, map it (command line)

From this folder:

- `dancr verify "Maize trial & data stewardship.json" --record trial.att.json` — a checkable record of
  exactly what produced every number.
- `dancr verify "Maize trial & data stewardship.json" --manifest trial.att.json` — re-run and confirm it
  still reproduces.
- `dancr fair "Maize trial & data stewardship.json" --format schema.org` — a FAIR descriptor.
- `dancr package "Maize trial & data stewardship.json" --out trial.rocrate.zip --copy metadata` — a
  self-contained crate a colleague can verify.
- `dancr graph build .` then `dancr graph shared-keys .` — index the whole folder and see which keys
  link the projects.
- `dancr mcp --root .` — hand an AI agent the same tools, with a policy gate (see `docs/GATEWAY.md`).

Change one value in `trials/phenotype_measurements.csv`, run `verify --manifest` again, and watch the
verdict turn from **verified** to **mismatch**. That is the point.
"""
    (directory / "START-HERE.md").write_text(text)


def write_data(directory: Path | str) -> list[str]:
    """Generate the whole pack's data under ``directory``; return the files the main project loads."""
    directory = Path(directory)
    for sub in ("trials", "environment", "germplasm", "gene_editing", "governance", "ai_knowledge"):
        (directory / sub).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    _write_weather(directory, rng)
    _write_trials(directory, rng)
    _write_registry(directory, rng)
    _write_editing(directory, rng)
    _write_governance(directory, rng)
    _write_knowledge(directory, rng)
    return list(MAIN_FILES)
