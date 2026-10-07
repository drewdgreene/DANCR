# Open

- Windows and macOS builds are validated only by the CI workflow, not on a real machine.
- Releases are not code-signed (Windows SmartScreen and macOS Gatekeeper warn on first open): that needs a
  signing certificate and an Apple Developer ID, then `signtool` and `codesign`/`notarytool` steps in CI. The
  macOS build is for Apple silicon only (the `macos-latest` runner); an Intel Mac needs an Intel runner or a
  universal2 Python and wheels.
- `contribution` in change mode with an average (`stat` mean) adds up each group's change in its average, which
  is not a share of the change in the overall average; a change in an average does not split by group.
- The seasonal forecast's band uses the straight-line fit's error, so it is a rough band.
- Comparing groups: with three or more groups there is no test of which pairs differ (no post-hoc test); a paired
  comparison is of two groups only (no repeated-measures ANOVA); a two-way design (light × species) is compared one
  factor at a time. Merged cells are inferred from where banner text sits (the cell grid has no merge ranges), and
  a sheet's formulas are not read, so a summary row is found by its label (AVERAGE, Total) rather than by its
  formula. A table laid out sideways (one variable per row) is not turned round. On a sheet over 20,000 rows only
  its top 300 and bottom 60 rows are used to work out the layout.
- Answers cannot yet: group by two things at once (a cross-tab; refused), link keys written differently
  (`#10234` against `10234`), treat survey answers as ordered, say which month was over budget, join two
  different conditions with "or", keep two separate periods (`in March and April`), exclude a month
  (`not in March`) or a range of dates, or read a decimal comma (`30,5` is 30 and 5). Relative dates count from
  the data's latest date, not today.
- Times written at several UTC offsets (daylight saving) stay in UTC unless a time zone is chosen: the
  offsets alone do not say which zone they came from.
- A cache sweep in one process can move a result aside in the instant between another process seeing it and
  reading it; that step then fails once with a missing-file error and succeeds when run again.
- Cancel takes effect between steps: a step writing a very large result finishes first (Polars 1.44 cannot
  stop a write safely).
- The project file's lock is an operating-system file lock: on a network drive shared by two machines it may
  not be seen by the other machine (each save still checks the file did not change since it was read).
- Places: distances are great-circle on a fixed sphere (mean radius), not ellipsoidal or projected, so they
  differ from geodesic answers by a few tenths of a percent; a grid cell given in km is turned into *rectangular*
  degrees, so away from the equator a "5km" cell is narrower east–west than north–south. Coordinate columns are
  taken as WGS84 lon/lat; a GeoPackage or shapefile layer whose CRS is known is reprojected to WGS84 on load, and
  `project` turns UTM (built in) or another EPSG code (optional pyproj) into lon/lat. Points can be tested
  against polygon layers (a feature's `geometry` WKT, point-in-polygon with holes handled), but polygons are not
  drawn, there are no line geometries and no polygon maths beyond containment, and a nearest-place match compares
  every point with the places within the distance (fine for thousands of places, not millions). The basemap is
  1:110m Natural Earth outlines, so small islands and borders are simplified. The map draws an equirectangular
  or Mercator view, not a tiled web map; a point with a coordinate outside its range is drawn only if the
  coordinate columns were not cleaned with 'Make a point' first.
- GeoPackage and shapefile loading, reprojection of a non-UTM EPSG code, database servers, and the
  NetCDF/HDF5 readers all ship in the one install now (pyogrio, shapely, pyproj, SQLAlchemy, xarray, h5py).
  The `[project.optional-dependencies]` groups (`geo`, `db`, `science`, `assistant`) remain only as aliases
  for source installs; nothing asks the person to add an extra.

- Load folder reads each file with the same reader as Load file; `diagonal` takes the union of the columns and
  relaxes types (a column that is a number in one file and text in another becomes text; missing values are blank),
  `strict` needs every file to have exactly the same columns, and `text` reads every column as text. Files are read
  in path order. A change to any member file, or a file added or removed, makes the step run again.
- Batch rebinds one file setting of one source step, so a project whose shape differs per file (a different sheet,
  a lookup that also changes) needs one run per shape rather than a batch. Results are written next to the project;
  the per-file results are kept in the project's cache (not swept), so a batch over very many distinct files grows
  the cache. `--jobs` runs files at once; the shared cache is safe for it, but progress lines interleave.
- The FAIR descriptors describe the project's tables, not a hosted release: schema.org uses each table's source
  file as its distribution and there is no persistent identifier (push the export to Zenodo/OSF for one). RO-Crate
  is a metadata graph only (`ro-crate-metadata.json`), not a zipped crate with the files. The license is recorded
  as given; an SPDX id is linked to its canonical URL.
- A context document's `content_hash` is the step's plan hash: `--changed` compares a fresh document with one
  exported earlier, so it sees content and code changes, but a project edited without being saved is compared as it
  is on disk. A change to DANCR itself (or a library) marks every document changed, which is the safe answer.

- Connectors reach outside the machine: a project that uses Load from a URL or Load from a database is no longer
  strictly offline. A URL source reads the whole file into the results folder before reading it (no streaming from
  the server), and asks the server (a HEAD request, remembered for 30 seconds) whether it changed; with that off,
  it reruns only when its settings change. A server database has no cheap, generic way to notice new rows, so it
  reruns when a named `version_column`'s maximum changes, or when its settings change, or when you force it.
  Credentials written as `${ENV_VAR}` are never stored; a password typed straight into the project file is stored
  as written (it is redacted from everything shown, but not from the file itself).
- Load from a database materializes the result (Polars' `read_database`); very large queries should be narrowed by
  the query or aggregated in the database, not pulled whole.
- Load folder with "every table" unions tables of different shapes into one (missing values blank) and marks the
  origin in a `source_table` column; a workbook larger than 30 MB is not looked through for several tables on one
  sheet, so only its sheets/plain table are read.
- Watch polls (every two seconds by default) and reruns after a quiet poll, so it does not use the operating
  system's file notifications; a folder with a great many files is re-checked each poll.
- The catalog reads every project under a folder; with statistics or samples it computes each table, so a folder of
  many large projects takes a while (use `--jobs`). It lists file names only, never complete paths, as the profile
  export does.
- A packaged RO-Crate is metadata-only by default; including the data (`--copy data`) or the result files
  (`--copy results`) copies them into the crate, which can be large. Units are recorded as UCUM codes only for the
  units `core/units.py` knows; an unknown unit keeps the form it was written in.
- Load from a database (a server) and Load NetCDF / Load HDF5 are included in the install (SQLAlchemy, xarray,
  h5py). SQLite and URL reading need nothing beyond the standard library.

- Bio/lab ingestion (`load_sequences`, `load_variants`, `load_features`, `load_genbank`, `load_markers`)
  turns FASTA/FASTQ, VCF, GFF/GTF/BED, GenBank and PLINK `.map`/`.ped` files into ordinary tables (plain or
  gzip `.gz`/`.bgz` text). Each step reads the file into memory, so a very large one (millions of variants)
  should be limited with `limit` or split first — unlike the streaming table readers. It reads text VCF only
  (not binary `.bcf`), the non-binary PLINK forms (`.map`/`.ped`, not `.bed`/`.bim`/`.fam`), and each GenBank
  record's features (the sequence itself is not read). A PLINK `.bed` is treated as a BED interval file, not a
  PLINK binary genotype file. Variant genotypes are kept as written (`0/1`, `1|1`) with a computed `dosage`
  column (0/1/2, null when missing). Load folder reads a directory of these as one table, like it does for
  tables; `bio_params` passes reader settings (e.g. VCF `samples`) through.

- Documents (`load_document`) are read through MinerU (Apache-2.0). The packaged DANCR bundles a relocatable
  MinerU env (`.dancr-mineru`, with `basic` models) beside the app via `packaging/build-mineru.sh`, so documents
  work with no separate install; a source checkout detects a repo `/.dancr-mineru`, `$DANCR_MINERU_HOME`,
  `$DANCR_MINERU_CMD`, or `mineru-kit`/`mineru` on PATH, or a MinerU V1 endpoint (`$DANCR_MINERU_ENDPOINT`). It
  can also read a folder MinerU already produced with no install (`engine=output`). Reading is local unless `allow_remote` is on. Block/table output shape depends on the
  MinerU version; the parser is tolerant (middle_json, structured_content, content_list v1/v2, markdown, and the
  `mineru parse --json` response). `what=tables` writes one CSV per table under `out_dir`. `dancr doctor` reports
  whether MinerU is present; `dancr inspect` and the MCP `read_document` tool preview/read documents.

- The cross-project graph (`dancr graph`, `docs/adr/0001-graph-identity.md`) has limits worth knowing. A
  dataset's identity is its project file's path relative to the repository root plus the step id, so **moving or
  renaming a project file inside the repository is a new identity** (its datasets and edges are re-created). The
  graph describes relations from a *sample* of each table (the same `understand` pass the answer engine uses);
  row counts and key uniqueness may be estimates. Cross-project links are **by key name only** — two datasets in
  different projects whose resolved key columns share a normalized name get a proposal edge, and the match
  percentage is left unmeasured (0), so it is a hypothesis to check, not a measured join. The build reads each
  changed project's sources; it therefore takes longer the more data changed. It holds the repository lock while
  writing (one writer per repository) but never a project lock, because it does not change project files.
- Hybrid retrieval (`docs/adr/0004-retrieval-embedder-policy.md`) keeps the deterministic offline embedder as the
  default; BM25 and the hybrid fusion are computed at query time from the stored passage text, so their cost
  grows with the index size and a very large index is slower to search than the vector-only path. The hybrid
  weight is fixed (0.5 lexical / 0.5 keyword). There is no neural embedder: the interfaces carry an embedder id
  and the persistent index records it in a `.meta.json` sidecar, so a neural one could be added and recorded
  later, pinned by id and version, without changing the retrieval contract.

- The repository watcher (`dancr watch --repo`, `docs/GRAPH.md` and `docs/GATEWAY.md`) polls (every two seconds
  by default), like the single-project watcher; it uses no operating-system file notifications. It records
  `source_changed` / `dataset_invalidated` events and ripples invalidation across projects through the graph,
  but it **does not re-run by default**: `--rerun` recomputes only the changed projects that are safe to
  recompute (never one whose step would write outside its folder or over its data — those are reported instead).
  The event log is append-only and never compacted, so a very long-running watch grows the file.
- Governance (`dancr policy/approvals/audit`, `docs/GATEWAY.md`) is **opt-in and manual** on the local surfaces:
  nothing is enforced until `<root>/.dancr/gateway/policy.json` exists, and the stdio server/CLI tools are not
  wrapped by default — a caller must use `enforce`/`policy_check` to be governed. The policy language is
  deliberately tiny. The **network transport** (`dancr gateway`) *does* enforce per call: it is loopback-only by
  default (remote needs `--allow-remote` and a TLS-terminating proxy, since TLS is not built in), requires a
  bearer token from the policy, and refuses to start without a policy and a token. It serves one repository root;
  multi-root is out of scope.

- Cross-project question answering (`dancr graph ask`, `docs/adr/0005-cross-project-scope.md`) answers only
  **structural** questions — a shared key, a join path, where a dataset comes from, what it feeds, a name lookup.
  It does **not** execute a pipeline across projects; the natural follow-up ("build this join") materialises into
  one project and reuses the existing engine. The grammar is small and fixed; a question it does not read is
  refused with the dataset candidates, never guessed. Upstream/downstream follow edge orientation, except a
  `link`, which is treated as relating both ways because a shared key does not have a direction.
- Scenario sets (`dancr scenarios`, `docs/SCENARIOS.md`) run each scenario as a separate run — there is no
  fan-out inside the LazyFrame executor — so a large grid costs proportionally and writes one output per
  scenario. Monte Carlo is uniform (or a fixed choice) only, and its seed lives in the spec, not the project.

- Static typing (`uv run mypy dancr`, now in CI alongside ruff and pytest). The target is **Python 3.12**
  because the installed numpy stubs use 3.12 syntax (PEP 695) that mypy cannot parse with a 3.11 target;
  runtime support stays 3.11 and the pytest matrix tests 3.11 and 3.14, so 3.11 compatibility is covered
  behaviourally. Heavy third-party stubs (polars, numpy, scipy, matplotlib, pyproj, shapely, pyogrio, xarray,
  h5py, fastexcel) are not analyzed — their types are stricter/newer than this code's usage — while DANCR's own
  annotations are checked. The **remaining allowance** is a per-module `ignore_errors` list in
  `pyproject.toml`: the pre-existing modules not yet typed to mypy's satisfaction (the Qt UI against PySide6's
  stubs, the answer engine's dynamic dicts, and a handful of older modules; ~440 findings, concentrated in
  `core/recipes/_plan.py`). New modules (identity, repo, graph, events, scenarios, crossask, gateway, rag,
  verify) are checked and clean. Burn the list down by fixing the modules it names; do not add to it without a
  note here.

Design and extension of the answer engine: `docs/ANSWERS.md`. Earlier plans and review worklists
(all items done): `docs/history/`.