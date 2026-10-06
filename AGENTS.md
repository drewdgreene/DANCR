# DANCR for AI coding agents

DANCR is a desktop app where a person's data is a table and every change to it
is a recorded step. Everything the window does is also available headlessly, so
an agent can answer "analyse this dataset with DANCR" by building a project
file, running it, reading the results, and leaving the project open for the
person to explore in the window.

Two interfaces, same engine:

- **MCP server** (preferred): `dancr mcp` (stdio). Register it once in your MCP
  client (for example: `claude mcp add dancr -- dancr mcp`); `--root DIR` sets the
  folder pipelines may be created in (default: the folder it starts in; started in
  or above the home folder, or at the top of a drive, it creates and changes no pipelines
  until `--root` names a folder). Tools: `ask`,
  `suggest_answers`, `change_answer`, `remove_answer`, `understand_data`, `profile`, `list_node_types`,
  `formula_reference`, `inspect_file`, `create_pipeline`, `build_template`,
  `describe_pipeline`, `add_node`, `set_params`, `connect_nodes`,
  `disconnect_nodes`, `remove_node`, `rename_node`, `set_input`, `remove_input`,
  `set_column_label`, `get_dataset_meta`, `set_dataset_meta`, `assistant`, `apply_edits`, `connections`, `run_pipeline`, `run_batch`,
  `node_status`, `get_schema`, `get_sample`,
  `get_stats`, `render_chart` (returns a PNG image), `render_map` (a map PNG), `read_document` (a PDF/Office
  document as content blocks or a table catalog, via MinerU),
  `export_node`, `export_fair` (schema.org / Frictionless / manifest / RO-Crate), `package_project` (a
  self-contained RO-Crate), `record_attestation` (a checkable record of what produced the results),
  `verify_pipeline` (re-run and check it still reproduces), `lineage` (what produced a step, what depends on it),
  `get_proof` (a proof card for one step), `search_knowledge` (search a project's own text index),
  `get_trace` (the Assistant conversation as an audit log), `run_eval` (score questions against the project),
  `catalog` (every project under a folder),
  `graph_build` / `graph_query` / `graph_neighbors` / `graph_path` / `graph_shared_keys` / `graph_ask` /
  `graph_suggest` (the cross-project entity graph and structural QA — see below), `get_events` (the repository
  event log), `policy_check` (whether a governance policy would allow an action), `run_scenarios` (run a step
  across many named/sweep/monte_carlo/sensitivity scenarios), `open_in_gui`.
- **CLI**: `dancr --json <command> …` prints JSON. `dancr ask p.json "total sales by region" --file a.csv`,
  `dancr suggest p.json [--build N]` (`--file` on a workbook adds every sheet),   `dancr answer p.json [ID] [--set stat=mean] [--choose N M] [--remove --steps]`
  and `dancr understand p.json` are the answer commands. `dancr context p.json [--samples] [--jsonl] [--changed old.jsonl]` (alias
  `dancr profile`) prints a knowledge-base document for the project's datasets (see below). `dancr batch p.json --files 'folder/*.csv' --out-dir results`
  runs one step over many files (an output each plus a combined table); `dancr dataset p.json --set license=CC-BY-4.0`
  sets dataset metadata; `dancr fair p.json --format schema.org|frictionless|manifest|rocrate [--out f]` writes a FAIR
  descriptor; `dancr package p.json --out study.rocrate.zip [--copy data|results|all]` writes a self-contained crate;
  `dancr catalog DIR [--jsonl --output index.jsonl]` catalogs every project under a folder; `dancr watch p.json
  [--once] [--batch --files … --out-dir …]` reruns when the data changes.
  `dancr verify p.json --record att.json` records an **attestation** (a checkable record of exactly what produced
  the results); `dancr verify p.json --manifest att.json [--rerun] [--strict-sources]` re-runs and reports whether
  it still reproduces (`verified` | `mismatch` | `incomplete` | `engine-changed`); `dancr lineage p.json NODE
  [--up|--down]` shows what produced a step and what depends on it;   `dancr proof p.json NODE` prints a proof card.
  Build a search index with a `build_index` step, search a node with a `retrieve` step, or `dancr search p.json "query"
  [--node NODE] [--k N] [--retriever lexical|bm25|hybrid]` searches the project's own text index (offline, deterministic);
  the result carries a `provenance` block (the retrievers, their scores and the embedder).
  `dancr graph build DIR [--project FILE] [--force] [--jobs N]` builds the repository's cross-project graph
  under `<DIR>/.dancr/graph/graph.db`; `dancr graph summary|query|neighbors|path|shared-keys DIR …` reads it,
  and `dancr graph ask DIR "what joins A and B?"` / `dancr graph suggest DIR` answer structural cross-project
  questions over it (no execution).
  A project whose file and source files have not changed is carried over from the stored graph without being
  read again (the build is incremental and deterministic). See `docs/GRAPH.md`.
  `dancr events DIR [--follow] [--type T] [--since N]` prints the repository event log; `dancr watch DIR --repo
  [--rerun]` watches every project and records `source_changed` / `dataset_invalidated` (rippling across
  projects through the graph) and, with `--rerun`, recomputes the changed projects that are safe to recompute.
  `dancr policy check DIR PRINCIPAL TOOL`, `dancr approvals DIR [--approve ID|--deny ID]` and `dancr audit DIR`
  are the governance surface; without a `<DIR>/.dancr/gateway/policy.json` nothing is enforced. `dancr gateway
  DIR [--host H] [--port N] [--path /mcp] [--allow-remote]` serves the same MCP tools over the SDK's
  streamable-HTTP transport, loopback-only and bearer-token gated (a principal's policy entry carries `tokens`);
  it refuses to start without a policy and a token, and audits every call. See `docs/GATEWAY.md`.
  `dancr scenarios p.json --set spec.json --out-dir DIR` runs one step across many scenarios (named, sweep,
  monte_carlo, sensitivity), one output each plus a combined table with a `scenario` column; `dancr verify
  p.json --record att.json --scenarios spec.json` (and `--manifest att.json --scenarios spec.json`) folds the
  per-scenario plan and output hashes into the attestation and re-checks them. See `docs/SCENARIOS.md`.
  `dancr eval p.json --set cases.json [--model]` scores questions against the project (the engine, or the Assistant);
  `dancr trace p.json` prints the saved Assistant conversation as an audit log.
  `dancr assistant p.json "…" [--file …] [--build]`
  is the conversational front end (needs a model key in the environment); with `--build` it applies and runs
  the proposal. Same core as the in-app Assistant: `docs/ASSISTANT.md`. `dancr nodes -v` documents
  every step type and setting; `dancr formulas` documents the formula language;
  `dancr template --list` lists starter projects. `KEY=VALUE` settings are
  parsed as JSON when they look like JSON (`sheet=1` is the number 1).

Both interfaces report a step the same way: `node_id`, `title`, `type`,
`status`, `rows`, `columns` (`[{name, dtype}]`), `error`, `messages`, `report`,
`elapsed`, `from_cache`. A run reports `{ok, failed, nodes: {id: step}, problems,
elapsed, cache_dir}` (`dancr --json run`, `run_pipeline`); `dancr --json status p.json
id` and `node_status` report one step. CLI exit codes: 0 ok, 1 a step failed, 2 a
usage or input error, 3 an internal error (details in `dancr log`). `dancr doctor` checks that every
capability (geo, database, NetCDF/HDF5, the model client) is present in this build; `dancr formats` lists the
table formats DANCR reads and what to do with the ones it doesn't (`dancr/help/formats.md`); `dancr inspect FILE`
prints any file's columns, types and first rows (bio formats included). The MCP
reading tools (`get_sample`, `get_stats`, `render_chart`, `export_node`) run the
pipeline automatically when a node is not computed yet; their CLI equivalents
(`dancr sample/stats/chart/export`) need `--run` to do the same.

From a packaged install (the `dist/DANCR` folder, `DANCR.app`), use the
`dancr-cli` binary (`dancr-cli.exe`; `DANCR.app/Contents/MacOS/dancr-cli`) for
every command above, including `dancr-cli mcp`. The `DANCR` binary is the
window and has no console, so it cannot serve JSON or the MCP protocol.

Where MCP may write: pipeline files (`create_pipeline`, `build_template`, and
every tool that edits one) must be `.json` files inside the server's root
folder and outside any `.dancr` folder; `create_pipeline(overwrite=true)` replaces only a file that is
already a DANCR pipeline. Every other file must be inside the folder that holds the pipeline
file, outside its `.dancr` folder, and must not be a data file the pipeline reads: `export_node(out_path)`,
`export_fair(out_path)`, `package_project(out_path)`, `render_chart(out_png)`, the `out_dir` and `manifest` of
`run_batch`, and the `path` of `export`, `workbook` and `report` steps (checked when the step is added or changed,
and before `run_pipeline`, `run_batch` or a reading tool runs it); a step pointing elsewhere fails with a message.
Relative paths are taken from that folder, and the `files` of `run_batch` too. Put the pipeline next to the data
and results you want. `catalog` reads projects under the server's root folder. Reading (`inspect_file`,
`load_file` paths, and the `load_url`/`load_sql` connectors) is not confined — a connector step reaches the
network or a database, so a project using one is no longer strictly offline.

One writer at a time: the MCP server, the CLI and the window each hold a project
file's lock (`.dancr/locks/` next to it) while they read, change and save it, so
none saves over another's change. A command that cannot get the lock within 30 s,
or finds the file changed by something that does not take it, fails with a message
and saves nothing: run it again. When the window reloads a project changed
elsewhere, it does not run by itself a step that was added or repointed to save
outside the project folder or over the project's data; the person runs it.

## Mental model

- A project is a JSON file (`"dancr": 2`) with `nodes` (id, type, title,
  params), `edges` (source → target, input port), `inputs` (named values with
  units) and `columns` (display labels and units). Paths in params are relative
  to the project file's folder (templates store them that way too); saving the
  project into another folder rewrites them so they point at the same files. Node ids are yours to choose (`--id` /
  `node_id`), else `type_N`.
- Every step type is a **transform of Polars LazyFrames**. Sources have no
  inputs; most steps have one input `in`; `combine` has `left` and `right`;
  `stack`, `report` and `workbook` have a multi-input (`tables` / `items`);
  `predict` has `data` and `model`.
- **Running** materialises each step's output to Parquet under
  `<project dir>/.dancr/cache/<project file name>/<node>/` (a project with no file yet uses
  the user cache folder instead, so always save the project file first). Re-runs skip steps whose
  settings, inputs and upstream data did not change (a source file counts as changed when its size,
  time or a sample of its content differs); `export`, `workbook` and
  `report` steps also run again when their file was deleted or changed, or when
  the titles or column labels they show changed. `force` (`run --force`,
  `run_pipeline(force=true)`) recomputes everything and replaces the stored
  results. `dancr clear-cache` deletes the stored results except those another DANCR program is
  using right now (a window showing them, a run or a read in progress). Results of 100M-row steps stay on disk; `get_sample`, `get_stats` and
  `render_chart` read them lazily.
- **Reports**: steps attach findings (fit equation, R², RMSE; gap statistics;
  PASS/FAIL counts) to their status. Read them with `node_status` /
  `dancr status`.
- **Inputs**: `set_input(name, value, unit, note)`. Use them in formulas as
  `[name]`, as filter values, and as `min`/`max` of `check_limits` or chart
  `limits`. Changing one recomputes only the steps that use it. A value with
  leading zeros (`007`) stays text, like a code in a loaded file.
- Errors are plain English and name the column or setting. Fix and re-run.
- A project may also hold **Answers**: a question and the steps that answer it. An Answer has an
  `id`, a `title`, a `terminal` node id (the step whose output answers the question), a `view`, the
  question as a `spec`, the `steps` it built (`{plan key: {node, made, title, inputs, hand?}}`), and the
  `assumptions` it made (each with `choices` that change it). Answers are *not* part of the dataflow —
  the executor and cache ignore them entirely — so removing one never changes what a run computes.
  `dancr --json show` and MCP `describe_pipeline` list them under `answers`.

## Fastest route: let DANCR answer

`ask` / `dancr ask` reads a plain question against the project's own words (column names and labels,
table names, category values, stack labels) with a fixed grammar — no model, fully deterministic —
and builds ordinary steps plus an Answer. `suggest_answers` / `dancr suggest` lists what DANCR can
answer on its own (compare two logs, trend, totals by group, top N, relationship, gaps, unusual
readings, spread, linked or stacked tables, describe), best first. Both work from `understand_data` /
`dancr understand`: each column's role (time, id, category, measure, flag, text), each table's shape
(series, lookup, events, table), and relations (links with cardinality and match %, stacks with labels,
time alignments with a tolerance), computed from every row.

```bash
dancr new shop.json
dancr --json suggest shop.json --file orders.csv --file customers.xlsx   # adds loaders, lists answers
dancr --json ask shop.json "top 10 customers by sales"                   # builds it; reports chips + assumptions
dancr --json answer shop.json answer_1 --set stat=mean                   # change a chip; steps update in place
dancr --json answer shop.json answer_1 --choose 0 0                      # take assumption 0's first alternative
dancr --json run shop.json
```

A question word DANCR does not know fails with `did you mean` hints instead of a guess. A spec is JSON
(`{"recipe": "breakdown", "table": "<node>", "measure": ["<node>", "<column>"], "stat": "sum", "by": ["<node>",
"<column>"], "filters": [{"column": [..], "op": "gt", "value": 2}]}`); column references are
`[table node id, column]`, and the planner adds the links needed to reach columns of other tables.
Changing an answer keeps steps the person edited by hand (reported as `kept_by_hand`) and never touches
steps another answer uses.

Besides the named recipes (`trend`, `breakdown`, `top`, `compare`, `gaps`, `outliers`, `single`, `distribution`,
`linked`, `stacked`, `rows`, `describe`), the answer engine can build the questions people ask most:
`groups` (`compare treated and control`, `is mass higher in treated samples`, `t test inner area`, `compare before and after`:
the groups compared on every number with a test, an effect size and a sentence; spec `by` or `columns` for groups kept
in a column each, `test`, `paired`; it is the first suggestion for a small table of measurements in groups — a study),
`change` (`what changed`, `this month vs last`), `explain` (`what drives sales`, `why did it drop`),
`drivers` (`what relates to price`), `quality` (`check the data`, `is the data clean`), `forecast`
(`where is pressure heading`, `when will we hit the limit`), and the place recipes: `map` (`map the water points`,
`map the villages by status`), `density` (`where are the clusters`, `density of water points`) and `nearest`
(`nearest clinic to each village`, `how far to the nearest clinic`). A table is a place when its columns name a
latitude and a longitude (lat/lon, latitude/longitude, clat/clon, easting/northing, and plain x/y only when the
ranges leave no doubt). Each emits a **finding** — one plain sentence
attached to the step's `report["finding"]["statement"]`; a run reports the best one as `headline`.

A question the grammar cannot read is repaired and re-read before it is ever refused: a mistyped word is
matched to the project's own words (`core/lexicon.py`), and if that fails the question is matched against a
bank of every answer the project can build (`core/bank.py`, threshold-gated, so a loose question finds its
answer and a gibberish one still fails). A project remembers the words it has learned and the questions
recently asked (`core/memory.py`, kept in the project's `meta`; the format is unchanged). `dancr --json ask`
reports `source` (`grammar`|`matched`), `matched`, and any `corrected` spellings.

## Recipe: two logs of the same quantity, one noisier than the other

Both logs here have the columns `time` and `value` (after `combine`, B's
column is `value_2`); use the names in your files (MCP `inspect_file`, or `dancr --json schema compare.json a` after the first step).

```bash
dancr new compare.json
dancr add compare.json load_file --id a --title "Log A" --set path=log_A.csv
dancr add compare.json load_file --id b --title "Log B" --set path=log_B.csv
dancr add compare.json find_gaps --id gaps_a --after a
dancr add compare.json combine --id aligned --after a --port left --also-after b \
      --params '{"method":"nearest_time","tolerance":"40ms"}'
dancr add compare.json remove_outliers --id clean --after aligned \
      --params '{"columns":["value_2"],"method":"rolling","window":101,"threshold":8}'
dancr add compare.json fit_curve --id fit --after clean --params '{"x":"value","y":"value_2","kind":"linear"}'
dancr add compare.json time_buckets --id per_min --after fit --params '{"every":"1m","default_stats":["mean","min","max"]}'
dancr add compare.json chart --id chart --after per_min --params '{"x":"time","series":[{"column":"value_2_residual_mean","label":"B minus fitted A"}],"y_label":"units"}'
dancr add compare.json report --id report --after chart --port items --also-after gaps_a --params '{"title":"Comparison of A and B","path":"report.html"}'
dancr --json run compare.json
dancr --json status compare.json fit         # equation, r2, rmse per fit
dancr sample compare.json gaps_a --rows 10
dancr chart compare.json chart --out residual.png
dancr open compare.json                      # hand it to the person; the window reloads on every file change
```

The same flow via MCP: `create_pipeline`, `add_node(... after=..., port=...,
also_after=[...])`, `run_pipeline`, `node_status`, `render_chart`, `open_in_gui`.
For a quick demo on generated data: `dancr template compare demo.json` or
`build_template(path, "compare")`.

## Recipe: occasional samples against a continuous log

```bash
dancr new samples.json
dancr add samples.json load_file --id log --set path=log.csv
dancr add samples.json enter_data --id samples --params '{"columns":[{"name":"sampled_at","type":"datetime"},{"name":"result","type":"number"}],"rows":[["2024-06-03 09:00",4.1],["2024-06-10 09:00",3.6]]}'
# one statistic keeps the column names (value, temperature); several add a suffix (value_mean, value_max)
dancr add samples.json summarise_around --id around --after samples --also-after log \
      --params '{"window":"24h","side":"before","columns":["value","temperature"],"stats":["mean"]}'
dancr inputs samples.json "area" 12.5 --unit m2 --note "from the drawing"
dancr add samples.json calculate --id per_area --after around --params '{"formulas":[{"name":"result_per_area","expr":"[result] * [value] / [area]"}]}'
dancr add samples.json fit_curve --id fit --after per_area --params '{"x":"temperature","y":"result_per_area","kind":"linear"}'
dancr --json run samples.json
```

Here `log.csv` has the columns `time`, `value` and `temperature`.

## Recipe: water points on a map, and the nearest clinic to each village

```bash
dancr new places.json
dancr add places.json enter_data --id points --params '{"columns":[{"name":"lat","type":"number"},{"name":"lon","type":"number"},{"name":"status","type":"text"}],"rows":[[-1.29,36.82,"on"],[-1.25,36.90,"on"],[-1.40,36.70,"off"],[-1.20,37.10,"on"]]}'
dancr add places.json make_point --id pt --after points --set lat=lat --set lon=lon
dancr add places.json map --id map --after pt --params '{"lat":"latitude","lon":"longitude","color_by":"status","title":"Water points"}'
dancr add places.json enter_data --id clinics --params '{"columns":[{"name":"clat","type":"number"},{"name":"clon","type":"number"},{"name":"clinic","type":"text"}],"rows":[[-1.29,36.82,"A"],[-1.20,37.10,"B"]]}'
dancr add places.json combine --id near --after pt --port left --also-after clinics \
      --params '{"method":"nearest_feature","left_lat":"latitude","left_lon":"longitude","right_lat":"clat","right_lon":"clon","max_distance":"10km","units":"km"}'
dancr add places.json points_grid --id grid --after pt --params '{"lat":"latitude","lon":"longitude","size":"5km","count_column":"points"}'
dancr add places.json map --id density --after grid --params '{"lat":"cell_lat","lon":"cell_lon","cell_size":"5km","color_by":"points","title":"Water points per 5km cell"}'
dancr add places.json report --id report --after map --port items --also-after density --params '{"title":"Where the water points are","path":"places.html"}'
dancr --json run places.json
dancr map places.json map --out points.png     # or the map node's own settings
```

The same flow via MCP: `add_node("make_point", …)`, `add_node("map", …)`,
`add_node("combine", {"method": "nearest_feature", …}, after=…, port="left", also_after=[…])`,
then `render_map`. Offline country outlines are drawn when `basemap` is on; nothing reaches the network.

## Settings cheat-sheet (full list: `dancr nodes -v` or `list_node_types`)

- `load_file`: `path` (CSV/TSV/text, Excel, Parquet, GeoJSON, GeoPackage `.gpkg` or shapefile `.shp`), `sheet`, `layer` (GeoPackage: which layer, blank = the first; `dancr suggest --file` adds one step per layer), `has_header`, `layout` (`auto`: reads the sheet as laid out — title lines, tables side by side under banners or in blocks under titles become one table with a `group` column, section lines become a `group` column, labels written once per run are filled down, names in two rows are joined, summary rows under the data (AVERAGE, Total …) and empty template rows are left out and the summary checked against the data; the step's `report["layout"]` and messages say what was done; `as_is`: every row under the column names), `table` (which table on a sheet that holds several, from 1; `dancr suggest --file` adds one step per table), `skip_rows`, `separator` (auto), `parse_dates` (also joins a Date and a time-of-day column into `<Date> <Time>`), `parse_numbers` (reads `1,234.50` `£99` `31.5%` `(120)`; codes with leading zeros stay text), `date_format`, `day_first` (only for dates like 01/05/2024 that read either way; default month/day, and a run reads the whole column to choose), `time_zone` (for times written with a UTC offset: empty keeps the file's own offset when it has one throughout, else UTC; or a name such as `Europe/London`), `decimal_comma`, `encoding` utf8|latin1, `infer_rows`, `ignore_errors`, `columns`. A file that is not a table is refused with a plain reason and a route forward, never misread as CSV: HDF5/NetCDF point at `load_hdf5`/`load_netcdf`; the scientific formats (FASTA/FASTQ, VCF, GFF/GTF/BED, GenBank, PLINK) point at the bio steps below and documents (PDF/Office/EPUB/HTML) at the 'Load document' step (a dropped file gets the right step automatically); the rest (JSONL, zip, XML, Markdown…) say to convert first.
- `load_folder` (all the files in a folder or glob as one table): `path` (a folder or glob), `pattern` (`*.csv`), `recursive`, `source_column` (the file name column; blank for none), `tables` (`first` | `all` every sheet/table/layer | `match` a named one), `table_match`, `table_name_column`, `unify` (`diagonal` union of columns | `strict` same columns only | `text` every column as text), `on_error` (fail | skip), and the read options of `load_file` (`has_header`, `layout`, `parse_dates`, `parse_numbers`, `encoding`, `decimal_comma`…). Each file is read by the same reader as `load_file`, and a scientific file (FASTA/FASTQ, VCF, GFF/GTF/BED, GenBank, PLINK, plain or `.gz`) by its own bio step — so a folder of VCFs is one table. `bio_params` (a mapping) is passed to that reader, e.g. `{"samples": "genotype"}`. The cache notices when any member file changes, is added or removed.
- `load_url` (a table at a URL): `url`, `format` (auto|csv|tsv|parquet|json), `headers` (a mapping; put tokens in `${ENV_VAR}`), `check_remote` (ask the server whether it changed), `timeout`, and the CSV read options. http(s); standard library only.
- `load_sql` (a database query or table): `connection` (a SQLite file or `${PG_DSN}`-style server URL; **secret**, redacted wherever settings are shown), `query` or `table`+`schema`, `version_column` (a value whose max says new rows arrived). SQLite is built in; a server (SQLAlchemy and a driver) is included in the install.
- `load_netcdf` / `load_hdf5` (scientific arrays): `path` and a `variable` / `dataset` (xarray and h5py are included in the install).
- `load_sequences` (FASTA/FASTQ): `path` (plain or `.gz`), `format` auto|fasta|fastq, `include_sequence`, `limit`; one row per sequence (`id`, `description`, `length`, `gc_percent`, `sequence`).
- `load_variants` (VCF, plain or `.gz`): `path`, `info` (split INFO into typed columns), `samples` none|genotype|full, `limit`; one row per variant, or one per variant×sample carrying the FORMAT fields and a computed `dosage` (0/1/2).
- `load_features` (GFF3/GTF/BED): `path`, `format` auto|gff|bed, `attributes` (keys lifted into columns), `limit`; one row per feature.
- `load_genbank`: `path`, `qualifiers` (keys lifted), `limit`; one row per feature (locus, type, start/end, strand, gene/product/note).
- `load_markers` (PLINK .map/.ped): `path`, `genotypes`, `limit`; one row per marker, a genotype column per sample.
- `load_document` (PDF/Office/EPUB/HTML/images, via MinerU): `path` (file/folder/glob, or a MinerU result for `engine=output`), `what` blocks|tables, `engine` auto|output|command|endpoint, `tier` flash|basic|standard|advanced, `pages`, `include`, `out_dir`, `mineru_cmd`, `endpoint`/`token`, `allow_remote`, `timeout`. Blocks give `doc, doc_id, page, block, type, text, bbox, locator`; tables give `doc, page, table_index, caption, n_rows, n_cols, csv, locator` (each table written to a CSV under `out_dir`). MinerU is an external Apache-2.0 tool, detected (env `DANCR_MINERU_CMD`, else `mineru-kit`/`mineru` on PATH) or via `DANCR_MINERU_ENDPOINT`. The packaged DANCR bundles MinerU in `.dancr-mineru` (built by `packaging/build-mineru.sh`, found next to the DANCR/dancr-cli binaries); a source checkout can use a repo-local `.dancr-mineru` or `uv sync --extra mineru`. `dancr doctor` reports whether it is present.
- `choose_columns`: `mode` keep|drop, `columns`, `rename`. `sort`: `columns`, `descending`. `remove_duplicates`: `columns`, `keep` first|last|none.
- `fix_missing`: `method` drop|drop_all|value|forward|backward|interpolate|mean|zero, `value`, `columns`. `change_type`: `columns`, `to` number|integer|text|datetime|bool, `date_format`, `epoch_unit`, `time_zone` (as `load_file`).
- `unpivot` (columns into rows): `columns` (e.g. Jan … Dec), `name_column` (month), `value_column` (value), `year` (month columns then also get a `date`). Answers add it by themselves in front of a wide table.
- `take_sample`: `mode` first|last|every|random, `rows`, `every`, `fraction`, `seed`. `stack`: `label_column`, `labels`; connect tables to `tables`.
- `pivot` (rows into columns, the inverse of `unpivot`): `index` (columns kept as rows), `columns` (whose values become the new columns), `values` (cell values; blank = count rows), `agg` sum|mean|min|max|median|first|last|count, `fill_zero`, `max_columns`. Report: `new_columns`, `rows`.
- `describe_dataset` (a data dictionary): `definitions` (a column→description mapping). Output one row per column with `role`, `type`, `unit`, `definition`, `missing`, `missing_percent`, `distinct`, `min`, `max`, `mean`, `example`; `report` has `columns`, `with_units`, `with_blanks`, `undocumented`.
- `build_index` (offline text index): inputs `items` (tables/documents); `text_column` (blank = a `text` column, else all columns joined), `chunk_chars`, `chunk_overlap`, `dim`, `id_column`, `index_path`, `changed_path`, `max_chunks`. Output one row per passage with `dataset`, `doc_key`, `content_hash`, `chunk_id`, `chunk_index`, `text`, `vector` (a fixed-size float array) and the source columns. With `index_path` it keeps a persistent index: each document's text+metadata is hashed, and only documents whose `content_hash` changed are embedded again — the report's `index` block gives `added`/`changed`/`unchanged`/`removed` documents and `reused_chunks`/`embedded_chunks`. `id_column` names the column that identifies a document across runs (blank falls back to the source row's position). With `changed_path` (a `dancr context --jsonl` export, or its `--changed` form) a whole dataset whose content has not moved is **skipped** and carried over from the persistent index without being read again; the report adds `skipped_datasets` and `carried_chunks`. A `--changed` feed lists only what moved (datasets absent are skipped); a full context is compared per dataset by `content_hash`.
- `retrieve` (search an index): input `in` (a `build_index` table); `query`, `k`, `min_score`, `retriever` (`lexical` offline embedding, the default | `bm25` keyword | `hybrid`), `allow_restricted`. Output the best passages ranked, each with `rank`, `score` and its source columns; the report carries a `retrieval` provenance block. `dancr search p.json "query" [--node NODE] [--k N] [--retriever …]` and the MCP `search_knowledge` tool run the same search. When the index carries a `sensitivity` column, confidential/restricted passages are withheld unless `allow_restricted`.
- `check_contract` (a data contract): `in` (a table) plus optional `references` (other tables); `contract` (JSON text) or `contract_path` (a file); `output` rows|issues. Contract JSON: `{"columns": {"c": {"kind": "number"|"text"|"date/time", "required": true, "unique": true, "min": .., "max": .., "allowed": [..], "regex": ".."}}, "rules": [{"name": "..", "expr": ".."}], "references": [{"column": "..", "to_node": "..", "to_column": ".."}]}`. With no contract it infers a draft into `report["inferred_contract"]`. Report: `checks`, `issues`, `errors`.
- `diff_tables` (two versions): inputs `a` (before), `b` (after); `key`, `compare`, `tolerance`, `schema_only`. Output rows `change_type` added|removed|changed, `column`, `before`, `after`, `delta`. Report: `changed`, `added`, `removed`, `columns_added/removed/retyped`.
- `label_sensitivity`: `level` public|internal|confidential|restricted (whole table) or `column` (per-row labels); `name` (default `sensitivity`). The label is a column that travels with the rows.
- `redact`: `columns`, `method` mask|hash|drop, `keep`, `mask` — mask keeps the first few characters, hash gives a stable pseudonym.
- `enter_data`: `columns` = `[{"name","type": text|number|datetime|bool}]`, `rows` = list of lists.
- `keep_rows`: `mode` keep|remove, `conditions` = `{"match":"all"|"any","rules":[{"column","op","value","value2"}]}`  (also **`strict_columns`**, as in New column)
  with ops `eq ne gt lt ge le between contains not_contains starts ends in empty not_empty true false year month` (`year`/`month` on date columns: `2024`, `3` or `March`); values may be input names; or `formula`. As in Excel (and as `=`/`<>` in formulas), text matches ignoring case, a blank cell counts as not equal to (and not containing) any value, blank cells are matched with `empty` / `not_empty`; a rule without the value it needs is left out until it has one (the step says so in `messages`). `in` takes a list, or text separated by `;` or `,` (`"1,000; 2,500"` for numbers with thousands separators).
- `calculate`: `formulas` = `[{"name": "diff", "expr": "[b] - [a]"}]`, `only_new`, **`strict_columns`** (a column name in a formula that is not exact — only spaces/underscores differ — is refused rather than matched; a forgiving match is otherwise named in the step's `messages`). Whole-number `+ - *` work in 64 bits.
- `fix_values`: `fixes` = `[{"row": 1-based, "column", "value", "was", "note"}]`. `was` is what the cell held (`null`: blank); when a cell no longer holds it (rows moved upstream) the step fails and names those corrections instead of changing the wrong cell. Leave `was` out to skip the check.
- `combine`: `method` match|nearest_time|side_by_side; match: `on`, `right_on`, `how` (text keys match ignoring case, like VLOOKUP; the key keeps the values as written); nearest_time: `left_time`, `right_time`, `direction`, `tolerance` (e.g. `500ms`); `suffix`.
- `time_buckets`: `every` (`1s 1m 15m 1h 1d`, calendar `1mo 1q 1y`), `by` (also split per value of these columns), `columns` (empty = every number column), `default_stats` (mean median min max std sum count first last n_unique rows — `count` is filled values, `rows` every row; one statistic keeps column names, several add `_stat`), `time_column`, `aggregations`, `count_column`.
- `rolling`: `columns`, `stat` mean|median|min|max|std|sum, `window` (rows like `20` or a span like `30s`), `time_column`, `centered` (default true, as the `ROLLING_*` formulas; a span `w` then covers `[t - w/2, t + w/2]`; otherwise `(t - w, t]`), `replace`.
- `rate_of_change`: `columns`, `time_column`, `per` s|m|h|d, `span`. `find_gaps`: `time_column`, `expected`, `factor`. `regular_grid`: `time_column`, `every`, `method` nearest|backward|forward|interpolate.
- `summarise_around`: `window`, `side` before|after|around, `columns`, `stats`, `sample_time`, `log_time`.
- `remove_outliers`: `columns`, `method` rolling|zscore|iqr|range, `window`, `threshold`, `iqr_factor`, `local_spread`, `min`, `max`, `action` remove|blank|flag|clip, `flag_column`, `by` (zscore/iqr: each group judged against its own values).
- `fit_curve`: `x`, `y`, `kind` linear|saturating|exponential|power|logarithmic|polynomial, `degree`, `group`, `predicted_column`. Report: `fits` = list of {equation, r2, rmse, params, n, group}.
- `predict`: inputs `data` (x values) and `model` (a fit_curve step); `x`, `output`, `group`.
- `check_limits`: `column`, `min`, `max` (numbers or input names; a minimum above the maximum fails), `action` flag|remove|keep_failing, `flag_column`. Report: verdict PASS|FAIL|NOTHING CHECKED (every value blank), checked, outside, blank, rows.
- `group_summary`: `by`, `columns`, `default_stats`, `aggregations`, `count_column`. `summarize`: `columns`.
- `chart`: `kind` line|scatter|histogram|bar, `x`, `series` `[{"column","color","label"}]`, `color_by`, `split_by` (one panel per value), `limits` `[{"value","label"}]`, `fit`, `mean_line`, `column`, `bins`, `category`, `value`, `stat` mean|sum|count|min|max|median, `error` (bar of averages: `se`|`sd`|`ci95` error bars, groups in the order they appear), `title`, `y_label`, `break_gaps`, `log_y`.
- `export`: `path` (.csv | .tsv | .txt | .parquet | .xlsx). `workbook`: `path` (.xlsx); connect tables to `items`.
- `report`: `title`, `path` (.html), `notes`, `company`, `author`, `blocks` (`[{"type":"heading"|"text","text"}, {"type":"item","index"}]`, optional), `pdf`, `max_rows`, `include_stats`; connect charts/tables to `items`.
- `compare_periods` (this period vs the one before): `time_column`, `every` (the period, e.g. `1mo`), `measure` (blank = count rows), `stat`, `by` (optional groups). Output: `previous`, `current`, `change`, `change_percent` per group. Says what rose and fell.
- `contribution` (what drives a total or a change): `by`, `measure`, `stat`, and optional `time_column` + `every` to explain the change since. Output: each group's value and `share_percent`/`cumulative_percent`, or `previous`/`current`/`change`/`contribution_percent`.
- `associations` (what relates to what): `target` (blank = every pair), `columns`. Uses every row of the first 100,000: Pearson r for numbers, eta (the root of eta², on r's scale) for a number against a group, Cramér's V for two categories. Output ranked by strength.
- `check_data` (is the data trustworthy): `columns` (blank = all). One pass: blanks, exact-duplicate rows, **repeated values in a key/id column** (`report["duplicate_keys"]`), numbers stored as text, columns with one value, and calculated columns (`D = PA - LA`, `m / LA * 10000`, found from the numbers) with the rows that break the rule (`report["calculated"]`). Output one row per column with an `issue` and `severity`.
- `compare_groups` (do the groups differ): `by` (the group column), `columns` (numbers; blank = all), `test` `auto`|`welch`|`student`|`rank`|`none`, `pair_by` (a column matching the same thing across two groups: paired t-test / Wilcoxon), `label` (names rows in notes), `size_check` (default true), `relative_to`. Output one row per number: `measure`, `unit`, per group `<g> n/mean/SD/SE/median`, `difference`, `difference (%)`, `test`, `statistic`, `p value`, `effect size` (Hedges' g, dz or eta²), `effect`, `result`; rows `<number> per <size>` when the size check changed a result. Report: `results`, `relative`, a `finding`.
- `forecast` (where it is heading): `time_column`, `column`, `horizon` (steps), `method` `linear`|`seasonal`, `cycle` (`weekday`|`hour`|`month`), `every` (step; blank = inferred), `threshold` (optional). Output: future times and the projected value with `_lower`/`_upper` (about 95%).
- `make_point` (Location): `lat`, `lon` (numeric columns), `validate` (blank impossible coordinates), `drop_invalid`, `lat_out`, `lon_out`. Adds clean `latitude`/`longitude` columns.
- `distance` (Location): `method` `between`|`from`; between: `lat1`,`lon1`,`lat2`,`lon2`; from: `lat`,`lon` + `to_lat`,`to_lon`; `units` `km`|`m`|`mi`|`nmi`|`ft`; `output`. Great-circle, one fixed sphere radius, deterministic.
- `points_grid` (Location): `lat`, `lon`, `size` (degrees like `0.1`, or a length like `5km`), `columns`, `default_stats`, `count_column`, `aggregations`. Output has `cell_lat`, `cell_lon` and the statistics — ready to map.
- `map` (Location, sink, no materialise): `lat`, `lon`, `color_by`, `size_by`, `label`, `cell_size` (blank = points, else grid squares), `basemap`, `extent` `auto`|`world`, `projection` `equirectangular`|`mercator`, `title`. Draws in the views; `render_map` / `dancr map` produce a PNG. Offline Natural Earth outlines; no network.
- `project` (Location): `easting`, `northing`, `utm_zone` (or `crs`, e.g. `EPSG:32737`), `south`, `lon_out`, `lat_out`. Turns projected coordinates into longitude/latitude. UTM is built in; any other EPSG code uses pyproj, included in the install. Blank coordinates stay blank.
- A GeoJSON, GeoPackage or shapefile loads as one row per feature: its attributes are columns, plus `geometry` (WKT) and a `longitude`/`latitude` centre. A layer in another CRS is reprojected to WGS84 lon/lat on load (the projection and geometry libraries are included in the install; a `.gpkg` with several layers becomes one load step per layer).
- `combine` also has `method` `nearest_feature`: `left_lat`,`left_lon`,`right_lat`,`right_lon`,`max_distance` (e.g. `10km`, blank = any), `units`, `distance_column`, `near_how` `left`|`inner`. Adds a distance column and the nearest place's columns.
- `combine` also has `method` `within` (point-in-polygon): `left_lat`,`left_lon`,`right_geometry` (a WKT polygon column), `place_column`, `near_how` `left`|`inner`. Looks up which polygon each point falls inside (holes handled) and adds the place's columns.
- `combine` also has `method` `fuzzy` (text keys that are almost the same): `left_key`, `right_key`, `algorithm` `normalized`|`nearest`, `threshold` (nearest), `max_candidates`, `score_column`, `how`, `suffix`. `normalized` matches after tidying case, spaces and punctuation; `nearest` picks the closest key within the similarity. Rows that match nothing keep a blank key and score; the report gives the match percentage.
- `export` accepts `.geojson` as well as `.csv`/`.tsv`/`.parquet`/`.xlsx` (needs latitude/longitude, or a `geometry` column).

## Indexing a project for a knowledge base (and the Python API)

`profile` (MCP) / `dancr context` (alias `dancr profile`) emits one **knowledge-base document** per project: for
each dataset its schema (every column's role, type, unit and range), the per-column statistics, up to 100 example
rows (`--samples`), and a one-paragraph **doc card** — deterministic prose (shape, size, source, time span,
columns, relations) a search index can embed as language. Together they let an agent's RAG retrieve DANCR's
*documents* while the engine keeps computing exact figures through the other tools; nothing is inferred by the
model. Shape: `{kind: "dancr.context", version, engine_version, fingerprint, generated_at, project, dataset,
tables[], relations[], inputs[], skipped{}, documents[]}`, where each `tables[]`/`documents[]` entry carries a
**`content_hash`** (the step's plan hash, which folds in the code fingerprint, settings, inputs and source files)
and `documents[]` is `{id, node, title, text, content_hash, rows, source, stats?, sample?}`. `--jsonl` /
`context_jsonl` gives one JSON object per dataset, the unit an index ingests. `--changed OLD.jsonl` (MCP
`profile(changed=…)`) returns only the datasets whose `content_hash` differs from an earlier context, so a RAG
re-indexes what moved, never a stale document. Statistics and samples need a table's result; a source not run
yet is computed first (samples alone may fall back to a preview of its first rows). This shares its schema layer
with the Assistant's own profile (`dancr/core/profile.py`).

```bash
dancr context shop.json                         # human summary: a doc card per dataset
dancr --json context shop.json                   # the whole document (schema + stats), for a script
dancr context shop.json --samples --jsonl --output kb.jsonl   # one line per dataset, to embed
dancr context shop.json --jsonl --changed kb.jsonl            # only what changed since the last export
```

**FAIR descriptors.** `dancr fair p.json --format …` / MCP `export_fair` (Python `dancr.export_fair`) turns the
same context into a standards-shaped record: `schema.org` (JSON-LD `Dataset`, for Google Dataset Search),
`frictionless` (a Data Package `datapackage.json`), `manifest` (provenance: engine and library versions, the code
fingerprint, each source file's size/time/content sample, and every step's plan hash, rows and elapsed time), and
`rocrate` (a metadata-only RO-Crate graph). Dataset-level metadata (creator, license, description, keywords,
citation…) is set with `dancr dataset p.json --set license=CC-BY-4.0` / MCP `set_dataset_meta` and lives in the
project's `meta["dataset"]`, so it travels with the file (an older DANCR preserves it). `dancr package p.json
--out study.rocrate.zip [--copy data|results|all]` / MCP `package_project` writes a self-contained RO-Crate
(directory or `.zip`): the descriptors, the pipeline file and the run manifest, plus optionally the source data
and the files the project wrote. Column units are emitted as UCUM codes where known (`g`, `g/m2`, `Cel`, `%`…),
with the written unit kept beside them. `dancr catalog DIR [--jsonl] [--changed old]` / MCP `catalog` describes
every project under a folder and indexes each dataset's `content_hash`, so a team can re-index only what changed.

The same engine is importable as a Python SDK (`import dancr`), for scripts and notebooks — an agent should
still prefer MCP. Names are imported on first use, so `import dancr` stays light:

```python
import dancr
project = dancr.read_project("shop.json")
with dancr.editing("shop.json") as p:                  # saved under the file's lock
    dancr.add_step(p, "load_file", {"path": "orders.csv"}, node_id="orders")
ctx = dancr.build_context(dancr.read_project("shop.json"), samples=True)   # the KB document
answer = dancr.ask_question(project, "total sales by region")              # spec + steps, deterministic
```

The public names: `read_project`, `editing`, `project_lock`, `ProjectBusy`, `Pipeline`, `Executor`, `NodeState`,
`add_step`, `build_template`, `data_model`, `suggestions`, `build_answer`, `ask_question`, `change_answer`,
`assistant_turn`, `connection_map`, `run_record`, `node_record`, `run_batch`, `build_context`, `context_jsonl`,
`context_text`, `context_changes`, `export_fair`, `dataset_jsonld`, `datapackage`, `run_manifest`,
`project_profile`, `table_card`, and the cross-project graph (`Graph`, `build_graph`, `load_graph`,
`graph_summary`, `graph_query`, `graph_neighbors`, `graph_path`, `graph_shared_keys`, `graph_slice`), the
repository event log (`read_events`, `append_event`, `invalidate`, `watch_repo`) and governance (`Policy`,
`Principal`, `evaluate_policy`, `policy_category`, `load_policy`, `save_policy`, `policy_check`, `enforce`,
`request_approval`, `list_approvals`, `decide_approval`, `audit_records`), cross-project structural QA
(`cross_ask`, `cross_suggest`) and scenario sets (`run_scenarios`, `scenario_set`).

## The cross-project entity graph (A1)

A repository (the folder holding your projects) can be indexed once into a **graph** of datasets, columns and
sources and the relations between them — each link, stack, time alignment, nearest-place and containment edge
carrying its evidence and confidence. It is derived state under `<root>/.dancr/graph/graph.db` (SQLite), always
rebuildable, never a source of truth.

```bash
dancr graph build .                     # one vertex per project/dataset/column/source, one edge per relation
dancr --json graph query . --text orders
dancr graph neighbors . shop.json#orders
dancr graph path . a.json#orders b.json#customers
dancr graph shared-keys .               # keys that link datasets across different projects
```

- **Incremental and deterministic.** A project whose file bytes and source-file stamps have not changed is
  carried over from the stored graph without being read again; the same repository always produces the same
  graph (see `docs/adr/0001-graph-identity.md`).
- **Identity is readable.** A dataset is `<project file relative path>#<node id>` (the catalog key); a column is
  scoped to its dataset. Moving a project file inside the repo is, by decision, a new identity.
- **Cross-project links are proposals, not joins.** Two datasets in *different* projects whose resolved key
  columns have the same normalized name get a `link` edge with a fixed confidence; the match percentage is left
  unmeasured (0) and the evidence says so. No value scan, no model, no silent join.
- **Sensitivity is respected.** A project can declare `meta["sensitivity"]`, or a `Label sensitivity` step sets
  its own level; confidential/restricted datasets (and the edges touching them) are withheld from `graph_query`
  and exports unless `allow_restricted` is passed — matching `search_knowledge`.

## The Assistant (the in-app AI, for people, not for agents)

The window has an optional **Assistant** (button next to *Ask a question*, Ctrl+Shift+J): a chat whose model
proposes answers and the deterministic engine proves them. It is the app's own client of the same engine —
an agent should still drive the CLI/MCP directly, not the Assistant. Under the hood:

- `dancr/core/assistant/` is pure core (no Qt): `client.py` (provider + OpenAI-style HTTP + fake),
  `context.py` (re-exports the project profile from `dancr/core/profile.py`, which the knowledge-base export
  shares), `tools.py` (read tools, `list_connections`, and the terminal `ask_choice`/`propose`), `session.py`
  (the turn loop and the unverified-figure check), `store.py` (thread in `meta["assistant"]`).
- Key, endpoint and model live in QSettings (`assistant/api_key`, `assistant/base_url`, `assistant/model`),
  or `DANCR_ASSISTANT_API_KEY` / `DANCR_ASSISTANT_BASE_URL` / `DANCR_ASSISTANT_MODEL` /
  `FIREWORKS_API_KEY`. `DANCR_ASSISTANT_FAKE=1` runs it with a scripted fake and no key.
- Trust contract: the model never states a number that no tool result contains (unbacked figures are
  flagged), `propose` changes nothing (the window applies it, undoably), data is delimited and never
  treated as instructions, and sample rows are off by default. The first time data would leave the machine
  it says what is sent and asks; the ⋮ menu can revoke it.
- Headless: `dancr --json assistant p.json "question" [--file …] [--build] [--samples]` and the MCP
  `assistant` tool run one turn (both call `headless.assistant_turn`); `--build` applies and runs the
  proposal. The window's built card can also reveal the steps on the canvas, save them as a new project,
  or replace the canvas (one undo step). The engine's connection map (links with match % and cardinality)
  is saved with the project and available as `dancr --json connections p.json [--recompute]` and the MCP
  `connections` tool. Design and extension: `docs/ASSISTANT.md`.

## Conventions that keep people happy

- Give steps readable titles (`--title` / `rename_node`); they are what the person sees in the project list.
- Prefer `time_buckets` before charting or exporting anything with millions of rows, and pass `columns` so the output stays readable.
- Set display names and units with `set_column_label` / `dancr columns`; they appear in the table header, on charts and in reports.
- Put thresholds and constants in Inputs rather than hard-coding them in formulas, so the person can change them on the Inputs page.
- For a deliverable, end with a `report` step and run it; also `open_in_gui` / `dancr open` so the person can explore.
- The window and the CLI/MCP may run and change the same project at the same time; the cache and the project file's lock are safe for that.
- Never write into the `.dancr` cache folder yourself. Exports and chart files go next to the project file (MCP enforces this).
- If something goes wrong, `dancr log` prints the log file path and its last lines; `dancr doctor` checks the build's capabilities.
