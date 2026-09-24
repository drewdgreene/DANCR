# DANCR for AI coding agents

DANCR is a desktop app where a person's data is a table and every change to it
is a recorded step. Everything the window does is also available headlessly, so
an agent can answer "analyse this dataset with DANCR" by building a project
file, running it, reading the results, and leaving the project open for the
person to explore in the window.

Two interfaces, same engine:

- **MCP server** (preferred): `dancr mcp` (stdio). Register it once in your MCP
  client (for example: `claude mcp add dancr -- dancr mcp`); `--root DIR` sets the
  folder pipelines may be created in (default: the folder it starts in). Tools: `ask`,
  `suggest_answers`, `change_answer`, `remove_answer`, `understand_data`, `list_node_types`,
  `formula_reference`, `inspect_file`, `create_pipeline`, `build_template`,
  `describe_pipeline`, `add_node`, `set_params`, `connect_nodes`,
  `disconnect_nodes`, `remove_node`, `rename_node`, `set_input`, `remove_input`,
  `set_column_label`, `run_pipeline`, `node_status`, `get_schema`, `get_sample`,
  `get_stats`, `render_chart` (returns a PNG image), `export_node`, `open_in_gui`.
- **CLI**: `dancr --json <command> …` prints JSON. `dancr ask p.json "total sales by region" --file a.csv`,
  `dancr suggest p.json [--build N]`, `dancr answer p.json [ID] [--set stat=mean] [--choose N M] [--remove --steps]`
  and `dancr understand p.json` are the answer commands. `dancr nodes -v` documents
  every step type and setting; `dancr formulas` documents the formula language;
  `dancr template --list` lists starter projects. `KEY=VALUE` settings are
  parsed as JSON when they look like JSON (`sheet=1` is the number 1).

Both interfaces report a step the same way: `node_id`, `title`, `type`,
`status`, `rows`, `columns` (`[{name, dtype}]`), `error`, `messages`, `report`,
`elapsed`, `from_cache`. A run reports `{ok, failed, nodes: {id: step}, problems,
elapsed, cache_dir}` (`dancr --json run`, `run_pipeline`); `dancr --json status p.json
id` and `node_status` report one step. CLI exit codes: 0 ok, 1 a step failed, 2 a
usage or input error, 3 an internal error (details in `dancr log`). The MCP
reading tools (`get_sample`, `get_stats`, `render_chart`, `export_node`) run the
pipeline automatically when a node is not computed yet; their CLI equivalents
(`dancr sample/stats/chart/export`) need `--run` to do the same.

From a packaged install (the `dist/DANCR` folder, `DANCR.app`), use the
`dancr-cli` binary (`dancr-cli.exe`; `DANCR.app/Contents/MacOS/dancr-cli`) for
every command above, including `dancr-cli mcp`. The `DANCR` binary is the
window and has no console, so it cannot serve JSON or the MCP protocol.

Where MCP may write: pipeline files (`create_pipeline`, `build_template`, and
every tool that edits one) must be `.json` files inside the server's root
folder. Every other file must be inside the folder that holds the pipeline
file: `export_node(out_path)`, `render_chart(out_png)`, and the `path` of
`export`, `workbook` and `report` steps when `run_pipeline` (or a reading tool)
runs them; a step pointing elsewhere fails with a message. Relative paths are
taken from that folder. Put the pipeline next to the data and results you
want. Reading (`inspect_file`, `load_file` paths) is not confined.

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
  `<project dir>/.dancr/cache/<name>/<node>/` (a project with no file yet uses
  the user cache folder instead, so always save the project file first). Re-runs skip steps whose
  settings, inputs and upstream data did not change; `export`, `workbook` and
  `report` steps also run again when their file was deleted or changed, or when
  the titles or column labels they show changed. `force` (`run --force`,
  `run_pipeline(force=true)`) recomputes everything and replaces the stored
  results. Results of 100M-row steps stay on disk; `get_sample`, `get_stats` and
  `render_chart` read them lazily.
- **Reports**: steps attach findings (fit equation, R², RMSE; gap statistics;
  PASS/FAIL counts) to their status. Read them with `node_status` /
  `dancr status`.
- **Inputs**: `set_input(name, value, unit, note)`. Use them in formulas as
  `[name]`, as filter values, and as `min`/`max` of `check_limits` or chart
  `limits`. Changing one recomputes only the steps that use it.
- Errors are plain English and name the column or setting. Fix and re-run.
- A project may also hold **Answers**: a question and the steps that answer it. An Answer has an
  `id`, a `title`, a `terminal` node id (the step whose output answers the question), a `view`, the
  question as a `spec`, the `steps` it built (`{plan key: {node, made, title, hand?}}`), and the
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

## Settings cheat-sheet (full list: `dancr nodes -v` or `list_node_types`)

- `load_file`: `path`, `sheet`, `has_header`, `skip_rows`, `separator` (auto), `parse_dates`, `date_format`, `day_first` (only for dates like 01/05/2024 that read either way; default month/day, and a run reads the whole column to choose), `decimal_comma`, `encoding` utf8|latin1, `infer_rows`, `ignore_errors`, `columns`.
- `choose_columns`: `mode` keep|drop, `columns`, `rename`. `sort`: `columns`, `descending`. `remove_duplicates`: `columns`, `keep` first|last|none.
- `fix_missing`: `method` drop|drop_all|value|forward|backward|interpolate|mean|zero, `value`, `columns`. `change_type`: `columns`, `to` number|integer|text|datetime|bool, `date_format`, `epoch_unit`.
- `take_sample`: `mode` first|last|every|random, `rows`, `every`, `fraction`, `seed`. `stack`: `label_column`, `labels`; connect tables to `tables`.
- `enter_data`: `columns` = `[{"name","type": text|number|datetime|bool}]`, `rows` = list of lists.
- `keep_rows`: `mode` keep|remove, `conditions` = `{"match":"all"|"any","rules":[{"column","op","value","value2"}]}`
  with ops `eq ne gt lt ge le between contains not_contains starts ends in empty not_empty true false`; values may be input names; or `formula`. As in Excel, a blank cell counts as not equal to (and not containing) any value. `in` takes a list, or text separated by `;` or `,` (`"1,000; 2,500"` for numbers with thousands separators).
- `calculate`: `formulas` = `[{"name": "diff", "expr": "[b] - [a]"}]`, `only_new`. Whole-number `+ - *` work in 64 bits.
- `fix_values`: `fixes` = `[{"row": 1-based, "column", "value", "was", "note"}]`.
- `combine`: `method` match|nearest_time|side_by_side; match: `on`, `right_on`, `how`; nearest_time: `left_time`, `right_time`, `direction`, `tolerance` (e.g. `500ms`); `suffix`.
- `time_buckets`: `every` (`1s 1m 15m 1h 1d`, calendar `1mo 1q 1y`), `columns` (empty = every number column), `default_stats` (mean median min max std sum count first last; one statistic keeps column names, several add `_stat`), `time_column`, `aggregations`, `count_column`.
- `rolling`: `columns`, `stat` mean|median|min|max|std|sum, `window` (rows like `20` or a span like `30s`), `time_column`, `centered` (a span `w` then covers `[t - w/2, t + w/2]`; otherwise `(t - w, t]`), `replace`.
- `rate_of_change`: `columns`, `time_column`, `per` s|m|h|d, `span`. `find_gaps`: `time_column`, `expected`, `factor`. `regular_grid`: `time_column`, `every`, `method` nearest|backward|forward|interpolate.
- `summarise_around`: `window`, `side` before|after|around, `columns`, `stats`, `sample_time`, `log_time`.
- `remove_outliers`: `columns`, `method` rolling|zscore|iqr|range, `window`, `threshold`, `iqr_factor`, `local_spread`, `min`, `max`, `action` remove|blank|flag|clip, `flag_column`.
- `fit_curve`: `x`, `y`, `kind` linear|saturating|exponential|power|logarithmic|polynomial, `degree`, `group`, `predicted_column`. Report: `fits` = list of {equation, r2, rmse, params, n, group}.
- `predict`: inputs `data` (x values) and `model` (a fit_curve step); `x`, `output`, `group`.
- `check_limits`: `column`, `min`, `max` (numbers or input names), `action` flag|remove|keep_failing, `flag_column`. Report: verdict PASS|FAIL, outside, rows.
- `group_summary`: `by`, `columns`, `default_stats`, `aggregations`, `count_column`. `summarize`: `columns`.
- `chart`: `kind` line|scatter|histogram|bar, `x`, `series` `[{"column","color","label"}]`, `color_by`, `split_by` (one panel per value), `limits` `[{"value","label"}]`, `fit`, `mean_line`, `column`, `bins`, `category`, `value`, `stat` mean|sum|count|min|max|median, `title`, `y_label`, `break_gaps`, `log_y`.
- `export`: `path` (.csv | .tsv | .txt | .parquet | .xlsx). `workbook`: `path` (.xlsx); connect tables to `items`.
- `report`: `title`, `path` (.html), `notes`, `company`, `author`, `blocks` (`[{"type":"heading"|"text","text"}, {"type":"item","index"}]`, optional), `pdf`, `max_rows`, `include_stats`; connect charts/tables to `items`.

## Conventions that keep people happy

- Give steps readable titles (`--title` / `rename_node`); they are what the person sees in the project list.
- Prefer `time_buckets` before charting or exporting anything with millions of rows, and pass `columns` so the output stays readable.
- Set display names and units with `set_column_label` / `dancr columns`; they appear in the table header, on charts and in reports.
- Put thresholds and constants in Inputs rather than hard-coding them in formulas, so the person can change them on the Inputs page.
- For a deliverable, end with a `report` step and run it; also `open_in_gui` / `dancr open` so the person can explore.
- The window and the CLI/MCP may run the same project at the same time; the cache is safe for that.
- Never write into the `.dancr` cache folder yourself. Exports and chart files go next to the project file (MCP enforces this).
- If something goes wrong, `dancr log` prints the log file path and its last lines.
