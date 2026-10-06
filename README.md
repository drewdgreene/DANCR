# DANCR

DANCR (Data Analysis Node-based Canvas for Research) is a desktop app for
tables that are too large for Excel. It opens CSV, text, Excel and Parquet
files, and you build the analysis as a chain of steps: filter rows, calculate
columns, join tables, resample time series, fit curves, chart and report.

A project is a small JSON file. Run it on next month's data and it repeats the
same steps.

## Download and install

Get the latest installer from the [Releases](../../releases/latest) page.

On Windows 10 or 11 (64-bit), download `DANCR-Setup-<version>.exe`, run it and
follow the wizard. It installs for the current user, so you don't need
administrator rights, and adds a Start Menu shortcut. Linux and macOS packages
are on the same Releases page.

The app is self-contained. You don't need to install anything else first.

## What it does

- Opens large CSV, text, Excel, Parquet, GeoJSON, GeoPackage or shapefile data
  as a table you can sort, filter and search, with statistics for each column.
- Reads spreadsheets laid out for people, not machines: title lines, tables side
  by side under banners, blocks under their own titles, labels written once per
  run, AVERAGE and Total rows under the data (checked against the data, then left
  out), and several tables on one sheet.
- Works out how a pile of spreadsheets fits together when you drop them on the
  window: which files link on a key, which are one table split up, and which
  logs record the same thing. It then offers answers it can build right away.
- Knows a small study when it sees one (treated and control plots, before and after)
  plots, before and after) and compares the groups: averages, spreads, standard
  errors, the right test and why, how big the difference is, values unusual for
  their group, and when a difference is only down to size. A calculated column
  with a mistyped value is pointed out, and fixed with one click.
- Answers questions typed in your own words, like "total sales by region" or
  "average pressure per hour". It builds the steps, tells you what it assumed,
  and lets you change any choice. The same files and the same question always
  build the same steps.
- An optional **Assistant** you talk to: it proposes the same kind of steps and
  answers, you approve them, and the engine does every calculation. It sends your
  tables' shape and statistics to a model you configure — never your rows, unless
  you allow sample rows. Every reply shows how to trust it.
- Builds an analysis as steps on a map. You can filter rows, make columns with
  formulas, average over time, smooth, find gaps, fit a curve, predict, check
  against limits, chart, and put together a one-page report.
- Works with places: measures great-circle distances, counts points into grid
  cells for a density, matches each row to its nearest place by coordinates, and
  draws it all on a map with offline country outlines — no internet, no tile
  server, nothing leaving the machine.
- Reads a whole folder or glob of files as one table (with a column naming each row's file, and every sheet, table
  or layer inside a workbook or GeoPackage), and runs one project over many files at once — an output each plus a
  combined table — so this month's exports or a directory of trials is one command.
- Pulls data in from a URL (CSV, Parquet, JSON), a database (SQLite built in; PostgreSQL and others with an extra),
  or a NetCDF/HDF5 file, alongside local files, keeping credentials in the environment and out of logs.
- Watches a project and its data and reruns when a file arrives or changes, and catalogs a whole folder of projects
  with each dataset's content hash — so a team can find everything and re-index only what moved.
- Searches a project's own text: a **Build search index** step turns tables and documents into offline,
  deterministic vectors; **Search index** (and `dancr search` / the `search_knowledge` MCP tool) returns the
  closest passages with their source rows — withholding confidential/restricted ones unless you ask for them.
  With a persistent index file, re-indexing is **incremental**: each document is hashed and only the ones whose
  content changed are embedded again (the step reports added / changed / unchanged / removed). Point it at a
  `dancr context --changed` feed and whole datasets that did not move are skipped and carried over from the
  index without being read again.
- Stewards data: **Check data contract** (column kinds, blanks, uniqueness, range, allowed values, patterns,
  cross-column rules and keys that must exist elsewhere), **Compare two versions** (what changed), a
  data-dictionary step, **Label sensitivity**, and **Redact** columns for a shareable copy.
- Trusts its AI: `dancr eval` scores a question set against the engine or the Assistant, and `dancr trace`
  prints the saved Assistant conversation — the tools it called, the step it built, and any figure no tool backed.
- Also adds **Rows into columns** (pivot) and **fuzzy text-key matching** (join keys that are almost the same).
- Records every step, so the same project reruns on a new file.
- Proves a result: writes an **attestation** (the engine, each source file's content sample, and every step's
  plan hash, output content hash and the numbers it found), **verifies** the project still reproduces it, and
  shows a step's **lineage** (what produced it, what depends on it) and a one-node **proof card**. A result can
  be re-checked by someone else, and the CLI exits non-zero when a claim no longer holds.
- Writes FAIR metadata for your datasets: set who made it, the license and how to cite it, then export a
  schema.org/Dataset record (Google Dataset Search), a Frictionless data package, or a run manifest that records
  exactly what produced the results — or package the whole thing as a self-contained RO-Crate. Units are written as
  UCUM codes, and the knowledge-base export stamps each dataset with a content hash and a `--changed` mode, so a
  search index or agent refreshes only what moved.
- Tells you, in a sentence, what changed, what drives it, what relates to what, whether the data is
  trustworthy, and where it is heading — with the chart underneath as evidence.
- Saves results to CSV, Excel, Parquet or GeoJSON, and exports reports as HTML or PDF.
- Comes with a command line, a Python API and an MCP server for scripts and AI agents. `dancr context`
  exports a project's datasets as a knowledge base (schema, statistics, sample rows and a written summary
  per table) for a search index or an agent to read. See `AGENTS.md`.

## License

Proprietary. See [LICENSE](LICENSE).
