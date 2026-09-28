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

- Opens large CSV, text, Excel or Parquet files as a table you can sort,
  filter and search, with statistics for each column.
- Reads spreadsheets laid out for people, not machines: title lines, tables side
  by side under banners, blocks under their own titles, labels written once per
  run, AVERAGE and Total rows under the data (checked against the data, then left
  out), and several tables on one sheet.
- Works out how a pile of spreadsheets fits together when you drop them on the
  window: which files link on a key, which are one table split up, and which
  logs record the same thing. It then offers answers it can build right away.
- Knows a small study when it sees one (sun and shade leaves, treated and control
  plots, before and after) and compares the groups: averages, spreads, standard
  errors, the right test and why, how big the difference is, values unusual for
  their group, and when a difference is only down to size. A calculated column
  with a mistyped value is pointed out, and fixed with one click.
- Answers questions typed in your own words, like "total sales by region" or
  "average pressure per hour". It builds the steps, tells you what it assumed,
  and lets you change any choice. The same files and the same question always
  build the same steps.
- Builds an analysis as steps on a map. You can filter rows, make columns with
  formulas, average over time, smooth, find gaps, fit a curve, predict, check
  against limits, chart, and put together a one-page report.
- Records every step, so the same project reruns on a new file.
- Tells you, in a sentence, what changed, what drives it, what relates to what, whether the data is
  trustworthy, and where it is heading — with the chart underneath as evidence.
- Saves results to CSV, Excel or Parquet, and exports reports as HTML or PDF.
- Comes with a command line and an MCP server for scripts and AI agents. See
  `AGENTS.md`.

## License

Proprietary. See [LICENSE](LICENSE).
