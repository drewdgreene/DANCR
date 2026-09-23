# DANCR

DANCR (Data Analysis Node-based Canvas for Research) is a desktop app for
tables that are too large for Excel. It opens CSV, text, Excel and Parquet
files and builds the analysis as a chain of steps: filter rows, calculate
columns, join tables, resample time series, fit curves, chart, and report. A
project is a small JSON file, so rerunning it on next month's data repeats
every step exactly.

## Download and install

Get the latest installer from the [Releases](../../releases/latest) page.

- **Windows 10 / 11 (64-bit):** download `DANCR-Setup-<version>.exe`, run it, and
  follow the wizard. It installs for the current user (no administrator needed)
  and adds a Start Menu shortcut. A Linux package and a macOS package are on the
  same Releases page.

The app is self-contained; nothing else needs to be installed first.

## What it does

- Open CSV, text, Excel or Parquet files, however large, and work on them as a
  table with sorting, filtering, find, and per-column statistics.
- Build an analysis as a chain of steps on the map: filter rows, make columns
  with formulas, average over time, smooth, find gaps, fit a curve, predict,
  check against limits, chart, and assemble a one-page report.
- Every step is recorded, so the same project reruns on a new file.
- Save results to CSV, Excel or Parquet; export a report as HTML or PDF.
- Work with a built-in command line and an MCP server for scripting and AI
  agents (see `AGENTS.md`).

## License

Proprietary. See [LICENSE](LICENSE).
