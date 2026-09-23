# DANCR

**Data Analysis Node-based Canvas for Research.** A desktop app for people who
live in Excel and have data that is too big or too repetitive for it: logs with
tens of millions of rows, weekly results, monthly reports.

Open a file and see a table; every change you make becomes a recorded step, so
the same analysis reruns on the next file and everyone can see what was done.

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
