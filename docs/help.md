# DANCR user guide

DANCR is a desktop app for tables that are too large for Excel. It opens a
file as a table and the analysis is built as a chain of steps: filter rows,
calculate columns, join tables, average over time, fit curves, check limits,
chart, report. Every step is saved with the project, so running it again on a
new file repeats the whole analysis, and anyone who opens the project file can
see exactly what was done.

## The window

- **Left: your project.** Your steps as a **tree that follows their
  connections**: each step sits under the step that feeds it, and a step with
  more than one input lists the other inputs beneath it. The **Flow / Type**
  button in the header switches between this tree and a plain list grouped into
  Tables, Charts and Reports; **Inputs** are always at the bottom. Click a step
  to open it — the map centres on it.
- **Centre: the thing you clicked.** A table shows *Rows* and a *Describe* tab
  (one line per column: blanks, average, min, max…). A chart shows the chart
  with its settings as chips along the top. A report shows the report builder.
- **Map (bottom, Ctrl+M):** every step, with arrows showing what came from
  what. Each card is tinted with the colour of its category (Get data, Time,
  Clean up…). Drag a step to move it, drag the background to move around, and
  scroll to zoom. To rewire, **drag an input dot**: pick up a connection and
  drop it on another step, or drop it on empty space to remove it.
- **Settings (right, Ctrl+,):** every option of the selected step, with a
  *Run to here* button for big data.

DANCR follows your system's light or dark appearance. To choose by hand, use
*View → Appearance* (Follow the system / Light / Dark); the choice is remembered.

## Start

Open a CSV, Excel or Parquet file (*Open data file*, or drop it anywhere on
the window). It appears in the project tree straight away, however big it is.

Or pick a **template** on the start screen. Each one builds a small project on
a generated sample file (two values recorded for a few days) so you can see how the pieces fit,
then swap in your own file in the loader's Settings.

## Working with a table

Right-click a **column header**:

| Menu item | What happens |
|---|---|
| Filter rows by this column… | A *Filter rows* step is added; fill in the condition in Settings. |
| Sort by this column | A *Sort* step. |
| Chart this column | A line chart of that column over time (or row number). |
| Check against a limit… | A *Check against limits* step: flags rows outside the limit and counts them. |
| New column from a formula… | A *New column* step with the formula editor open. |
| Rename, set units… | A display name and unit for the column (used on charts and reports; the data is unchanged). |
| Fix numbers and dates… | Converts text that should be numbers or dates. |
| Hide this column | A *Pick columns* step that drops it. |

Right-click a **cell** to *Fix this value*: you type the new value and a note
saying why. Corrections are collected in a *Fix values* step with the old
value, the new value and your note, so nothing is hidden. Select two number
columns and right-click to chart them together or fit a curve between them.

Other things in the table: **Ctrl+F** finds text or jumps to a time such as
`2024-06-05 14:00`. Hover a header to see blanks, min, max and average. The
copy button (or Ctrl+C) puts the selection on the clipboard ready for Excel.

## Steps

Each step takes a table, does one thing, and hands the result on. Add one from
the header menus above, from **+ Add step** (Ctrl+K), or from the **+** on a
step in the map. New steps attach to the table you are looking at.

| Category | Step | What it does |
|---|---|---|
| Get data | Load file | CSV, TSV, text, Excel, Parquet. Separators and dates are detected. |
| Get data | Type in a table | A small table you fill in yourself (a short list of items, a lookup table). Paste from Excel works; a pasted header row becomes the column names. |
| Filter & sort | Filter rows | Keep or remove rows that match plain-English conditions or a formula. |
| Filter & sort | Pick columns | Keep, remove, reorder or rename columns. |
| Filter & sort | Sort | Order rows. |
| Filter & sort | Take a sample | First/last N rows, every Nth row, or a random fraction. |
| Calculate | New column (formula) | Excel-style formulas: `[Price] * [Quantity]`, `IF(x > 1, "hi", "lo")`, `ROLLING_MEAN(x, 20)`. Inputs can be used by name. |
| Clean up | Fill blanks | Remove blank rows or fill blanks (value, previous, next, interpolate, average). |
| Clean up | Fix numbers and dates | Text → number / date, number → text. |
| Clean up | Remove duplicates | Drop repeated rows. |
| Clean up | Remove spikes | Values far from the rolling median, z-score, IQR or a fixed range. Remove, blank, flag or clip. |
| Clean up | Fix values | Individual cell corrections with notes (made from the cell menu). |
| Combine | Combine two tables | Match on a key (like VLOOKUP), line up two time series by nearest time, or side by side. |
| Combine | Stack tables | Append rows of several tables. |
| Time | Average over time | Bucket rows by second/minute/hour/day and summarise each bucket. The fastest way to shrink millions of rows. |
| Time | Smooth out noise | Rolling average/median/min/max/std over N rows or a time span. |
| Time | Rate of change | Change per second/minute/hour/day. |
| Time | Find gaps | One row per dropout: when it started, ended, how long. |
| Time | Even out the timing | Resample onto evenly spaced timestamps. |
| Time | Summarise around each sample | For each row of a short table (weekly samples), average a continuous log over a window before, after or around it. |
| Analyse & model | Describe the columns | Count, blanks, mean, std, min, quartiles, max per column. |
| Analyse & model | Totals by group | Pivot-table style: one row per group. |
| Analyse & model | Fit a curve | Straight line, levelling-off, exponential, power, logarithmic or polynomial between two columns, optionally per group. Reports the equation, R² and typical error; adds fitted and residual columns. |
| Analyse & model | Predict from a fit | Apply a fit to new x values (a typed-in table, say). |
| Analyse & model | Check against limits | Flag, keep or drop rows outside a minimum/maximum. Limits can be numbers or Inputs. Reports PASS/FAIL and counts. |
| Share | Chart | Line, scatter, histogram, bar. Summarised per pixel so 100 million points draw instantly. |
| Share | Save to file | CSV, Excel or Parquet. |
| Share | Excel workbook | Several tables into one .xlsx, one sheet each. |
| Share | Report | Charts and tables on one page, with a title block, introduction and your own headings and text. Saves HTML and PDF. |

## Charts

Open a chart and use the chips: **type**, **X**, **Y** (tick several columns),
**Colour by** (one line per category), **Split by** (one panel per category,
stacked with a shared X axis), **Limit line** (a number or an Input),
**Fitted curve** (scatter charts, one fit per panel) and **Average line**. Hover to read
values; click a legend entry to hide a series. Drag to pan, scroll to zoom,
right-drag a box to zoom into it, double-click to see everything. Zooming
re-queries the data, so you always see the true minimum and maximum per pixel.

The buttons at the top right copy the chart as an image, save it as PNG or SVG,
and **Add to report**.

## Reports

A report collects charts and tables. Fill in the title, company and
introduction, put the items in order, add headings and text between them, and
press **Build report**. DANCR writes an HTML file that opens in any browser and
a PDF next to it. Rebuild after the data changes; the layout is kept.

## Inputs

Inputs are named values with units: a maximum allowed value, an alarm
threshold, a calibration factor. Type `[maximum allowed]` in a formula, choose
an input as a filter value, or type its name as a limit. Change the value on the Inputs page
and only the steps that use it recompute. Inputs are saved with the project
and appear in reports.

## Big data

Small files recompute after every change. For big files DANCR shows a
**preview** from 50,000 rows spread across the data (counts and totals are
approximate; the header says so) and waits until you press **Run** (Ctrl+R) or
*Run to here*. Results are kept next to the project in a `.dancr` folder (in
your user cache folder until the project is first saved), so only steps you
changed run again. *Run → Run automatically after every change*
switches the behaviour by hand.

If the file on disk changes (a logger appended more rows), DANCR notices and
marks the steps as needing a run.

## Saving, versions and undo

Projects are small `.json` files; data files stay where they are and are
referenced relative to the project. Once saved, DANCR saves again every minute
and keeps the previous 50 versions under *File → Earlier versions…*. A project
you have not saved yet is kept aside every minute too, and offered back the
next time DANCR starts. Ctrl+Z
undoes anything, including deleting a step (a toast offers Undo too).

## Keyboard shortcuts

| Keys | Action |
|---|---|
| Ctrl+I | Open data file |
| Ctrl+K / Insert | Add step |
| Ctrl+F | Find in the table / jump to a time |
| Ctrl+R / F5 | Run everything |
| Ctrl+Shift+R | Run up to this step |
| Esc | Stop the run |
| Ctrl+S / Ctrl+Shift+S | Save / Save as |
| Ctrl+Z / Ctrl+Y | Undo / Redo |
| Ctrl+D | Duplicate step |
| Delete | Delete step |
| Ctrl+M | Show / hide the map |
| Ctrl+, | Show / hide settings |
| Ctrl+Shift+I | Inputs |
| Ctrl+0 | Fit the map in view |
| F1 | This guide |

## Troubleshooting

- *File not found*: the project stores paths relative to itself. Move the data
  next to the project or re-choose the file in the loader.
- *No column called X* after changing an earlier step: the later step still
  refers to the old name. Open it and re-pick the column.
- *Dates came in as text*: use *Fix numbers and dates → date/time* and give the
  format, e.g. `%d/%m/%Y %H:%M:%S`.
- *Excel says the file is too big*: Excel stops at 1,048,576 rows. Use *Average
  over time* or *Take a sample* first, or save as CSV/Parquet.
- *The disk that holds DANCR's results is full*: results are kept next to the
  project (or, before the first save, in your user cache folder). Free some
  space or use *Run → Clear cached results*, then run again. Every step runs
  on the Polars streaming engine and writes its result as it goes, so a step
  is not limited by memory. The exceptions are *Smooth out noise* (rolling windows),
  ranks, exact quantiles, whole-column totals in formulas (`SUM(x)` on its
  own) and *Sort*, which hold one column (or the sorted table) in memory or
  on the spill disk.
- *Something crashed or looks wrong*: *Help → Show log file* opens the log
  (also `dancr log`). Send it with your report.

## For scripts and AI agents

Everything above can be driven without the window. In the installed app the
command line is **`dancr-cli`** (`dancr-cli.exe` on Windows, in the folder you
installed to); from a source checkout it is `dancr`. Run it with `--help` to
list the commands, or start the MCP server with `dancr-cli mcp` for AI coding
agents. See *Help → For AI agents and the command line*. When a script or agent
edits the project file, the open window reloads it.

## Credits

DANCR was designed and developed by Drew Greene (drewdgreene@gmail.com) and
commissioned by Jens Dancer. Engine: Polars. UI: Qt / PySide6 / pyqtgraph.
Icons: Phosphor (MIT).
