# DANCR user guide

DANCR is a desktop app for tables that are too large for Excel. You open a file
and it appears as a table. Then you build the analysis as a chain of steps:
filter rows, calculate columns, join tables, average over time, fit curves,
check limits, chart, report.

Every step is saved with the project. Run it again on a new file and the whole
analysis repeats. Anyone who opens the project file can see what was done.

## The window

The **project tree** is on the left. It shows your steps as a tree that follows
their connections, so each step sits under the step that feeds it. If a step
has more than one input, the other inputs are listed beneath it. The
**Flow / Type** button in the header switches to a plain list grouped into
Tables, Charts and Reports. **Inputs** are always at the bottom. Click a step to
open it, and the map centres on it.

The centre shows whatever you clicked. A table has a *Rows* tab and a
*Describe* tab, which gives one line per column with blanks, average, min, max
and so on. A chart shows its settings as chips along the top. A report opens
the report builder.

**Settings** is on the right (Ctrl+,). It lists every option of the selected
step and has a **Run to here** button for big data.

The **map** runs along the bottom (Ctrl+M). It shows every step, with arrows
from each step to the ones it feeds. The project tree and the settings panel
sit above it and stop at its top edge, so the map gets the full width of the
window. Each card is tinted with the colour of its category (Get data, Time,
Clean up…).

On the map you can:

- drag a step to move it
- drag the background to move around
- scroll to zoom
- drag an input dot to rewire. Drop the connection on another step, or on empty
  space to remove it.

Drag the divider between the pages and the map to give the map more or less
room.

DANCR follows your system's light or dark appearance. To choose by hand, use
*View → Appearance* (Follow the system / Light / Dark). DANCR remembers your
choice.

## Start

Open a CSV, Excel or Parquet file with *Open data files…*, or drop it anywhere
on the window. It shows up in the project tree right away, even when it's very
large.

You can open several files at once, such as orders and customers or a month of
logs. DANCR works out how they fit together.

To look around first, open one of the **Examples** on the start screen (or
**File → Examples**): shop sales, two sensor logs, a department budget, or batch
test results. Each is a finished project with its questions already answered
and a report. They're saved in `DANCR samples` in your home folder, so you can
change them freely.

**File → New from template** has a few starter projects built on a generated
sample file. When you're ready, swap in your own file in the loader's Settings.

## Answers: ask, or pick one DANCR offers

Press **Ask a question** (Ctrl+J). A bar opens above the table with answers
DANCR can build right away, such as *Total sales by region*, *Average pressure
per hour*, *probe B minus probe A*, *Gaps in the log* or *Top 10 customers*.
Each card has a small preview.

Click a card and DANCR builds it. You get ordinary steps on the map, plus an
**Answer** card next to them. If you select a table in the project tree first,
the suggestions are about that table.

The bar also opens when you select an Answer. Close it with × or Esc.

You can also type a question in the bar in your own words. Use the names of
your columns, tables and values:

| You type | You get |
|---|---|
| `total sales by region` | a bar chart of sales added up per region |
| `average pressure per hour` | a line per hour (one line per file when several files have the same columns) |
| `top 10 customers by sales` · `bottom 5 products by qty` | the ten biggest (or smallest), ranked |
| `monthly sales` · `sales per week` | totals per calendar month or week |
| `compare probe_A and probe_B` | the two logs paired reading by reading, B fitted to A, and the difference over time |
| `temperature against pressure` | a scatter with the fitted line |
| `orders where qty above 2` · `sales between 100 and 200` | only the rows that match |
| `sales by store for Leeds` | any question, limited to one value of a category |
| `gaps in probe_A` · `spikes in pressure` · `spread of weight` · `describe customers` | gaps, unusual readings, a histogram, a summary |
| `sales in March` · `revenue by region in 2024` · `amount after 2024-06-01` · `orders between 2024-03-01 and 2024-03-31` | any question, limited to a month, a year or dates |
| `biggest orders` · `top 5 employees by salary` | the rows themselves, largest first |
| `which supplier has the most items` · `max temperature by device` | rows counted per lookup, or per file |
| `items where stock level below reorder level` | one column compared with another |
| `hottest day` · `top 3 months by quantity` | the days, weeks or months with the highest average or total |
| `fastest vehicle` · `cheapest order` | the one row with the highest or lowest value |
| `share of sales by region` · `what share agree` | each group's part of the whole, in per cent |
| `tips by hour of day` · `busiest day of the week` · `sales by month of the year` | totals by a part of the time, across every day, week or year |

### How DANCR decides

It reads every row before building anything. The suggestions come from a quick
look at the first rows, but links, row counts and time spans are worked out
from the whole file. On very large files this takes a few seconds, and the bar
tells you.

Links follow the keys. Two tables are linked when a column's values in one are
found in the other, and each value appears there only once (a customer id in
the customers table, say). That way linking never repeats rows. If a key
repeats in both tables, DANCR won't link them and tells you why.

Readings are averaged and amounts are added up. A logger's pressure is averaged
per hour, while sales are totalled per month. A number that describes a lookup
row, like a product's unit cost, is averaged.

If DANCR doesn't know a word, it says so instead of guessing ("I don't know
“colour”. Did you mean …?"). If it knows a word but can't fit it into the
question, it names that word too, so nothing is quietly left out. Everyday words
work: *revenue* or *sales* for an Amount column, *pays* for salary, *started*
for a start date.

It also tidies up things spreadsheets often get wrong, and lists each fix in the
answer's assumptions, where one click undoes it. A TOTAL row at the bottom is
left out, empty rows are left out, and North, north and "North " count as one
region.

### Messy files

A wide sheet with a column per month (Jan, Feb … Dec) is read as one row per
item and month. If the file name gives the year, as in *budget_2024.xlsx*, you
get dates too. So *total per month*, *per month by department* and *compare
budget and actual* all work.

A title line or notes above the column names are skipped, and so are rows the
file uses for nothing. When a file is opened:

- text that is really numbers (`1,373.10`, `£1,200`, `31.5%`, `(120)`) is read as numbers
- codes with leading zeros (`00042`) stay as typed
- a Date column and a Time column are also combined into one date/time
- every sheet of a workbook becomes a table.

### Changing an answer

When you select an answer, the strip above the result shows its choices as
chips (*Total*, *qty*, *by region*, *per hour*, a filter). Click a chip to change
it and the answer's steps are updated in place. **Assumptions** lists what DANCR
decided for you, such as which link it used, putting similar files together, or
the time step. Each alternative is one click away.

Build as many answers as you like. Steps they share, like a link to the
customers table, are used once and not duplicated.

Your own edits win. If you change one of an answer's steps by hand and then
change the answer, your settings are kept and the strip tells you so. A step
you attached to an answer's steps is never removed.

**Delete** asks whether to remove only the card or also the steps that only it
uses. Your tables, and steps that other answers need, are always kept. Every
build and change is one undo step.

## Working with a table

Right-click a column header for these:

| Menu item | What happens |
|---|---|
| Filter rows by this column… | Adds a *Filter rows* step. Fill in the condition in Settings. |
| Sort by this column | Adds a *Sort* step. |
| Chart this column | A line chart of that column over time (or row number). |
| Check against a limit… | Adds a *Check against limits* step, which flags rows outside the limit and counts them. |
| New column from a formula… | Adds a *New column* step with the formula editor open. |
| Rename, set units… | Gives the column a display name and unit for charts and reports. The data doesn't change. |
| Fix numbers and dates… | Converts text that should be numbers or dates. |
| Hide this column | Adds a *Pick columns* step that drops it. |

Right-click a cell and choose *Fix this value* to type a new value and a note
saying why. Corrections are collected in a *Fix values* step with the old value,
the new value and your note, so nothing is hidden.

Select two number columns and right-click to chart them together or fit a curve
between them.

**Ctrl+F** finds text, or jumps to a time such as `2024-06-05 14:00`. Hover over
a header to see blanks, min, max and average.

The copy button (or Ctrl+C) puts the selection on the clipboard, ready to paste
into Excel. It copies from the table itself, so every digit is kept and blank
cells stay blank. You can copy up to 200,000 rows.

## Steps

Each step takes a table, does one thing to it, and passes the result on. You can
add a step from the header menus above, from **+ Add step** (Ctrl+K), or from
the **+** on a step in the map. New steps attach to the table you're looking at.

| Category | Step | What it does |
|---|---|---|
| Get data | Load file | CSV, TSV, text, Excel, Parquet. Separators and dates are detected. |
| Get data | Type in a table | A small table you fill in yourself, like a short list of items or a lookup table. You can paste from Excel, and a pasted header row becomes the column names. |
| Filter & sort | Filter rows | Keep or remove rows that match plain-English conditions or a formula. |
| Filter & sort | Pick columns | Keep, remove, reorder or rename columns. |
| Filter & sort | Sort | Order rows. |
| Filter & sort | Take a sample | First/last N rows, every Nth row, or a random fraction. |
| Calculate | New column (formula) | Excel-style formulas such as `[Price] * [Quantity]`, `IF(x > 1, "hi", "lo")` or `ROLLING_MEAN(x, 20)`. You can use Inputs by name. |
| Clean up | Fill blanks | Remove blank rows, or fill blanks with a value, the previous or next value, an interpolation or the average. |
| Clean up | Fix numbers and dates | Text → number / date, number → text. |
| Clean up | Remove duplicates | Drop repeated rows. |
| Clean up | Remove spikes | Finds values far from the rolling median, by z-score, by IQR or outside a fixed range. Remove, blank, flag or clip them. |
| Clean up | Fix values | Individual cell corrections with notes, made from the cell menu. |
| Combine | Combine two tables | Match on a key (like VLOOKUP), line up two time series by nearest time, or put tables side by side. |
| Combine | Columns into rows | Turns a wide sheet with a column per month (Jan, Feb, Mar …) into one row per item and month, so it can be totalled and charted over time. |
| Combine | Stack tables | Append the rows of several tables. |
| Time | Average over time | Groups rows by second, minute, hour or day and summarises each group. It's the fastest way to shrink millions of rows. |
| Time | Smooth out noise | Rolling average/median/min/max/std over N rows or a time span. |
| Time | Rate of change | Change per second/minute/hour/day. |
| Time | Find gaps | One row per dropout, with when it started, when it ended and how long it lasted. |
| Time | Even out the timing | Resample onto evenly spaced timestamps. |
| Time | Summarise around each sample | For each row of a short table (weekly samples, say), averages a continuous log over a window before, after or around it. |
| Analyse & model | Describe the columns | Count, blanks, mean, std, min, quartiles and max for each column. |
| Analyse & model | Totals by group | Like a pivot table, with one row per group. |
| Analyse & model | Fit a curve | Fits a straight line, levelling-off, exponential, power, logarithmic or polynomial curve between two columns, optionally per group. Reports the equation, R² and typical error, and adds fitted and residual columns. |
| Analyse & model | Predict from a fit | Apply a fit to new x values, from a typed-in table for example. |
| Analyse & model | Check against limits | Flag, keep or drop rows outside a minimum/maximum. Limits can be numbers or Inputs. Reports PASS/FAIL and counts. |
| Share | Chart | Line, scatter, histogram, bar. Data is summarised per pixel, so even 100 million points draw quickly. |
| Share | Save to file | CSV, Excel or Parquet. |
| Share | Excel workbook | Several tables in one .xlsx, one sheet each. |
| Share | Report | Charts and tables on one page, with a title block, an introduction and your own headings and text. Saves HTML and PDF. |

## Charts

Open a chart and change it with the chips along the top:

- **type**
- **X**
- **Y** (tick several columns)
- **Colour by** draws one line per category
- **Split by** draws one panel per category, stacked with a shared X axis
- **Limit line** takes a number or an Input
- **Fitted curve** adds one fit per panel on scatter charts
- **Average line**

Hover to read values, and click a legend entry to hide a series. Drag to pan,
scroll to zoom, right-drag a box to zoom into it, and double-click to see
everything. Zooming re-reads the data, so you always see the true minimum and
maximum for each pixel.

The buttons at the top right copy the chart as an image, save it as PNG or SVG,
or **Add to report**.

## Reports

A report collects charts and tables on one page. Fill in the title, company and
introduction, put the items in order, and add headings and text between them.
Then press **Build report**. DANCR writes an HTML file that opens in any browser,
with a PDF next to it.

If the data changes, build it again. The layout is kept.

## Inputs

Inputs are named values with units, like a maximum allowed value, an alarm
threshold or a calibration factor. You can type `[maximum allowed]` in a
formula, choose an input as a filter value, or type its name as a limit.

When you change a value on the Inputs page, only the steps that use it
recompute. Inputs are saved with the project and appear in reports.

## Big data

Small files recompute after every change. Big files are different. DANCR shows a
preview built from 50,000 rows spread across the data, and waits until you
press **Run** (Ctrl+R) or **Run to here**. In the preview, counts and totals are
approximate, and the header says so.

Results are kept next to the project in a `.dancr` folder, so only the steps you
changed run again. Until you first save the project, they go in your user cache
folder. To switch the behaviour by hand, use *Run → Run automatically after
every change*.

If the file on disk changes, for example because a logger appended more rows,
DANCR notices and marks the steps as needing a run.

## Saving, versions and undo

Projects are small `.json` files. Data files stay where they are, and the
project refers to them by a path relative to itself. If you use *Save as* to
put the project in another folder, every file setting still points at the same
file.

The project file changes only when you save. To have DANCR save a saved project
every minute as well, turn on *File → Autosave*; the choice is remembered. Earlier
versions are under *File → Earlier versions…*. It keeps the last 50 you saved
plus the last 30 autosaves, so autosaving can't push out a version you saved
yourself.

Changes that aren't in the file yet are set aside every minute, whether autosave
is on or off. That covers a project you haven't saved, edits you haven't saved,
and edits made while autosave is paused. If DANCR closes unexpectedly, it offers
them back the next time it starts.

Ctrl+Z undoes anything, including deleting a step, and puts connections back in
their original order. A toast also offers Undo when you delete.

Steps that save a file (*Save to file*, *Excel workbook*, *Report*) write it
again if it was deleted or changed, or if the titles, column names or units they
show have changed. *Run → Run everything again (ignore cached results)*
recomputes every step and replaces the stored results.

## Keyboard shortcuts

| Keys | Action |
|---|---|
| Ctrl+I | Open data file |
| Ctrl+K / Insert | Add step |
| Ctrl+J | Ask a question about your data |
| Ctrl+F | Find in the table / jump to a time |
| Ctrl+R / F5 | Run everything |
| Ctrl+Shift+R | Run up to this step |
| Ctrl+. | Stop the run |
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

*File not found.* The project stores paths relative to itself. Move the data
next to the project, or choose the file again in the loader.

*No column called X* after you changed an earlier step. A later step still
uses the old name. Open it and pick the column again.

*Dates came in as text.* Use *Fix numbers and dates → date/time* and give the
format, for example `%d/%m/%Y %H:%M:%S`.

*Excel says the file is too big.* Excel stops at 1,048,576 rows. Use *Average
over time* or *Take a sample* first, or save as CSV or Parquet.

*The disk that holds DANCR's results is full.* Results are kept next to the
project, or in your user cache folder before the first save. Free some space
or use *Run → Clear cached results*, then run again.

Every step runs on the Polars streaming engine and writes its result as it
goes, so memory doesn't limit how big a step can be. A few things are the
exception: *Smooth out noise* (rolling windows), ranks, exact quantiles,
whole-column totals in formulas (`SUM(x)` on its own) and *Sort*. These hold one
column, or the sorted table, in memory or on the spill disk.

*Something crashed or looks wrong.* *Help → Show log file* opens the log. You
can also run `dancr log`. Send the log with your report.

## For scripts and AI agents

You can do everything in this guide without the window. In the installed app
the command line is **`dancr-cli`**, or `dancr-cli.exe` on Windows, in the
folder you installed to. From a source checkout it's `dancr`.

Run it with `--help` to list the commands. For AI coding agents, start the MCP
server with `dancr-cli mcp`. There's more under *Help → For AI agents and the
command line*.

If a script or agent edits the project file, the open window reloads it.

## Credits

DANCR was designed and developed by Drew Greene (drewdgreene@gmail.com) and
commissioned by Jens Dancer. Engine: Polars. UI: Qt / PySide6 / pyqtgraph.
Icons: Phosphor (MIT).
