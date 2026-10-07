# DANCR

DANCR was built for the spreadsheet problem.

Picture forty spreadsheets from six people. Between them is the answer to one question you need. Every file is laid out a little differently. Combining them is a job for someone who writes SQL, and if you do not have that person, the question goes unanswered, or somebody merges the files by hand and the result is out of date by the next batch.

DANCR lets the person closest to the data do it themselves. Read the spreadsheets as they are, combine and clean them with steps instead of queries, get an answer, and run the same thing again on next month's files.

![version](https://img.shields.io/badge/version-2.2.0-blue) ![license](https://img.shields.io/badge/license-source--available-blue)

## What it solves

### Too many spreadsheets, and no SQL
The files are laid out for people. A title row. Two tables sharing a sheet. A label written once and left blank below it. A total row at the bottom. DANCR reads that as a table anyway, with nothing to reformat first.

Then you build what you need from steps: pick the columns, filter rows, add a calculated column, join two sheets, stack a folder of workbooks into one table. There is no query language to learn. You can also type "total sales by region" and it builds the steps for you. The same question on the same files always builds the same pipeline, because it runs on rules, not a model.

### An AI that makes up your numbers
Point a chatbot at a spreadsheet and it will answer confidently and invent the figures. In DANCR the AI never does the arithmetic. It can suggest which steps to take; the engine runs them. An answer cannot contain a number that no calculation produced. That is what makes it safe to put an AI near your data at all.

### Hundreds of millions of rows
Excel gives up around a million rows, and slows down long before that. DANCR handles hundreds of millions. Results are kept on disk rather than held in memory, so the size of the data is not the wall, and running the project again only recomputes the steps whose data changed.

### Giving an AI agent the run of your data
Handing an agent a folder of your files is a hard thing to feel good about. It does not know how the files relate to each other, and it can touch anything. DANCR gives it a map and a gate.

The map is a description of the data the agent can read: what each table is, its columns, how the tables join, and a hash of each one so it knows what changed.

Linking datasets across separate projects is a newer and thinner feature. It works for keys a project is already joining on, and it does not yet measure how well the values match across projects, so do not lean on it yet. Everything inside a single project is finished and tested.

The gate is a policy that decides which tools the agent may run against which project. Anything consequential needs approval, and every call is logged. You decide what it can reach.

## What that looks like

```bash
dancr ask trials.json "compare rainfed and drought_stress grain yield" --file phenotype_measurements.csv
dancr run trials.json
```

```
rainfed has 19% more grain_yield_q_ha than drought_stress (77.29 vs 65.07; p < 0.001)
```

It picked the right test on its own and said, in one sentence, what the difference was.

## What you get out the other end

Clean tables you can export to Excel, CSV or Parquet. Group comparisons with the test chosen for you. Outliers and gaps. A chart, a map, or a one-page report you can send. And a record of exactly how each number was produced, so somebody else can re-run it later and check it still comes out the same.

## What it does well

- Reads spreadsheets the way people actually make them: a title row, two tables on one sheet, a total row at the bottom, and a folder of files that are each a little different.
- Handles hundreds of millions of rows, holding results on disk instead of in memory.
- Builds the whole pipeline by connecting steps, with no SQL and nothing hidden behind code.
- Answers a question typed in plain words by building those steps, and does it without a model, so the same question on the same files always builds the same pipeline.
- Keeps the AI away from the arithmetic. It can suggest, but every number has to come from a calculation the engine actually ran.
- Runs again on next month's files, recomputing only the steps whose data changed.
- Records what produced a result, so anyone can re-run it later and confirm it still comes out the same.
- Catches the defects a spreadsheet hides: a duplicated ID, a value outside its range, free text where a controlled code is expected, a reference that points nowhere.
- Marks rows sensitive through restricted, keeps the restricted ones out of searches and shared exports, and can redact a copy you hand to someone else.
- Searches your own files offline, and re-indexes only the documents that changed.
- Exports FAIR metadata (schema.org, Frictionless, RO-Crate) so the data can be found and cited.
- Runs from a desktop window, the command line, Python, or an AI agent, where the agent gets a fixed set of tools and a policy over what it may touch, with every call logged.

## For the technical reader

The same engine runs as a command line, a Python library, and an MCP server for AI agents. There is a desktop app too, for when you would rather click than type.

## Install

Download an installer from the [Releases](../../releases/latest) page for Windows, Linux or macOS. The app is self-contained.

From source:

```bash
git clone <repo> DANCR && cd DANCR
uv sync                 # or: pip install -e .
uv run dancr doctor     # reports which capabilities this build has
```

## Status

DANCR is under active development. The agent gateway (policy, approvals, audit) is finished and tested. Linking datasets across projects is early. The running list of what is unfinished or fragile is kept in `docs/OPEN.md`.

## License

Source-available. The code is published so anyone can read it, and the app is free to use. You may not fork it or resell it, and you may not build a paid product, service, or support business on top of it. See [LICENSE](LICENSE).
