# Answers: how DANCR answers on its own

Replaces the four-page "Build it for me" wizard (removed). Deterministic throughout: no model, no randomness;
ties are broken by project order, column order and recipe order.

## The pieces

| Module | Does |
|---|---|
| `core/nodes/load.py` | Reads files the way people write them: title lines above the header skipped, numbers written as text (`1,234.50`, `£99`, `31.5%`, `(120)`), codes with leading zeros kept, a Date + Time pair combined, every sheet of a workbook offered as a table (`tables_in`). |
| `core/understand.py` | The **data model** of a project's tables, read from step outputs (so a loader's settings are what is described). Column roles (time, id, category, measure, flag, text, constant), table shapes (series, lookup, events, table), total rows, empty rows, spelling variants, wide month columns (described as the long table they stand for), and relations: links oriented by key cardinality with match %, stacks labelled from file names, alignments of logs. `understand()` samples; `deepen()` reads every row once. Answers are only built from the deep model. |
| `core/recipes.py` | The **catalogue**: compare, trend, breakdown, top, toprows, relationship, gaps, outliers, single, distribution, linked, stacked, rows, describe. `suggest()` ranks candidates; `plan(model, spec)` turns a spec into steps, with the assumptions made (each with alternatives) and the chips. `_Builder.table()` puts the cleaning in front of every table used (empty rows, total row, unpivot, spellings). |
| `core/ask.py` | Reads a typed question against the project's own words with a fixed grammar. Meanings are chosen from context (`_pick`); every word must end up in the spec or the question is refused naming it (`_finish`). |
| `core/planner.py` | `apply_plan`: puts a plan into a project, updating an answer's own steps in place, never changing a step it did not make or one another answer uses, setting aside a hand-edited step the change conflicts with. |
| `core/answers.py` | Build / change / remove an Answer (shared by CLI, MCP and the window). |
| `ui/answering.py` | `Understanding` (the model kept current off the GUI thread; keyed on what the tables contain), `AskBar` (question box and tray of suggestions with previews), `AnswerPanel` (chips and assumptions). |

A **spec** is JSON: `{"recipe", "table", "measure"|"measures", "by", "by_part", "every", "stat", "filters", "n",
"bottom", "together", "share", "keep_*"…}` with column references `[table node id, column]`.

## Adding to it

- A new kind of answer: a `_plan_<name>` function in `recipes.py`, its name in `RECIPES` and `WEIGHT`,
  a candidate in `_candidates`, chips in `chips()`, and the words that ask for it in `ask.py`.
- A new word: `STATS`, `RECIPE_WORDS`, `OPS`, `PARTS`, `SYNONYMS`, `ADJECTIVES` in `ask.py`.
- Every change to what is understood or suggested shows in `tests/test_corpus.py`; if the new result is better,
  store it with `DANCR_UPDATE_CORPUS=1`. Bump `RULES_VERSION` when the same spec would build different steps.

## Not yet

See `docs/OPEN.md`.
