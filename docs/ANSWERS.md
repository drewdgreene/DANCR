# Answers: how DANCR answers on its own

Replaces the four-page "Build it for me" wizard (removed). Deterministic throughout: no model, no randomness;
ties are broken by project order, column order and recipe order.

## The pieces

| Module | Does |
|---|---|
| `core/nodes/load.py` | Reads files the way people write them: title lines above the header skipped, numbers written as text (`1,234.50`, `£99`, `31.5%`, `(120)`), codes with leading zeros kept, a Date + Time pair combined, every sheet of a workbook offered as a table (`tables_in`). |
| `core/understand.py` | The **data model** of a project's tables, read from step outputs (so a loader's settings are what is described). Column roles (time, id, category, measure, flag, text, constant), table shapes (series, lookup, events, table), total rows, empty rows, spelling variants, wide month columns (described as the long table they stand for), and relations: links oriented by key cardinality with match %, stacks labelled from file names, alignments of logs. Numbers that name things are ids, not measures: a name about the code itself (`zip`, `zip_code`, `phone_number`, `customer_phone`, `account_no`; not `account_balance` or `phone_calls`), or whole numbers of seven or more digits of one width numbered from one base (their first three digits the same); a population or revenue spread across its width stays a measure. A link needs its names to agree when both keys are small counting ids (1, 2, 3 … overlap whatever they count), and two key names with different stems (`stock_id`, `store_id`) never link. `understand()` samples; `deepen()` reads each table in one streaming pass (counts, span, blank rows, the last rows, key counts; a second pass only to count keys exactly in a source not run yet) and each link in one pass of its two key columns, after links are re-pointed at the true lookup. Answers are only built from the deep model. |
| `core/recipes.py` | The **catalogue**: compare, trend, breakdown, top, toprows, relationship, gaps, outliers, single, distribution, linked, stacked, rows, describe, change, explain, drivers, forecast, quality. `suggest()` ranks candidates; `plan(model, spec)` turns a spec into steps, with the assumptions made (each with alternatives) and the chips. `_Builder.table()` puts the cleaning in front of every table used (empty rows, total row, unpivot, spellings). A step whose result is a *finding* (compare_periods, contribution, associations, check_data, forecast, fit_curve, check_limits, find_gaps) attaches one plain sentence to its report, which the report and the result card show as a headline. |
| `core/ask.py` | Reads a typed question against the project's own words with a fixed grammar, in three stages: `_read` matches the longest known phrase at each point (meanings chosen from context by `_pick`; an unknown word is refused with "did you mean"), `_clauses` joins phrases into clauses, and `_assemble` puts the clauses into a spec. Every word must end up in the spec or the question is refused naming it (`_finish`). |
| `core/lexicon.py` | Deterministic spelling repair: a word the project does not know is matched to one it does — an adjacent swap (`regoin` → `region`), a near miss, or a word the project has learned — and the question is read again. Every fix is recorded (`corrected`), never silent. |
| `core/bank.py` | The recall layer *behind* the grammar: every answer the recipes can build for the project's tables, indexed by its words. Used only when the grammar fails, and only when a match clears `MATCH_THRESHOLD`; a question must mention the columns it is about (`_required_groups`), so an omitted number cannot match the wrong phrasing. No model. |
| `core/memory.py` | What a project learns: the words it has learned to read (from repairs) and the questions recently asked, kept in the project file's `meta` so it travels with the project and never changes the format. |
| `core/planner.py` | `apply_plan`: puts a plan into a project, updating an answer's own steps in place, never changing a step it did not make or one another answer uses, setting aside a hand-edited step the change conflicts with. Each step's record keeps the settings and inputs it was made with, so a step the person put between two answer steps stays there (moved onto the new input when the answer's step is to read from somewhere else, so it stays in the answer's path), and a step the person edited is kept (set aside) when the change no longer needs it. |
| `core/answers.py` | Build / change / remove an Answer (shared by CLI, MCP and the window). Removing an answer with its steps hands the steps another answer uses over to that answer, so they change with it rather than being left behind. |
| `ui/answering.py` | `Understanding` (the model kept current off the GUI thread; keyed on what the tables contain), `AskBar` (question box and tray of suggestions with previews), `AnswerPanel` (chips and assumptions). |

## What a question can say

- Statistics (total, average, count, highest …), a group (`by region`, `per customer`), a time step (`per week`,
  `monthly`), top N, parts of the day or week, shares, and the recipe words (trend, compare, gaps, spikes …).
- Comparisons on a column: `price above 30`, `price is between 10 and 20`, `price from 10 to 20`, `price not
  between 10 and 20`, `price is not above 30`, `customer_id is 1, 2 or 3` (any of them), `stock below reorder level`.
  A comma between digits groups thousands only before exactly three digits (`1,234`, `12,345.5`); otherwise it
  separates a list (`1,2` is 1 and 2). A decimal comma is not read: write `30.5`. "is" belongs to the comparison after it, or means "equals" before a value (`region is North`),
  and is nothing otherwise (`what is the average price`).
- Values bind to the column that holds them: `total quantity except North` filters region, not quantity.
  Several values of one column are either (`North and South`, `Leeds or York`, one `in` rule); `not North or
  South` is neither. `or` between two different conditions is refused. `compare Leeds and York` (or `Leeds vs
  York`) groups by that column, kept to those values; values of a lookup (`compare North and South`, regions of
  customers) compare the rows that point at them (orders), by their first number unless one is named (`… by price`).
- A number compared is also the one added up when no other is named: `total quantity above 3`.
- A group named by an id in short (`per product` for `product_id`) is shown by the linked lookup's name column.
- Dates: `in March` (any year), `in March 2024`, `in 2024-03`, `in 2024`, `on 2024-03-01` (also `2024/03/01`,
  `1 Mar 2024`, `Mar 1`, and `25/03/2024` when only one reading is a date; `01/03/2024` is refused as it reads
  either way), `before/after/since/until` any of those, ranges (`from 2024-02-01 to 2024-03-01`, `between March
  and May`, `from January to March`, `from March and April`, `from 1 Feb to 10 Feb`), and relative dates (`this
  month`, `last week`, `yesterday`, `in the last 7 days`, `past 3 months`). A month with no year (in a comparison
  or range) is the latest such month that starts at or before the data's latest date (`since December` on data
  from January 2024 is since December 2023). Relative dates are taken from the
  **latest date in the data**, not today's date, so the same question on the same data always keeps the same
  rows; `last month` is the calendar month before the one holding that date, `the last 7 days` the seven days up
  to it. The filter's text says the dates chosen (`last month (March 2024)`). Two separate periods (`in March and
  April`), dates no row can be in at once (`before February and after March`), a range written backwards and
  `the last 0 days` are refused rather than answered with nothing.
- Superlatives: `biggest orders`, `cheapest orders`, `orders with the highest price` are rows; `lowest temperature
  in site B` is one value; `hottest day`, `highest sales day`, `which day had the highest sales`, `best day` over a
  table with several rows per day add each day up (or average it, for readings) first and give the top one
  (recipe `top` with `every`); `top 3 days by sales`, `best 3 weeks` give the top three.
- After `top`, `biggest` or `compare`, `by <a number>` is what is ranked or compared (`top 5 orders by price`,
  `biggest orders by price`), unless another number is named; otherwise a number after `by` is a group (a numbered
  table). `top 3 customers in March` ranks customers by their rows (orders) in March.
- One group at a time: `by region by segment` and `by region and segment` are refused.

A **spec** is JSON: `{"recipe", "table", "measure"|"measures", "by", "by_part", "every", "stat", "filters", "n",
"bottom", "together", "share", "keep_*"…}` with column references `[table node id, column]`.

## Adding to it

- A new kind of answer: a `_plan_<name>` function in `recipes.py`, its name in `RECIPES`, `PLANNERS` and `WEIGHT`,
  a candidate in `_candidates`, chips in `chips()`, and the words that ask for it in `ask.py`.
- A new word: `STATS`, `RECIPE_WORDS`, `OPS`, `COPULA`, `RELATIVE`, `DAY_WORDS`, `PARTS`, `SYNONYMS`, `ADJECTIVES` in
  `ask.py`. `STOP` (words that carry no meaning) must not repeat any of them; `tests/test_ask.py` checks.
- Every change to what is understood or suggested shows in `tests/test_corpus.py`; if the new result is better,
  store it with `DANCR_UPDATE_CORPUS=1`. The corpus freezes titles; what an answer computes is checked in
  `tests/test_ask.py` (and `test_understand.py`, `test_planner.py`). Bump `RULES_VERSION` when the same spec would build different steps.

## Not yet

See `docs/OPEN.md`.
