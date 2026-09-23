# Formula reference

Formulas are used by **Calculate** (new columns) and by **Keep rows → Or a formula**
(true/false tests). They look like Excel formulas but work on whole columns.

## Syntax

- **Columns**: `Total`, `[Unit price]` (brackets when the name has spaces
  or symbols), or `` `Total` ``. Matching is forgiving about case and spaces.
  A column always wins over an Input of the same name; write the column in
  brackets and rename the Input if you need both.
- **Numbers and text**: `2.5`, `"warm"`, `'warm'`.
- **Arithmetic**: `+ - * / ^ %` (`^` is power, `%` is remainder). As in
  Excel, a minus sign on a value binds tighter than `^`, so `-2^2` is 4;
  write `-(2^2)` for −4. `2^3^2` is `2^(3^2)`.
- **Comparison**: `= != < > <= >=` (also `==`, `<>`).
- **Logic**: `and`, `or`, `not`, or `AND(a, b)`, `OR(a, b)`, `NOT(a)`.
- **Text joining**: `&` (e.g. `name & " (" & unit & ")"`).
- **Dates**: comparing a date column with text works: `time > "2024-06-01 12:00"`.
- `TRUE`, `FALSE`, `NULL`.

One-argument `SUM`, `AVERAGE`, `MIN`, `MAX`, `MEDIAN`, `STDEV`, `COUNT` work on
the whole column and repeat the answer on every row (`x - AVERAGE(x)` centres
a column). With several arguments, `SUM`, `MIN`, `MAX`, `AVERAGE` work across
the row (`MAX(a, b, c)`).

## Functions

| Function | Meaning |
|---|---|
| `ABS(x)` `SQRT(x)` `EXP(x)` `LN(x)` `LOG(x)` `LOG(x, base)` `LOG2(x)` `POW(x, y)` | Maths |
| `ROUND(x, digits)` `FLOOR(x)` `CEIL(x)` `CEILING(x)` `SIGN(x)` `MOD(a, b)` `CLIP(x, lo, hi)` | Rounding and limits |
| `SIN COS TAN ASIN ACOS ATAN ATAN2(y, x) PI()` | Trigonometry (radians) |
| `SUM MIN MAX AVERAGE MEAN MEDIAN STDEV STD VAR COUNT` | Column aggregates (one arg) or row-wise (several) |
| `PERCENTILE(x, 0.95)` `ZSCORE(x)` `RANK(x)` | Statistics |
| `CUMSUM(x)` `CUMMAX(x)` `CUMMIN(x)` | Running totals |
| `ROW()` | Row number from 1 |
| `LAG(x, n)` `LEAD(x, n)` `DIFF(x, n)` `PCT_CHANGE(x, n)` | Previous/next rows (n defaults to 1). `DIFF` of a date gives seconds |
| `ROLLING_MEAN(x, n)` `ROLLING_MEDIAN` `ROLLING_STD` `ROLLING_MIN` `ROLLING_MAX` `ROLLING_SUM` | Rolling windows over n rows; add `, TRUE` to centre |
| `FILL_FORWARD(x)` `INTERPOLATE(x)` | Fill blanks |
| `IF(test, a, b)` `ISBLANK(x)` `ISNULL(x)` `COALESCE(a, b, …)` `IFNULL(x, fallback)` | Conditions and blanks. Both branches must be the same kind of value (both numbers, both text …) |
| `LEN UPPER LOWER TRIM LEFT(s, n) RIGHT(s, n) MID(s, start, n)` | Text |
| `CONTAINS(s, part)` `STARTSWITH ENDSWITH REPLACE(s, old, new) CONCAT(a, b, …)` | Text |
| `TEXT(x)` `TEXT(date, "%Y-%m-%d")` `VALUE(s)` | Convert to text / number |
| `DATE(text)` `DATE(text, "%d/%m/%Y")` | Parse a date |
| `YEAR MONTH DAY HOUR MINUTE SECOND WEEKDAY DAYOFYEAR` | Parts of a date |
| `ELAPSED(time, "s"|"min"|"h"|"d")` | Time since the first row |
| `SECONDS_BETWEEN(start, end)` | Difference of two date columns |

## Examples

```
[Price] * [Quantity]
ROUND(([Value B] - [Value A]) / [Value A] * 100, 3)
IF(value > 100 and status = "ok", "high", "normal")
ELAPSED(time, "h")
ROLLING_MEDIAN(value, 101, TRUE)
ZSCORE(value) > 4
HOUR(time) >= 6 and HOUR(time) < 18
```
