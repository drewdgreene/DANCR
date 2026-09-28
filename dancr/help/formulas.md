# Formula reference

You write formulas in the **New column (formula)** step, and in **Filter rows → Or a formula**,
where they're true/false tests. They look like Excel formulas, but each one works on a whole column.

## Syntax

- Columns are written `Total`, `[Unit price]` or `` `Total` ``. Use brackets when the name has
  spaces or symbols. Case and spaces don't have to match exactly. If a column and an Input have
  the same name, the column wins. To use both, write the column in brackets and rename the Input.
- Numbers and text look like `2.5`, `"warm"` or `'warm'`. As in Excel, you write a quote inside
  text twice (`"say ""hi"""`), and a backslash is an ordinary character (`"C:\data"`).
- Arithmetic uses `+ - * / ^ %`. `^` is power and `%` is remainder. As in Excel, a minus sign
  on a value binds tighter than `^`, so `-2^2` is 4. Write `-(2^2)` for −4. `^` works left to
  right, so `2^3^2` is `(2^3)^2` = 64.
- Whole numbers are added, subtracted and multiplied in 64 bits, so small or unsigned number
  columns from Parquet files never wrap around. `/` always gives a decimal.
- Comparisons are `= != < > <= >=`, and `==` and `<>` work too. As in Excel, text is compared
  ignoring case, so `"abc" = "ABC"` is TRUE. A blank cell counts as empty text. It equals `""`
  and isn't equal to `"a"`. A blank number or date isn't equal to any value and passes no other
  comparison. Filter rules behave the same way.
- If you compare a text column with a number, they're compared as numbers, so `code > 5` works
  when `code` holds "10". Text that isn't a number gives a blank answer.
- For logic, use `and`, `or`, `not`, or `AND(a, b)`, `OR(a, b)`, `NOT(a)`.
- `&` joins text, as in `name & " (" & unit & ")"`. As in Excel, a blank joins as nothing, and a
  number is written with up to 15 significant digits. So `0.1 + 0.2 & ""` is "0.3", and a whole
  number has no ".0". `CONCAT` and `TEXT(x)` write numbers the same way.
- You can compare a date column with text, as in `time > "2024-06-01 12:00"`.
- `TRUE`, `FALSE` and `NULL` are available as values.

With one argument, `SUM`, `AVERAGE`, `MIN`, `MAX`, `MEDIAN`, `STDEV` and `COUNT` work on the whole
column and repeat the answer on every row. For example, `x - AVERAGE(x)` centres a column. With
several arguments, `SUM`, `MIN`, `MAX` and `AVERAGE` work across the row, as in `MAX(a, b, c)`.

## Functions

| Function | Meaning |
|---|---|
| `ABS(x)` `SQRT(x)` `EXP(x)` `LN(x)` `LOG(x)` `LOG(x, base)` `LOG2(x)` `POW(x, y)` | Maths |
| `ROUND(x, digits)` `FLOOR(x)` `CEIL(x)` `CEILING(x)` `SIGN(x)` `MOD(a, b)` `CLIP(x, lo, hi)` | Rounding and limits. ROUND works as in Excel. Halves go away from zero (2.5 → 3, and 2.675 → 2.68 to two places), and negative digits round to tens, hundreds… MOD takes the sign of b, like Excel's. `MOD(a, 0)` and `a % 0` are blank |
| `SIN COS TAN ASIN ACOS ATAN ATAN2(y, x) PI()` | Trigonometry (radians) |
| `SUM MIN MAX AVERAGE MEAN MEDIAN STDEV STD VAR COUNT` | Column aggregates with one argument, or row-wise with several |
| `PERCENTILE(x, 0.95)` `ZSCORE(x)` `RANK(x)` | Statistics. RANK gives the largest value rank 1, like Excel, and `RANK(x, 1)` ranks from the smallest. PERCENTILE interpolates like Excel's PERCENTILE.INC. Give it 0–1 or 0–100. A number above 1 is a percentage, so 1.5 means 1.5 % |
| `CUMSUM(x)` `CUMMAX(x)` `CUMMIN(x)` | Running totals |
| `ROW()` | Row number from 1 |
| `LAG(x, n)` `LEAD(x, n)` `DIFF(x, n)` `PCT_CHANGE(x, n)` | Previous or next rows. n defaults to 1. `DIFF` of a date gives seconds |
| `ROLLING_MEAN(x, n)` `ROLLING_MEDIAN` `ROLLING_STD` `ROLLING_MIN` `ROLLING_MAX` `ROLLING_SUM` | Rolling windows over n rows centred on each row, as the Smooth step does. Add `, FALSE` for the n rows ending at each row |
| `FILL_FORWARD(x)` `INTERPOLATE(x)` | Fill blanks |
| `IF(test, a, b)` `ISBLANK(x)` `ISNULL(x)` `COALESCE(a, b, …)` `IFNULL(x, fallback)` | Conditions and blanks. Both branches must be the same kind of value, such as both numbers or both text |
| `LEN UPPER LOWER TRIM LEFT(s, n) RIGHT(s, n) MID(s, start, n)` | Text |
| `CONTAINS(s, part)` `STARTSWITH ENDSWITH CONCAT(a, b, …)` | Text |
| `SUBSTITUTE(s, old, new)` `SUBSTITUTE(s, old, new, k)` | As in Excel. Every `old` becomes `new`, or only the k-th one |
| `REPLACE(s, start, n, new)` | As in Excel. The n characters from position `start` become `new`. Position 1 is the first character |
| `TEXT(x)` `TEXT(date, "%Y-%m-%d")` `VALUE(s)` `NUMBER(s)` | Convert to text (numbers get up to 15 significant digits) or to a number |
| `DATE(text)` `DATE(text, "%d/%m/%Y")` | Parse a date |
| `YEAR MONTH DAY HOUR MINUTE SECOND DAYOFYEAR` | Parts of a date |
| `WEEKDAY(date)` `WEEKDAY(date, type)` | As in Excel, 1 = Sunday … 7 = Saturday. Type 2 gives 1 = Monday … 7 = Sunday, and type 3 gives 0 = Monday … 6 = Sunday. With types 11–17 the week starts on Monday … Sunday |
| `ELAPSED(time, unit)` | Time since the first row that has a time, in `"s"`, `"min"`, `"h"` or `"d"`. That isn't always the earliest time, so sort first if the rows aren't in time order |
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

Dividing by zero gives a blank, where Excel would show #DIV/0!. You won't get infinity, so totals
and charts stay correct.

A NaN read from a file counts as blank in filters, in statistics and in ISBLANK.
