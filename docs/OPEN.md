# Open

- Windows and macOS builds are validated only by the CI workflow, not on a real machine.
- Releases are not code-signed (Windows SmartScreen and macOS Gatekeeper warn on first open): that needs a
  signing certificate and an Apple Developer ID, then `signtool` and `codesign`/`notarytool` steps in CI. The
  macOS build is for Apple silicon only (the `macos-latest` runner); an Intel Mac needs an Intel runner or a
  universal2 Python and wheels.
- `contribution` in change mode with an average (`stat` mean) adds up each group's change in its average, which
  is not a share of the change in the overall average; a change in an average does not split by group.
- The seasonal forecast's band uses the straight-line fit's error, so it is a rough band.
- Comparing groups: with three or more groups there is no test of which pairs differ (no post-hoc test); a paired
  comparison is of two groups only (no repeated-measures ANOVA); a two-way design (light × species) is compared one
  factor at a time. Merged cells are inferred from where banner text sits (the cell grid has no merge ranges), and
  a sheet's formulas are not read, so a summary row is found by its label (AVERAGE, Total) rather than by its
  formula. A table laid out sideways (one variable per row) is not turned round. On a sheet over 20,000 rows only
  its top 300 and bottom 60 rows are used to work out the layout.
- Answers cannot yet: group by two things at once (a cross-tab; refused), link keys written differently
  (`#10234` against `10234`), treat survey answers as ordered, say which month was over budget, join two
  different conditions with "or", keep two separate periods (`in March and April`), exclude a month
  (`not in March`) or a range of dates, or read a decimal comma (`30,5` is 30 and 5). Relative dates count from
  the data's latest date, not today.
- Times written at several UTC offsets (daylight saving) stay in UTC unless a time zone is chosen: the
  offsets alone do not say which zone they came from.
- A cache sweep in one process can move a result aside in the instant between another process seeing it and
  reading it; that step then fails once with a missing-file error and succeeds when run again.
- Cancel takes effect between steps: a step writing a very large result finishes first (Polars 1.44 cannot
  stop a write safely).
- The project file's lock is an operating-system file lock: on a network drive shared by two machines it may
  not be seen by the other machine (each save still checks the file did not change since it was read).

Design and extension of the answer engine: `docs/ANSWERS.md`. Earlier plans and review worklists
(all items done): `docs/history/`.
