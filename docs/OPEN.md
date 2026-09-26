# Open

- Windows and macOS builds are validated only by the CI workflow, not on a real machine.
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
