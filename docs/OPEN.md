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
- Places: distances are great-circle on a fixed sphere (mean radius), not ellipsoidal or projected, so they
  differ from geodesic answers by a few tenths of a percent; a grid cell given in km is turned into *rectangular*
  degrees, so away from the equator a "5km" cell is narrower east–west than north–south. Coordinate columns are
  taken as WGS84 lon/lat; a GeoPackage or shapefile layer whose CRS is known is reprojected to WGS84 on load, and
  `project` turns UTM (built in) or another EPSG code (optional pyproj) into lon/lat. Points can be tested
  against polygon layers (a feature's `geometry` WKT, point-in-polygon with holes handled), but polygons are not
  drawn, there are no line geometries and no polygon maths beyond containment, and a nearest-place match compares
  every point with the places within the distance (fine for thousands of places, not millions). The basemap is
  1:110m Natural Earth outlines, so small islands and borders are simplified. The map draws an equirectangular
  or Mercator view, not a tiled web map; a point with a coordinate outside its range is drawn only if the
  coordinate columns were not cleaned with 'Make a point' first.
- GeoPackage and shapefile loading, and reprojection of a non-UTM EPSG code, use the optional `dancr[geo]`
  packages (pyogrio, shapely, pyproj). A source install adds them with `uv sync --extra geo`; the packaged
  installers do not bundle them yet, so on a packaged install those two things show a message asking for the
  extra (everything else about places works without them).

- Load folder reads each file with the same reader as Load file; `diagonal` takes the union of the columns and
  relaxes types (a column that is a number in one file and text in another becomes text; missing values are blank),
  `strict` needs every file to have exactly the same columns, and `text` reads every column as text. Files are read
  in path order. A change to any member file, or a file added or removed, makes the step run again.
- Batch rebinds one file setting of one source step, so a project whose shape differs per file (a different sheet,
  a lookup that also changes) needs one run per shape rather than a batch. Results are written next to the project;
  the per-file results are kept in the project's cache (not swept), so a batch over very many distinct files grows
  the cache. `--jobs` runs files at once; the shared cache is safe for it, but progress lines interleave.
- The FAIR descriptors describe the project's tables, not a hosted release: schema.org uses each table's source
  file as its distribution and there is no persistent identifier (push the export to Zenodo/OSF for one). RO-Crate
  is a metadata graph only (`ro-crate-metadata.json`), not a zipped crate with the files. The license is recorded
  as given; an SPDX id is linked to its canonical URL.
- A context document's `content_hash` is the step's plan hash: `--changed` compares a fresh document with one
  exported earlier, so it sees content and code changes, but a project edited without being saved is compared as it
  is on disk. A change to DANCR itself (or a library) marks every document changed, which is the safe answer.

- Connectors reach outside the machine: a project that uses Load from a URL or Load from a database is no longer
  strictly offline. A URL source reads the whole file into the results folder before reading it (no streaming from
  the server), and asks the server (a HEAD request, remembered for 30 seconds) whether it changed; with that off,
  it reruns only when its settings change. A server database has no cheap, generic way to notice new rows, so it
  reruns when a named `version_column`'s maximum changes, or when its settings change, or when you force it.
  Credentials written as `${ENV_VAR}` are never stored; a password typed straight into the project file is stored
  as written (it is redacted from everything shown, but not from the file itself).
- Load from a database materializes the result (Polars' `read_database`); very large queries should be narrowed by
  the query or aggregated in the database, not pulled whole.
- Load folder with "every table" unions tables of different shapes into one (missing values blank) and marks the
  origin in a `source_table` column; a workbook larger than 30 MB is not looked through for several tables on one
  sheet, so only its sheets/plain table are read.
- Watch polls (every two seconds by default) and reruns after a quiet poll, so it does not use the operating
  system's file notifications; a folder with a great many files is re-checked each poll.
- The catalog reads every project under a folder; with statistics or samples it computes each table, so a folder of
  many large projects takes a while (use `--jobs`). It lists file names only, never complete paths, as the profile
  export does.
- A packaged RO-Crate is metadata-only by default; including the data (`--copy data`) or the result files
  (`--copy results`) copies them into the crate, which can be large. Units are recorded as UCUM codes only for the
  units `core/units.py` knows; an unknown unit keeps the form it was written in.
- Load from a database (a server) and Load NetCDF / Load HDF5 use the optional `dancr[db]` / `dancr[science]`
  packages; like `dancr[geo]`, the packaged installers do not bundle them yet, so a packaged install asks for the
  extra. SQLite and URL reading need nothing beyond the standard library.

Design and extension of the answer engine: `docs/ANSWERS.md`. Earlier plans and review worklists
(all items done): `docs/history/`.
