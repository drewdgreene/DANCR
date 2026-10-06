# What DANCR can read

DANCR analyses **tables**. It reads a file the way a person would (title lines, tables side by side, summary
rows and all), and it can read a whole folder or glob of files as one table. Scientific formats that are not
tables get their **own step** that turns them into a table — one row per sequence, variant, feature or marker —
which the rest of DANCR can filter, join, chart and report like any other. Anything it truly cannot read is
refused with a plain reason, never quietly misread as CSV.

## Tables DANCR reads

| You can open | Notes |
|---|---|
| CSV, TSV, TXT, DAT, TAB, LOG | Separator and types are detected; titles, blocks and summary rows are handled. |
| Excel (`.xlsx`, `.xlsm`, `.xls`, `.xlsb`, `.ods`) | Every sheet becomes a table; several tables on one sheet are found. |
| Parquet (`.parquet`, `.pq`) | |
| GeoJSON (`.geojson`, or a `.json` that is a GeoJSON file) | One row per feature, with `geometry` and a centre point. |
| GeoPackage (`.gpkg`) and shapefile (`.shp`) | One row per feature; a layer in another CRS is reprojected. |
| A folder or glob of the above | **Load folder** reads them as one table, with a column naming each file. |
| A table at a URL (CSV/TSV/Parquet/JSON) | **Load from a URL** (needs the network). |
| A database query or table | **Load from a database**: SQLite is built in; PostgreSQL and others are included. |
| NetCDF (`.nc`) | **Load NetCDF**: one variable. |
| HDF5 (`.h5`, `.hdf5`) | **Load HDF5**: one dataset. |
| A small table you type | **Type in a table**. |

## Scientific formats — each has its own step

| Format | Step | One row per… |
|---|---|---|
| FASTA / FASTQ (`.fa`, `.fasta`, `.fna`, `.fq`, `.fastq`; also `.gz`) | **Load sequences (FASTA/FASTQ)** | sequence: id, description, length, GC%, and the sequence |
| VCF (`.vcf`, `.vcf.gz`) | **Load variants (VCF)** | variant (chrom, pos, id, ref, alt, qual, filter, INFO columns) — or variant×sample genotype rows |
| GFF3 / GTF / BED (`.gff3`, `.gff`, `.gtf`, `.bed`) | **Load features (GFF/GTF/BED)** | feature, with attributes (ID, Name, Parent, …) lifted into columns |
| GenBank (`.gb`, `.gbk`) | **Load GenBank features** | feature: locus, type, position, strand, gene/product/note |
| PLINK (`.map` + `.ped`) | **Load markers (PLINK)** | marker, with a genotype column per sample |
Dropping one of these on the window (or `dancr suggest --file x.vcf`) adds the right step automatically, and
**Load folder** reads a whole directory of them as one table (with a column naming each file).

## Documents — via MinerU

| Format | Step | One row per… |
|---|---|---|
| PDF, Word (`.doc/.docx`), PowerPoint (`.ppt/.pptx`), RTF, OpenDocument, EPUB, OFD, HTML/MHTML, images | **Load document (PDF/Office)** | content block (`text`, `heading`, `table`, `formula`, …) with a page/block locator — or, with **Output = tables**, a catalog of the tables found (each written to a CSV) |

Document reading uses [MinerU](https://github.com/opendatalab/MinerU) (Apache-2.0). The packaged DANCR ships
MinerU beside the app, so it works with nothing to install; a source checkout detects a bundled or system
MinerU (or a MinerU V1 endpoint). If MinerU is missing, the step says how to add it, or you can point it at a
folder MinerU already produced (engine **an existing MinerU result**) with no install at all. Documents stay
on your machine unless you turn on **Allow upload** for an endpoint.

## Not tables — what to do instead

| You tried | What to do |
|---|---|
| JSON Lines (`.jsonl`, `.ndjson`) or plain JSON | Not a table. Spread the objects into rows and columns and save as CSV/Parquet; a GeoJSON feature collection loads as a table. |
| zip, XML, Markdown, YAML, NumPy/MATLAB files | Not a table. Export the data part to CSV or Excel first. |
| HDF5 (`.h5`) or NetCDF (`.nc`) opened with **Load file** | Use **Load HDF5** / **Load NetCDF** instead and name the dataset or variable — both are included in DANCR. |

**Tip.** `dancr formats` prints this list from the command line, `dancr doctor` confirms every reader is present
(and whether MinerU is installed), and `dancr inspect FILE` previews any readable file.
