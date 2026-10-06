"""File-format constants, the non-table refusal, separator sniffing and sheet listing."""
from __future__ import annotations

from pathlib import Path


CSV_EXT = {".csv", ".tsv", ".txt", ".dat", ".tab", ".log"}
EXCEL_EXT = {".xlsx", ".xlsm", ".xls", ".xlsb", ".ods"}
PARQUET_EXT = {".parquet", ".pq"}
GEOJSON_EXT = {".geojson", ".json"}
VECTOR_EXT = {".gpkg", ".shp"}

_COMPRESS_SUFFIXES = (".gz", ".bgz")


def effective_ext(path: Path) -> str:
    """The format extension, seeing through a trailing .gz/.bgz (so x.vcf.gz is a .vcf)."""
    if path.suffix.lower() in _COMPRESS_SUFFIXES and len(path.suffixes) >= 2:
        return path.suffixes[-2].lower()
    return path.suffix.lower()

# Formats that are not tables. They are refused with a plain reason and a route forward, so a file is never
# silently misread as CSV (a FASTA became thousands of rows; an HDF5 file loaded as empty; a JSONL row as columns).
H5_EXT = {".h5", ".hdf5", ".hdf", ".he5"}
NC_EXT = {".nc", ".nc4", ".cdf", ".netcdf"}
NON_TABLE_EXT = {
    ".bcf": "a BCF variant file", ".embl": "an EMBL file",
    ".jsonl": "a line-delimited JSON file", ".ndjson": "a line-delimited JSON file",
    ".zip": "a zip archive", ".gz": "a compressed file", ".bz2": "a compressed file", ".xz": "a compressed file",
    ".tar": "a tar archive", ".xml": "an XML file", ".md": "a text document",
    ".yaml": "a YAML file", ".yml": "a YAML file",
    ".npy": "a NumPy array", ".npz": "a NumPy archive", ".mat": "a MATLAB file",
}

# Documents and images: not tables, read with the 'Load document' step (MinerU) — see dancr/core/nodes/document.py.
DOC_EXT = {
    ".pdf": "a PDF", ".doc": "a Word document", ".docx": "a Word document",
    ".ppt": "a PowerPoint file", ".pptx": "a PowerPoint file", ".rtf": "an RTF document",
    ".odt": "an OpenDocument text", ".ods": "an OpenDocument spreadsheet", ".odp": "an OpenDocument presentation",
    ".epub": "an EPUB", ".ofd": "an OFD document", ".html": "a web page", ".htm": "a web page",
    ".mhtml": "a web archive", ".mht": "a web archive",
}
IMAGE_EXT = {".png": "an image", ".jpg": "an image", ".jpeg": "an image", ".webp": "an image", ".gif": "an image",
             ".bmp": "an image", ".tif": "an image", ".tiff": "an image", ".jp2": "an image"}


# Scientific formats that DANCR reads with their own step (see dancr/core/nodes/bio.py): the guard points at it.
BIO_EXT = {
    ".fa": ("a FASTA sequence file", "Load sequences (FASTA/FASTQ)"),
    ".fasta": ("a FASTA sequence file", "Load sequences (FASTA/FASTQ)"),
    ".fna": ("a FASTA sequence file", "Load sequences (FASTA/FASTQ)"),
    ".faa": ("a FASTA protein file", "Load sequences (FASTA/FASTQ)"),
    ".ffn": ("a FASTA sequence file", "Load sequences (FASTA/FASTQ)"),
    ".fq": ("a FASTQ sequence file", "Load sequences (FASTA/FASTQ)"),
    ".fastq": ("a FASTQ sequence file", "Load sequences (FASTA/FASTQ)"),
    ".vcf": ("a VCF variant file", "Load variants (VCF)"),
    ".gff": ("a GFF/GTF feature file", "Load features (GFF/GTF/BED)"),
    ".gff3": ("a GFF3 feature file", "Load features (GFF/GTF/BED)"),
    ".gtf": ("a GTF feature file", "Load features (GFF/GTF/BED)"),
    ".bed": ("a BED interval file", "Load features (GFF/GTF/BED)"),
    ".gb": ("a GenBank file", "Load GenBank features"),
    ".gbk": ("a GenBank file", "Load GenBank features"),
    ".genbank": ("a GenBank file", "Load GenBank features"),
    ".ped": ("a PLINK genotype file", "Load markers (PLINK)"),
    ".bim": ("a PLINK .bim file", "Load markers (PLINK)"),
    ".fam": ("a PLINK .fam file", "Load markers (PLINK)"),
}


def _refuse_non_table(path: Path, ext: str) -> None:
    """Raise a plain reason for a file that is not a table, so it is never misread as CSV."""
    if ext in H5_EXT:
        raise ValueError(f"{path.name} is an HDF5 file, not a table. Add a 'Load HDF5' step and name the dataset inside it.")
    if ext in NC_EXT:
        raise ValueError(f"{path.name} is a NetCDF file, not a table. Add a 'Load NetCDF' step and name the variable.")
    if ext in BIO_EXT:
        what, step = BIO_EXT[ext]
        raise ValueError(f"{path.name} is {what}, not a plain table. Add a '{step}' step to read it.")
    if ext in DOC_EXT:
        raise ValueError(f"{path.name} is {DOC_EXT[ext]}, not a table. Add a 'Load document (PDF/Office)' step "
                         "to read it with MinerU.")
    if ext in IMAGE_EXT:
        raise ValueError(f"{path.name} is {IMAGE_EXT[ext]}. If it is a scanned page, screenshot or chart of data, "
                         "add a 'Load document (PDF/Office)' step to read it (OCR); otherwise it is not a table.")
    if ext in NON_TABLE_EXT:
        raise ValueError(f"{path.name} is {NON_TABLE_EXT[ext]}, not a table DANCR can read. Convert it to CSV, "
                         "Parquet or Excel, or read it with the tool that writes it.")


def _open_text(path: Path, encoding: str):
    """A text handle that decompresses .gz/.bgz, so a compressed file is sniffed as the text it holds."""
    if path.suffix.lower() in _COMPRESS_SUFFIXES:
        import gzip
        return gzip.open(path, "rt", encoding=encoding, errors="replace", newline="")
    return open(path, "r", encoding=encoding, errors="replace", newline="")


def sniff_separator(path: Path, encoding: str = "utf8") -> str:
    with _open_text(path, encoding) as f:
        head = f.read(64 * 1024)
    lines = [ln for ln in head.splitlines() if ln.strip()][:50]
    if not lines:
        return ","
    best, best_score = ",", -1.0
    for sep in [",", "\t", ";", "|"]:
        counts = [ln.count(sep) for ln in lines]
        if not counts or max(counts) == 0:
            continue
        # consistent count across lines and > 0
        mode = max(set(counts), key=counts.count)
        consistency = counts.count(mode) / len(counts)
        score = consistency * 10 + min(mode, 20) / 20
        if mode > 0 and score > best_score:
            best, best_score = sep, score
    return best


NULLS = ["", "NA", "N/A", "NaN", "nan", "NULL", "null", "#N/A"]


def list_sheets(path: str | Path) -> list[str]:
    import logging
    try:
        import fastexcel
        return list(fastexcel.read_excel(str(path)).sheet_names)
    except Exception as e:  # noqa: BLE001 - the caller reports "no such sheet"; log the underlying cause
        logging.getLogger("dancr").debug("could not list the sheets of %s: %s", path, e)
        return []
