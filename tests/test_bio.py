"""The bio/lab ingestion steps: scientific formats as ordinary tables (one row per sequence/variant/feature/marker)."""
from pathlib import Path

import polars as pl
import pytest

from dancr.core import Pipeline
from dancr.core.executor import Executor
from conftest import run_one


def frame(tmp_path: Path, type_key: str, body: str, name: str, **params) -> pl.DataFrame:
    (tmp_path / name).write_text(body)
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node(type_key, params={"path": name, **params}, id="n")
    ex = Executor(p)
    st = ex.run(targets=["n"])["n"]
    assert st.status == "done", st.error
    return ex.frame("n").collect()


# ---------------------------------------------------------------- sequences
FASTA = ">seq1 first record\nACGTACGT\nAAAA\n>seq2\nGGGGCCCC\n"


def test_load_sequences_fasta(tmp_path):
    df = frame(tmp_path, "load_sequences", FASTA, "s.fa")
    assert df.height == 2
    row = df.row(0, named=True)
    assert row["id"] == "seq1" and row["description"] == "first record"
    assert row["length"] == 12 and row["sequence"] == "ACGTACGTAAAA"
    assert row["gc_percent"] == pytest.approx(33.3)


def test_load_sequences_fastq(tmp_path):
    body = "@r1 read one\nACGTACGT\n+\nIIIIIIII\n@r2\nGGCC\n+\nIIII\n"
    df = frame(tmp_path, "load_sequences", body, "s.fastq")
    assert df.height == 2 and set(df["id"]) == {"r1", "r2"}
    assert df.row(0, named=True)["description"] == "read one"


def test_load_sequences_rejects_a_non_fasta(tmp_path):
    (tmp_path / "x.fa").write_text("just some text\nwith no records\n")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_sequences", params={"path": "x.fa"}, id="n")
    st = Executor(p).run(targets=["n"])["n"]
    assert st.status == "failed" and "FASTA" in (st.error or "")


# ---------------------------------------------------------------- variants
VCF = (
    "##fileformat=VCFv4.2\n"
    '##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\n"
    "chr1\t100\trs1\tA\tG\t50\tPASS\tDP=20;AF=0.5\tGT:DP\t0/1:9\t1/1:11\n"
    "chr1\t200\t.\tC\tT\t.\tq10\tDP=8\tGT:DP\t0/0:8\t0/1:0\n"
)


def test_load_variants_wide_splits_info(tmp_path):
    df = frame(tmp_path, "load_variants", VCF, "v.vcf")
    assert df.height == 2
    assert set(["chrom", "pos", "id", "ref", "alt", "qual", "filter", "DP", "AF"]) <= set(df.columns)
    r = df.row(0, named=True)
    assert r["chrom"] == "chr1" and r["pos"] == 100 and r["id"] == "rs1" and r["qual"] == 50.0
    assert r["DP"] == 20 and r["AF"] == 0.5                 # INFO values are typed, not text


def test_load_variants_genotype_long_form(tmp_path):
    df = frame(tmp_path, "load_variants", VCF, "v.vcf", samples="genotype")
    assert df.height == 4                                   # 2 variants x 2 samples
    assert set(df["sample"]) == {"S1", "S2"}
    r = df.filter((pl.col("pos") == 100) & (pl.col("sample") == "S2")).row(0, named=True)
    assert r["gt"] == "1/1"


def test_load_variants_full_form(tmp_path):
    df = frame(tmp_path, "load_variants", VCF, "v.vcf", samples="full")
    assert "dp" in df.columns and df.height == 4


# ---------------------------------------------------------------- features
GFF = (
    "##gff-version 3\n"
    "chr1\tanno\tgene\t100\t500\t.\t+\t.\tID=gene1;Name=DREB2A\n"
    "chr1\tanno\texon\t100\t200\t.\t+\t.\tParent=gene1\n"
)


def test_load_features_gff3_lifts_attributes(tmp_path):
    df = frame(tmp_path, "load_features", GFF, "f.gff3")
    assert df.height == 2
    gene = df.filter(pl.col("type") == "gene").row(0, named=True)
    assert gene["seqid"] == "chr1" and gene["start"] == 100 and gene["end"] == 500
    assert gene["ID"] == "gene1" and gene["Name"] == "DREB2A"
    exon = df.filter(pl.col("type") == "exon").row(0, named=True)
    assert exon["Parent"] == "gene1"


def test_load_features_bed(tmp_path):
    df = frame(tmp_path, "load_features", "chr1\t0\t100\tpeakA\t50\t+\n", "x.bed")
    r = df.row(0, named=True)
    assert r["chrom"] == "chr1" and r["chrom_start"] == 0 and r["chrom_end"] == 100
    assert r["name"] == "peakA" and r["strand"] == "+"


# ---------------------------------------------------------------- genbank
GB = (
    "LOCUS       VYL-CON-0001        100 bp    DNA     circular SYN 01-JAN-2026\n"
    "DEFINITION  test vector.\n"
    "     source         1..100\n"
    '                     /organism="synthetic DNA construct"\n'
    "     CDS            10..90\n"
    '                     /label="Cas9"\n'
    '                     /product="nuclease"\n'
    "ORIGIN\n"
    "        1 acgtacgtac\n"
    "//\n"
)


def test_load_genbank_features(tmp_path):
    df = frame(tmp_path, "load_genbank", GB, "c.gb")
    assert df.height == 2
    cds = df.filter(pl.col("type") == "CDS").row(0, named=True)
    assert cds["locus"] == "VYL-CON-0001" and cds["start"] == 10 and cds["end"] == 90
    assert cds["length"] == 81 and cds["product"] == "nuclease"


# ---------------------------------------------------------------- plink
MAP = "1\tm1\t0\t100\n1\tm2\t0\t200\n"
PED = "F1\tindA\t0\t0\t0\t-9\tA\tA\tG\tG\nF1\tindB\t0\t0\t0\t-9\tA\tG\tG\tG\n"


def test_load_markers_with_genotypes(tmp_path):
    (tmp_path / "c.map").write_text(MAP)
    (tmp_path / "c.ped").write_text(PED)
    df = frame(tmp_path, "load_markers", MAP, "c.map")
    assert df.height == 2
    assert df.columns[:4] == ["chrom", "marker", "cm", "pos"]
    assert set(df.columns) >= {"indA", "indB"}
    ind = df.filter(pl.col("marker") == "m1").row(0, named=True)
    assert ind["indA"] == "AA" and ind["indB"] == "AG"


def test_load_markers_positions_only(tmp_path):
    (tmp_path / "c.map").write_text(MAP)
    df = frame(tmp_path, "load_markers", MAP, "c.map", genotypes=False)
    assert df.columns == ["chrom", "marker", "cm", "pos"]


# ---------------------------------------------------------------- routing + guard
def test_guard_points_at_the_bio_step(tmp_path):
    (tmp_path / "v.vcf").write_text(VCF)
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "v.vcf"}, id="n")
    st = Executor(p).run(targets=["n"])["n"]
    assert st.status == "failed"
    assert "Load variants (VCF)" in (st.error or "")


def test_add_files_routes_by_extension(tmp_path):
    import dancr.headless as hl
    (tmp_path / "v.vcf").write_text(VCF)
    (tmp_path / "s.fa").write_text(FASTA)
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    ids = hl.add_files(p, [str(tmp_path / "v.vcf"), str(tmp_path / "s.fa")])
    types = [p.nodes[i].type for i in ids]
    assert types == ["load_variants", "load_sequences"]


def test_bio_node_for_maps_extensions():
    from dancr.core.nodes.bio import bio_node_for
    assert bio_node_for("x.vcf") == "load_variants"
    assert bio_node_for("x.gff3") == "load_features"
    assert bio_node_for("x.fq") == "load_sequences"
    assert bio_node_for("x.csv") is None


@pytest.mark.parametrize("ext", sorted(__import__("dancr.core.nodes.load", fromlist=["BIO_EXT"]).BIO_EXT))
def test_every_bio_extension_points_at_its_step(ext, tmp_path):
    (tmp_path / f"x{ext}").write_bytes(b"x")
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": f"x{ext}"}, id="n")
    st = Executor(p).run(targets=["n"])["n"]
    assert st.status == "failed"
    assert "step" in (st.error or "") and "not a plain table" in (st.error or "")


# ---------------------------------------------------------------- dosage, gzip, inspect
def test_variant_dosage(tmp_path):
    df = frame(tmp_path, "load_variants", VCF, "v.vcf", samples="genotype")
    s2 = df.filter((pl.col("pos") == 100) & (pl.col("sample") == "S2")).row(0, named=True)
    s1 = df.filter((pl.col("pos") == 100) & (pl.col("sample") == "S1")).row(0, named=True)
    assert s1["dosage"] == 1 and s2["dosage"] == 2


def test_gzip_is_read_through(tmp_path):
    import gzip

    def load(name: str, data: bytes, type_key: str, **params):
        (tmp_path / name).write_bytes(data)
        p = Pipeline("t"); p.path = tmp_path / "p.json"
        p.add_node(type_key, params={"path": name, **params}, id="n")
        ex = Executor(p)
        st = ex.run(targets=["n"])["n"]
        assert st.status == "done", st.error
        return ex.frame("n").collect()

    df = load("v.vcf.gz", gzip.compress(VCF.encode()), "load_variants", samples="genotype")
    assert df.height == 4 and "dosage" in df.columns
    seq = load("s.fa.gz", gzip.compress(FASTA.encode()), "load_sequences")
    assert seq.height == 2


def test_gzip_extension_routes_and_is_guarded(tmp_path):
    from dancr.core.nodes.bio import bio_node_for
    import gzip
    assert bio_node_for("x.vcf.gz") == "load_variants"
    (tmp_path / "v.vcf.gz").write_bytes(gzip.compress(VCF.encode()))
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_file", params={"path": "v.vcf.gz"}, id="n")
    st = Executor(p).run(targets=["n"])["n"]
    assert st.status == "failed" and "Load variants (VCF)" in (st.error or "")


def test_inspect_file_reads_bio_and_tables(tmp_path):
    import dancr.headless as hl
    (tmp_path / "v.vcf").write_text(VCF)
    out = hl.inspect_file(tmp_path / "v.vcf", rows=2)
    assert any(c["name"] == "chrom" for c in out["columns"])
    assert out["messages"] and "variant" in out["messages"][0]
    (tmp_path / "t.csv").write_text("a,b\n1,x\n2,y\n")
    out2 = hl.inspect_file(tmp_path / "t.csv", rows=1)
    assert [c["name"] for c in out2["columns"]] == ["a", "b"]


# ---------------------------------------------------------------- a folder of scientific files
def _folder(tmp_path, pattern="*", **params):
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_folder", params={"path": str(tmp_path), "pattern": pattern,
                                      "source_column": "source_file", **params}, id="n")
    ex = Executor(p)
    st = ex.run(targets=["n"])["n"]
    assert st.status == "done", st.error
    return ex.frame("n").collect(), st


def test_load_folder_reads_a_folder_of_vcfs(tmp_path):
    (tmp_path / "a.vcf").write_text(VCF)
    (tmp_path / "b.vcf").write_text(VCF)
    df, st = _folder(tmp_path, "*.vcf")
    assert df.height == 4 and set(df["source_file"]) == {"a.vcf", "b.vcf"}
    assert any("scientific file" in m for m in st.messages)


def test_load_folder_passes_bio_params(tmp_path):
    (tmp_path / "a.vcf").write_text(VCF)
    df, _ = _folder(tmp_path, "*.vcf", bio_params={"samples": "genotype"})
    assert {"sample", "dosage"} <= set(df.columns) and df.height == 4


def test_load_folder_includes_gzip_bio_files(tmp_path):
    import gzip
    (tmp_path / "a.vcf").write_text(VCF)
    (tmp_path / "b.vcf.gz").write_bytes(gzip.compress(VCF.encode()))
    df, _ = _folder(tmp_path, "*.vcf*")
    assert df.height == 4 and set(df["source_file"]) == {"a.vcf", "b.vcf.gz"}


GB_JOIN = (
    "LOCUS       X 1000 bp DNA linear SYN 01-JAN-2026\n"
    "     CDS             join(10..90,200..290)\n"
    '                     /label="x"\n'
    "     CDS             complement(join(400..500,600..650))\n"
    '                     /label="y"\n'
    "ORIGIN\n"
    "//\n"
)


def test_load_genbank_counts_every_part_of_a_join(tmp_path):
    """join(10..90,200..290) is 172 bp over positions 10-290, not the first block alone."""
    df = frame(tmp_path, "load_genbank", GB_JOIN, "j.gb")
    x = df.filter(pl.col("label") == "x").row(0, named=True)
    y = df.filter(pl.col("label") == "y").row(0, named=True)
    assert x["start"] == 10 and x["end"] == 290 and x["length"] == 81 + 91 and x["strand"] == "+"
    assert y["start"] == 400 and y["end"] == 650 and y["length"] == 101 + 51 and y["strand"] == "-"


def test_load_folder_diagonal_unions_mixed_bio_files(tmp_path):
    (tmp_path / "a.vcf").write_text(VCF)
    (tmp_path / "s.fa").write_text(FASTA)
    df, _ = _folder(tmp_path, "*")
    assert set(df["source_file"]) == {"a.vcf", "s.fa"}
    assert "id" in df.columns and "chrom" in df.columns     # diagonal union of the two schemas


def test_load_sequences_reads_a_gzip_file(tmp_path):
    import gzip
    with gzip.open(tmp_path / "s.fa.gz", "wt") as fh:
        fh.write(FASTA)
    p = Pipeline("t"); p.path = tmp_path / "p.json"
    p.add_node("load_sequences", params={"path": "s.fa.gz"}, id="n")
    out = run_one(p, "n")
    assert out.height == 2 and out["id"].to_list() == ["seq1", "seq2"]


def test_load_features_limit_stops_early(tmp_path):
    body = "".join(f"chr1\tsrc\tgene\t{i}\t{i + 10}\t.\t+\t.\tID=g{i}\n" for i in range(1, 50))
    df = frame(tmp_path, "load_features", body, "f.gff", limit=4)
    assert df.height == 4
