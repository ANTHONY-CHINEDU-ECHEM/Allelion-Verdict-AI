"""Layer 1: acquisition.

Fetches every public source from a pinned Git commit (raw file or sparse
checkout) or a pinned PyPI wheel, and records the byte count and SHA256 of
every file in data/raw/MANIFEST.json. Nothing downstream reads the network.

Run:  python -m allelion.data.acquire            (fetch what is missing)
      python -m allelion.data.acquire --verify   (recompute and compare hashes)
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from allelion.config import RAW

RAWGH = "https://raw.githubusercontent.com"


@dataclass
class Source:
    key: str
    title: str
    repo: str              # owner/name on GitHub, or pypi:<name>==<version>
    commit: str            # Git commit, or wheel file name for PyPI
    files: dict[str, str]  # remote path -> local path under data/raw
    licence: str
    citation: str
    sparse_dirs: list[str] = field(default_factory=list)  # whole directories (sparse checkout)


SOURCES: list[Source] = [
    Source(
        key="clinvar",
        title="ClinVar allele level tables joined to gnomAD v2 exomes and genomes (GRCh37)",
        repo="macarthur-lab/clinvar",
        commit="2dd39f98022b45adebdf767631251110a00a2ebc",
        files={
            "output/b37/single/clinvar_alleles_with_gnomad_exomes.single.b37.tsv.gz": "clinvar_gnomad_exomes.b37.tsv.gz",
            "output/b37/single/clinvar_alleles_with_gnomad_genomes.single.b37.tsv.gz": "clinvar_gnomad_genomes.b37.tsv.gz",
        },
        licence="ClinVar data are public domain (NCBI); gnomAD data are released under CC0; pipeline code MIT",
        citation="Landrum MJ et al. ClinVar: improving access to variant interpretations. Nucleic Acids Res 2018. Zhang X et al. macarthur-lab/clinvar pipeline.",
    ),
    Source(
        key="intervar",
        title="InterVar evidence database (gene lists, OMIM inheritance, Orphanet summaries, 1000 Genomes zygosity)",
        repo="WGLab/InterVar",
        commit="4801682d37982bb3e99b50392790878a568b42cd",
        files={f"intervardb/{n}": f"intervar/{n}" for n in [
            "PVS1.LOF.genes.hg19", "PS1.AA.change.patho.hg19", "knownGeneCanonical.txt.hg19",
            "orpha.txt.utf8", "PM1_domains_with_benigns.hg19", "mim_orpha.txt", "mim_pheno.txt",
            "mim_recessive.txt", "mim_domin.txt", "mim_adultonset.txt", "PS4.variants.hg19",
            "BP1.genes.hg19", "PP2.genes.hg19", "BS2_hom_het.hg19"]},
        licence="InterVar is free for academic use; derived tables used for gene level annotation only",
        citation="Li Q, Wang K. InterVar: clinical interpretation of genetic variants by the 2015 ACMG-AMP guidelines. Am J Hum Genet 2017.",
    ),
    Source(
        key="gene_lists",
        title="Curated gene lists (inheritance, haploinsufficiency, essential genes, OMIM disease genes)",
        repo="macarthur-lab/gene_lists",
        commit="fcc849637bd71e683bffc618e1a48081a8df08f8",
        files={f"lists/{n}": f"gene_lists/{n}" for n in [
            "all_ar.tsv", "all_ad.tsv", "berg_ar.tsv", "berg_ad.tsv", "clingen_level3_genes_2015_02_27.tsv",
            "clingen_level3_genes_2018_09_13.tsv", "universe.tsv", "mgi_essential.tsv", "clinvar_path_likelypath.tsv",
            "haploinsufficiency_severe_curated_2016.tsv", "homozygous_lof_tolerant_twohit.tsv"]}
        | {"other_data/omim.use.tsv": "gene_lists/omim.use.tsv"},
        licence="MIT (lists); source annotations retain their own terms",
        citation="MacArthur Lab gene_lists repository; Blekhman 2008; Berg 2013; ClinGen dosage sensitivity.",
    ),
    Source(
        key="constraint",
        title="Gene constraint (ExAC pLI) and LoFtool gene intolerance scores",
        repo="Ensembl/VEP_plugins",
        commit="59b34e58b41c052e220b2476c829e692d28b794d",
        files={"pLI_values.txt": "vep/pLI_values.txt", "LoFtool_scores.txt": "vep/LoFtool_scores.txt"},
        licence="Apache 2.0 (plugins); scores from ExAC (open) and Fadista et al.",
        citation="Lek M et al. Analysis of protein coding genetic variation in 60,706 humans. Nature 2016. Fadista J et al. LoFtool. Bioinformatics 2017.",
    ),
    Source(
        key="transcripts",
        title="Canonical GENCODE transcript exon models for GRCh37",
        repo="Illumina/SpliceAI",
        commit="03f42437aaf56dc5dfd822c4ccee5aec1a705079",
        files={"spliceai/annotations/grch37.txt": "spliceai/grch37.txt"},
        licence="Annotation table distributed with SpliceAI (PolyForm Strict for models; table used for exon structure only)",
        citation="Jaganathan K et al. Predicting splicing from primary sequence with deep learning. Cell 2019.",
    ),
    Source(
        key="hpo",
        title="Human Phenotype Ontology, disease annotations and gene to phenotype map (release 2025-01-16)",
        repo="pypi:pyhpo==4.0.0",
        commit="pyhpo-4.0.0-py3-none-any.whl",
        files={"pyhpo/data/hp.obo": "hpo/hp.obo", "pyhpo/data/genes_to_phenotype.txt": "hpo/genes_to_phenotype.txt",
               "pyhpo/data/phenotype.hpoa": "hpo/phenotype.hpoa"},
        licence="HPO licence (free with attribution)",
        citation="Gargano MA et al. The Human Phenotype Ontology in 2024. Nucleic Acids Res 2024.",
    ),
    Source(
        key="corpora",
        title="Annotated variant mention corpora (tmVar, SETH, OSIRIS, Variome, MutationFinder, SNP)",
        repo="Rostlab/nala",
        commit="96a9efade9f7c0da7f8be204c25479e8c660877e",
        files={},
        sparse_dirs=["resources/corpora/tmvar", "resources/corpora/seth", "resources/corpora/osiris",
                     "resources/corpora/variome", "resources/corpora/snp", "resources/corpora/mutationfinder"],
        licence="Each corpus keeps its original licence (research use); PubMed abstracts are NLM records",
        citation="Wei CH et al. tmVar 2013; Thomas P et al. SETH 2016; Furlong LI et al. OSIRIS 2008; Verspoor K et al. Variome 2013; Caporaso JG et al. MutationFinder 2007; Cejuela JM et al. nala 2017.",
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=300) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)


def _fetch_github(src: Source) -> None:
    for remote, local in src.files.items():
        dest = RAW / local
        if dest.exists():
            continue
        print(f"  {src.repo}@{src.commit[:8]}  {remote}")
        _download(f"{RAWGH}/{src.repo}/{src.commit}/{remote}", dest)
    if src.sparse_dirs:
        target = RAW / src.key
        if target.exists():
            return
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["git", "clone", "-q", "--filter=blob:none", "--sparse", "--no-checkout",
                            f"https://github.com/{src.repo}", tmp], check=True)
            subprocess.run(["git", "-C", tmp, "sparse-checkout", "set", *src.sparse_dirs], check=True)
            subprocess.run(["git", "-C", tmp, "checkout", "-q", src.commit], check=True)
            for d in src.sparse_dirs:
                shutil.copytree(Path(tmp) / d, target / Path(d).name)


def _fetch_pypi(src: Source) -> None:
    if all((RAW / v).exists() for v in src.files.values()):
        return
    name, version = src.repo.split(":", 1)[1].split("==")
    meta = json.load(urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json", timeout=60))
    url = next(u["url"] for u in meta["urls"] if u["filename"] == src.commit)
    print(f"  {src.commit}")
    blob = urllib.request.urlopen(url, timeout=300).read()
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for remote, local in src.files.items():
            dest = RAW / local
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(remote))


def _files_for(src: Source) -> list[Path]:
    out = [RAW / v for v in src.files.values()]
    if src.sparse_dirs:
        out += sorted(p for p in (RAW / src.key).rglob("*") if p.is_file())
    return out


def acquire(verify: bool = False) -> dict:
    manifest_path = RAW / "MANIFEST.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {"sources": []}
    for src in SOURCES:
        print(f"[{src.key}] {src.title}")
        if not verify:
            (_fetch_pypi if src.repo.startswith("pypi:") else _fetch_github)(src)
        files = []
        for p in _files_for(src):
            rel = str(p.relative_to(RAW))
            files.append({"path": rel, "bytes": p.stat().st_size, "sha256": sha256(p)})
        manifest["sources"].append({"key": src.key, "title": src.title, "repo": src.repo, "commit": src.commit,
                                    "licence": src.licence, "citation": src.citation, "files": files})
    if verify and old:
        before = {f["path"]: f["sha256"] for s in old["sources"] for f in s["files"]}
        bad = [f["path"] for s in manifest["sources"] for f in s["files"] if before.get(f["path"]) != f["sha256"]]
        print("verify:", "all files match" if not bad else f"{len(bad)} mismatches: {bad[:5]}")
        if bad:
            sys.exit(1)
        return old
    manifest["total_files"] = sum(len(s["files"]) for s in manifest["sources"])
    manifest["total_bytes"] = sum(f["bytes"] for s in manifest["sources"] for f in s["files"])
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"manifest: {manifest['total_files']} files, {manifest['total_bytes'] / 1e6:.1f} MB")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    acquire(ap.parse_args().verify)
