# Data card

Every input to Allelion Verdict AI is public and fetched by `make acquire` from a pinned Git commit or a pinned PyPI wheel. The acquisition step writes the byte count and SHA256 of each file to `data/raw/MANIFEST.json`, and `make verify` recomputes them. Nothing after acquisition reads the network. The raw layer is 1,187 files and 236 MB.

## Sources

| Source | Pinned at | Files | Size | Used for | Terms |
|---|---|---|---|---|---|
| ClinVar allele tables joined to gnomAD v2 exomes and genomes, GRCh37 (macarthur-lab/clinvar) | `2dd39f98` | 2 | 90.3 MB | variants, classifications, review status, submissions, allele counts, homozygotes, continental frequencies | ClinVar public domain; gnomAD CC0; pipeline MIT |
| InterVar evidence database (WGLab/InterVar) | `4801682d` | 14 | 26.2 MB | inheritance (OMIM dominant, recessive, adult onset), Orphanet summaries, gene lists for PVS1, PP2 and BP1 cross checks, 1000 Genomes zygosity | free for academic use |
| Curated gene lists (macarthur-lab/gene_lists) | `fcc84963` | 12 | 1.0 MB | autosomal dominant and recessive genes, ClinGen haploinsufficiency, essential genes, OMIM disease genes | MIT; source annotations keep their own terms |
| Gene constraint (Ensembl/VEP_plugins) | `59b34e58` | 2 | 0.7 MB | ExAC pLI and LoFtool per gene | Apache 2.0 |
| Canonical transcript exon table (Illumina/SpliceAI) | `03f42437` | 1 | 4.4 MB | exon boundaries for 2,782 genes, used only as coordinates | terms of the repository at the pinned commit |
| Human Phenotype Ontology, release of 16 January 2025 (pyhpo 4.0.0 wheel) | wheel file | 3 | 63.4 MB | ontology, disease annotations, gene to phenotype map | HPO licence, attribution required |
| Variant mention corpora: tmVar, SETH, OSIRIS, Variome, MutationFinder (Rostlab/nala) | `96a9efad` | 1,153 | 49.9 MB | training and testing the variant mention extractor, the literature layer | each corpus keeps its original research terms |

NCBI, EBI and Hugging Face downloads were not reachable from the build environment, which is why the newest ClinVar release, gnomAD v4, ClinGen allele registry records, SpliceAI and conservation scores are absent. Every source above was chosen because it can be fetched again byte for byte.

## Knowledge base

| Table | Rows | Notes |
|---|---|---|
| Variants | 341,374 | one row per ClinVar variation, 179,425 observed in gnomAD |
| Genes | 5,561 | constraint, inheritance, dosage, disease count, mechanism |
| Genes with an anchored exon model | 2,782 | exon boundaries anchored to coding coordinates from ClinVar's own genomic and HGVS pairs |
| Diseases with phenotype profiles | 12,687 | from HPO annotations, with term frequencies |
| HPO terms | 19,034 | with information content for semantic similarity |
| Literature documents | 2,168 | with 1,623 links from extracted mentions to ClinVar variants |
| Knowledge graph | 445,822 nodes, 2,203,144 edges | variants, genes, diseases, phenotypes, traits, publications |
| Copilot passages | 37,881 passages, 2.17 million words | phenotype 16,449, disease 12,687, literature 4,466, gene 4,203, guidance 76 |

### Variants by ClinVar review status

| Stars | Variants | Meaning |
|---|---|---|
| 0 | 44,027 | no assertion criteria |
| 1 | 241,160 | single submitter, or conflicting submissions |
| 2 | 47,293 | multiple submitters, no conflicts |
| 3 | 8,871 | reviewed by an expert panel |
| 4 | 23 | practice guideline |

### Variants by consequence

| Consequence | Variants |
|---|---|
| Missense | 144,228 |
| Synonymous | 59,891 |
| Intronic | 44,072 |
| 3 prime UTR | 34,624 |
| Frameshift | 19,646 |
| Nonsense | 14,269 |
| 5 prime UTR | 7,894 |
| Canonical splice donor | 4,200 |
| Canonical splice acceptor | 3,188 |
| In frame insertion or deletion | 2,395 |
| Other (upstream, start loss, stop loss, downstream, non coding, unparsed) | 6,967 |

Consequence is derived by parsing the HGVS expression ClinVar displays, then refined with the exon model where one exists (for example a stop gained in the last exon or within 50 nucleotides of the last exon junction is marked as escaping nonsense mediated decay).

## Population frequency

Exome and genome counts from gnomAD v2 are summed per variant. Three frequencies are derived:

1. **Filtering allele frequency (FAF95).** For each continental group (African, Latino, East Asian, South Asian, non Finnish European) the lower bound of the 95% Poisson confidence interval on the allele count divided by the allele number; the maximum across groups is kept. This is the frequency compared with BA1 and BS1 thresholds, as recommended by Whiffin and colleagues (2017).
2. **Group maximum with an allele number of at least 2,000.** Used for BA1 so that a single observation in a small group does not trigger a stand alone benign call.
3. **Overall allele frequency, allele count, homozygotes and hemizygotes.** Used for PM2, BS2 and the interface.

A variant absent from gnomAD is recorded as unobserved rather than as frequency zero, and PM2 is applied at supporting strength as recommended by ClinGen SVI (2020).

## Evaluation sets

| Set | Variants | Definition |
|---|---|---|
| Blind | 8,733 in 17 genes | every variant with three or four stars and a five tier classification, frozen and hashed before any model existed |
| Development | 3,300 in 1,112 genes | two star variants, stratified to 700 per class (500 likely pathogenic) |
| Evidence pool | 269,419 | every remaining classified variant; source of allelic evidence, gene spectra and predictor training data |
| Blind questions | 30 | hashed separately |
| Development questions | 60 | used for retrieval and writer choices |

Blind set composition by panel: ENIGMA (BRCA1, BRCA2) 6,031; InSiGHT (MLH1, MSH2, MSH6, PMS2) 2,113; CFTR2 254; ClinGen RASopathy expert panel 232; ClinGen inherited cardiomyopathy expert panel (MYH7) 102; PharmGKB 1. By class: pathogenic 4,868, likely pathogenic 207, uncertain significance 774, likely benign 1,307, benign 1,577.

Variants with conflicting interpretations, drug response, risk factor or "not provided" labels are kept in the knowledge base and the interface but are not used as references.

## Known issues in the data

1. **Display transcript.** ClinVar reports each variant on one transcript. For BRCA1 this is often an exon 11 skipping isoform, which turns exon 11 truncating variants into deep intronic changes. See [evaluation.md](evaluation.md).
2. **Snapshot age.** The join dates from 2018. Classifications have been revised since, and gnomAD v4 has roughly six times as many exomes.
3. **Ancestry coverage.** gnomAD v2 under represents several populations. A variant rare in the five groups used for FAF95 may be common in a group that is not represented, which biases toward pathogenic calls for those groups.
4. **Literature corpora.** The five corpora are annotated abstracts chosen to train mention extraction, not a sample of the literature relevant to any gene. The copilot shows them as literature evidence but they never contribute to a classification.
5. **Exon models.** Exon boundaries are anchored per exon from ClinVar coordinate pairs, and 2,779 genes in the knowledge base have no reliable model. For those genes the PVS1 decision tree cannot test nonsense mediated decay or exon position, so the engine falls back to evidence from the pool (pathogenic truncating variants downstream of the new stop) or reduces PVS1 to strong strength, and states which path it took in the rationale.

## Privacy

The knowledge base contains no patient level data. ClinVar records are aggregate submissions, and gnomAD frequencies are aggregate counts. The copilot redacts identifiers typed into a question (names after a title, dates and dates of birth, NHS numbers, record numbers, email addresses, phone numbers, postcodes, ages over 89) before retrieval, and the audit log stores only the redacted question and a SHA256 of the original.
