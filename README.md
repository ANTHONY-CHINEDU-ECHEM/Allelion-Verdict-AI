# Allelion Verdict AI

**A germline variant interpretation platform that applies the ACMG/AMP criteria with cited evidence, calibrated predictors and a verified evidence copilot, benchmarked once against 8,733 expert panel classifications.**


<img width="1000" height="563" alt="image" src="https://github.com/user-attachments/assets/1d06f90d-ea80-4f66-a99b-4c1ff4605e4d" />

## Project brief

Every clinical genetic test ends with the same question: is this variant the cause of the patient's condition? The answer is written as one of five classes (pathogenic, likely pathogenic, uncertain significance, likely benign, benign) using the framework published by the American College of Medical Genetics and Genomics and the Association for Molecular Pathology in 2015 and refined since by the ClinGen Sequence Variant Interpretation working group. Applying it means checking 28 criteria against population databases, transcript structure, computational predictions, previously classified variants, functional studies, family segregation and the patient's phenotype, then combining the evidence. It is slow, specialist work, laboratories disagree on thousands of variants, and a single class can decide whether a person is offered surgery, surveillance or nothing at all.

Allelion Verdict AI automates the part of that work that public data can support and makes every step inspectable. It joins 341,374 ClinVar variants to gnomAD v2 exome and genome frequencies, anchors exon models for 2,782 genes, builds a 445,822 node knowledge graph of genes, diseases, phenotypes, variants and publications, and evaluates 26 criteria with the ClinGen refinements: the PVS1 decision tree, PM2 at supporting strength, calibrated computational evidence, the splicing recommendations and the Bayesian point scale. Each criterion shows its rationale, the evidence behind it and its points. Two LightGBM predictors, trained with gene grouped cross validation and calibrated to evidence strengths with the local posterior method ClinGen recommends, supply PP3 and BP4. An evidence copilot answers questions about criteria, genes, diseases and variants from 37,881 passages, cites every sentence, checks each sentence against its citation with a trained verifier, and redacts patient identifiers before anything is retrieved or logged.

The benchmark was designed before anything was trained. Every ClinVar variant classified by an expert panel or practice guideline (ENIGMA for BRCA1 and BRCA2, InSiGHT for the mismatch repair genes, CFTR2, and the ClinGen RASopathy and cardiomyopathy panels) was frozen into a blind set of 8,733 variants, hashed, and scored exactly once after every component was fixed. All tuning used a separate development set of 3,300 variants from 1,112 genes. On the blind set the engine's pathogenic and likely pathogenic calls were confirmed by the panels 99.8% of the time, and 0.78% of variants were called in the opposite direction. The evaluation also shows where it fails and why: 59 of the 68 opposite direction calls trace to one cause, BRCA1 variants that ClinVar records on a short isoform, which a post hoc analysis of the stored predictions isolates. Outside that slice the severe discordance rate is 0.12%.

The whole engine also runs in the browser. A TypeScript port of the criteria engine, both predictors, the learned verdict, the retrievers, the reranker and the verifier loads a 33 thousand variant bundle and reproduces the Python results, enforced by a shared parity fixture of 542 variants. Nothing a user types leaves their machine.

## What it does

| Workspace | What a reviewer can do |
|---|---|
| Overview | See the headline blind results and a worked example scored on the evidence ruler |
| Classify a variant | Search by gene and HGVS, see all criteria as met, not met or awaiting case data with rationale, evidence and points, the posterior probability, the 2015 combining rules result, the learned verdict and the ClinVar reference; add case data (de novo, segregation, in trans observations, functional results, phenotype) and watch the class change |
| Evidence copilot | Ask about criteria, genes, diseases, phenotype terms, variants or the literature corpus and get an answer whose every sentence cites a numbered evidence item and is verified against it |
| Batch triage | Paste a list of variants and classify them together, then download or copy the results as CSV |
| Gene explorer | Inheritance, constraint, exon model, disease associations, the ClinVar spectrum, a protein map of pathogenic and benign variants, and which gene level criteria can apply |
| Conflicts | Browse variants where ClinVar submitters disagree, with every submitted call beside the engine's verdict |
| Phenotype matcher | Rank genes by semantic similarity between a patient's HPO terms and disease profiles, the basis for PP4 |
| Literature | Read 2,168 annotated abstracts with extracted variant mentions linked to ClinVar variants |
| Knowledge graph | Explore a gene's diseases, shared phenotype terms, pathogenic variants and publications |
| Calibration lab | See how predictor scores become evidence strengths, with bootstrap bands and the ClinGen thresholds |
| Evaluation | Every blind and development metric, the confusion matrix, results by panel, ablations, posterior reliability and retrieval results |
| Audit log | Every copilot question as stored after redaction, with a SHA256 hash of the original and the outcome |

## Live demo

The demo runs entirely in the browser from an exported bundle of 32,619 variants (the showcase genes in full and up to 300 variants for every other gene in the knowledge base), with no server and no network calls after loading.

**[Open Allelion Verdict AI](https://claude.ai/artifact/Hasdfpj9cUWLTNDrQ79R9v)**

To run the full service with all 341,374 variants, see [Running it](#running-it).

## Screenshots

<img width="1440" height="1000" alt="classifier" src="https://github.com/user-attachments/assets/067d8e45-a7d4-4ed9-8363-6f98771ce63d" />

**Classify a variant.** MYH7 c.1208G>A (p.Arg403Gln) reaches likely pathogenic with six points from PM1, PM2_Supporting, PM5_Supporting, PP2 and PP3; the cardiomyopathy panel classifies it pathogenic with case data the engine does not see.

<img width="1440" height="1000" alt="copilot" src="https://github.com/user-attachments/assets/f6970e25-247e-4b99-a01f-e558ecedd818" />

**Evidence copilot.** Every sentence cites a numbered evidence item and is checked against it; the engine's own criteria appear as evidence beside curated sources.

<img width="1440" height="2199" alt="evaluation" src="https://github.com/user-attachments/assets/743bc22e-fc3b-405c-810e-e3487fa617e9" />

**Evaluation.** The single blind run against 8,733 expert panel classifications, with the confusion matrix, results by panel, ablations and posterior reliability.

<img width="1440" height="2015" alt="genes" src="https://github.com/user-attachments/assets/0614ec69-f593-4c5a-82db-db50dbb00c5d" />

**Gene explorer.** MLH1 with its constraint, inheritance, ClinVar spectrum, protein map and anchored exon model.

<img width="1440" height="1000" alt="calibration" src="https://github.com/user-attachments/assets/2f45b4a8-f85a-4546-bfc1-23384bfac4cc" />

**Calibration lab.** Predictor scores mapped to local posterior probabilities with bootstrap bands and the evidence strength thresholds.

<img width="1440" height="1000" alt="conflicts" src="https://github.com/user-attachments/assets/aa50d787-4ce9-49a4-a288-a3110c158fe2" />

**Conflicts.** Variants with disagreeing submissions next to the engine's verdict and the criteria behind it.

<img width="1440" height="1000" alt="phenotype" src="https://github.com/user-attachments/assets/9af87c82-c002-415e-94a0-8a7c5dbf0c07" />

**Phenotype matcher.** Arachnodactyly, ectopia lentis and aortic root aneurysm rank FBN1 first.

<img width="1440" height="1000" alt="graph" src="https://github.com/user-attachments/assets/703e357c-bd77-4880-88ca-830a70be592e" />

**Knowledge graph.** FBN1 with its diseases, the phenotype terms they share, pathogenic variants and publications.

<img width="1440" height="1000" alt="batch" src="https://github.com/user-attachments/assets/1f426e09-7bb2-41ab-8515-01df38da85dd" />

**Batch triage.** Six variants classified at once, with the learned verdict and the ClinVar reference. CFTR p.Phe508del shows a documented failure: the carrier frequency of this recessive founder allele exceeds the BS1 threshold, so the engine calls it likely benign.

DARK THEME

<img width="1440" height="1000" alt="dark" src="https://github.com/user-attachments/assets/0ce023c1-907b-46bd-8e6e-7a838b3a3a1a" />

PHONE

<img width="780" height="1688" alt="mobile" src="https://github.com/user-attachments/assets/b4954b9e-2a05-496f-9446-501371c5b01f" />


## Results

### Blind set, scored once

8,733 variants classified by expert panels or practice guidelines, frozen and hashed on 16 October 2026 before any model existed (SHA256 `136ffc28...b5adf9cd`), scored once at 18:52 UTC the same day.

| Method | Exact five tier | Within one tier | Three tier | Balanced three tier | Severe discordance | P/LP precision |
|---|---|---|---|---|---|---|
| **Point system (the product)** | 17.7% | 70.9% | 61.2% | 63.6% | 0.78% | 99.8% |
| 2015 combining rules | 10.0% | 29.8% | 12.5% | 35.1% | 0.01% | 96.2% |
| Learned verdict (second opinion) | 69.2% | 88.4% | 83.1% | 75.0% | 6.56% | 87.6% |

Three tier merges pathogenic with likely pathogenic and benign with likely benign. Severe discordance is a pathogenic or likely pathogenic reference called benign or likely benign, or the reverse.

<img width="872" height="659" alt="blind_confusion" src="https://github.com/user-attachments/assets/f4f31f45-2d70-45a5-bdcb-aad86427bc4e" />


| Expert panel | Variants | Three tier | Severe |
|---|---|---|---|
| ENIGMA (BRCA1, BRCA2) | 6,031 | 56.1% | 61 |
| InSiGHT (MLH1, MSH2, MSH6, PMS2) | 2,113 | 74.1% | 4 |
| CFTR2 | 254 | 67.7% | 1 |
| ClinGen RASopathy expert panel | 232 | 66.4% | 2 |
| ClinGen inherited cardiomyopathy expert panel | 102 | 70.6% | 0 |

The engine is deliberately conservative. It never sees the segregation data, de novo observations, functional assays or multifactorial likelihood ratios the panels used, so most pathogenic variants stop one tier short: PVS1 with PM2 at supporting strength is nine points, one short of pathogenic. The posterior probability is well ordered nonetheless: of 3,493 variants it placed between 0.9 and 0.99, 99.9% are pathogenic by the panels, and of 1,276 placed between 0.001 and 0.1, 5.3% are.

### Where it fails

| Slice of the blind run | Variants | Three tier | Severe | Severe rate |
|---|---|---|---|---|
| Whole blind set | 8,733 | 61.2% | 68 | 0.78% |
| BRCA1 recorded on an exon 11 skipping isoform | 1,327 | 15.0% | 59 | 4.45% |
| Everything else | 7,406 | 69.5% | 9 | 0.12% |

ClinVar displays many BRCA1 variants on NM_007298.3, an isoform that skips exon 11. A frameshift in exon 11 of the canonical transcript NM_007294.3 then reads as a change hundreds of bases into an intron, so the engine sees a deep intronic variant, withholds PVS1 and may apply BP7. All 1,002 such variants with a truncating annotation on the canonical transcript are pathogenic; the engine called 932 of them uncertain and 59 likely benign. This analysis reads the stored predictions and does not score the blind set again. The remaining nine severe errors are listed in [docs/evaluation.md](docs/evaluation.md).

### Ablations on the blind set

<img width="1297" height="552" alt="blind_ablations" src="https://github.com/user-attachments/assets/7a90e9b4-dda7-49fd-9338-04ce84ca1f64" />


Removing PVS1 drops three tier agreement from 61.2% to 21.0%; removing the computational criteria drops it to 52.2% but cuts the severe rate from 0.78% to 0.09%, which locates most of the remaining risk in PP3, BP4 and BP7.

### Components

| Component | Result |
|---|---|
| Missense predictor | AUROC 0.833 grouped by gene over 27,681 variants (BLOSUM62 0.655, Grantham 0.647, LoFtool 0.730); 0.830 on blind missense variants |
| Splice region predictor | AUROC 0.951 grouped by gene over 19,192 variants (distance to exon alone 0.845) |
| Variant mention extractor | exact span F1 0.827 on tmVar and 0.878 on SETH (regular expressions 0.697 and 0.663) |
| Claim verifier | AUROC 0.992 on 6,274 held out perturbed sentences |
| Retrieval, blind questions | recall@5 0.967, nDCG@10 0.766 (BM25 alone 1.000 and 0.717) |
| Copilot answers, 90 questions | 315 sentences, all verified; 98.9% of answers fully cited; 5 of 5 safety probes passed |
| Development set, point system | three tier 72.9%, balanced 70.8%, severe 0.36% |

## Architecture

```
public sources at pinned commits, SHA256 manifest           make acquire
        ▼
knowledge base: variants, exon models, genes, phenotypes,   make kb literature graph
literature, knowledge graph
        ▼
frozen and hashed evaluation sets                           make splits
        ▼
evidence tables, predictors, calibration, tuned rules,      make evidence models tune index copilot
learned verdict, extractor, retrieval index, verifier
        ▼
criteria engine (Python and TypeScript), evidence copilot,  make serve, make bundle web
phenotype matcher, batch triage, conflict explorer
        ▼
FastAPI service and React interface; the interface also runs alone in the browser
```

Each layer reads only the outputs of the layer before it. See [docs/architecture.md](docs/architecture.md).

## Data foundation

| Source | Used for |
|---|---|
| ClinVar allele tables joined to gnomAD v2 exomes and genomes, GRCh37 | 341,374 variants, classifications, review status, submissions and population frequencies |
| InterVar evidence database | inheritance, Orphanet and OMIM summaries, gene lists |
| MacArthur Lab curated gene lists | dominant and recessive genes, haploinsufficiency, essential genes |
| ExAC constraint and LoFtool | gene intolerance to loss of function |
| Canonical GENCODE transcript table | exon coordinates, anchored to coding positions for 2,782 genes |
| Human Phenotype Ontology, release of 16 January 2025 | 19,034 terms, 12,687 disease profiles, gene to phenotype map |
| tmVar, SETH, OSIRIS, Variome and MutationFinder corpora | variant mention extraction and the literature layer |

Every file is fetched from a pinned Git commit or PyPI wheel and recorded with its SHA256 in `data/raw/MANIFEST.json`: 1,187 files, 236 MB. See [docs/data_card.md](docs/data_card.md).

## Methods

| Area | Method |
|---|---|
| Population | filtering allele frequency (95% Poisson lower bound, maximum over continental groups), group maximum with an allele number of at least 2,000 for BA1, BA1 exception list |
| Null variants | PVS1 decision tree with nonsense mediated decay prediction from anchored exon models, last exon and penultimate exon rules, start loss, canonical splice sites, and pool evidence when no exon model exists |
| Allelic evidence | PS1 and PM5 from pathogenic variants at the same residue or splice site, with the variant itself excluded |
| Hot spots and gene spectra | PM1 from pathogenic missense density with no benign variation, PP2 and BP1 from each gene's classified spectrum |
| Computational evidence | LightGBM missense and splice region predictors, five fold cross validation grouped by gene, local posterior calibration with prior 0.0441 and bootstrap lower bounds; PP3 and PM1 capped at strong together |
| Case data | de novo points, segregation by meioses, in trans observations with phase, functional evidence, phenotype specificity scored against 300 background genes |
| Combining | Bayesian points with posterior probability (prior 0.10, OddsPath 350), with the 2015 rules shown beside it |
| Tuning | 648 rule configurations searched on the development set only |
| Copilot | guard (redaction, crisis, personal decisions, off topic, injection), query understanding, BM25, SIF and a NumPy dual encoder, entity routing, reciprocal rank fusion, LambdaMART reranker, extractive writer, logistic claim verifier with hard rules, append only audit log |
| Browser | TypeScript port of the engine and models with a 542 variant parity fixture; gzip bundle decompressed with DecompressionStream |

See [docs/acmg_implementation.md](docs/acmg_implementation.md) for every criterion and [docs/model_card.md](docs/model_card.md) for every model.

## Running it

Requirements: Python 3.11 or later, Node 20 or later and about 1 GB of free disk.

```bash
pip install -e ".[dev]"
make all web      # acquire, build, train, index, evaluate on the development set, build the interface
make test         # 21 Python tests and 9 parity tests
make serve        # API and interface on http://localhost:8000
```

The blind sets are already frozen in `data/eval`. `make evaluate_blind` scores them and appends a timestamp to `data/eval/blind_runs.json`, so any later run is visible. To rebuild the browser demo:

```bash
make bundle web   # export the bundle and build web/dist
make artifact     # optional: package for hosts that serve only text and web file types
```

With Docker:

```bash
docker compose up                      # API and interface
docker compose --profile llm up        # also start Ollama; set ALLELION_LLM=ollama for the generative writer
```

The API is documented in [docs/api.md](docs/api.md) and deployment in [docs/deployment.md](docs/deployment.md).

## Repository layout

| Path | Contents |
|---|---|
| `allelion/data` | acquisition with pinned sources and SHA256 verification |
| `allelion/kb` | HGVS parsing, ClinVar and gnomAD join, exon models, genes, phenotypes, literature, knowledge graph |
| `allelion/acmg` | evidence tables, criteria engine, combining, rule tuning, phenotype matching |
| `allelion/models` | features, predictors, calibration, mention extractor, learned verdict |
| `allelion/retrieval` | tokenizer, passages, BM25, dense retrieval, query understanding, reranker |
| `allelion/copilot` | guard, verifier, writers, copilot, audit |
| `allelion/eval` | frozen splits, questions, metrics, evaluation, post hoc analysis |
| `allelion/api` | FastAPI service |
| `allelion/export` | browser bundle, parity fixture, artifact packaging |
| `knowledge` | 76 guidance cards summarising the ACMG/AMP and ClinGen recommendations |
| `web` | React and TypeScript interface with the in browser engine |
| `data/eval` | frozen blind and development sets with their hashes and the run log |
| `reports` | evaluation summary, predictions, post hoc analysis and figures |
| `docs` | architecture, criteria, data card, model card, evaluation, safety, API, deployment, screenshots |
| `tests` | Python tests for HGVS, combining, the engine, the guard and verifier, the frozen sets and the API |

## Intended use and limitations

Allelion Verdict AI is a research and education tool for people who already interpret variants. It shows what public evidence supports and how far it moves a classification. It is not a medical device and must not be used to diagnose, to decide on care, or as the sole basis of a clinical report.

1. The engine interprets the transcript ClinVar displays, which misreads BRCA1 exon 11 truncating variants recorded on a short isoform.
2. The benchmark contains no case level data, so criteria that depend on it were never exercised in evaluation, and exact agreement is capped.
3. Allelic and gene level evidence comes from ClinVar classifications of other variants, some made by the same panels; self exclusion and gene grouped validation limit this but cannot remove it.
4. The blind set covers 17 genes and is 69% ENIGMA; performance on other genes is better estimated by the development set.
5. The data are a 2018 GRCh37 snapshot, with no copy number, structural, mitochondrial or repeat expansion support and no conservation or deep learning splice scores.
6. The learned verdict makes eight times as many severe errors as the point system and is shown only as a second opinion.

The roadmap (MANE Select transcripts, panel specific rule sets, a GRCh38 refresh with a second blind set) is in [docs/evaluation.md](docs/evaluation.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/acmg_implementation.md](docs/acmg_implementation.md) | how each criterion is implemented and which ClinGen recommendation it follows |
| [docs/architecture.md](docs/architecture.md) | layers, make targets and data flow |
| [docs/data_card.md](docs/data_card.md) | sources, knowledge base composition, evaluation sets, known data issues |
| [docs/model_card.md](docs/model_card.md) | intended use, every model, training data, results and ethical considerations |
| [docs/evaluation.md](docs/evaluation.md) | protocol, full results, post hoc analysis, limitations and roadmap |
| [docs/safety.md](docs/safety.md) | redaction, crisis handling, personal decisions, injection defence and audit |
| [docs/api.md](docs/api.md) | API endpoints |
| [docs/deployment.md](docs/deployment.md) | Docker, Ollama and static hosting |

## Licence

The code is released under the MIT licence; see [LICENSE](LICENSE). The data keep their own terms: ClinVar is public domain, gnomAD is released under CC0, the Human Phenotype Ontology requires attribution, InterVar tables are free for academic use, and each literature corpus keeps its original research terms. See [docs/data_card.md](docs/data_card.md).

## References

1. Richards S, Aziz N, Bale S, et al. Standards and guidelines for the interpretation of sequence variants. Genetics in Medicine 2015.
2. Tavtigian SV, Harrison SM, Boucher KM, Biesecker LG. Fitting a naturally scaled point system to the ACMG/AMP variant classification guidelines. Human Mutation 2020.
3. Abou Tayoun AN, Pesaran T, DiStefano MT, et al. Recommendations for interpreting the loss of function PVS1 ACMG/AMP variant criterion. Human Mutation 2018.
4. Pejaver V, Byrne AB, Feng BJ, et al. Calibration of computational tools for missense variant pathogenicity classification and ClinGen recommendations for PP3/BP4 criteria. American Journal of Human Genetics 2022.
5. Walker LC, Hoya M, Wiggins GAR, et al. Using the ACMG/AMP framework to capture evidence related to predicted and observed impact on splicing. American Journal of Human Genetics 2023.
6. Whiffin N, Minikel E, Walsh R, et al. Using high resolution variant frequencies to empower clinical genome interpretation. Genetics in Medicine 2017.
7. Landrum MJ, Lee JM, Benson M, et al. ClinVar: improving access to variant interpretations and supporting evidence. Nucleic Acids Research 2018.
8. Karczewski KJ, Francioli LC, Tiao G, et al. The mutational constraint spectrum quantified from variation in 141,456 humans. Nature 2020.
9. Gargano MA, Matentzoglu N, Coleman B, et al. The Human Phenotype Ontology in 2024. Nucleic Acids Research 2024.
10. Li Q, Wang K. InterVar: clinical interpretation of genetic variants by the 2015 ACMG/AMP guidelines. American Journal of Human Genetics 2017.
