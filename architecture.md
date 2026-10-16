# Architecture

Allelion Verdict AI has five layers. Each is rebuilt by one make target and reads only the outputs of the layer before it, so any layer can be inspected, replaced or rerun in isolation.

```
public sources (pinned Git commits and a PyPI wheel, SHA256 manifest)
        │  make acquire
        ▼
knowledge base  ─ variants (ClinVar + gnomAD exomes and genomes, parsed HGVS, consequence, FAF95)
                ─ exon models anchored to coding coordinates
                ─ genes (constraint, inheritance, dosage, diseases)
                ─ phenotypes (HPO ontology, disease profiles, information content)
                ─ literature (five corpora, mentions, links) and knowledge graph
        │  make kb literature graph
        ▼
frozen evaluation sets (blind expert panel variants, development variants, questions; hashed)
        │  make splits (never overwritten)
        ▼
evidence and models ─ pool evidence tables (allelic, splice site, density, truncating, gene spectra)
                    ─ missense and splice region predictors with local posterior calibration
                    ─ tuned rule thresholds and the learned verdict
                    ─ variant mention CRF, passages, BM25, SIF, dual encoder, LambdaMART reranker, verifier
        │  make evidence models tune index copilot
        ▼
services ─ criteria engine (Python and an identical TypeScript port)
         ─ evidence copilot (guard, retrieval, variant evaluation, writer, verifier, audit)
         ─ phenotype matcher, batch triage, conflict explorer
        │  make serve  /  make bundle web
        ▼
delivery ─ FastAPI service and a React interface that also runs entirely in the browser from an exported bundle
```

## Design decisions

**Evidence comes from a pool that excludes the test sets.** Expert panel classifications are the closest public reference standard, so every one of them is held out. The evidence tables that PS1, PM5, PM1, PP2, BP1 and the PVS1 mechanism test draw on are built from the remaining classified variants, and each variant is removed from those tables before its own evaluation. A unit test flips the label of pool variants and checks that their criteria do not change.

**Rules first, learning second.** The ACMG/AMP criteria are evaluated by explicit code with a written rationale for every decision. Machine learning is used where the guidelines themselves call for a model (computational evidence), and the models are calibrated to evidence strengths rather than used as classifiers. A learned verdict over the engine's evidence is reported next to the rules as a comparison, never as a replacement.

**Calibrated, not thresholded by eye.** Predictor scores become evidence strength only through the local posterior procedure recommended by ClinGen, on out of fold predictions from cross validation grouped by gene, so the strength a score earns reflects how often variants with similar scores are pathogenic in genes the model has not seen.

**Exon models from the data.** Decay prediction and exon skipping need the reporting transcript's exon structure in coding coordinates. Instead of shipping a transcript database, the engine reconstructs it from the agreement between genomic and coding positions of thousands of ClinVar variants, exon by exon, and refuses genes where the evidence is inconsistent.

**One contract, two runtimes.** The criteria engine, retrieval pipeline, verifier and phenotype matcher are implemented in Python and in TypeScript. A shared fixture of 600 variants, 16 variants entered by HGVS, case level scenarios, phenotype matches, verifier probabilities and retrieval results is checked in both, so the browser demo is the system, not a mock up.

**Local by default.** Every component trains on a laptop CPU from public data. The copilot's default writer is extractive; a local language model can be plugged in through Ollama, and its sentences go through the same verifier. Nothing is sent to an external service.

## Repository layout

| Path | Contents |
| --- | --- |
| `allelion/data` | acquisition and the SHA256 manifest |
| `allelion/kb` | ClinVar and gnomAD loading, HGVS parsing, exon models, genes, phenotypes, literature, graph |
| `allelion/acmg` | evidence tables, the criteria engine, combining rules, phenotype matcher, threshold tuning |
| `allelion/models` | predictor features and training, calibration, mention extractor, learned verdict |
| `allelion/retrieval` | tokeniser, passages, BM25, dense retrieval, query understanding, reranker |
| `allelion/copilot` | guard, writers, verifier, orchestration and audit |
| `allelion/eval` | frozen sets, concordance metrics, retrieval and answer evaluation, the evaluation runner |
| `allelion/api` | FastAPI service |
| `allelion/export` | browser bundle and the parity fixture |
| `knowledge` | 76 guidance cards written for the copilot |
| `web` | React interface and the TypeScript engine |
| `tests` | Python tests; `web/src/engine/parity.test.ts` for the browser engine |
| `data/eval` | frozen sets, hashes and the blind run counter (committed) |
| `reports` | evaluation summary, predictions and figures |
