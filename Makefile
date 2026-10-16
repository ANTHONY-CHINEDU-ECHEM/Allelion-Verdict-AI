PY ?= python3
.PHONY: all acquire verify kb splits evidence models literature graph tune index copilot evaluate evaluate_blind postmortem bundle artifact web test serve clean_models

all: acquire kb splits evidence models literature graph tune index copilot evaluate bundle

acquire:            ## layer 1: pinned sources with SHA256 manifest
	$(PY) -m allelion.data.acquire

verify:             ## recompute every raw file hash against the manifest
	$(PY) -m allelion.data.acquire --verify

kb:                 ## layer 2: variants, exon models, phenotypes, genes
	$(PY) -m allelion.kb.clinvar
	$(PY) -m allelion.kb.transcripts
	$(PY) -m allelion.kb.phenotypes
	$(PY) -m allelion.kb.genes

splits:             ## freeze the blind and development sets (never overwritten once written)
	test -f data/eval/blind_hash.txt || $(PY) -m allelion.eval.splits
	test -f data/eval/questions_blind_hash.txt || $(PY) -m allelion.eval.questions

evidence:           ## evidence tables from the pool only
	$(PY) -m allelion.acmg.tables

models:             ## in silico predictors with calibration
	$(PY) -m allelion.models.insilico

literature:         ## corpora, mention extractor, literature links
	$(PY) -m allelion.kb.literature
	$(PY) -m allelion.models.extractor
	$(PY) -m allelion.kb.link_literature

graph:
	$(PY) -m allelion.kb.graph

tune:               ## rule thresholds on the development set, then the learned verdict
	$(PY) -m allelion.acmg.tune
	$(PY) -m allelion.models.meta

index:              ## passages, BM25, SIF vectors, dual encoder, reranker
	$(PY) -m allelion.retrieval.build_index
	$(PY) -m allelion.retrieval.reranker

copilot:            ## claim verifier
	$(PY) -m allelion.copilot.verifier

evaluate:           ## development sets only
	$(PY) -m allelion.eval.evaluate

evaluate_blind:     ## blind sets, scored once after every component is frozen
	$(PY) -m allelion.eval.evaluate --blind
	$(PY) -m allelion.figures

postmortem:         ## slices of the stored blind predictions, never re-scores
	$(PY) -m allelion.eval.postmortem

bundle:             ## in browser engine bundle
	$(PY) -m allelion.export.bundle

web:
	cd web && npm ci && npm run build

test:
	$(PY) -m pytest -q tests
	cd web && npx vitest run

serve:
	uvicorn allelion.api.main:app --host 0.0.0.0 --port 8000

artifact:           ## package web/dist for hosts that serve only web and text types (base64 gzip bundle)
	$(PY) -m allelion.export.artifact build/artifact
