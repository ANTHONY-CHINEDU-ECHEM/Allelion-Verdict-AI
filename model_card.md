# Model card

Allelion Verdict AI combines a rule engine with six trained components. The rule engine makes the classification; the trained components supply evidence to it (the missense and splice predictors), give a second opinion beside it (the learned verdict), or run the evidence copilot (the extractor, the retrievers, the reranker and the verifier). This card describes each one. All of them were trained and frozen before the blind set was scored: the last model file, the reranker, was written at 18:39 UTC and the single blind run took place at 18:52 UTC on 16 October 2026.

## Intended use

Allelion Verdict AI is a research and education tool for people who already interpret germline variants: clinical scientists, genetic counsellors, bioinformaticians and students of the ACMG/AMP framework. It shows which criteria the public evidence supports for a variant, at what strength and why, and how far that evidence alone moves the classification. It is a starting point for curation and a way to audit a classification, not a source of clinical reports.

It must not be used to make or communicate a diagnosis, to decide on treatment, surveillance or reproductive options, or as the sole basis for a classification in a clinical report. The engine never sees the case level evidence that decides most real classifications, its data date from 2018, and the evaluation shows systematic failures (see [evaluation.md](evaluation.md)).

## Criteria engine

| Item | Detail |
|---|---|
| Type | deterministic rules, implemented twice (Python and TypeScript) with a shared parity fixture of 542 variants |
| Framework | ACMG/AMP 2015 with ClinGen SVI refinements: the PVS1 decision tree (Abou Tayoun 2018), PM2 at supporting strength (2020), PP5 and BP6 retired, BA1 exception list, de novo and PM3 point systems, segregation counting, calibrated computational evidence (Pejaver 2022), splicing guidance (Walker 2023) |
| Combining | Bayesian points (Tavtigian 2020): supporting 1, moderate 2, strong 4, very strong 8; pathogenic at 10 or more, likely pathogenic 6 to 9, uncertain 0 to 5, likely benign minus 1 to minus 6, benign minus 7 or less; posterior from prior 0.10 and OddsPath 350. The 2015 combining rules are reported beside it |
| Tuned settings | seven thresholds selected from 648 configurations on the development set: BS1 frequency 0.003 dominant and 0.01 recessive, PM1 window 10 residues with at least 3 pathogenic missense variants, PP2 needs 5 pathogenic missense variants and at most 10% benign, BS2 needs 5 homozygotes |
| Evidence it uses | population frequency (FAF95, group maximum), exon models, nonsense mediated decay prediction, allelic evidence and hot spot density from the pool, gene spectra, inheritance and constraint, the two predictors, and optional case data entered by the user |
| Evidence it never uses | the ClinVar classification of the variant being evaluated, which is excluded from every evidence table |

## Missense predictor

| Item | Detail |
|---|---|
| Model | LightGBM binary classifier, learning rate 0.05, 31 leaves, at least 40 rows per leaf, early stopping per fold |
| Features | 69: substitution scores (BLOSUM62, Grantham, hydropathy, volume, charge), special residues (proline, glycine, cysteine, aromatic), position in the protein, codon position, transition or transversion, reference and alternate amino acid, and gene features (pLI, LoFtool, haploinsufficiency, essentiality, inheritance, disease count, protein length) |
| Training data | 27,681 pool missense variants in 3,011 genes classified P or LP (12,484) or B or LB |
| Validation | five fold cross validation grouped by gene, so no gene is in both training and validation folds |
| Result | AUROC 0.833 out of fold (best single baseline LoFtool 0.730); AUROC 0.830 on the 558 blind missense variants |
| Calibration | local posterior probability with bootstrap lower bounds, prior 0.0441 (Pejaver 2022). PP3 supporting from 0.671, moderate from 0.803; BP4 supporting below 0.184, moderate below 0.073; strong never reached |
| Most important features | LoFtool, protein length, pLI, Grantham distance, gene disease count |

The model leans on gene level features: it learns which genes tolerate missense change as much as which substitutions are damaging. That is why it misses gain of function variants in genes otherwise tolerant of missense variation (SHOC2 p.Ser2Gly in the blind set), and why its score must be read alongside PP2 and BP1, which also use gene level evidence. The engine caps PP3 and PM1 together at strong strength, as ClinGen SVI recommends, but does not cap PP3 against PP2 or BP4 against BP1, so gene level evidence can be counted twice. The blind ablation without PP2 and BP1 shows the net effect is small.

## Splice region predictor

| Item | Detail |
|---|---|
| Model | LightGBM binary classifier with the same settings |
| Features | 25: distance to the nearest exon boundary and its side, the reference and alternate base, substitution or indel, indel length, and the gene features |
| Training data | 19,192 pool variants between 3 and 50 nucleotides from a coding exon boundary (outside the canonical splice dinucleotides), 741 pathogenic |
| Result | AUROC 0.951 grouped by gene (distance alone 0.845); average precision 0.668 |
| Calibration | PP3 for splicing at supporting from 0.094, moderate from 0.217, strong from 0.644; BP7 support below 0.0086 |

It has no sequence context beyond the changed base, so it cannot detect cryptic splice sites or deep intronic pseudoexon activation. Deep intronic variants are scored as quiet, which feeds BP7.

## Learned verdict

| Item | Detail |
|---|---|
| Model | LightGBM multiclass over the five classes |
| Features | the engine's points, points per criterion, frequency, predictor scores, consequence and gene features |
| Training data | 60,000 pool variants, 12,000 per class |
| Validation | grouped cross validation on the pool: three tier agreement 82.8%, severe discordance 0.89% |
| Result | development three tier 90.2%; blind three tier 83.1% with 573 severe errors (6.56%) |

The learned verdict reproduces what ClinVar submitters tend to conclude given the same features. That is useful as a check on the rules, but it inherits submitter habits rather than the framework, and on the blind set it called 438 benign variants pathogenic. It is shown only as a labelled second opinion.

## Variant mention extractor

A linear chain CRF over word and character features, trained on 838 documents from tmVar and SETH and tested on 292 held out documents. Exact span F1 is 0.827 on tmVar and 0.878 on SETH, against 0.697 and 0.663 for regular expressions alone. On MutationFinder abstracts the combined extractor reaches document level F1 0.82. Extracted mentions link literature to ClinVar variants in the knowledge graph and the copilot; they never feed a classification.

## Retrieval and reranker

| Component | Detail |
|---|---|
| BM25 | 68,738 terms over 37,881 passages, with a tokenizer that keeps HGVS expressions, gene symbols, HPO identifiers and criterion codes whole |
| SIF vectors | word2vec (skip gram) trained on the corpus, smooth inverse frequency weighting with the first principal component removed |
| Dual encoder | two linear projections over SIF vectors trained in NumPy with a symmetric in batch InfoNCE loss on title and passage pairs mined from the corpus; loss fell from 1.22 to 0.62 over 876 steps |
| Entity routing | query understanding finds genes, HGVS, HPO terms, diseases and criterion codes and restricts candidates to passages that mention them |
| Fusion | reciprocal rank fusion of all generators |
| Reranker | LightGBM LambdaMART over 38 features, trained on 2,995 synthetic queries generated from passages; validation nDCG@10 0.991 |
| Result | blind questions: recall@5 0.967, nDCG@10 0.766 (BM25 alone 1.000 and 0.717) |

## Claim verifier

A logistic regression over twelve features comparing a sentence with the evidence it cites: token and bigram recall, best matching span of up to two consecutive evidence sentences, and agreement on numbers, genes, criterion codes, HGVS expressions, negation, class words and antonyms. Hard rules reject a sentence that has no citation, or whose numbers, genes, criterion codes or HGVS expressions are absent from the cited evidence, or that contains a word whose antonym appears in the best matching evidence sentence. Trained on 19,080 sentences and their perturbations (number and gene changes, negation, swapped arguments, wrong citations, paraphrases), tested on 6,274: AUROC 0.992, accuracy 96.5%. Argument swaps are detected only 53.5% of the time.

## Writers

The default writer is extractive: it selects evidence sentences that answer the question, ranks them by overlap with the query focus, entity match and retrieval rank, and cites each. An optional writer calls a local Ollama model and is held to the same verifier; anything the verifier cannot support is marked in the interface. No external API receives a question.

## Ethical considerations

1. **Severity asymmetry.** A pathogenic variant called benign can end surveillance for a person at risk; a benign variant called pathogenic can lead to unnecessary surgery. The engine is tuned to be conservative in both directions, and the interface always shows the criteria, points and the ClinVar reference together so a reader can see disagreement immediately.
2. **Ancestry.** Frequency based benign evidence is weaker for populations under represented in gnomAD v2, and pool based evidence reflects the populations that have been tested most. Variants seen mostly in under represented groups are more likely to stay uncertain or be over called.
3. **Gene coverage.** Performance is measured on 17 genes with expert panels. Most genes in the knowledge base have no expert classified variants and fewer pool variants, so allelic and gene level criteria apply less often.
4. **Personal questions.** The copilot routes crisis language to crisis support (Samaritans, NHS 111, 999) instead of answering. Questions about personal care decisions are answered only with general evidence and a notice that a clinical genetics service or genetic counsellor should advise on the decision.
