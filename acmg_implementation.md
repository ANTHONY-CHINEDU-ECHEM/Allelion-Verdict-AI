# How each criterion is implemented

This document describes, criterion by criterion, exactly what the engine in `allelion/acmg/engine.py` (and its browser port in `web/src/engine/engine.ts`) does. Thresholds marked *tuned* were chosen on the development set only (`allelion/acmg/tune.py`, values in `data/models/rules.json`); everything else follows the published recommendation cited.

Two principles hold throughout. First, the evidence pool (ClinVar variants with assertion criteria, a five tier class and no conflict, excluding every blind and development variant) is the only source of evidence drawn from other variants, and the variant under review is removed from every pool table before its own criteria are evaluated. Second, PP5 and BP6 are never used, following the 2018 SVI recommendation, and the variant's own ClinVar classification is never an input.

## Population criteria

| Criterion | Rule |
| --- | --- |
| BA1 | Allele frequency above 5% in any continental group of gnomAD v2 (exomes and genomes combined) with at least 2000 observed alleles. A short exception list (HFE p.Cys282Tyr and p.His63Asp, GJB2 p.Val37Ile, BTD p.Asp444His) is honoured. BA1 alone gives Benign. |
| BS1 | Filtering allele frequency (the lower bound of the one sided 95% Poisson interval of the allele frequency, maximum over the AFR, AMR, EAS, SAS and NFE groups) above a threshold that depends on the gene's recorded inheritance: dominant 0.003, recessive 0.01, undetermined 0.005 (*tuned*). |
| PM2 | Absent from gnomAD exomes and genomes, or for recessive only genes a filtering allele frequency below 0.0001; always applied at Supporting strength (SVI 2020). |
| BS2 | For recessive only genes without adult onset disorders, at least five homozygous gnomAD individuals (*tuned*), or hemizygous males for X linked recessive genes. Not applied to dominant genes because reduced penetrance makes healthy heterozygotes uninformative. |

When a variant is entered by HGVS without population data, all four are reported as not evaluated rather than guessed.

## PVS1 (decision tree after Abou Tayoun and colleagues 2018)

1. The variant must be a predicted null allele: nonsense, frameshift, canonical plus or minus one or two splice site, or initiation codon.
2. Loss of function must be an established mechanism for the gene: curated haploinsufficiency, or at least three pathogenic truncating variants in the pool, or at least one for a recessive gene. It is refused when pathogenic variation is predominantly missense (at least 20 pathogenic missense and under 10% truncating), the signature of gain of function or dominant negative genes such as MYH7 and the RASopathy genes.
3. Nonsense and frameshift: the premature stop codon is located (for frameshifts, the new stop given by fsTer), mapped onto the anchored exon model, and tested against the 50 nucleotide rule. Predicted decay gives Very strong. Escape gives Strong when pathogenic truncating variants lie further downstream (the removed region is critical) or when more than 10% of the protein is removed, and Moderate otherwise. Without an exon model, downstream pathogenic truncation keeps Very strong and its absence gives Strong.
4. Canonical splice sites: the affected exon is located; if its coding length is not a multiple of three, skipping it shifts the frame (Very strong); an in frame skip is Strong when it removes more than 10% of the coding sequence or a region with pathogenic missense variation, and Moderate otherwise.
5. Initiation codon variants: Moderate.

### Exon models anchored to coding coordinates

GENCODE canonical exon models carry genomic coordinates only, and the GENCODE canonical transcript is not always the reporting transcript. Every exonic ClinVar substitution gives a pair of genomic and c. positions, so each exon is placed in c. space by the modal offset of its anchors. Exons absent from the reporting transcript appear as a jump equal to their length and are dropped; alternative splice sites are reconciled within 300 bases; a gene is accepted only when at least 90% of anchors agree and the resulting exons are contiguous. 2,782 genes pass, including BRCA1, BRCA2, MLH1, MSH2, MSH6, PMS2, CFTR and MYH7.

## Allelic criteria

| Criterion | Rule |
| --- | --- |
| PS1 (missense) | The same amino acid change from a different nucleotide change is pathogenic or likely pathogenic in the pool: Strong if any match is Pathogenic, Moderate if all are Likely pathogenic. |
| PS1 (splicing) | A different change at the same intronic position is pathogenic or likely pathogenic in the pool (SVI splicing 2023): Moderate at the canonical dinucleotides, Supporting elsewhere. |
| PM5 | A different missense change at the same residue is pathogenic or likely pathogenic: Strong with two or more distinct pathogenic changes, Moderate with one Pathogenic, Supporting when only Likely pathogenic comparators exist. Not combined with PS1. |

## Region and protein length criteria

| Criterion | Rule |
| --- | --- |
| PM1 | Within 10 residues (*tuned*) at least 3 pathogenic or likely pathogenic missense variants (*tuned*) and no benign missense variants: Moderate; a weaker cluster (at least 2) gives Supporting. Capped together with PP3 at 4 points. |
| PM4 | In frame indels and stop loss variants without benign in frame variation within 10 residues: Moderate. Not combined with PVS1. |
| BP3 | In frame indels with benign in frame indels within 10 residues in the pool (a proxy for repetitive or tolerant sequence): Supporting. |

## Gene level criteria

| Criterion | Rule |
| --- | --- |
| PP2 | Missense variant in a gene with at least 5 pathogenic missense variants (*tuned*) and a benign share of classified missense variants of at most 10%. |
| BP1 | Missense variant in a gene with at least 10 pathogenic truncating variants where missense makes up at most 5% of pathogenic variants. |

## Computational criteria

| Criterion | Rule |
| --- | --- |
| PP3, BP4 (missense) | The Allelion missense predictor (LightGBM on substitution properties, protein position, codon change and ClinVar independent gene constraint) is calibrated with the local posterior method of Pejaver and colleagues (2022) on out of fold scores from gene grouped cross validation. A strength is awarded only where the one sided 95% bound clears the posterior implied by the likelihood ratio thresholds 2.406, 5.790, 33.53 and 1124 at prior 0.0441. Capped at Strong. Not combined with PVS1. |
| PP3, BP4 (splice region) | A splice region predictor for intronic offsets 3 to 50, calibrated the same way. |
| BP7 | Synonymous variants outside the exonic splice region (not in the last three bases or the first base of an exon); intronic variants at or beyond plus 7 or minus 21 with a benign splice prediction; deep intronic variants beyond 50 bases. |

## Case level criteria (supplied by the curator)

| Criterion | Rule |
| --- | --- |
| PS2, PM6 | SVI de novo points: confirmed with a specific phenotype 2, confirmed with a consistent phenotype 1, assumed 1 or 0.5. Supporting 0.5, Moderate 1, Strong 2, Very strong 4. |
| PM3 | SVI in trans points: in trans with Pathogenic 1, with Likely pathogenic 0.5, phase unknown 0.5 or 0.25, homozygous 0.5. Same strength scale. |
| PP1 | Informative meioses: 3 Supporting, 5 Moderate, 7 Strong. |
| BS4 | Non segregation in affected relatives: Strong. |
| PS3, BS3 | Functional assay with the strength the curator assigns under the SVI OddsPath framework. |
| PS4 | Case control odds ratio of at least 5 with a lower confidence limit above 1 (Strong), or proband counting (1 Supporting, 2 Moderate, 4 Strong). |
| PP4 | Best match average Resnik similarity of the patient's HPO terms to the gene's disease profiles, empirical p below 0.01 against 300 background genes, and a best matching disease with at most two associated genes. |
| BP5, BP2 | Alternate molecular cause; phase with a pathogenic variant. |

## Combining

The Bayesian point system is the primary result (Supporting 1, Moderate 2, Strong 4, Very strong 8; Pathogenic at 10 or more, Likely pathogenic 6 to 9, Uncertain 0 to 5, Likely benign minus 1 to minus 6, Benign minus 7 or fewer). The posterior probability is reported with prior 0.10 and OddsPath 350 raised to points over 8. The 2015 combining rules are computed alongside.
