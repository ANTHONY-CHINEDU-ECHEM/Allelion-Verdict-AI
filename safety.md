# Safety, privacy and intended use

## Intended use

Allelion Verdict AI is a research and education tool for people who already interpret variants: clinical scientists, genetic counsellors in training, bioinformaticians and students. It drafts evidence summaries and shows how each ACMG/AMP criterion would apply. It is not a medical device, it has not been validated for clinical reporting, and its output must not be used for individual clinical decisions.

## Guard (runs before retrieval and before logging)

| Control | Behaviour |
| --- | --- |
| Identifier redaction | NHS numbers (with or without the label), other long numeric identifiers, dates of birth and other dates, email addresses, UK phone numbers, postcodes, hospital record numbers, titled names and ages above 89 are replaced by a labelled placeholder. HGVS descriptions, rs identifiers, HPO and OMIM identifiers and PubMed identifiers are protected from redaction. |
| Crisis language | First person language about self harm routes to a fixed message with UK crisis contacts instead of retrieval. |
| Personal decisions | Questions about the asker's own or a relative's care get a fixed notice that the tool cannot advise on individual decisions; the answer stays at the level of published evidence. |
| Off topic | Questions with no genetics or medicine content are declined with a description of what the copilot covers. |
| Prompt injection | Instruction patterns are removed from the question and from every passage before anything reaches a language model, and a notice is shown. |

## Grounding

Every answer sentence must cite a numbered evidence item. A verifier scores each sentence against the item it cites and hard rules override the score: a number, gene symbol, criterion code or HGVS description not present in the cited evidence, a word that contradicts the best matching evidence sentence, or a missing citation always marks the sentence unsupported. Unsupported sentences are shown underlined, never hidden, and lower the confidence label.

## Audit

The server appends one JSON line per question to `data/audit/audit.jsonl`: time, the redacted question, the SHA256 hash of the original text, the action taken, the writer, the number of evidence items, the verified share, the confidence and the classification if a variant was evaluated. The original text is never stored. The browser demo keeps the same record in local storage.

## Known risks

The engine has no access to case level data unless a curator enters it, so pathogenic variants frequently land in likely pathogenic or uncertain; this is conservative by design but means the tool understates certainty for well characterised variants. The evidence pool reflects ClinVar submission biases: heavily tested genes, founder variants and laboratory specific practices. The missense predictor relies partly on gene level constraint, which overlaps with gene level criteria (PP2), and this overlap is disclosed in the rationale.
