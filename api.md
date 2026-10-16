# API

Start the service with `make serve` (or `uvicorn allelion.api.main:app --port 8000`). Interactive documentation is served at `/docs`.

| Method and path | Purpose |
| --- | --- |
| `GET /api/health` | liveness |
| `POST /api/classify` | classify a variant; body `{"variation_id": "..."}` or `{"gene": "MYH7", "hgvs_c": "c.1208G>A", "hgvs_p": "p.Arg403Gln", "population": {...}, "case": {...}}` |
| `GET /api/variants/search?q=&gene=&limit=` | find knowledge base variants by gene and HGVS fragment |
| `GET /api/genes/{symbol}` | gene record, pool statistics, missense positions, truncating positions and exon model |
| `POST /api/copilot` | `{"question": "..."}`; returns redacted question, notices, cited sentences with verification, evidence and engine result |
| `POST /api/phenotype/match` | `{"gene": "FBN1", "terms": ["HP:0001166"]}`; score, empirical p and specificity for PP4 |
| `POST /api/phenotype/rank` | `{"terms": [...]}`; genes ranked by phenotype similarity |
| `POST /api/batch` | `{"lines": ["MLH1 c.350C>T", ...]}`; up to 500 variants |
| `GET /api/conflicts?gene=` | conflicting ClinVar variants with submitter calls and the engine verdict |
| `GET /api/calibration` | both calibration curves and thresholds |
| `GET /api/evaluation` | the evaluation summary |
| `GET /api/audit` | recent audit records |

## Case evidence in `/api/classify`

```json
{
  "gene": "FBN1", "hgvs_c": "c.3037G>A",
  "case": {
    "de_novo": [{"confirmed": true, "phenotype": "specific"}],
    "in_trans": [{"partner_class": "Pathogenic", "phase": "trans"}],
    "segregation_meioses": 5,
    "nonsegregation": false,
    "functional": {"direction": "damaging", "strength": "Moderate", "assay": "fibrillin secretion assay"},
    "probands": 3,
    "phenotype_terms": ["HP:0001166", "HP:0001083"],
    "alternate_cause": false
  }
}
```

## Response of `/api/classify`

`criteria` lists all 26 criteria with `code`, `title`, `direction`, `status` (`met`, `not_met`, `not_evaluated`), `strength`, `label` (for example `PVS1_Strong`), `rationale` and `evidence`. The top level adds `met`, `points`, `classification`, `classification_2015`, `posterior`, `scores` (missense and splice region), `notes`, `learned_verdict`, `reference` (the ClinVar class, for comparison only) and `population`.
