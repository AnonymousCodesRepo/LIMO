# Datasets

Preprocessed copies of the three benchmark datasets used in the paper are
shipped in this directory. Each dataset directory contains the merged document
table, the query-name mapping, and (for opp115/hoc) the expert ground-truth
pairs file; for CUAD the reference pairs live in the large-model cache
`examples/predictions/zeroshot_all_queries/Llama-3.1-70B-Instruct_zeroshot_cuad.csv`
(registered in `eval/data.py`).

| Paper name | Directory | Registered name | Docs | Queries | Source |
|---|---|---|---|---|---|
| CUAD | `LegalBench/cuad/` | `cuad` | 482 | 38 | CUAD (Hendrycks et al., NeurIPS 2021 D&B), cleaned split via LegalBench (Guha et al., NeurIPS 2023) |
| Policy | `opp115/` | `opp115` | 3,273 | 9 | OPP-115 (Wilson et al., ACL 2016), cleaned version via LegalBench |
| Cancer | `hoc/` | `hoc` | 1,852 | 10 | Hallmarks of Cancer (Baker et al., Bioinformatics 2016) |

Files per dataset:

* `merged_df.csv` (`merged_df_paragraph.csv` for CUAD) — one row per document
  (`merged_text` / `merged_paragraph` column).
* `query_name_mapping.csv` — query id → natural-language predicate.
* `pairs_ground_truth.csv` — expert labels per (document, query) pair
  (opp115, hoc).
* `document_query_matrix.csv` (CUAD only) — the raw expert label matrix.

`scripts/split_para_into_cuad_contract.py` documents how the CUAD split was
derived from the LegalBench paragraph export.

## Licenses / attribution

These are public academic benchmarks redistributed here for review purposes
only: CUAD (CC BY 4.0), OPP-115 (released for research use — see the Usable
Privacy Policy Project terms), HoC (research use). Please cite the original
papers if you use the data.
