"""Shape our datasets into the format ScaleDoc's code expects.

ScaleDoc baseline port. This is the pure-pandas half (no torch / no embedding
server), so it runs anywhere.

Per dataset it writes, under ``--out`` (default scratchpad/scaledoc_data):
  doc/{ds}.json           : [{"id": <document_id>, "content": <text>}, ...]  (all docs)
  query.json              : {<ds>: [{"q_id": i, "query": <query_description>}, ...]}
  gt/{ds}_res_{qid}.json  : [[doc_id, {"label": "Yes"/"No", "prompt_tokens": t,
                                       "completion_tokens": 1}], ...]  sorted by doc_id
                            label = the *70B* zero-shot prediction (our large model
                            plays ScaleDoc's "oracle"); prompt_tokens = 70B input
                            tokens when known (for USD), else -1.
  gold/{ds}_q{qid}.json   : {str(doc_id): 0/1}  expert gold (for vs-gold eval only)
  meta/{ds}.json          : {qid: query_name}, n_docs, avg_large_in (USD), notes

The oracle labels come from the cached 70B predictions
(examples/predictions/zeroshot_all_queries/Llama-3.1-70B-Instruct_zeroshot_{ds}.csv),
so no live API call is ever made. Gold is the ``ground_truth`` column of the same
cache. doc_id is contiguous 0..N-1 for every dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PRED = REPO / "examples" / "predictions" / "zeroshot_all_queries"
TOKENS_DIR = REPO / "runs" / "cost_2026_06_22"

# dataset -> (merged_df dir, merged file, text column, 70B cache csv stem,
#             token-csv stem used to estimate avg 70B input tokens for USD)
DATASETS = {
    "cuad": ("datasets/LegalBench/cuad", "merged_df_paragraph.csv", "merged_paragraph",
             "Llama-3.1-70B-Instruct_zeroshot_cuad", "legalbench_para_tokens"),
    "contract": ("datasets/LegalBench/contract", "merged_df_paragraph.csv", "merged_paragraph",
                 "Llama-3.1-70B-Instruct_zeroshot_contract", "legalbench_para_tokens"),
    "opp115": ("datasets/opp115", "merged_df.csv", "merged_text",
               "Llama-3.1-70B-Instruct_zeroshot_opp115", "opp115_tokens"),
    "hoc": ("datasets/hoc", "merged_df.csv", "merged_text",
            "Llama-3.1-70B-Instruct_zeroshot_hoc", "hoc_tokens"),
    "sembench_medical": ("datasets/sembench_medical", "merged_df.csv", "merged_text",
                         "Llama-3.1-70B-Instruct_zeroshot_sembench_medical", None),
    "fever": ("datasets/fever", "merged_df.csv", "merged_text",
              "Llama-3.1-70B-Instruct_zeroshot_fever", None),
    # para (long-context) legalbench: 604 docs, 52 queries (38 cuad_ + 14
    # contract_nli_). Run ScaleDoc here, then split results by query prefix
    # into the cuad / contract_nli panels.
    "para": ("datasets/LegalBench", "merged_df_paragraph.csv", "merged_paragraph",
             "Llama-3.1-70B-Instruct_zeroshot", "legalbench_para_tokens"),
}


def _yn(v: str) -> str:
    return "Yes" if str(v).strip().lower() == "yes" else "No"


def avg_large_in(token_stem: str | None, docs: dict[int, str]) -> tuple[float, str]:
    """Average 70B input tokens per (doc,query) for the USD axis.

    Reuse our committed token CSVs where available; otherwise fall back to a
    char/4 estimate over the document text (flagged approximate)."""
    if token_stem is not None:
        p = TOKENS_DIR / f"{token_stem}.csv"
        if p.exists():
            t = pd.read_csv(p)
            return float(t["large_in"].mean()), f"avg from {token_stem}.csv"
    # fallback: ~4 chars/token over doc text + ~25 for the system+question template
    est = sum(len(c) for c in docs.values()) / max(1, len(docs)) / 4.0 + 25.0
    return float(est), "char/4 estimate (approximate)"


def build(ds: str, out: Path, query_json: dict, cache_prefix: str | None = None) -> None:
    doc_dir, merged_file, text_col, cache_stem, token_stem = DATASETS[ds]
    if cache_prefix is not None:
        # swap the oracle (large-model) label source, e.g. the 397B canonical
        # cache `Qwen3.5-397B-A17B_zeroshot_{ds}` in place of the 70B default.
        cache_stem = f"{cache_prefix}_{ds}"
    merged = pd.read_csv(REPO / doc_dir / merged_file)
    docs = {int(r.document_id): str(getattr(r, text_col)) for r in merged.itertuples()}

    cache = pd.read_csv(PRED / f"{cache_stem}.csv")
    for col in ("document_id", "query_name", "ground_truth", "prediction"):
        if col not in cache.columns:
            raise SystemExit(f"{ds}: cache missing column {col!r} ({list(cache.columns)})")

    qnames = sorted(cache["query_name"].dropna().unique().tolist())
    qmap = pd.read_csv(REPO / doc_dir / "query_name_mapping.csv")
    q_desc = dict(zip(qmap["query_name"].str.strip(), qmap["query_description"].str.strip()))

    (out / "doc").mkdir(parents=True, exist_ok=True)
    (out / "gt").mkdir(parents=True, exist_ok=True)
    (out / "gold").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "embeds").mkdir(parents=True, exist_ok=True)

    # documents (all, document_id order)
    doc_list = [{"id": i, "content": docs[i]} for i in sorted(docs)]
    (out / "doc" / f"{ds}.json").write_text(json.dumps(doc_list))

    # queries + per-query gt/gold
    query_json[ds] = []
    qid_to_name = {}
    for qid, qn in enumerate(qnames):
        query_json[ds].append({"q_id": qid, "query": q_desc.get(qn, qn)})
        qid_to_name[qid] = qn
        sub = cache[cache["query_name"] == qn]
        gt_rows, gold = [], {}
        for r in sub.itertuples():
            did = int(r.document_id)
            if did not in docs:
                continue
            gt_rows.append([did, {"label": _yn(r.prediction),
                                   "prompt_tokens": -1, "completion_tokens": 1}])
            gold[str(did)] = 1 if str(r.ground_truth).strip().lower() == "yes" else 0
        gt_rows.sort(key=lambda x: x[0])
        (out / "gt" / f"{ds}_res_{qid}.json").write_text(json.dumps(gt_rows))
        (out / "gold" / f"{ds}_q{qid}.json").write_text(json.dumps(gold))

    avg_in, note = avg_large_in(token_stem, docs)
    (out / "meta" / f"{ds}.json").write_text(json.dumps({
        "dataset": ds,
        "n_docs": len(docs),
        "n_queries": len(qnames),
        "qid_to_name": qid_to_name,
        "avg_large_in_tokens": avg_in,
        "avg_large_in_note": note,
        "text_col": text_col,
    }, indent=2))
    print(f"{ds:16s} docs={len(docs):5d} queries={len(qnames):3d} "
          f"avg_large_in={avg_in:7.1f}  ({note})")


def main() -> None:
    default_out = ("/private/tmp/claude-501/-Users-user-Github-Projects-mop-research/"
                   "4a2b6136-56d7-407d-8b6a-9aacf92fa443/scratchpad/scaledoc_data")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--cache-prefix", default=None,
                    help="override the oracle label cache stem to "
                         "f'{prefix}_{ds}' (e.g. 'Qwen3.5-397B-A17B_zeroshot' "
                         "to run with 397B as the large model). Default = the "
                         "per-dataset Llama-3.1-70B stem.")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    query_json: dict = {}
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        build(ds, out, query_json, cache_prefix=args.cache_prefix)
    (out / "query.json").write_text(json.dumps(query_json, indent=2))
    print(f"\nwrote ScaleDoc-format data to {out}")


if __name__ == "__main__":
    main()
