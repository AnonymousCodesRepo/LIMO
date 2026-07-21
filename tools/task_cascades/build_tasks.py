"""Build per-query Task-Cascades task packages from our datasets.

For each query of a dataset this emits:
  tc_data/<dataset>/tasks.json          — query -> {question, prefix, instruction,
                                          suffix, n_docs}
  tc_data/<dataset>/<query>.csv         — doc_id, gold, oracle_pred,
                                          oracle_prompt_tokens, oracle_output_tokens

``oracle_pred`` comes from the cached 70B zero-shot predictions (the dataset's
pairs file), and ``oracle_prompt_tokens`` is the token count of the exact clean
2-way prompt the zs-70B anchor saw — this is what "dev labeling counts as one
70B call" is priced at. Document texts are NOT copied here;
run_tc.py reloads them via eval.data.load_points, so the packages stay small
and committable.

Run (needs the Llama tokenizer dir):
  PYTHONPATH=. $MOP_PYTHON tools/task_cascades/build_tasks.py --dataset cuad
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

LLAMA_DIR_DEFAULT = "models/Meta-Llama-3.1-70B-Instruct"

# Document intro sentence, following upstream's PROMPT_PREFIX_SUFFIX_DICT idiom.
DOC_INTRO = {
    "cuad": "I will give you an excerpt from a legal contract. Here is the excerpt: {text}\n\n",
    "contract": "I will give you an excerpt from a legal contract. Here is the excerpt: {text}\n\n",
    "opp115": "I will give you a segment of a website privacy policy. Here is the segment: {text}\n\n",
    "hoc": "I will give you a biomedical abstract. Here is the abstract: {text}\n\n",
}
SUFFIX = "You must respond with ONLY True or False:"

# Oracle (70B) predictions ALWAYS come from the cached zero-shot 70B CSV, keyed
# by (query_name, document_id). For cuad/contract this file IS the dataset's
# pairs_file; for opp115/hoc the pairs_file is a self-contained
# pairs_ground_truth.csv whose `prediction` column is EMPTY (gold only), so the
# oracle must be read from the separate zero-shot CSV instead.
REPO_PRED = Path(__file__).resolve().parents[2] / "examples" / "predictions" / "zeroshot_all_queries"
ORACLE_PRED_CSV = {
    "cuad": REPO_PRED / "Llama-3.1-70B-Instruct_zeroshot_cuad.csv",
    "contract": REPO_PRED / "Llama-3.1-70B-Instruct_zeroshot_contract.csv",
    "opp115": REPO_PRED / "Llama-3.1-70B-Instruct_zeroshot_opp115.csv",
    "hoc": REPO_PRED / "Llama-3.1-70B-Instruct_zeroshot_hoc.csv",
}


def make_instruction(question: str) -> str:
    q = question.strip().replace("{", "{{").replace("}", "}}")
    return (
        "Your task is to answer the following question about the given document.\n\n"
        f"Question: {q}\n\n"
        "- Return True if the answer to the question is Yes.\n"
        "- Return False if the answer is No."
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DOC_INTRO))
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "tc_data"))
    ap.add_argument("--llama-dir", default=LLAMA_DIR_DEFAULT)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from eval.data import DATASETS, load_points, resolve_query_list
    from pipeline.common.prompts import build_replay_messages

    tl = AutoTokenizer.from_pretrained(args.llama_dir)
    spec = DATASETS[args.dataset]
    queries = resolve_query_list("all_eval", dataset=args.dataset)
    pts = load_points(queries, dataset=args.dataset)

    # oracle predictions from the cached zero-shot 70B CSV (NOT pairs_file: for
    # opp115/hoc pairs_file carries gold with an empty prediction column)
    oracle: dict[tuple[str, int], str] = {}
    with open(ORACLE_PRED_CSV[args.dataset]) as f:
        for r in csv.DictReader(f):
            pred = r["prediction"]
            if pred not in ("Yes", "No"):
                raise SystemExit(
                    f"{ORACLE_PRED_CSV[args.dataset].name}: bad prediction "
                    f"{pred!r} for ({r['query_name']}, {r['document_id']})")
            oracle[(r["query_name"], int(r["document_id"]))] = pred

    out = Path(args.out_dir) / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    tasks: dict[str, dict] = {}
    by_q: dict[str, list] = {}
    for p in pts:
        by_q.setdefault(p.query_name, []).append(p)

    for q in queries:
        rows = by_q.get(q, [])
        if not rows:
            print(f"[warn] {q}: no docs, skipped")
            continue
        question = rows[0].query_description
        tasks[q] = {
            "question": question,
            "prefix": DOC_INTRO[args.dataset],
            "instruction": make_instruction(question),
            "suffix": SUFFIX,
            "n_docs": len(rows),
        }

        prompts = [
            tl.apply_chat_template(
                build_replay_messages(p.doc_text, p.query_description,
                                      system_base=spec.large_system_prompt),
                add_generation_prompt=True, tokenize=False)
            for p in rows
        ]
        ntok = [len(ids) for ids in tl(prompts, add_special_tokens=False)["input_ids"]]

        with open(out / f"{q}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["doc_id", "gold", "oracle_pred",
                        "oracle_prompt_tokens", "oracle_output_tokens"])
            for p, n in zip(rows, ntok):
                pred = oracle.get((q, p.doc_id))
                if pred is None:
                    raise SystemExit(
                        f"missing cached 70B prediction for ({q}, doc {p.doc_id})")
                otok = len(tl(pred, add_special_tokens=False)["input_ids"])
                w.writerow([
                    p.doc_id,
                    1 if p.ground_truth == "Yes" else 0,
                    1 if pred == "Yes" else 0,
                    n, otok,
                ])
        print(f"{q}: {len(rows)} docs")

    with open(out / "tasks.json", "w") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)
    print(f"wrote {out}/tasks.json ({len(tasks)} queries)")


if __name__ == "__main__":
    main()
