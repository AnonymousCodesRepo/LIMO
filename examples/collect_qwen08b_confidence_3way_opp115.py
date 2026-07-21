"""Collect Qwen3.5-0.8B 3-way confidence features for ALL opp115 (doc, query) pairs.

Same 3-way prompt + logprob extraction as collect_qwen08b_confidence_3way.py,
for the opp115 dataset (which ships a self-contained pairs_ground_truth.csv).

Output: examples/predictions/zeroshot_all_queries/Qwen3.5-0.8B_confidence_3way_opp115.csv

Emits per-point small-model features (p_true/p_false/logprobs/confidence/pred)
used to compute the offline lightgbm router's agreement score.
"""

from __future__ import annotations
import os

import argparse
import csv
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from openai import OpenAI

DATASET_DIR = Path(__file__).resolve().parent.parent / "datasets" / "opp115"
OUTPUT_DIR = Path(__file__).resolve().parent / "predictions" / "zeroshot_all_queries"

MODEL_PATH = os.environ.get("MOP_SMALL_MODEL", "Qwen3.5-0.8B")
BASE_URL = os.environ.get("MOP_SMALL_URL", "http://localhost:8105/v1")
MAX_WORKERS = 8

SYSTEM_PROMPT = (
    "You are a precise data filtering assistant. "
    "Determine whether the given document satisfies the condition. "
    "If you are uncertain, it is better to say so rather than guessing.\n"
    'Respond with ONLY one of:\n'
    '  {"match": true}\n'
    '  {"match": false}\n'
    '  {"match": "unsure"}'
)

TRUE_TOKENS = {"true", "true}"}
FALSE_TOKENS = {"false", "false}"}
UNSURE_TOKENS = {"unsure", '"unsure"', 'unsure"', '"unsure', "unsure}"}


def make_messages(query_description: str, document_text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"Condition: {query_description}\n\nDocument:\n{document_text}"
        )},
    ]


def extract_3way_confidence(client: OpenAI, model_path: str,
                            query_desc: str, doc_text: str) -> dict:
    messages = make_messages(query_desc, doc_text)
    try:
        resp = client.chat.completions.create(
            model=model_path, messages=messages, temperature=0.0, max_tokens=20,
            logprobs=True, top_logprobs=20,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = resp.choices[0].message.content or ""
        try:
            result = json.loads(raw.strip())
            match_val = result.get("match", False)
            if isinstance(match_val, str) and match_val.lower() == "unsure":
                prediction = "unsure"
            else:
                prediction = "Yes" if bool(match_val) else "No"
        except (json.JSONDecodeError, AttributeError):
            lower = raw.lower()
            if "unsure" in lower:
                prediction = "unsure"
            elif "true" in lower and "false" not in lower:
                prediction = "Yes"
            elif "false" in lower:
                prediction = "No"
            else:
                prediction = "PARSE_FAIL"

        raw_lp_true = raw_lp_false = raw_lp_unsure = -100.0
        logprobs_data = resp.choices[0].logprobs
        if logprobs_data and logprobs_data.content:
            for token_info in logprobs_data.content:
                tok_lower = token_info.token.strip().lower()
                is_decision = (
                    tok_lower in TRUE_TOKENS or tok_lower in FALSE_TOKENS or
                    tok_lower in UNSURE_TOKENS or tok_lower == '"'
                )
                if not is_decision:
                    continue
                if tok_lower in TRUE_TOKENS:
                    raw_lp_true = token_info.logprob
                elif tok_lower in FALSE_TOKENS:
                    raw_lp_false = token_info.logprob
                elif tok_lower in UNSURE_TOKENS:
                    raw_lp_unsure = token_info.logprob
                if token_info.top_logprobs:
                    for alt in token_info.top_logprobs:
                        alt_lower = alt.token.strip().lower()
                        if alt_lower in TRUE_TOKENS:
                            raw_lp_true = max(raw_lp_true, alt.logprob)
                        elif alt_lower in FALSE_TOKENS:
                            raw_lp_false = max(raw_lp_false, alt.logprob)
                        elif alt_lower in UNSURE_TOKENS:
                            raw_lp_unsure = max(raw_lp_unsure, alt.logprob)
                break

        lps = [raw_lp_true, raw_lp_false, raw_lp_unsure]
        active = [lp for lp in lps if lp > -100.0]
        if len(active) >= 2:
            max_lp = max(lps)
            exps = [math.exp(lp - max_lp) if lp > -100.0 else 0.0 for lp in lps]
            total = sum(exps)
            p_true, p_false, p_unsure = exps[0]/total, exps[1]/total, exps[2]/total
        elif len(active) == 1:
            p_true = 1.0 if raw_lp_true > -100.0 else 0.0
            p_false = 1.0 if raw_lp_false > -100.0 else 0.0
            p_unsure = 1.0 if raw_lp_unsure > -100.0 else 0.0
        else:
            p_true = p_false = p_unsure = 1/3

        if prediction == "Yes":
            confidence = p_true
        elif prediction == "No":
            confidence = p_false
        elif prediction == "unsure":
            confidence = p_unsure
        else:
            confidence = max(p_true, p_false, p_unsure)

        entropy = 0.0
        for p in [p_true, p_false, p_unsure]:
            if p > 1e-10:
                entropy -= p * math.log2(p)

        return {
            "prediction": prediction, "raw_response": raw.strip()[:200],
            "p_true": p_true, "p_false": p_false, "p_unsure": p_unsure,
            "confidence": confidence, "entropy": entropy,
            "logprob_true": raw_lp_true, "logprob_false": raw_lp_false,
            "logprob_unsure": raw_lp_unsure,
        }
    except Exception as e:
        return {
            "prediction": "PARSE_FAIL", "raw_response": f"ERROR: {e}"[:200],
            "p_true": 1/3, "p_false": 1/3, "p_unsure": 1/3,
            "confidence": 1/3, "entropy": math.log2(3),
            "logprob_true": -100.0, "logprob_false": -100.0, "logprob_unsure": -100.0,
        }


def load_data():
    merged = pd.read_csv(DATASET_DIR / "merged_df.csv")
    docs = {int(r.document_id): {"name": str(r.document_name), "text": str(r.merged_text)}
            for r in merged.itertuples()}
    qmap = pd.read_csv(DATASET_DIR / "query_name_mapping.csv")
    queries = {str(r.query_name).strip(): str(r.query_description).strip()
               for r in qmap.itertuples() if str(r.query_name).strip()}
    pairs = pd.read_csv(DATASET_DIR / "pairs_ground_truth.csv")
    gt, pair_list = {}, []
    for r in pairs.itertuples():
        did, q = int(r.document_id), str(r.query_name)
        gt[(did, q)] = str(r.ground_truth)
        pair_list.append((did, q))
    return docs, queries, gt, pair_list


def load_checkpoint(output_file: Path) -> set:
    done = set()
    if output_file.exists():
        with open(output_file) as f:
            for row in csv.DictReader(f):
                done.add((int(row["document_id"]), row["query_name"]))
    return done


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-docs", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    docs, queries, gt, pair_list = load_data()
    print(f"Loaded {len(docs)} docs, {len(queries)} queries, {len(pair_list)} pairs")

    client = OpenAI(base_url=BASE_URL, api_key="dummy")
    client.chat.completions.create(
        model=MODEL_PATH, messages=[{"role": "user", "content": "hi"}],
        max_tokens=3, temperature=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    print(f"[OK] connected to 0.8B at {BASE_URL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "Qwen3.5-0.8B_confidence_3way_opp115.csv"

    if args.max_docs:
        keep = set(sorted(docs)[:args.max_docs])
        pair_list = [(d, q) for d, q in pair_list if d in keep]

    done = set()
    if args.resume:
        done = load_checkpoint(output_file)
    tasks = [(d, q) for d, q in pair_list if (d, q) not in done]
    if not tasks:
        print("All done!"); return
    total = len(tasks)
    print(f"Total: {total}  Output: {output_file}")

    write_header = not output_file.exists() or not args.resume
    f_out = open(output_file, "a" if args.resume and output_file.exists() else "w", newline="")
    fieldnames = ["document_id", "document_name", "query_name", "ground_truth",
                  "prediction", "correct", "p_true", "p_false", "p_unsure",
                  "confidence", "entropy", "logprob_true", "logprob_false",
                  "logprob_unsure", "raw_response"]
    writer = csv.DictWriter(f_out, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()

    completed = correct_count = unsure_count = 0
    t_start = time.time()

    def predict_one(doc_id, qname):
        res = extract_3way_confidence(client, MODEL_PATH, queries[qname], docs[doc_id]["text"])
        g = gt.get((doc_id, qname), "N/A")
        pm = res["prediction"]
        is_correct = (pm == g) if pm in ("Yes", "No") else False
        return {
            "document_id": doc_id, "document_name": docs[doc_id]["name"],
            "query_name": qname, "ground_truth": g, "prediction": pm,
            "correct": is_correct,
            "p_true": f"{res['p_true']:.6f}", "p_false": f"{res['p_false']:.6f}",
            "p_unsure": f"{res['p_unsure']:.6f}", "confidence": f"{res['confidence']:.6f}",
            "entropy": f"{res['entropy']:.6f}",
            "logprob_true": f"{res['logprob_true']:.5f}",
            "logprob_false": f"{res['logprob_false']:.5f}",
            "logprob_unsure": f"{res['logprob_unsure']:.5f}",
            "raw_response": res["raw_response"],
        }

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futs = {ex.submit(predict_one, d, q): (d, q) for d, q in tasks}
        for fut in as_completed(futs):
            row = fut.result()
            writer.writerow(row); f_out.flush()
            completed += 1
            if row["correct"]:
                correct_count += 1
            if row["prediction"] == "unsure":
                unsure_count += 1
            if completed % 500 == 0 or completed == total:
                el = time.time() - t_start
                print(f"  [{completed}/{total}] acc={correct_count/completed:.4f} "
                      f"unsure={unsure_count/completed:.2%} rate={completed/el:.1f}/s "
                      f"eta={(total-completed)/(completed/el)/60:.1f}min")
    f_out.close()
    print(f"DONE: {completed} preds, acc={correct_count/completed:.4f}, "
          f"unsure={unsure_count/completed:.2%}, {(time.time()-t_start)/60:.1f}min")


if __name__ == "__main__":
    main()
